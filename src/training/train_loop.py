"""Full training loop: supports AST and CNN14 backbones.

Config keys:
    model_type      str     "ast" | "cnn14"              (default: "ast")
    lr              float   AdamW learning rate           (baseline: 1e-5)
    wd              float   AdamW weight decay            (baseline: 1e-4)
    rho             float   SAM perturbation radius       (baseline: 0.05)
    batch_size      int     training batch size           (baseline: 8)
    epochs          int     number of training epochs
    seed            int     random seed
    checkpoint      str     HuggingFace model ID          (ast/beats only)
    save_dir        str     directory to write .pt files  (optional)
    augment         bool    enable SpecAugment            (ast only)
    augment_kwargs  dict    SpecAugment parameters
    label_smoothing float   CrossEntropyLoss smoothing    (default: 0.0)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.eval.metrics import compute_metrics, confusion_matrix_4class, format_metrics
from src.training.sam import SAM
from src.training.sampler import make_sampler


def _build_model_and_datasets(config: dict, cache_path: Path, device: torch.device):
    """Return (model, train_ds, val_ds, cache) based on config model_type."""
    model_type = config.get("model_type", "ast")

    if model_type == "cnn14":
        from src.data.waveform_dataset import build_waveform_datasets
        from src.models.cnn14_model import build_cnn14_model
        train_ds, val_ds, cache = build_waveform_datasets(cache_path)
        model = build_cnn14_model(device=str(device))

    else:  # ast (default)
        from src.data.icbhi_dataset import build_datasets
        from src.models.ast_model import CHECKPOINT as AST_CKPT, build_model
        augment = config.get("augment", False)
        augment_kwargs = config.get("augment_kwargs", {})
        train_ds, val_ds, cache = build_datasets(cache_path, augment=augment,
                                                  augment_kwargs=augment_kwargs)
        ckpt = config.get("checkpoint", AST_CKPT)
        model = build_model(checkpoint=ckpt)

    return model.to(device), train_ds, val_ds, cache


def _forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Generic forward: supports HF models (.logits) and plain nn.Modules."""
    out = model(x) if not hasattr(model, "module") else model(x)
    if hasattr(out, "logits"):
        return out.logits
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    config: dict,
    cache_path: str | Path,
    device: torch.device | str = "cpu",
) -> dict:
    """Run the full training loop. Returns a results dict with per-epoch metrics
    and the best checkpoint path (if save_dir is set in config).
    """
    device = torch.device(device)
    set_seed(config["seed"])

    model, train_ds, val_ds, cache = _build_model_and_datasets(config, cache_path, device)
    sampler = make_sampler(cache["y_train"])

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"] * 4,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"using {n_gpus} GPUs via DataParallel")

    # optimizer
    sam = SAM(
        model.parameters(),
        base_optimizer=torch.optim.AdamW,
        rho=config["rho"],
        lr=config["lr"],
        weight_decay=config["wd"],
    )
    label_smoothing = config.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # training loop
    save_dir = Path(config["save_dir"]) if config.get("save_dir") else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_score = -1.0
    best_ckpt_path = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"epoch {epoch:02d}/{config['epochs']}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            # SAM pass 1: forward + backward, then perturb weights
            loss = criterion(_forward(model, x), y)
            loss.backward()
            sam.first_step(zero_grad=True)
            running_loss += loss.item()

            # SAM pass 2: forward + backward at perturbed point, then restore + update
            criterion(_forward(model, x), y).backward()
            sam.second_step(zero_grad=True)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="val", leave=False):
                logits = _forward(model, x.to(device))
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels.append(y)

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        m = compute_metrics(preds, labels)
        m["epoch"] = epoch
        m["loss"] = avg_loss
        history.append(m)

        print(f"[{epoch:02d}/{config['epochs']}] loss={avg_loss:.4f}  {format_metrics(m)}")

        # checkpoint
        if save_dir and m["score"] > best_score:
            best_score = m["score"]
            best_ckpt_path = save_dir / f"best_score_epoch{epoch:02d}.pt"
            weights = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({"epoch": epoch, "model": weights, "metrics": m}, best_ckpt_path)

    cm = confusion_matrix_4class(preds, labels)
    return {"history": history, "confusion_matrix": cm, "best_checkpoint": best_ckpt_path}
