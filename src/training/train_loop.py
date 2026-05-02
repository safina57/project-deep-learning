"""Full training loop: AST + WeightedRandomSampler + SAM + ICBHI binary eval.

Config keys expected:
    lr          float   AdamW learning rate          (baseline: 1e-5)
    wd          float   AdamW weight decay            (baseline: 1e-4)
    rho         float   SAM perturbation radius       (baseline: 0.05)
    batch_size  int     training batch size           (baseline: 8)
    epochs      int     number of training epochs     (baseline: 20)
    seed        int     random seed                   (baseline: 42)
    checkpoint  str     HuggingFace model ID          (optional)
    save_dir    str     directory to write .pt files  (optional)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.icbhi_dataset import build_datasets
from src.eval.metrics import compute_metrics, confusion_matrix_4class, format_metrics
from src.models.ast_model import CHECKPOINT, build_model
from src.training.sam import SAM
from src.training.sampler import make_sampler


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

    train_ds, val_ds, cache = build_datasets(cache_path)
    sampler = make_sampler(cache["y_train"])

    # data
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

    # model
    ckpt = config.get("checkpoint", CHECKPOINT)
    model = build_model(checkpoint=ckpt).to(device)

    # optimizer
    sam = SAM(
        model.parameters(),
        base_optimizer=torch.optim.AdamW,
        rho=config["rho"],
        lr=config["lr"],
        weight_decay=config["wd"],
    )
    criterion = nn.CrossEntropyLoss()

    # training loop
    save_dir = Path(config["save_dir"]) if config.get("save_dir") else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_se = -1.0
    best_ckpt_path = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # SAM pass 1: forward + backward, then perturb weights
            loss = criterion(model(input_values=x).logits, y)
            loss.backward()
            sam.first_step(zero_grad=True)
            running_loss += loss.item()

            # SAM pass 2: forward + backward at perturbed point, then restore + update
            criterion(model(input_values=x).logits, y).backward()
            sam.second_step(zero_grad=True)

        avg_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(input_values=x.to(device)).logits
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
        if save_dir and m["se"] > best_se:
            best_se = m["se"]
            best_ckpt_path = save_dir / f"best_se_epoch{epoch:02d}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": m}, best_ckpt_path)

    cm = confusion_matrix_4class(preds, labels)
    return {"history": history, "confusion_matrix": cm, "best_checkpoint": best_ckpt_path}
