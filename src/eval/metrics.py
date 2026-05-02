"""Official ICBHI binary evaluation: Se / Sp / Score."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


NORMAL = 0
ABNORMAL_CLASSES = (1, 2, 3)  # Crackle, Wheeze, Both
CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def compute_metrics(
    preds: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Return Se, Sp, Score (all in [0, 1]) given predicted and true class indices."""
    preds = _to_numpy(preds).astype(int)
    labels = _to_numpy(labels).astype(int)

    normal_mask = labels == NORMAL
    abnormal_mask = ~normal_mask

    total_normal = int(normal_mask.sum())
    total_abnormal = int(abnormal_mask.sum())

    correct_normal = int(((preds == NORMAL) & normal_mask).sum())
    correct_abnormal = int((abnormal_mask & (preds != NORMAL)).sum())

    se = correct_abnormal / total_abnormal if total_abnormal > 0 else 0.0
    sp = correct_normal / total_normal if total_normal > 0 else 0.0
    score = (se + sp) / 2.0

    return {"se": se, "sp": sp, "score": score}


def confusion_matrix_4class(
    preds: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """Return a (4, 4) confusion matrix (rows=true, cols=pred) via sklearn."""
    preds = _to_numpy(preds).astype(int)
    labels = _to_numpy(labels).astype(int)
    return confusion_matrix(labels, preds, labels=[0, 1, 2, 3])


def format_metrics(m: dict[str, float]) -> str:
    return f"Se={m['se']*100:.2f}%  Sp={m['sp']*100:.2f}%  Score={m['score']*100:.2f}%"
