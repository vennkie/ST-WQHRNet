import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds[mask] == targets[mask]).float().mean().item()
    return correct


def macro_f1_from_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, num_classes: int) -> float:
    if mask.sum().item() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    preds = preds[mask].view(-1)
    targets = targets[mask].view(-1)

    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))
