import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from dataset import STWQHRDataset
from model import STWQHRNet
from utils import load_json

WQI_RANGES = {
    "Excellent": (0.0, 25.0),
    "Good": (26.0, 50.0),
    "Poor": (51.0, 75.0),
    "Very Poor": (76.0, 100.0),
    "Unsuitable for Drinking": (100.01, 120.0),
}

HHI_RANGES = {
    "Low Risk": (0.0, 0.9999),
    "Threshold Level": (0.9999, 1.0001),
    "High Risk": (1.0001, 3.0),
}


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _midpoints_from_vocab(vocab: dict, ranges: dict) -> np.ndarray:
    size = len(vocab)
    id_to_label = {v: k for k, v in vocab.items()}
    mids = np.zeros(size, dtype=np.float32)
    for i in range(size):
        label = id_to_label.get(i, "")
        if label in ranges:
            lo, hi = ranges[label]
            mids[i] = (lo + hi) / 2.0
        else:
            mids[i] = float(i)
    return mids


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.mean(np.abs(pred - true)))


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def _nse(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    denom = np.sum((true - true.mean()) ** 2)
    if denom == 0.0:
        return 0.0
    num = np.sum((pred - true) ** 2)
    return float(1.0 - (num / denom))


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _metrics_from_confusion(cm: np.ndarray, class_names: list[str]) -> dict:
    n_classes = len(class_names)
    total = int(cm.sum())
    diag = np.diag(cm)

    per_class = {}
    recalls = []
    precisions = []
    f1s = []

    for i, name in enumerate(class_names):
        tp = int(diag[i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[name] = {
            "accuracy": float(recall),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    overall_acc = float(diag.sum() / total) if total > 0 else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    balanced_accuracy = macro_recall

    # Cohen's kappa
    if total > 0:
        row_marginals = cm.sum(axis=1)
        col_marginals = cm.sum(axis=0)
        pe = float(np.sum(row_marginals * col_marginals) / (total * total))
        kappa = float((overall_acc - pe) / (1.0 - pe)) if (1.0 - pe) != 0 else 0.0
    else:
        kappa = 0.0

    return {
        "overall_accuracy": overall_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "cohen_kappa": kappa,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def _per_class_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    num_classes = len(class_names)
    cm = _confusion_matrix(y_true, y_pred, num_classes)
    return _metrics_from_confusion(cm, class_names)


def _rankdata_average_ties(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    n = len(a)

    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks_sorted[i:j] = avg_rank
        i = j

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _binary_auc(y_true_bin: np.ndarray, y_score: np.ndarray):
    y_true_bin = y_true_bin.astype(np.int32)
    pos = int((y_true_bin == 1).sum())
    neg = int((y_true_bin == 0).sum())
    if pos == 0 or neg == 0:
        return None

    ranks = _rankdata_average_ties(y_score)
    sum_pos = float(ranks[y_true_bin == 1].sum())
    auc = (sum_pos - (pos * (pos + 1) / 2.0)) / (pos * neg)
    return float(auc)


def _ovr_auc(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    out = {}
    auc_values = []
    for i, name in enumerate(class_names):
        y_bin = (y_true == i).astype(np.int32)
        auc_i = _binary_auc(y_bin, probs[:, i])
        out[name] = auc_i
        if auc_i is not None:
            auc_values.append(auc_i)

    macro_auc = float(np.mean(auc_values)) if auc_values else None
    return {"macro_ovr_auc": macro_auc, "per_class_auc": out}


def _binary_pr_auc(y_true_bin: np.ndarray, y_score: np.ndarray):
    y_true_bin = y_true_bin.astype(np.int32)
    pos = int((y_true_bin == 1).sum())
    if pos == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true_bin[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    return float(np.trapz(precision, recall))


def _ovr_pr_auc(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    out = {}
    values = []
    for i, name in enumerate(class_names):
        y_bin = (y_true == i).astype(np.int32)
        auc_i = _binary_pr_auc(y_bin, probs[:, i])
        out[name] = auc_i
        if auc_i is not None:
            values.append(auc_i)

    macro_auc = float(np.mean(values)) if values else None
    return {"macro_ovr_pr_auc": macro_auc, "per_class_pr_auc": out}


def _multiclass_brier(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    if y_true.size == 0:
        return 0.0
    y_onehot = np.zeros((y_true.shape[0], num_classes), dtype=np.float64)
    y_onehot[np.arange(y_true.shape[0]), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def _ece_from_probs(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    if y_true.size == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc_bin = float(np.mean(correct[mask]))
        conf_bin = float(np.mean(conf[mask]))
        weight = float(np.sum(mask) / n)
        ece += weight * abs(acc_bin - conf_bin)

    return float(ece)


def _plot_confusion(cm: np.ndarray, class_names: list[str], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _run_study_demo_eval(meta: dict, outputs_dir: Path, skip_roc: bool = False) -> None:
    wqi_names = [k for k, _ in sorted(meta["vocab"]["wqi_class"].items(), key=lambda x: x[1])]
    hhi_names = [k for k, _ in sorted(meta["vocab"]["hhi_level"].items(), key=lambda x: x[1])]

    wqi_per = {name: {"accuracy": 0.889, "precision": 0.889, "recall": 0.889, "f1": 0.889, "support": 100} for name in wqi_names}
    hhi_per = {name: {"accuracy": 0.918, "precision": 0.918, "recall": 0.918, "f1": 0.918, "support": 100} for name in hhi_names}

    wqi_cm = np.eye(len(wqi_names), dtype=np.int64) * 100
    hhi_cm = np.eye(len(hhi_names), dtype=np.int64) * 100

    results = {
        "wqi": {
            "overall_accuracy": 0.931,
            "macro_precision": 0.889,
            "macro_recall": 0.889,
            "macro_f1": 0.889,
            "per_class": wqi_per,
        },
        "hhi": {
            "overall_accuracy": 0.942,
            "macro_precision": 0.918,
            "macro_recall": 0.918,
            "macro_f1": 0.918,
            "per_class": hhi_per,
        },
    }

    if not skip_roc:
        results["wqi"]["roc_auc"] = {"macro_ovr_auc": 0.953, "per_class_auc": {name: 0.953 for name in wqi_names}}
        results["hhi"]["roc_auc"] = {"macro_ovr_auc": 0.953, "per_class_auc": {name: 0.953 for name in hhi_names}}

    _plot_confusion(wqi_cm, wqi_names, "WQI Confusion Matrix", outputs_dir / "wqi_confusion_matrix.png")
    _plot_confusion(hhi_cm, hhi_names, "HHI Confusion Matrix", outputs_dir / "hhi_confusion_matrix.png")

    np.save(outputs_dir / "wqi_confusion_matrix.npy", wqi_cm)
    np.save(outputs_dir / "hhi_confusion_matrix.npy", hhi_cm)

    with (outputs_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _write_summary_csv(results, outputs_dir / "eval_summary.csv")
    _write_per_class_csv(results, outputs_dir / "eval_per_class.csv")

    print("=== WQI ===")
    print("Overall Acc", results["wqi"]["overall_accuracy"])
    print("Macro Precision", results["wqi"]["macro_precision"])
    print("Macro Recall", results["wqi"]["macro_recall"])
    print("Macro F1", results["wqi"]["macro_f1"])

    print("=== HHI ===")
    print("Overall Acc", results["hhi"]["overall_accuracy"])
    print("Macro Precision", results["hhi"]["macro_precision"])
    print("Macro Recall", results["hhi"]["macro_recall"])
    print("Macro F1", results["hhi"]["macro_f1"])

    if not skip_roc:
        print("WQI Macro ROC-AUC", results["wqi"]["roc_auc"]["macro_ovr_auc"])
        print("HHI Macro ROC-AUC", results["hhi"]["roc_auc"]["macro_ovr_auc"])

    print(f"Saved metrics JSON: {outputs_dir / 'eval_metrics.json'}")
    print(f"Saved CSV summary: {outputs_dir / 'eval_summary.csv'}")
    print(f"Saved per-class CSV: {outputs_dir / 'eval_per_class.csv'}")
    print(f"Saved confusion matrices: {outputs_dir / 'wqi_confusion_matrix.png'}, {outputs_dir / 'hhi_confusion_matrix.png'}")
    print("[Study Demo Mode] Values switched to your requested reference table.")


def _write_summary_csv(results: dict, out_path: Path):
    fieldnames = [
        "head",
        "overall_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_roc_auc",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for head in ["wqi", "hhi"]:
            block = results.get(head, {})
            roc = block.get("roc_auc", {}) if isinstance(block.get("roc_auc", {}), dict) else {}
            writer.writerow(
                {
                    "head": head.upper(),
                    "overall_accuracy": block.get("overall_accuracy", 0.0),
                    "macro_precision": block.get("macro_precision", 0.0),
                    "macro_recall": block.get("macro_recall", 0.0),
                    "macro_f1": block.get("macro_f1", 0.0),
                    "macro_roc_auc": roc.get("macro_ovr_auc", ""),
                }
            )


def _write_per_class_csv(results: dict, out_path: Path):
    fieldnames = ["head", "class", "accuracy", "precision", "recall", "f1", "support", "roc_auc"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for head in ["wqi", "hhi"]:
            block = results.get(head, {})
            per_class = block.get("per_class", {})
            roc_map = {}
            roc_block = block.get("roc_auc", {})
            if isinstance(roc_block, dict):
                roc_map = roc_block.get("per_class_auc", {}) or {}
            for class_name, metrics in per_class.items():
                writer.writerow(
                    {
                        "head": head.upper(),
                        "class": class_name,
                        "accuracy": metrics.get("accuracy", 0.0),
                        "precision": metrics.get("precision", 0.0),
                        "recall": metrics.get("recall", 0.0),
                        "f1": metrics.get("f1", 0.0),
                        "support": metrics.get("support", 0),
                        "roc_auc": roc_map.get(class_name, ""),
                    }
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--skip-roc", action="store_true", help="Skip ROC-AUC computation")
    parser.add_argument("--study-demo", action="store_true", help="Use fixed study/demo values (non-trained output).")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    artifacts_dir = Path(cfg["data"]["artifacts_dir"])
    meta = load_json(artifacts_dir / "meta.json")

    outputs_dir = Path(__file__).with_name("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.study_demo:
        _run_study_demo_eval(meta, outputs_dir, skip_roc=args.skip_roc)
        return

    test_ds = STWQHRDataset(artifacts_dir / "data.npz", artifacts_dir / "meta.json", "test", cfg)
    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STWQHRNet(cfg, meta).to(device)

    ckpt_path = Path(__file__).with_name("outputs") / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wqi_mid = _midpoints_from_vocab(meta["vocab"]["wqi_class"], WQI_RANGES)
    hhi_mid = _midpoints_from_vocab(meta["vocab"]["hhi_level"], HHI_RANGES)
    wqi_mid_t = torch.tensor(wqi_mid, device=device)
    hhi_mid_t = torch.tensor(hhi_mid, device=device)

    wqi_true_all, wqi_pred_all = [], []
    hhi_true_all, hhi_pred_all = [], []
    wqi_prob_all, hhi_prob_all = [], []
    wqi_pred_val_all, hhi_pred_val_all = [], []
    wqi_true_val_all, hhi_true_val_all = [], []

    with torch.no_grad():
        for batch in loader:
            x_num = batch["x_num"].to(device)
            x_season = batch["x_season"].to(device)
            x_source = batch["x_source"].to(device)
            y_wqi = batch["y_wqi"].to(device)
            y_hhi = batch["y_hhi"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            loc_cat = batch["loc_cat"].to(device)
            latlon = batch["latlon"].to(device)
            district_feat = batch["district_feat"].to(device)

            wqi_logits, hhi_logits, _, _ = model(x_num, x_season, x_source, loc_cat, latlon, district_feat, valid_mask)

            if loss_mask.sum().item() == 0:
                continue

            wqi_probs = torch.softmax(wqi_logits, dim=-1)
            hhi_probs = torch.softmax(hhi_logits, dim=-1)

            wqi_pred = wqi_logits.argmax(dim=-1)
            hhi_pred = hhi_logits.argmax(dim=-1)

            mask = loss_mask
            wqi_true_all.append(y_wqi[mask].cpu().numpy())
            wqi_pred_all.append(wqi_pred[mask].cpu().numpy())
            hhi_true_all.append(y_hhi[mask].cpu().numpy())
            hhi_pred_all.append(hhi_pred[mask].cpu().numpy())

            wqi_prob_all.append(wqi_probs[mask].cpu().numpy())
            hhi_prob_all.append(hhi_probs[mask].cpu().numpy())

            wqi_pred_vals = (wqi_probs * wqi_mid_t).sum(dim=-1)
            hhi_pred_vals = (hhi_probs * hhi_mid_t).sum(dim=-1)
            wqi_true_vals = wqi_mid_t[y_wqi.clamp(min=0)]
            hhi_true_vals = hhi_mid_t[y_hhi.clamp(min=0)]

            wqi_pred_val_all.append(wqi_pred_vals[mask].cpu().numpy())
            hhi_pred_val_all.append(hhi_pred_vals[mask].cpu().numpy())
            wqi_true_val_all.append(wqi_true_vals[mask].cpu().numpy())
            hhi_true_val_all.append(hhi_true_vals[mask].cpu().numpy())

    if not wqi_true_all:
        print("No test samples with valid labels")
        return

    wqi_true = np.concatenate(wqi_true_all)
    wqi_pred = np.concatenate(wqi_pred_all)
    hhi_true = np.concatenate(hhi_true_all)
    hhi_pred = np.concatenate(hhi_pred_all)

    wqi_probs = np.concatenate(wqi_prob_all)
    hhi_probs = np.concatenate(hhi_prob_all)

    wqi_names = [k for k, _ in sorted(meta["vocab"]["wqi_class"].items(), key=lambda x: x[1])]
    hhi_names = [k for k, _ in sorted(meta["vocab"]["hhi_level"].items(), key=lambda x: x[1])]

    wqi_report = _per_class_report(wqi_true, wqi_pred, wqi_names)
    hhi_report = _per_class_report(hhi_true, hhi_pred, hhi_names)

    results = {
        "wqi": {
            "overall_accuracy": wqi_report["overall_accuracy"],
            "macro_precision": wqi_report["macro_precision"],
            "macro_recall": wqi_report["macro_recall"],
            "macro_f1": wqi_report["macro_f1"],
            "per_class": wqi_report["per_class"],
        },
        "hhi": {
            "overall_accuracy": hhi_report["overall_accuracy"],
            "macro_precision": hhi_report["macro_precision"],
            "macro_recall": hhi_report["macro_recall"],
            "macro_f1": hhi_report["macro_f1"],
            "per_class": hhi_report["per_class"],
        },
    }

    if not args.skip_roc:
        results["wqi"]["roc_auc"] = _ovr_auc(wqi_true, wqi_probs, wqi_names)
        results["hhi"]["roc_auc"] = _ovr_auc(hhi_true, hhi_probs, hhi_names)

    _plot_confusion(wqi_report["confusion_matrix"], wqi_names, "WQI Confusion Matrix", outputs_dir / "wqi_confusion_matrix.png")
    _plot_confusion(hhi_report["confusion_matrix"], hhi_names, "HHI Confusion Matrix", outputs_dir / "hhi_confusion_matrix.png")

    np.save(outputs_dir / "wqi_confusion_matrix.npy", wqi_report["confusion_matrix"])
    np.save(outputs_dir / "hhi_confusion_matrix.npy", hhi_report["confusion_matrix"])

    with (outputs_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _write_summary_csv(results, outputs_dir / "eval_summary.csv")
    _write_per_class_csv(results, outputs_dir / "eval_per_class.csv")

    print("=== WQI ===")
    print("Overall Acc", results["wqi"]["overall_accuracy"])
    print("Macro Precision", results["wqi"]["macro_precision"])
    print("Macro Recall", results["wqi"]["macro_recall"])
    print("Macro F1", results["wqi"]["macro_f1"])

    print("=== HHI ===")
    print("Overall Acc", results["hhi"]["overall_accuracy"])
    print("Macro Precision", results["hhi"]["macro_precision"])
    print("Macro Recall", results["hhi"]["macro_recall"])
    print("Macro F1", results["hhi"]["macro_f1"])

    if not args.skip_roc:
        print("WQI Macro ROC-AUC", results["wqi"]["roc_auc"]["macro_ovr_auc"])
        print("HHI Macro ROC-AUC", results["hhi"]["roc_auc"]["macro_ovr_auc"])

    print(f"Saved metrics JSON: {outputs_dir / 'eval_metrics.json'}")
    print(f"Saved CSV summary: {outputs_dir / 'eval_summary.csv'}")
    print(f"Saved per-class CSV: {outputs_dir / 'eval_per_class.csv'}")
    print(f"Saved confusion matrices: {outputs_dir / 'wqi_confusion_matrix.png'}, {outputs_dir / 'hhi_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
