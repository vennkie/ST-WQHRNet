import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from dataset import STWQHRDataset
from model import STWQHRNet
from utils import load_json

WQI_RANGES = {
    "Good": (0.0, 44.0),
    "Moderate": (45.0, 60.0),
    "Poor": (61.0, 80.0),
    "Unfit": (81.0, 100.0),
}

HHI_RANGES = {
    "Low": (0.0, 1.0),
    "Medium": (1.0, 1.5),
    "High": (1.5, 3.0),
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


def _confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, targets):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _plot_confusion(cm: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_scatter(obs: np.ndarray, pred: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(obs, pred, s=10, alpha=0.5, color="#1f77b4", edgecolor="none")
    if obs.size > 0:
        lo = min(obs.min(), pred.min())
        hi = max(obs.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
        if obs.size >= 2:
            slope, intercept = np.polyfit(obs, pred, 1)
            xs = np.array([lo, hi])
            ax.plot(xs, slope * xs + intercept, color="#d62728", linewidth=1)
    ax.set_xlabel("Observed (proxy midpoint)")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_timeseries(years: np.ndarray, obs: np.ndarray, pred: np.ndarray, cfg: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))

    start_year = int(cfg["data"]["start_year"])
    train_end = int(cfg["data"]["train_end_year"])
    val_year = int(cfg["data"]["val_year"])
    test_year = int(cfg["data"]["test_year"])

    ax.axvspan(start_year - 0.5, train_end + 0.5, color="#d9f2d9", alpha=0.4, label="Train")
    ax.axvspan(val_year - 0.5, val_year + 0.5, color="#fff2cc", alpha=0.6, label="Val")
    ax.axvspan(test_year - 0.5, test_year + 0.5, color="#f8d7da", alpha=0.6, label="Test")

    ax.plot(years, obs, marker="o", linewidth=1.5, label="Observed (proxy)")
    ax.plot(years, pred, marker="o", linewidth=1.5, label="Predicted")

    ax.set_xlabel("Year")
    ax.set_ylabel("Value (class midpoint)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _label_from_value(value: float, ranges: dict) -> str:
    for label, (lo, hi) in ranges.items():
        if lo <= value <= hi:
            return label
    return list(ranges.keys())[-1]


def _jitter_value(base_value: float, label: str, ranges: dict, seed_key: str):
    lo, hi = ranges[label]
    width = hi - lo
    rng = np.random.default_rng(abs(hash(seed_key)) % (2**32))
    target = lo + rng.uniform(0.05, 0.95) * width
    alpha = 0.75
    value = (1.0 - alpha) * base_value + alpha * target
    value = float(np.clip(value, lo, hi))
    label = _label_from_value(value, ranges)
    return value, label


def _collect_predictions_model(cfg: dict, meta: dict, split: str):
    artifacts_dir = Path(cfg["data"]["artifacts_dir"])
    ds = STWQHRDataset(artifacts_dir / "data.npz", artifacts_dir / "meta.json", split, cfg)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STWQHRNet(cfg, meta).to(device)
    ckpt = torch.load(Path(__file__).with_name("outputs") / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wqi_mid = _midpoints_from_vocab(meta["vocab"]["wqi_class"], WQI_RANGES)
    hhi_mid = _midpoints_from_vocab(meta["vocab"]["hhi_level"], HHI_RANGES)
    wqi_mid_t = torch.tensor(wqi_mid, device=device)
    hhi_mid_t = torch.tensor(hhi_mid, device=device)

    wqi_pred_list = []
    wqi_true_list = []
    hhi_pred_list = []
    hhi_true_list = []
    wqi_cls_pred = []
    wqi_cls_true = []
    hhi_cls_pred = []
    hhi_cls_true = []

    max_t = ds.max_t
    sum_wqi_pred = np.zeros((max_t,), dtype=np.float64)
    sum_wqi_true = np.zeros((max_t,), dtype=np.float64)
    sum_hhi_pred = np.zeros((max_t,), dtype=np.float64)
    sum_hhi_true = np.zeros((max_t,), dtype=np.float64)
    count = np.zeros((max_t,), dtype=np.float64)

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

            wqi_probs = torch.softmax(wqi_logits, dim=-1)
            hhi_probs = torch.softmax(hhi_logits, dim=-1)
            wqi_pred = (wqi_probs * wqi_mid_t).sum(dim=-1)
            hhi_pred = (hhi_probs * hhi_mid_t).sum(dim=-1)

            wqi_true = wqi_mid_t[y_wqi.clamp(min=0)]
            hhi_true = hhi_mid_t[y_hhi.clamp(min=0)]

            mask = loss_mask
            if mask.sum().item() > 0:
                wqi_pred_list.append(wqi_pred[mask].cpu().numpy())
                wqi_true_list.append(wqi_true[mask].cpu().numpy())
                hhi_pred_list.append(hhi_pred[mask].cpu().numpy())
                hhi_true_list.append(hhi_true[mask].cpu().numpy())

                wqi_cls_pred.append(wqi_logits.argmax(dim=-1)[mask].cpu().numpy())
                wqi_cls_true.append(y_wqi[mask].cpu().numpy())
                hhi_cls_pred.append(hhi_logits.argmax(dim=-1)[mask].cpu().numpy())
                hhi_cls_true.append(y_hhi[mask].cpu().numpy())

            mask_f = mask.float()
            sum_wqi_pred += (wqi_pred * mask_f).sum(dim=0).cpu().numpy()
            sum_wqi_true += (wqi_true * mask_f).sum(dim=0).cpu().numpy()
            sum_hhi_pred += (hhi_pred * mask_f).sum(dim=0).cpu().numpy()
            sum_hhi_true += (hhi_true * mask_f).sum(dim=0).cpu().numpy()
            count += mask_f.sum(dim=0).cpu().numpy()

    def _concat(xs):
        return np.concatenate(xs, axis=0) if xs else np.array([])

    return {
        "wqi_pred": _concat(wqi_pred_list),
        "wqi_true": _concat(wqi_true_list),
        "hhi_pred": _concat(hhi_pred_list),
        "hhi_true": _concat(hhi_true_list),
        "wqi_cls_pred": _concat(wqi_cls_pred),
        "wqi_cls_true": _concat(wqi_cls_true),
        "hhi_cls_pred": _concat(hhi_cls_pred),
        "hhi_cls_true": _concat(hhi_cls_true),
        "ts_wqi_pred": np.divide(sum_wqi_pred, np.maximum(count, 1.0)),
        "ts_wqi_true": np.divide(sum_wqi_true, np.maximum(count, 1.0)),
        "ts_hhi_pred": np.divide(sum_hhi_pred, np.maximum(count, 1.0)),
        "ts_hhi_true": np.divide(sum_hhi_true, np.maximum(count, 1.0)),
    }


def _collect_predictions_synthetic(cfg: dict, meta: dict, split: str, max_samples: int):
    artifacts_dir = Path(cfg["data"]["artifacts_dir"])
    ds = STWQHRDataset(artifacts_dir / "data.npz", artifacts_dir / "meta.json", split, cfg)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    wqi_labels = [k for k, _ in sorted(meta["vocab"]["wqi_class"].items(), key=lambda x: x[1])]
    hhi_labels = [k for k, _ in sorted(meta["vocab"]["hhi_level"].items(), key=lambda x: x[1])]
    wqi_mid = _midpoints_from_vocab(meta["vocab"]["wqi_class"], WQI_RANGES)
    hhi_mid = _midpoints_from_vocab(meta["vocab"]["hhi_level"], HHI_RANGES)

    max_t = ds.max_t
    sum_wqi_pred = np.zeros((max_t,), dtype=np.float64)
    sum_wqi_true = np.zeros((max_t,), dtype=np.float64)
    sum_hhi_pred = np.zeros((max_t,), dtype=np.float64)
    sum_hhi_true = np.zeros((max_t,), dtype=np.float64)
    count = np.zeros((max_t,), dtype=np.float64)

    wqi_pred_list = []
    wqi_true_list = []
    hhi_pred_list = []
    hhi_true_list = []
    wqi_cls_pred = []
    wqi_cls_true = []
    hhi_cls_pred = []
    hhi_cls_true = []

    samples_seen = 0
    with torch.no_grad():
        for batch in loader:
            y_wqi = batch["y_wqi"].numpy()
            y_hhi = batch["y_hhi"].numpy()
            loss_mask = batch["loss_mask"].numpy()
            loc_cat = batch["loc_cat"].numpy()

            bsz, tlen = y_wqi.shape
            for b in range(bsz):
                d_id, b_id, v_id = loc_cat[b].tolist()
                for t in range(tlen):
                    if not loss_mask[b, t]:
                        continue
                    seed_base = f"{d_id}-{b_id}-{v_id}-{t}"  # deterministic

                    # Observed values (proxy)
                    wqi_true = wqi_mid[y_wqi[b, t]] if y_wqi[b, t] >= 0 else 0.0
                    hhi_true = hhi_mid[y_hhi[b, t]] if y_hhi[b, t] >= 0 else 0.0

                    # Predicted labels (synthetic)
                    wqi_label = wqi_labels[abs(hash(seed_base + "-wqi")) % len(wqi_labels)]
                    hhi_label = hhi_labels[abs(hash(seed_base + "-hhi")) % len(hhi_labels)]

                    wqi_base = sum(WQI_RANGES[wqi_label]) / 2.0
                    hhi_base = sum(HHI_RANGES[hhi_label]) / 2.0

                    wqi_pred, wqi_label_final = _jitter_value(wqi_base, wqi_label, WQI_RANGES, seed_base + "-wqi")
                    hhi_pred, hhi_label_final = _jitter_value(hhi_base, hhi_label, HHI_RANGES, seed_base + "-hhi")

                    wqi_pred_list.append(wqi_pred)
                    wqi_true_list.append(wqi_true)
                    hhi_pred_list.append(hhi_pred)
                    hhi_true_list.append(hhi_true)

                    wqi_cls_pred.append(wqi_labels.index(wqi_label_final))
                    wqi_cls_true.append(int(y_wqi[b, t]))
                    hhi_cls_pred.append(hhi_labels.index(hhi_label_final))
                    hhi_cls_true.append(int(y_hhi[b, t]))

                    sum_wqi_pred[t] += wqi_pred
                    sum_wqi_true[t] += wqi_true
                    sum_hhi_pred[t] += hhi_pred
                    sum_hhi_true[t] += hhi_true
                    count[t] += 1

            samples_seen += bsz
            if samples_seen >= max_samples:
                break

    return {
        "wqi_pred": np.array(wqi_pred_list, dtype=np.float32),
        "wqi_true": np.array(wqi_true_list, dtype=np.float32),
        "hhi_pred": np.array(hhi_pred_list, dtype=np.float32),
        "hhi_true": np.array(hhi_true_list, dtype=np.float32),
        "wqi_cls_pred": np.array(wqi_cls_pred, dtype=np.int64),
        "wqi_cls_true": np.array(wqi_cls_true, dtype=np.int64),
        "hhi_cls_pred": np.array(hhi_cls_pred, dtype=np.int64),
        "hhi_cls_true": np.array(hhi_cls_true, dtype=np.int64),
        "ts_wqi_pred": np.divide(sum_wqi_pred, np.maximum(count, 1.0)),
        "ts_wqi_true": np.divide(sum_wqi_true, np.maximum(count, 1.0)),
        "ts_hhi_pred": np.divide(sum_hhi_pred, np.maximum(count, 1.0)),
        "ts_hhi_true": np.divide(sum_hhi_true, np.maximum(count, 1.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic jitter predictions")
    parser.add_argument("--samples", type=int, default=512, help="Sample count for synthetic mode")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    artifacts_dir = Path(cfg["data"]["artifacts_dir"])
    meta = load_json(artifacts_dir / "meta.json")

    output_dir = Path(__file__).with_name("outputs") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        all_res = _collect_predictions_synthetic(cfg, meta, "all", args.samples)
    else:
        all_res = _collect_predictions_model(cfg, meta, "all")

    time_year = np.load(artifacts_dir / "data.npz", allow_pickle=True)["time_year"]
    time_year = time_year[: all_res["ts_wqi_pred"].shape[0]]

    years = np.unique(time_year)
    wqi_obs_year = []
    wqi_pred_year = []
    hhi_obs_year = []
    hhi_pred_year = []

    for y in years:
        idx = np.where(time_year == y)[0]
        wqi_obs_year.append(np.nanmean(all_res["ts_wqi_true"][idx]))
        wqi_pred_year.append(np.nanmean(all_res["ts_wqi_pred"][idx]))
        hhi_obs_year.append(np.nanmean(all_res["ts_hhi_true"][idx]))
        hhi_pred_year.append(np.nanmean(all_res["ts_hhi_pred"][idx]))

    title_suffix = "(Synthetic jitter)" if args.synthetic else "(Model output)"

    _plot_timeseries(
        years,
        np.array(wqi_obs_year),
        np.array(wqi_pred_year),
        cfg,
        f"WQI Time Series {title_suffix}",
        output_dir / "wqi_timeseries.png",
    )

    _plot_timeseries(
        years,
        np.array(hhi_obs_year),
        np.array(hhi_pred_year),
        cfg,
        f"HHI Time Series {title_suffix}",
        output_dir / "hhi_timeseries.png",
    )

    if args.synthetic:
        test_res = _collect_predictions_synthetic(cfg, meta, "test", args.samples)
    else:
        test_res = _collect_predictions_model(cfg, meta, "test")

    _plot_scatter(
        test_res["wqi_true"],
        test_res["wqi_pred"],
        f"WQI Observed vs Predicted {title_suffix}",
        output_dir / "wqi_scatter.png",
    )

    _plot_scatter(
        test_res["hhi_true"],
        test_res["hhi_pred"],
        f"HHI Observed vs Predicted {title_suffix}",
        output_dir / "hhi_scatter.png",
    )

    wqi_labels = [k for k, _ in sorted(meta["vocab"]["wqi_class"].items(), key=lambda x: x[1])]
    hhi_labels = [k for k, _ in sorted(meta["vocab"]["hhi_level"].items(), key=lambda x: x[1])]

    wqi_cm = _confusion_matrix(test_res["wqi_cls_pred"], test_res["wqi_cls_true"], len(wqi_labels))
    hhi_cm = _confusion_matrix(test_res["hhi_cls_pred"], test_res["hhi_cls_true"], len(hhi_labels))

    _plot_confusion(wqi_cm, wqi_labels, f"WQI Confusion Matrix {title_suffix}", output_dir / "wqi_confusion.png")
    _plot_confusion(hhi_cm, hhi_labels, f"HHI Confusion Matrix {title_suffix}", output_dir / "hhi_confusion.png")

    print("Saved figures to", output_dir)


if __name__ == "__main__":
    main()
