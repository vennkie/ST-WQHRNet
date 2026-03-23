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


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return (u[:, :2] * s[:2])


def _plot_bar(ax, values, labels, title, ylabel="Value"):
    ax.bar(np.arange(len(values)), values, color="#1f77b4")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)


def _plot_heatmap(ax, mat, title, xlabel="Feature", ylabel="Time"):
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def _plot_line(ax, x, y, title, xlabel="Time", ylabel="Value"):
    ax.plot(x, y, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)


def _plot_line_multi(ax, x, y1, y2, title, labels=("Raw", "Smoothed")):
    ax.plot(x, y1, color="#1f77b4", label=labels[0])
    ax.plot(x, y2, color="#ff7f0e", label=labels[1])
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic jitter outputs for layers 6-11")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    artifacts_dir = Path(cfg["data"]["artifacts_dir"])
    meta = load_json(artifacts_dir / "meta.json")

    ds = STWQHRDataset(artifacts_dir / "data.npz", artifacts_dir / "meta.json", "all", cfg)
    if len(ds) == 0:
        print("No samples available")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STWQHRNet(cfg, meta).to(device)
    ckpt = torch.load(Path(__file__).with_name("outputs") / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    num_features = len(meta["numeric_cols"])
    max_t = ds.max_t
    mean = np.array(meta["mean"], dtype=np.float32)
    std = np.array(meta["std"], dtype=np.float32)

    sum_features = np.zeros((num_features,), dtype=np.float64)
    count_features = 0.0

    sum_x_embed = np.zeros((max_t, cfg["model"]["d_model"]), dtype=np.float64)
    sum_trans = np.zeros((max_t, cfg["model"]["d_model"]), dtype=np.float64)
    count_time = np.zeros((max_t,), dtype=np.float64)

    wqi_mid = _midpoints_from_vocab(meta["vocab"]["wqi_class"], WQI_RANGES)
    hhi_mid = _midpoints_from_vocab(meta["vocab"]["hhi_level"], HHI_RANGES)
    wqi_labels = [k for k, _ in sorted(meta["vocab"]["wqi_class"].items(), key=lambda x: x[1])]
    hhi_labels = [k for k, _ in sorted(meta["vocab"]["hhi_level"].items(), key=lambda x: x[1])]
    wqi_mid_t = torch.tensor(wqi_mid, device=device)
    hhi_mid_t = torch.tensor(hhi_mid, device=device)

    sum_state = np.zeros((max_t,), dtype=np.float64)
    sum_state_s = np.zeros((max_t,), dtype=np.float64)
    sum_risk = np.zeros((max_t,), dtype=np.float64)
    sum_wqi_exp = np.zeros((max_t,), dtype=np.float64)
    sum_hhi_exp = np.zeros((max_t,), dtype=np.float64)

    sum_wqi_prob = np.zeros((len(wqi_mid),), dtype=np.float64)
    sum_hhi_prob = np.zeros((len(hhi_mid),), dtype=np.float64)
    count_prob = 0.0

    spatial_vecs = []
    spatial_districts = []

    samples_seen = 0
    with torch.no_grad():
        for batch in loader:
            x_num = batch["x_num"].to(device)
            x_season = batch["x_season"].to(device)
            x_source = batch["x_source"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            loc_cat = batch["loc_cat"].to(device)
            latlon = batch["latlon"].to(device)
            district_feat = batch["district_feat"].to(device)

            wqi_logits, hhi_logits, risk, state_consistent, inter = model(
                x_num,
                x_season,
                x_source,
                loc_cat,
                latlon,
                district_feat,
                valid_mask,
                return_intermediates=True,
            )

            bsz = x_num.size(0)
            samples_seen += bsz

            x_num_denorm = x_num.cpu().numpy() * std + mean
            mask = valid_mask.cpu().numpy()
            mask_f = mask.astype(np.float32)

            sum_features += (x_num_denorm * mask_f[..., None]).sum(axis=(0, 1))
            count_features += mask_f.sum()

            x_embed = inter["x_embed"].abs().cpu().numpy()
            trans = inter["transformer_out"].abs().cpu().numpy()
            sum_x_embed += (x_embed * mask_f[..., None]).sum(axis=0)
            sum_trans += (trans * mask_f[..., None]).sum(axis=0)
            count_time += mask_f.sum(axis=0)

            spatial_vecs.append(inter["spatial_vec"].cpu().numpy())
            spatial_districts.append(loc_cat[:, 0].cpu().numpy())

            if not args.synthetic:
                state = inter["state"].squeeze(-1).cpu().numpy()
                state_s = inter["state_consistent"].squeeze(-1).cpu().numpy()
                risk_np = inter["risk"].squeeze(-1).cpu().numpy()
                sum_state += (state * mask_f).sum(axis=0)
                sum_state_s += (state_s * mask_f).sum(axis=0)
                sum_risk += (risk_np * mask_f).sum(axis=0)

                wqi_probs = torch.softmax(wqi_logits, dim=-1).cpu().numpy()
                hhi_probs = torch.softmax(hhi_logits, dim=-1).cpu().numpy()
                sum_wqi_prob += (wqi_probs * mask_f[..., None]).sum(axis=(0, 1))
                sum_hhi_prob += (hhi_probs * mask_f[..., None]).sum(axis=(0, 1))
                count_prob += mask_f.sum()

                wqi_exp = (torch.softmax(wqi_logits, dim=-1) * wqi_mid_t).sum(dim=-1).cpu().numpy()
                hhi_exp = (torch.softmax(hhi_logits, dim=-1) * hhi_mid_t).sum(dim=-1).cpu().numpy()
                sum_wqi_exp += (wqi_exp * mask_f).sum(axis=0)
                sum_hhi_exp += (hhi_exp * mask_f).sum(axis=0)
            else:
                loc_cat_np = loc_cat.cpu().numpy()
                for b in range(bsz):
                    d_id, b_id, v_id = loc_cat_np[b].tolist()
                    for t in range(mask.shape[1]):
                        if not mask[b, t]:
                            continue
                        seed_base = f"{d_id}-{b_id}-{v_id}-{t}"
                        wqi_label = wqi_labels[abs(hash(seed_base + "-wqi")) % len(wqi_labels)]
                        hhi_label = hhi_labels[abs(hash(seed_base + "-hhi")) % len(hhi_labels)]

                        wqi_base = sum(WQI_RANGES[wqi_label]) / 2.0
                        hhi_base = sum(HHI_RANGES[hhi_label]) / 2.0

                        wqi_pred, wqi_label_final = _jitter_value(wqi_base, wqi_label, WQI_RANGES, seed_base + "-wqi")
                        hhi_pred, hhi_label_final = _jitter_value(hhi_base, hhi_label, HHI_RANGES, seed_base + "-hhi")

                        sum_wqi_exp[t] += wqi_pred
                        sum_hhi_exp[t] += hhi_pred
                        sum_wqi_prob[wqi_labels.index(wqi_label_final)] += 1
                        sum_hhi_prob[hhi_labels.index(hhi_label_final)] += 1
                        count_prob += 1

            if samples_seen >= args.samples:
                break

    mean_features = sum_features / max(count_features, 1.0)
    mean_x_embed = sum_x_embed / np.maximum(count_time[:, None], 1.0)
    mean_trans = sum_trans / np.maximum(count_time[:, None], 1.0)

    if args.synthetic:
        mean_wqi_exp = sum_wqi_exp / np.maximum(count_time, 1.0)
        mean_hhi_exp = sum_hhi_exp / np.maximum(count_time, 1.0)
        mean_wqi_prob = sum_wqi_prob / max(count_prob, 1.0)
        mean_hhi_prob = sum_hhi_prob / max(count_prob, 1.0)

        mean_risk = mean_wqi_exp / 100.0
        mean_state = (mean_wqi_exp - 50.0) / 25.0
        # Smooth state with simple moving average
        kernel = np.ones(5) / 5.0
        mean_state_s = np.convolve(mean_state, kernel, mode="same")
    else:
        mean_state = sum_state / np.maximum(count_time, 1.0)
        mean_state_s = sum_state_s / np.maximum(count_time, 1.0)
        mean_risk = sum_risk / np.maximum(count_time, 1.0)
        mean_wqi_prob = sum_wqi_prob / max(count_prob, 1.0)
        mean_hhi_prob = sum_hhi_prob / max(count_prob, 1.0)
        mean_wqi_exp = sum_wqi_exp / np.maximum(count_time, 1.0)
        mean_hhi_exp = sum_hhi_exp / np.maximum(count_time, 1.0)

    spatial_vecs = np.concatenate(spatial_vecs, axis=0)
    spatial_districts = np.concatenate(spatial_districts, axis=0)

    output_dir = Path(__file__).with_name("outputs") / "figures_layers"
    _ensure_dir(output_dir)

    suffix = "(Synthetic jitter)" if args.synthetic else "(Model output)"

    fig, ax = plt.subplots(figsize=(7, 3.5))
    _plot_bar(ax, mean_features, meta["numeric_cols"], f"Layer 1: Mean Input Features {suffix}", ylabel="Value")
    fig.tight_layout()
    fig.savefig(output_dir / "layer_01_input.png", dpi=300)
    plt.close(fig)

    district_emb = model.district_emb.weight.detach().cpu().numpy()
    district_pca = _pca_2d(district_emb)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(district_pca[:, 0], district_pca[:, 1], c=np.arange(len(district_pca)), cmap="tab20", s=30)
    ax.set_title(f"Layer 2: District Embedding (PCA) {suffix}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_02_embedding.png", dpi=300)
    plt.close(fig)

    spatial_pca = _pca_2d(spatial_vecs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(spatial_pca[:, 0], spatial_pca[:, 1], c=spatial_districts, cmap="viridis", s=10, alpha=0.6)
    ax.set_title(f"Layer 3: Spatial Encoding (PCA) {suffix}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_03_spatial.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = _plot_heatmap(ax, mean_x_embed, f"Layer 4: Temporal Encoding (mean |x|) {suffix}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_04_temporal.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = _plot_heatmap(ax, mean_trans, f"Layer 5: Transformer Output (mean |x|) {suffix}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_05_transformer.png", dpi=300)
    plt.close(fig)

    time_index = np.arange(len(mean_state))

    fig, ax = plt.subplots(figsize=(6, 3))
    _plot_line(ax, time_index, mean_state, f"Layer 6: Contamination State (mean) {suffix}")
    fig.tight_layout()
    fig.savefig(output_dir / "layer_06_state.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    _plot_line_multi(ax, time_index, mean_state, mean_state_s, f"Layer 7: Physical Consistency (mean) {suffix}")
    fig.tight_layout()
    fig.savefig(output_dir / "layer_07_physical.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    _plot_line(ax, time_index, mean_risk, f"Layer 8: Risk Score (mean) {suffix}", ylabel="Risk")
    fig.tight_layout()
    fig.savefig(output_dir / "layer_08_risk.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    _plot_bar(ax, mean_wqi_prob, wqi_labels, f"Layer 9: WQI Head (mean prob) {suffix}", ylabel="Probability")
    fig.tight_layout()
    fig.savefig(output_dir / "layer_09_wqi_head.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    _plot_bar(ax, mean_hhi_prob, hhi_labels, f"Layer 10: HHI Head (mean prob) {suffix}", ylabel="Probability")
    fig.tight_layout()
    fig.savefig(output_dir / "layer_10_hhi_head.png", dpi=300)
    plt.close(fig)

    time_year = np.load(artifacts_dir / "data.npz", allow_pickle=True)["time_year"]
    time_year = time_year[: mean_wqi_exp.shape[0]]
    years = np.unique(time_year)
    wqi_year = []
    hhi_year = []
    for y in years:
        idx = np.where(time_year == y)[0]
        wqi_year.append(np.nanmean(mean_wqi_exp[idx]))
        hhi_year.append(np.nanmean(mean_hhi_exp[idx]))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(years, wqi_year, marker="o", label="WQI")
    ax.plot(years, hhi_year, marker="o", label="HHI")
    ax.set_title(f"Layer 11: Forecast Output (proxy mean) {suffix}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Expected value")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_11_forecast.png", dpi=300)
    plt.close(fig)

    paths = [
        output_dir / "layer_01_input.png",
        output_dir / "layer_02_embedding.png",
        output_dir / "layer_03_spatial.png",
        output_dir / "layer_04_temporal.png",
        output_dir / "layer_05_transformer.png",
        output_dir / "layer_06_state.png",
        output_dir / "layer_07_physical.png",
        output_dir / "layer_08_risk.png",
        output_dir / "layer_09_wqi_head.png",
        output_dir / "layer_10_hhi_head.png",
        output_dir / "layer_11_forecast.png",
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    for i, p in enumerate(paths):
        img = plt.imread(p)
        axes[i].imshow(img)
        axes[i].axis("off")
    for j in range(len(paths), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "layers_overview.png", dpi=300)
    plt.close(fig)

    print("Saved layer figures to", output_dir)


if __name__ == "__main__":
    main()
