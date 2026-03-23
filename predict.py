import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from model import STWQHRNet
from utils import load_json


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_future_inputs(data, meta, cfg):
    x_num = data["x_num"]
    x_season = data["x_season"]
    x_source = data["x_source"]
    mask = data["mask"]

    start_year = cfg["data"]["start_year"]
    future_end_year = cfg["data"]["future_end_year"]
    total_t = (future_end_year - start_year + 1) * 3

    n, t_existing, f = x_num.shape
    if total_t <= t_existing:
        total_t = t_existing

    x_num_ext = np.zeros((n, total_t, f), dtype=np.float32)
    x_season_ext = np.zeros((n, total_t), dtype=np.int64)
    x_source_ext = np.zeros((n, total_t), dtype=np.int64)
    valid_mask_ext = np.zeros((n, total_t), dtype=np.bool_)

    x_num_ext[:, :t_existing] = x_num
    x_season_ext[:, :t_existing] = x_season
    x_source_ext[:, :t_existing] = x_source
    valid_mask_ext[:, :t_existing] = mask

    # Fill future steps using last observed values
    last_idx = np.where(mask, np.arange(t_existing), -1).max(axis=1)
    last_idx[last_idx < 0] = 0

    for i in range(n):
        li = last_idx[i]
        x_num_ext[i, t_existing:] = x_num[i, li]
        x_source_ext[i, t_existing:] = x_source[i, li]

    # Set season ids by fixed order
    season_order = meta["season_order"]
    season_vocab = meta["vocab"]["season"]
    for t in range(total_t):
        season_name = season_order[t % 3]
        season_id = season_vocab[season_name]
        x_season_ext[:, t] = season_id

    valid_mask_ext[:, t_existing:] = True

    return x_num_ext, x_season_ext, x_source_ext, valid_mask_ext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--district", required=True)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    artifacts_dir = Path(cfg["data"]["artifacts_dir"])

    meta = load_json(artifacts_dir / "meta.json")
    data = np.load(artifacts_dir / "data.npz", allow_pickle=True)

    district_vocab = meta["vocab"]["district"]
    if args.district not in district_vocab:
        raise ValueError(f"Unknown district: {args.district}")

    district_id = district_vocab[args.district]

    loc_cat = data["loc_cat"]
    latlon = data["latlon"]
    district_feat = data["district_feat"]

    loc_indices = np.where(loc_cat[:, 0] == district_id)[0]
    if loc_indices.size == 0:
        raise ValueError("No locations found for this district")

    x_num_ext, x_season_ext, x_source_ext, valid_mask_ext = build_future_inputs(data, meta, cfg)

    start_year = cfg["data"]["start_year"]
    t0 = (args.year - start_year) * 3
    t_indices = [t0, t0 + 1, t0 + 2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STWQHRNet(cfg, meta).to(device)

    ckpt_path = Path(__file__).with_name("outputs") / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("Model checkpoint not found. Run train.py first.")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wqi_id_to_name = {v: k for k, v in meta["vocab"]["wqi_class"].items()}
    hhi_id_to_name = {v: k for k, v in meta["vocab"]["hhi_level"].items()}
    season_order = meta["season_order"]

    batch_size = 256
    wqi_counts = [Counter() for _ in range(3)]
    hhi_counts = [Counter() for _ in range(3)]

    with torch.no_grad():
        for i in range(0, len(loc_indices), batch_size):
            idx = loc_indices[i : i + batch_size]

            x_num = torch.from_numpy(x_num_ext[idx]).float().to(device)
            x_season = torch.from_numpy(x_season_ext[idx]).long().to(device)
            x_source = torch.from_numpy(x_source_ext[idx]).long().to(device)
            valid_mask = torch.from_numpy(valid_mask_ext[idx]).bool().to(device)
            loc_cat_batch = torch.from_numpy(loc_cat[idx]).long().to(device)
            latlon_batch = torch.from_numpy(latlon[idx]).float().to(device)
            district_feat_batch = torch.from_numpy(district_feat[idx]).float().to(device)

            wqi_logits, hhi_logits, _, _ = model(x_num, x_season, x_source, loc_cat_batch, latlon_batch, district_feat_batch, valid_mask)

            for s, t in enumerate(t_indices):
                if t < wqi_logits.size(1):
                    wqi_pred = wqi_logits[:, t].argmax(dim=-1).cpu().numpy().tolist()
                    hhi_pred = hhi_logits[:, t].argmax(dim=-1).cpu().numpy().tolist()
                    wqi_counts[s].update(wqi_pred)
                    hhi_counts[s].update(hhi_pred)

    print(f"District: {args.district}")
    print(f"Year: {args.year}")

    for s, season_name in enumerate(season_order):
        if not wqi_counts[s]:
            continue
        wqi_mode = max(wqi_counts[s].items(), key=lambda x: x[1])[0]
        hhi_mode = max(hhi_counts[s].items(), key=lambda x: x[1])[0]
        print(f"{season_name}: WQI={wqi_id_to_name[wqi_mode]}, HHI={hhi_id_to_name[hhi_mode]}")


if __name__ == "__main__":
    main()
