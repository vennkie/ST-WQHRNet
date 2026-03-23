import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils import save_json

MANDATORY_PARAMS = ["Nitrate", "Fluoride", "Chloride", "TDS", "pH"]
EMERGING_PARAMS = ["Iron", "Alkalinity"]

NUMERIC_BASE = [
    "seasonal_rainfall_mm",
    "avg_temperature_c",
    "population_density",
    "agricultural_intensity_pct",
]

NUMERIC_LATLON = ["latitude", "longitude"]

PARAM_PATTERN = re.compile(r"\s*([^\[]+)\[([^\]]+)\]")
NUM_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def load_config():
    cfg_path = Path(__file__).with_name("config.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_param_string(value: str):
    result = {}
    if not isinstance(value, str):
        return result
    parts = value.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = PARAM_PATTERN.match(part)
        if not m:
            continue
        name = m.group(1).strip()
        value_part = m.group(2)
        num_match = NUM_PATTERN.search(value_part)
        if not num_match:
            continue
        result[name] = float(num_match.group(0))
    return result


def build_vocab(series):
    unique = sorted(series.dropna().unique().tolist())
    return {v: i for i, v in enumerate(unique)}


def mode_series(series):
    if series.empty:
        return -1
    m = series.mode()
    return int(m.iloc[0]) if not m.empty else -1


def main():
    cfg = load_config()
    data_cfg = cfg["data"]

    csv_path = Path(data_cfg["csv_path"])
    artifacts_dir = Path(data_cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Parse chemical parameters into numeric columns
    def add_param_columns(df, col_name, prefix, params):
        extracted = {f"{prefix}_{p.lower()}": [] for p in params}
        for val in df[col_name].fillna("").tolist():
            parsed = parse_param_string(val)
            for p in params:
                extracted[f"{prefix}_{p.lower()}"] .append(parsed.get(p, np.nan))
        return pd.DataFrame(extracted)

    above_mand = add_param_columns(df, "above_p_mandatory", "above_mand", MANDATORY_PARAMS)
    below_mand = add_param_columns(df, "below_p_mandatory", "below_mand", MANDATORY_PARAMS)
    above_em = add_param_columns(df, "above_p_emerging", "above_em", EMERGING_PARAMS)
    below_em = add_param_columns(df, "below_p_emerging", "below_em", EMERGING_PARAMS)

    df = pd.concat([df, above_mand, below_mand, above_em, below_em], axis=1)

    numeric_cols = NUMERIC_BASE + list(above_mand.columns) + list(below_mand.columns) + list(above_em.columns) + list(below_em.columns)

    # Build vocabularies
    vocab = {
        "district": build_vocab(df["district"]),
        "block": build_vocab(df["block"]),
        "village": build_vocab(df["village"]),
        "season": build_vocab(df["season"]),
        "source_type": build_vocab(df["source_type"]),
        "wqi_class": build_vocab(df["wqi_class"]),
        "hhi_level": build_vocab(df["hhi_level"]),
    }

    # Map categorical to ids
    df["district_id"] = df["district"].map(vocab["district"]).astype(int)
    df["block_id"] = df["block"].map(vocab["block"]).astype(int)
    df["village_id"] = df["village"].map(vocab["village"]).astype(int)
    df["season_id"] = df["season"].map(vocab["season"]).astype(int)
    df["source_id"] = df["source_type"].map(vocab["source_type"]).astype(int)
    df["wqi_id"] = df["wqi_class"].map(vocab["wqi_class"]).astype(int)
    df["hhi_id"] = df["hhi_level"].map(vocab["hhi_level"]).astype(int)

    # Location ids
    loc_keys = pd.MultiIndex.from_frame(df[["district_id", "block_id", "village_id"]])
    loc_id, loc_uniques = pd.factorize(loc_keys)
    df["loc_id"] = loc_id

    # Time mapping
    time_year = df.groupby("time_index")["year"].agg(lambda x: x.mode().iloc[0]).sort_index()
    time_season = df.groupby("time_index")["season"].agg(lambda x: x.mode().iloc[0]).sort_index()

    max_time_index = int(df["time_index"].max())

    # Aggregate duplicates per location and time_index
    group_cols = ["loc_id", "time_index"]
    agg_num = df.groupby(group_cols)[numeric_cols].mean().reset_index()
    agg_cat = df.groupby(group_cols)[["season_id", "source_id", "wqi_id", "hhi_id"]].agg(mode_series).reset_index()
    agg_year = df.groupby(group_cols)[["year"]].agg(mode_series).reset_index()
    agg = pd.merge(agg_num, agg_cat, on=group_cols, how="left")
    agg = pd.merge(agg, agg_year, on=group_cols, how="left")

    # Build arrays
    num_locs = len(loc_uniques)
    num_features = len(numeric_cols)
    T = max_time_index

    x_num = np.full((num_locs, T, num_features), np.nan, dtype=np.float32)
    x_season = np.full((num_locs, T), -1, dtype=np.int64)
    x_source = np.full((num_locs, T), -1, dtype=np.int64)
    y_wqi = np.full((num_locs, T), -1, dtype=np.int64)
    y_hhi = np.full((num_locs, T), -1, dtype=np.int64)
    year_arr = np.full((num_locs, T), -1, dtype=np.int64)
    mask = np.zeros((num_locs, T), dtype=np.bool_)

    loc_idx = agg["loc_id"].to_numpy(dtype=np.int64)
    t_idx = agg["time_index"].to_numpy(dtype=np.int64) - 1
    x_num[loc_idx, t_idx] = agg[numeric_cols].to_numpy(dtype=np.float32)
    x_season[loc_idx, t_idx] = agg["season_id"].to_numpy(dtype=np.int64)
    x_source[loc_idx, t_idx] = agg["source_id"].to_numpy(dtype=np.int64)
    y_wqi[loc_idx, t_idx] = agg["wqi_id"].to_numpy(dtype=np.int64)
    y_hhi[loc_idx, t_idx] = agg["hhi_id"].to_numpy(dtype=np.int64)
    year_arr[loc_idx, t_idx] = agg["year"].to_numpy(dtype=np.int64)
    mask[loc_idx, t_idx] = True

    # Ensure categorical placeholders are valid for embedding layers
    x_season[~mask] = 0
    x_source[~mask] = 0

    # Location metadata
    loc_meta = df.groupby("loc_id").agg({
        "district_id": "first",
        "block_id": "first",
        "village_id": "first",
        "latitude": "mean",
        "longitude": "mean",
    }).reset_index()

    loc_cat = loc_meta[["district_id", "block_id", "village_id"]].to_numpy(dtype=np.int64)
    latlon = loc_meta[["latitude", "longitude"]].to_numpy(dtype=np.float32)

    # District-level aggregate features (static, before normalization)
    district_meta = df.groupby("district_id")[numeric_cols].mean().reset_index()
    district_meta = district_meta.set_index("district_id")
    district_feat = district_meta.loc[loc_meta["district_id"]].to_numpy(dtype=np.float32)

    # Normalization using training years
    train_end_year = data_cfg["train_end_year"]
    time_year_arr = np.array([time_year.get(i + 1, train_end_year) for i in range(T)], dtype=np.int64)
    agg_year = agg["year"].to_numpy(dtype=np.int64)
    train_rows = agg_year <= train_end_year

    train_values = agg[numeric_cols].to_numpy(dtype=np.float32)[train_rows]
    mean = np.nanmean(train_values, axis=0)
    std = np.nanstd(train_values, axis=0)
    std[std == 0] = 1.0

    x_num = (x_num - mean) / std
    x_num = np.nan_to_num(x_num, nan=0.0)

    district_feat = (district_feat - mean) / std
    district_feat = np.nan_to_num(district_feat, nan=0.0)

    # Class weights
    wqi_counts = Counter(agg["wqi_id"].to_numpy(dtype=np.int64)[train_rows])
    hhi_counts = Counter(agg["hhi_id"].to_numpy(dtype=np.int64)[train_rows])

    num_wqi = len(vocab["wqi_class"])
    num_hhi = len(vocab["hhi_level"])
    wqi_weights = [0.0] * num_wqi
    hhi_weights = [0.0] * num_hhi

    total_wqi = sum(wqi_counts.values())
    total_hhi = sum(hhi_counts.values())
    for i in range(num_wqi):
        count = wqi_counts.get(i, 1)
        wqi_weights[i] = total_wqi / (num_wqi * count)
    for i in range(num_hhi):
        count = hhi_counts.get(i, 1)
        hhi_weights[i] = total_hhi / (num_hhi * count)

    meta = {
        "numeric_cols": numeric_cols,
        "feature_index": {name: i for i, name in enumerate(numeric_cols)},
        "vocab": vocab,
        "time_year": time_year_arr.tolist(),
        "time_season": time_season.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "max_time_index": T,
        "start_year": data_cfg["start_year"],
        "season_order": data_cfg["season_order"],
        "class_weights": {"wqi": wqi_weights, "hhi": hhi_weights},
        "emerging_above_idx": [
            numeric_cols.index("above_em_iron"),
            numeric_cols.index("above_em_alkalinity"),
        ],
    }

    np.savez_compressed(
        artifacts_dir / "data.npz",
        x_num=x_num,
        x_season=x_season,
        x_source=x_source,
        y_wqi=y_wqi,
        y_hhi=y_hhi,
        mask=mask,
        loc_cat=loc_cat,
        latlon=latlon,
        district_feat=district_feat,
        year_arr=year_arr,
        time_year=time_year_arr,
    )

    save_json(meta, artifacts_dir / "meta.json")

    print("Preprocessing complete")
    print(f"Locations: {num_locs}")
    print(f"Time steps: {T}")
    print(f"Numeric features: {num_features}")
    print(f"Artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()
