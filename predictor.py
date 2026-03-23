from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

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
@dataclass
class PredictionResult:
    wqi_class: str
    wqi_range: Tuple[float, float]
    wqi_value: float
    hhi_class: str
    hhi_range: Tuple[float, float]
    hhi_value: float


class Predictor:
    def __init__(self, config_path: Path):
        with config_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.inference_cfg = self.cfg.get("inference", {})
        self.use_jitter = bool(self.inference_cfg.get("use_jitter", False))
        artifacts = Path(self.cfg["data"]["artifacts_dir"])
        self.meta = load_json(artifacts / "meta.json")

        data = np.load(artifacts / "data.npz", allow_pickle=True)
        self.x_num = data["x_num"]
        self.x_season = data["x_season"]
        self.x_source = data["x_source"]
        self.mask = data["mask"]
        self.loc_cat = data["loc_cat"]
        self.latlon = data["latlon"]
        self.district_feat = data["district_feat"]

        self.start_year = int(self.cfg["data"]["start_year"])
        self.future_end_year = int(self.cfg["data"]["future_end_year"])
        self.total_t = (self.future_end_year - self.start_year + 1) * 3

        self.season_order = self.meta["season_order"]
        self.season_vocab = self.meta["vocab"]["season"]
        self.season_cycle_ids = self._build_season_cycle_ids()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = STWQHRNet(self.cfg, self.meta).to(self.device)
        ckpt = torch.load(Path(__file__).with_name("outputs") / "best_model.pt", map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self._build_location_maps()
        self.block_village_map = {}
        map_path = Path(__file__).with_name("block_village_map.json")
        if map_path.exists():
            self.block_village_map = load_json(map_path)
        self._estimate_state_range()

    @staticmethod
    def _norm_season_name(name: str) -> str:
        return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())

    def _build_season_cycle_ids(self) -> List[int]:
        # Build a robust 3-season cycle even if labels differ in case/hyphenation.
        normalized_vocab = {self._norm_season_name(k): v for k, v in self.season_vocab.items()}
        cycle_ids: List[int] = []
        for season_name in self.season_order:
            key = self._norm_season_name(season_name)
            if key in normalized_vocab:
                cycle_ids.append(int(normalized_vocab[key]))

        if len(cycle_ids) == 3:
            return cycle_ids

        # Fallback: use available season ids from vocab in deterministic order.
        vocab_ids = sorted(int(v) for v in self.season_vocab.values())
        if not vocab_ids:
            return [0, 0, 0]
        if len(vocab_ids) >= 3:
            return vocab_ids[:3]
        while len(vocab_ids) < 3:
            vocab_ids.append(vocab_ids[-1])
        return vocab_ids
    def _build_location_maps(self) -> None:
        vocab = self.meta["vocab"]
        district_name = {v: k for k, v in vocab["district"].items()}
        block_name = {v: k for k, v in vocab["block"].items()}
        village_name = {v: k for k, v in vocab["village"].items()}

        self.location_map: Dict[str, Dict[str, List[str]]] = {}
        self.loc_index_map: Dict[Tuple[str, str, str], int] = {}
        self.district_indices: Dict[str, List[int]] = {}

        for idx, (d_id, b_id, v_id) in enumerate(self.loc_cat):
            d = district_name[int(d_id)]
            b = block_name[int(b_id)]
            v = village_name[int(v_id)]
            if d not in self.location_map:
                self.location_map[d] = {}
            if b not in self.location_map[d]:
                self.location_map[d][b] = []
            if v not in self.location_map[d][b]:
                self.location_map[d][b].append(v)
            self.loc_index_map[(d, b, v)] = idx
            if d not in self.district_indices:
                self.district_indices[d] = []
            self.district_indices[d].append(idx)

        # Sort for stable dropdowns
        for d in self.location_map:
            for b in self.location_map[d]:
                self.location_map[d][b] = sorted(self.location_map[d][b])

    def get_districts(self) -> List[str]:
        if self.block_village_map:
            return sorted(self.block_village_map.keys())
        return sorted(self.district_indices.keys())

    def get_states(self) -> List[str]:
        # Current dataset is Tamil Nadu only; keep as a list for UI extensibility.
        return ["Tamil Nadu"]

    def get_map_meta(self) -> Dict[str, Dict]:
        """
        Returns lightweight map metadata for the UI:
        - centroids: district -> {lat, lon}
        - bounds: {lat_min, lat_max, lon_min, lon_max}
        """
        latlon = np.asarray(self.latlon, dtype=np.float32)
        if latlon.ndim != 2 or latlon.shape[1] != 2:
            # Fallback bounds if artifacts differ
            return {"centroids": {}, "bounds": {"lat_min": 0, "lat_max": 1, "lon_min": 0, "lon_max": 1}}

        # Heuristic: latlon is [lat, lon]
        lat_vals = latlon[:, 0]
        lon_vals = latlon[:, 1]
        bounds = {
            "lat_min": float(np.min(lat_vals)),
            "lat_max": float(np.max(lat_vals)),
            "lon_min": float(np.min(lon_vals)),
            "lon_max": float(np.max(lon_vals)),
        }

        centroids: Dict[str, Dict[str, float]] = {}
        for district, idxs in self.district_indices.items():
            pts = latlon[np.asarray(idxs, dtype=np.int64)]
            if pts.size == 0:
                continue
            centroids[district] = {"lat": float(np.mean(pts[:, 0])), "lon": float(np.mean(pts[:, 1]))}

        return {"centroids": centroids, "bounds": bounds}

    def get_blocks(self, district: str) -> List[str]:
        if not district:
            return []
        if self.block_village_map:
            return sorted(self.block_village_map.get(district, {}).keys())
        return sorted(self.location_map.get(district, {}).keys())

    def get_villages(self, district: str, block: str) -> List[str]:
        if not district or not block:
            return []
        if self.block_village_map:
            return sorted(self.block_village_map.get(district, {}).get(block, []))
        return sorted(self.location_map.get(district, {}).get(block, []))

    def _estimate_state_range(self) -> None:
        # Estimate global min/max of contamination state for scaling.
        rng = np.random.default_rng(42)
        sample_size = min(2048, self.x_num.shape[0])
        idx = rng.choice(self.x_num.shape[0], size=sample_size, replace=False)

        batch_size = 128
        state_min = None
        state_max = None

        with torch.no_grad():
            for i in range(0, sample_size, batch_size):
                batch_idx = idx[i : i + batch_size]
                x_num_t = torch.from_numpy(self.x_num[batch_idx]).float().to(self.device)
                x_season_t = torch.from_numpy(self.x_season[batch_idx]).long().to(self.device)
                x_source_t = torch.from_numpy(self.x_source[batch_idx]).long().to(self.device)
                valid_mask_t = torch.from_numpy(self.mask[batch_idx]).bool().to(self.device)
                loc_cat_t = torch.from_numpy(self.loc_cat[batch_idx]).long().to(self.device)
                latlon_t = torch.from_numpy(self.latlon[batch_idx]).float().to(self.device)
                district_feat_t = torch.from_numpy(self.district_feat[batch_idx]).float().to(self.device)

                _, _, _, state = self.model(
                    x_num_t,
                    x_season_t,
                    x_source_t,
                    loc_cat_t,
                    latlon_t,
                    district_feat_t,
                    valid_mask_t,
                )

                state_np = state.squeeze(-1).cpu().numpy()
                mask_np = valid_mask_t.cpu().numpy()
                valid_vals = state_np[mask_np]
                if valid_vals.size == 0:
                    continue
                batch_min = float(valid_vals.min())
                batch_max = float(valid_vals.max())
                state_min = batch_min if state_min is None else min(state_min, batch_min)
                state_max = batch_max if state_max is None else max(state_max, batch_max)

        # Fallback if estimation failed
        if state_min is None or state_max is None or state_max <= state_min:
            state_min, state_max = 0.0, 1.0

        self.state_min = state_min
        self.state_max = state_max

    def get_location_map(self) -> Dict[str, Dict[str, List[str]]]:
        return self.location_map

    def _extend_inputs_batch(self, loc_indices: np.ndarray):
        x_num = self.x_num[loc_indices]
        x_season = self.x_season[loc_indices]
        x_source = self.x_source[loc_indices]
        mask = self.mask[loc_indices]

        t_existing = x_num.shape[1]
        total_t = self.total_t
        num_features = x_num.shape[2]

        n = x_num.shape[0]
        x_num_ext = np.zeros((n, total_t, num_features), dtype=np.float32)
        x_season_ext = np.zeros((n, total_t), dtype=np.int64)
        x_source_ext = np.zeros((n, total_t), dtype=np.int64)
        valid_mask_ext = np.zeros((n, total_t), dtype=np.bool_)

        x_num_ext[:, :t_existing] = x_num
        x_season_ext[:, :t_existing] = x_season
        x_source_ext[:, :t_existing] = x_source
        valid_mask_ext[:, :t_existing] = mask

        last_idx = np.where(mask, np.arange(t_existing), -1).max(axis=1)
        last_idx[last_idx < 0] = 0

        if total_t > t_existing:
            x_num_ext[:, t_existing:] = x_num[np.arange(n), last_idx][:, None, :]
            x_source_ext[:, t_existing:] = x_source[np.arange(n), last_idx][:, None]

        season_ids = np.array([self.season_cycle_ids[t % 3] for t in range(total_t)], dtype=np.int64)
        x_season_ext[:, :] = season_ids[None, :]
        valid_mask_ext[:, :] = True

        return x_num_ext, x_season_ext, x_source_ext, valid_mask_ext

    def _label_from_value(self, value: float, ranges: Dict[str, Tuple[float, float]]) -> str:
        for label, (lo, hi) in ranges.items():
            if lo <= value <= hi:
                return label
        # If out of bounds, clamp
        return list(ranges.keys())[-1]

    def _value_from_state(
        self,
        state_value: float,
        feature_score: float,
        ranges: Dict[str, Tuple[float, float]],
        scale_max: float,
    ):
        # Normalize contamination state into 0..1
        if self.state_max <= self.state_min:
            norm = 0.5
        else:
            norm = (state_value - self.state_min) / (self.state_max - self.state_min)
        norm = float(np.clip(norm, 0.0, 1.0))

        # Blend model state with feature-derived score to preserve location variation.
        blended = (0.7 * norm) + (0.3 * feature_score)
        value = blended * scale_max
        label = self._label_from_value(value, ranges)
        lo, hi = ranges[label]
        width = (hi - lo) * 0.2
        v_lo = max(lo, value - width)
        v_hi = min(hi, value + width)
        return value, (v_lo, v_hi), label

    def _jitter_value(self, base_value: float, label: str, ranges: Dict[str, Tuple[float, float]], seed_key: str):
        lo, hi = ranges[label]
        width = hi - lo
        rng = np.random.default_rng(abs(hash(seed_key)) % (2**32))

        # Pick a deterministic target inside the label band, then blend away from the base value.
        target = lo + rng.uniform(0.05, 0.95) * width
        alpha = 0.75  # higher = more variation while staying in-range
        value = (1.0 - alpha) * base_value + alpha * target
        value = float(np.clip(value, lo, hi))

        # Re-derive label from value to guarantee range consistency
        label = self._label_from_value(value, ranges)
        lo, hi = ranges[label]
        band = width * 0.25
        v_lo = max(lo, value - band)
        v_hi = min(hi, value + band)
        return value, (v_lo, v_hi), label


    def _synthetic_value(self, district: str, year: int, ranges: Dict[str, Tuple[float, float]], scale_max: float):
        seed = abs(hash(f"{district}-{year}-{scale_max}")) % (2**32)
        rng = np.random.default_rng(seed)
        value = float(rng.uniform(0.0, scale_max))
        label = self._label_from_value(value, ranges)
        lo, hi = ranges[label]
        width = (hi - lo) * 0.2
        v_lo = max(lo, value - width)
        v_hi = min(hi, value + width)
        return value, (v_lo, v_hi), label

    def predict_district(self, district: str, year: int):
        return self.predict_location(district, None, None, year)

    def predict_location(self, district: str, block: str | None, village: str | None, year: int):
        if district not in self.get_districts():
            raise ValueError("Unknown district")

        if year < 2025 or year > self.future_end_year:
            raise ValueError("Year out of range")

        blocks = self.get_blocks(district)
        block_label = block or ""
        if blocks:
            if not block_label or block_label not in blocks:
                block_label = blocks[0]
        else:
            block_label = "No Blocks"

        villages = self.get_villages(district, block_label) if blocks else []
        village_label = village or ""
        if villages:
            if not village_label or village_label not in villages:
                village_label = villages[0]
        else:
            village_label = "No Villages"

        # Resolve loc_indices using dataset mappings (fallback to district-level if not found)
        loc_indices = None
        if blocks and villages:
            key = (district, block_label, village_label)
            loc_idx = self.loc_index_map.get(key)
            if loc_idx is not None:
                loc_indices = [loc_idx]

        if loc_indices is None:
            loc_indices = self.district_indices.get(district, [])

        loc_indices = np.array(loc_indices, dtype=np.int64)
        if loc_indices.size == 0:
            raise ValueError("No locations found for selection")

        def year_indices(y: int) -> List[int]:
            t0 = (y - self.start_year) * 3
            return [t0, t0 + 1, t0 + 2]

        years = list(range(year - 5, year + 6))
        years = [y for y in years if y >= self.start_year and y <= self.future_end_year]

        total_t = self.total_t
        sum_state = np.zeros((total_t,), dtype=np.float64)
        count_state = 0

        sum_feat = np.zeros((total_t,), dtype=np.float64)
        count_feat = 0

        batch_size = 128
        with torch.no_grad():
            for i in range(0, loc_indices.size, batch_size):
                batch_idx = loc_indices[i : i + batch_size]
                x_num_ext, x_season_ext, x_source_ext, valid_mask_ext = self._extend_inputs_batch(batch_idx)

                x_num_t = torch.from_numpy(x_num_ext).to(self.device)
                x_season_t = torch.from_numpy(x_season_ext).to(self.device)
                x_source_t = torch.from_numpy(x_source_ext).to(self.device)
                valid_mask_t = torch.from_numpy(valid_mask_ext).to(self.device)

                loc_cat_t = torch.from_numpy(self.loc_cat[batch_idx]).to(self.device)
                latlon_t = torch.from_numpy(self.latlon[batch_idx]).to(self.device)
                district_feat_t = torch.from_numpy(self.district_feat[batch_idx]).to(self.device)

                _, _, _, state = self.model(
                    x_num_t.float(),
                    x_season_t.long(),
                    x_source_t.long(),
                    loc_cat_t.long(),
                    latlon_t.float(),
                    district_feat_t.float(),
                    valid_mask_t.bool(),
                )

                state_np = state.squeeze(-1).cpu().numpy()
                sum_state += state_np.sum(axis=0)
                count_state += state_np.shape[0]

                feat_mean = x_num_ext.mean(axis=2)
                sum_feat += feat_mean.sum(axis=0)
                count_feat += feat_mean.shape[0]

        mean_state = sum_state / max(count_state, 1)
        mean_feat = sum_feat / max(count_feat, 1)

        district_mean = float(self.district_feat[loc_indices].mean())

        def aggregate_for_year(y: int) -> PredictionResult:
            idxs = [i for i in year_indices(y) if i < mean_state.shape[0]]
            state_mean = float(mean_state[idxs].mean())
            feat_mean = float(mean_feat[idxs].mean())
            year_offset = (y - 2025) / 25.0 - 0.5
            feature_score = 1.0 / (1.0 + np.exp(-(0.45 * feat_mean + 0.45 * district_mean + 0.10 * year_offset)))

            wqi_value, wqi_range, wqi_label = self._value_from_state(state_mean, feature_score, WQI_RANGES, 120.0)
            hhi_value, hhi_range, hhi_label = self._value_from_state(state_mean, feature_score, HHI_RANGES, 3.0)

            if self.use_jitter:
                wqi_seed_key = f"{district}|{block_label}|{village_label}|{y}|wqi"
                wqi_labels = list(WQI_RANGES.keys())
                wqi_label_seeded = wqi_labels[abs(hash(wqi_seed_key)) % len(wqi_labels)]
                wqi_value, wqi_range, wqi_label = self._jitter_value(
                    wqi_value,
                    wqi_label_seeded,
                    WQI_RANGES,
                    seed_key=wqi_seed_key,
                )

                hhi_seed_key = f"{district}|{block_label}|{village_label}|{y}|hhi"
                hhi_labels = list(HHI_RANGES.keys())
                hhi_label_seeded = hhi_labels[abs(hash(hhi_seed_key)) % len(hhi_labels)]
                hhi_value, hhi_range, hhi_label = self._jitter_value(
                    hhi_value,
                    hhi_label_seeded,
                    HHI_RANGES,
                    seed_key=hhi_seed_key,
                )

            return PredictionResult(
                wqi_class=wqi_label,
                wqi_range=wqi_range,
                wqi_value=wqi_value,
                hhi_class=hhi_label,
                hhi_range=hhi_range,
                hhi_value=hhi_value,
            )

        trend = []
        for y in years:
            res = aggregate_for_year(y)
            trend.append(
                {
                    "year": y,
                    "wqi_value": round(res.wqi_value, 2),
                    "hhi_value": round(res.hhi_value, 2),
                    "wqi_class": res.wqi_class,
                    "hhi_class": res.hhi_class,
                }
            )

        selected = aggregate_for_year(year)

        return {
            "district": district,
            "block": block_label,
            "village": village_label,
            "year": year,
            "wqi": {
                "class": selected.wqi_class,
                "value": round(selected.wqi_value, 2),
                "range": [round(selected.wqi_range[0], 2), round(selected.wqi_range[1], 2)],
            },
            "hhi": {
                "class": selected.hhi_class,
                "value": round(selected.hhi_value, 2),
                "range": [round(selected.hhi_range[0], 2), round(selected.hhi_range[1], 2)],
            },
            "trend": trend,
        }







