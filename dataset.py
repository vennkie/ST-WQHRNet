from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_json


def load_npz(npz_path: Path):
    # Load arrays and close the underlying file handle so the dataset is pickle-safe.
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


class STWQHRDataset(Dataset):
    def __init__(self, npz_path: Path, meta_path: Path, split: str, config: dict):
        data = load_npz(npz_path)
        self.meta = load_json(meta_path)

        self.split = split
        self.start_year = config["data"]["start_year"]
        self.train_end_year = config["data"]["train_end_year"]
        self.val_year = config["data"]["val_year"]
        self.test_year = config["data"]["test_year"]

        self.x_num = data["x_num"]
        self.x_season = data["x_season"]
        self.x_source = data["x_source"]
        self.y_wqi = data["y_wqi"]
        self.y_hhi = data["y_hhi"]
        self.mask = data["mask"]
        self.loc_cat = data["loc_cat"]
        self.latlon = data["latlon"]
        self.time_year = data["time_year"]
        self.district_feat = data["district_feat"]
        self.year_arr = data["year_arr"]

        self.max_time_index = int(self.meta["max_time_index"])

        self.max_t = self._max_t_for_split(split)
        self.year_mask = self._year_mask_for_split(split)[:, : self.max_t]

        # Keep only locations that have at least one labeled step in this split.
        split_mask = self.mask[:, : self.max_t] & self.year_mask
        self.indices = np.where(split_mask.sum(axis=1) > 0)[0]

    def _max_t_for_split(self, split: str) -> int:
        if split == "train":
            return (self.train_end_year - self.start_year + 1) * 3
        if split == "val":
            return (self.val_year - self.start_year + 1) * 3
        if split == "test":
            return (self.test_year - self.start_year + 1) * 3
        return self.max_time_index

    def _year_mask_for_split(self, split: str):
        if split == "train":
            years = set(range(self.start_year, self.train_end_year + 1))
        elif split == "val":
            years = {self.val_year}
        elif split == "test":
            years = {self.test_year}
        else:
            years = set(np.unique(self.year_arr[self.year_arr >= 0]).tolist())
        return np.isin(self.year_arr, list(years))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x_num = torch.from_numpy(self.x_num[real_idx, : self.max_t]).float()
        x_season = torch.from_numpy(self.x_season[real_idx, : self.max_t]).long()
        x_source = torch.from_numpy(self.x_source[real_idx, : self.max_t]).long()
        y_wqi = torch.from_numpy(self.y_wqi[real_idx, : self.max_t]).long()
        y_hhi = torch.from_numpy(self.y_hhi[real_idx, : self.max_t]).long()
        valid_mask = torch.from_numpy(self.mask[real_idx, : self.max_t]).bool()
        loss_mask = valid_mask & torch.from_numpy(self.year_mask[real_idx]).bool()
        loc_cat = torch.from_numpy(self.loc_cat[real_idx]).long()
        latlon = torch.from_numpy(self.latlon[real_idx]).float()
        district_feat = torch.from_numpy(self.district_feat[real_idx]).float()

        return {
            "x_num": x_num,
            "x_season": x_season,
            "x_source": x_source,
            "y_wqi": y_wqi,
            "y_hhi": y_hhi,
            "valid_mask": valid_mask,
            "loss_mask": loss_mask,
            "loc_cat": loc_cat,
            "latlon": latlon,
            "district_feat": district_feat,
        }
