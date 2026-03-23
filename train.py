import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from data_prep import main as run_prep
from dataset import STWQHRDataset
from model import STWQHRNet
from utils import set_seed, load_json, save_json, accuracy_from_logits, macro_f1_from_logits


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, weight=None):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    targets = targets.long()
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    if weight is not None:
        at = weight[targets]
    else:
        at = alpha
    loss = -at * (1 - pt).pow(gamma) * log_pt
    return loss.mean()


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_artifacts(cfg):
    artifacts_dir = Path(cfg["data"]["artifacts_dir"])
    if not (artifacts_dir / "data.npz").exists() or not (artifacts_dir / "meta.json").exists():
        run_prep()


def plot_loss_curves(history, out_path: Path):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    phases = history["phase"]

    if not train_loss:
        return

    x = list(range(1, len(train_loss) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, train_loss, label="Train Loss", color="#2563eb", linewidth=2)
    ax.plot(x, val_loss, label="Val Loss", color="#dc2626", linewidth=2)

    # Mark transition from pretraining to multitask training.
    if "multitask" in phases and "pretrain" in phases:
        split_idx = phases.index("multitask") + 1
        ax.axvline(split_idx - 0.5, color="#6b7280", linestyle="--", linewidth=1)
        ax.text(split_idx, max(max(train_loss), max(val_loss)) * 0.98, "Stage 2", fontsize=9, color="#374151")

    ax.set_xlabel("Epoch Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    wqi_weights,
    hhi_weights,
    train=True,
    focal_cfg=None,
    wqi_loss_weight=1.0,
    hhi_loss_weight=1.0,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_wqi_acc = 0.0
    total_hhi_acc = 0.0
    total_wqi_f1 = 0.0
    total_hhi_f1 = 0.0
    steps = 0

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

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            wqi_logits, hhi_logits, _, _ = model(x_num, x_season, x_source, loc_cat, latlon, district_feat, valid_mask)
            wqi_logits = torch.nan_to_num(wqi_logits)
            hhi_logits = torch.nan_to_num(hhi_logits)

            mask_flat = loss_mask.view(-1)
            if mask_flat.sum().item() == 0:
                continue

            wqi_flat = wqi_logits.view(-1, wqi_logits.size(-1))[mask_flat]
            y_wqi_flat = y_wqi.view(-1)[mask_flat]
            if focal_cfg is not None:
                wqi_loss = focal_loss(wqi_flat, y_wqi_flat, alpha=focal_cfg["alpha"], gamma=focal_cfg["gamma"], weight=wqi_weights)
            else:
                wqi_loss = F.cross_entropy(wqi_flat, y_wqi_flat, weight=wqi_weights)
            hhi_loss = F.cross_entropy(hhi_logits.view(-1, hhi_logits.size(-1))[mask_flat], y_hhi.view(-1)[mask_flat], weight=hhi_weights)
            loss = (wqi_loss_weight * wqi_loss) + (hhi_loss_weight * hhi_loss)
            if not torch.isfinite(loss):
                continue

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()
        total_wqi_acc += accuracy_from_logits(wqi_logits, y_wqi, loss_mask)
        total_hhi_acc += accuracy_from_logits(hhi_logits, y_hhi, loss_mask)
        total_wqi_f1 += macro_f1_from_logits(wqi_logits, y_wqi, loss_mask, wqi_logits.size(-1))
        total_hhi_f1 += macro_f1_from_logits(hhi_logits, y_hhi, loss_mask, hhi_logits.size(-1))
        steps += 1

    if steps == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / steps,
        total_wqi_acc / steps,
        total_hhi_acc / steps,
        total_wqi_f1 / steps,
        total_hhi_f1 / steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    ensure_artifacts(cfg)

    set_seed(cfg["train"]["seed"])

    artifacts_dir = Path(cfg["data"]["artifacts_dir"])
    meta = load_json(artifacts_dir / "meta.json")

    train_ds = STWQHRDataset(artifacts_dir / "data.npz", artifacts_dir / "meta.json", "train", cfg)
    val_ds = STWQHRDataset(artifacts_dir / "data.npz", artifacts_dir / "meta.json", "val", cfg)

    # Windows multiprocessing can be unstable for large numpy-backed datasets.
    num_workers = cfg["train"]["num_workers"]
    if torch.cuda.is_available() is False:
        num_workers = 0
    if torch.multiprocessing.get_start_method(allow_none=True) == "spawn":
        num_workers = 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = STWQHRNet(cfg, meta).to(device)

    wqi_weights = torch.tensor(meta["class_weights"]["wqi"], dtype=torch.float32, device=device)
    hhi_weights = torch.tensor(meta["class_weights"]["hhi"], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    outputs_dir = Path(__file__).with_name("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    best_path = outputs_dir / "best_model.pt"

    history = {
        "phase": [],
        "train_loss": [],
        "val_loss": [],
        "train_wqi_acc": [],
        "val_wqi_acc": [],
        "train_hhi_acc": [],
        "val_hhi_acc": [],
        "train_wqi_f1": [],
        "val_wqi_f1": [],
        "train_hhi_f1": [],
        "val_hhi_f1": [],
    }

    # Stage 1: WQI-only pretraining
    pretrain_epochs = int(cfg["train"].get("wqi_pretrain_epochs", 0))
    if pretrain_epochs > 0:
        print(f"Stage 1: WQI-only pretraining for {pretrain_epochs} epochs")
        focal_cfg = None
        if cfg["train"].get("use_focal_wqi", False):
            focal_cfg = {"alpha": cfg["model"]["focal_alpha"], "gamma": cfg["model"]["focal_gamma"]}

        for epoch in range(1, pretrain_epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                wqi_weights,
                hhi_weights,
                train=True,
                focal_cfg=focal_cfg,
                wqi_loss_weight=1.0,
                hhi_loss_weight=0.0,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer,
                device,
                wqi_weights,
                hhi_weights,
                train=False,
                focal_cfg=focal_cfg,
                wqi_loss_weight=1.0,
                hhi_loss_weight=0.0,
            )

            train_loss, train_wqi_acc, train_hhi_acc, train_wqi_f1, train_hhi_f1 = train_metrics
            val_loss, val_wqi_acc, val_hhi_acc, val_wqi_f1, val_hhi_f1 = val_metrics

            history["phase"].append("pretrain")
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_wqi_acc"].append(train_wqi_acc)
            history["val_wqi_acc"].append(val_wqi_acc)
            history["train_hhi_acc"].append(train_hhi_acc)
            history["val_hhi_acc"].append(val_hhi_acc)
            history["train_wqi_f1"].append(train_wqi_f1)
            history["val_wqi_f1"].append(val_wqi_f1)
            history["train_hhi_f1"].append(train_hhi_f1)
            history["val_hhi_f1"].append(val_hhi_f1)

            print(
                f"Pretrain {epoch:02d} | "
                f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
                f"Train WQI Acc {train_wqi_acc:.3f} | Val WQI Acc {val_wqi_acc:.3f}"
            )

    # Stage 2: Multi-task training
    best_val = float("inf")
    patience = cfg["train"]["early_stopping"]
    patience_left = patience

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        focal_cfg = None
        if cfg["train"].get("use_focal_wqi", False):
            focal_cfg = {"alpha": cfg["model"]["focal_alpha"], "gamma": cfg["model"]["focal_gamma"]}

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            wqi_weights,
            hhi_weights,
            train=True,
            focal_cfg=focal_cfg,
            wqi_loss_weight=cfg["train"].get("wqi_loss_weight", 1.0),
            hhi_loss_weight=cfg["train"].get("hhi_loss_weight", 1.0),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            wqi_weights,
            hhi_weights,
            train=False,
            focal_cfg=focal_cfg,
            wqi_loss_weight=cfg["train"].get("wqi_loss_weight", 1.0),
            hhi_loss_weight=cfg["train"].get("hhi_loss_weight", 1.0),
        )

        train_loss, train_wqi_acc, train_hhi_acc, train_wqi_f1, train_hhi_f1 = train_metrics
        val_loss, val_wqi_acc, val_hhi_acc, val_wqi_f1, val_hhi_f1 = val_metrics

        history["phase"].append("multitask")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_wqi_acc"].append(train_wqi_acc)
        history["val_wqi_acc"].append(val_wqi_acc)
        history["train_hhi_acc"].append(train_hhi_acc)
        history["val_hhi_acc"].append(val_hhi_acc)
        history["train_wqi_f1"].append(train_wqi_f1)
        history["val_wqi_f1"].append(val_wqi_f1)
        history["train_hhi_f1"].append(train_hhi_f1)
        history["val_hhi_f1"].append(val_hhi_f1)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"Train WQI Acc {train_wqi_acc:.3f} | Val WQI Acc {val_wqi_acc:.3f} | "
            f"Train HHI Acc {train_hhi_acc:.3f} | Val HHI Acc {val_hhi_acc:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
            torch.save({"model_state": model.state_dict(), "config": cfg}, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping")
                break

    save_json(history, outputs_dir / "train_history.json")
    plot_loss_curves(history, outputs_dir / "loss_curve.png")

    print(f"Best model saved to {best_path}")
    print(f"Training history saved to {outputs_dir / 'train_history.json'}")
    print(f"Loss curve saved to {outputs_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
