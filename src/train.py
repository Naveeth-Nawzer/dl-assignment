import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import WindowLoader, split_indices
from features import FEATURE_NAMES, prepare_dataset
from metrics import horizon_metrics, regression_metrics
from models import build_model


BASE = Path(__file__).resolve().parent.parent


def load_config(model_config_path, experiment_config_path=BASE / "configs" / "experiment.yaml"):
    """
    Merge the shared experiment config with one model config.
    """
    with open(experiment_config_path) as f:
        cfg = yaml.safe_load(f)
    with open(model_config_path) as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict(model, loader, scale):
    """
    Return forecasts and targets in demand units, with forecasts clipped at 0.
    """
    model.eval()
    predictions, targets, rows = [], [], []

    for x, y, index in loader:
        predictions.append(model(x).cpu())
        targets.append(y.cpu())
        rows.append(index.cpu())

    rows = torch.cat(rows).numpy()
    series_scale = scale[rows[:, 0]][:, None]
    y_pred = np.clip(torch.cat(predictions).numpy(), 0, None) * series_scale
    y_true = torch.cat(targets).numpy() * series_scale

    return y_pred, y_true, rows


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn):
    model.eval()
    total, count = 0.0, 0
    for x, y, _ in loader:
        total += loss_fn(model(x), y).item() * len(y)
        count += len(y)
    return total / count


def train_model(cfg, data, seed=42, evaluate_test=False, run_name=None,
                output_dir=BASE / "results", verbose=True):
    """
    Train one model with early stopping on validation loss and save its results.

    The test windows are used only when `evaluate_test` is True.
    """
    set_seed(seed)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    device = get_device(train_cfg["device"])

    if run_name is None:
        params = "_".join(
            f"{key}{value}" for key, value in cfg["model"].items() if key != "name"
        )
        run_name = f"{cfg['model']['name']}_{params}_seed{seed}"

    indices = split_indices(data, data_cfg)
    loaders = {
        split: WindowLoader(
            data, index, data_cfg, train_cfg["batch_size"], device,
            shuffle=(split == "train"), seed=seed,
            max_batches=train_cfg["max_train_batches"] if split == "train" else None,
        )
        for split, index in indices.items()
    }

    model = build_model(cfg["model"], len(FEATURE_NAMES), data_cfg["horizon"]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=train_cfg["lr_factor"], patience=train_cfg["lr_patience"]
    )

    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    train_start = time.perf_counter()

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        epoch_start = time.perf_counter()
        total, count = 0.0, 0

        for x, y, _ in loaders["train"]:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)

        train_loss = total / count
        val_loss = evaluate_loss(model, loaders["val"], loss_fn)
        scheduler.step(val_loss)
        synchronize(device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        })
        if verbose:
            print(
                f"[{run_name}] epoch {epoch:02d} "
                f"train {train_loss:.4f} val {val_loss:.4f} "
                f"({history[-1]['epoch_seconds']:.1f}s)"
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= train_cfg["patience"]:
                break

    train_seconds = time.perf_counter() - train_start
    model.load_state_dict(best_state)

    synchronize(device)
    inference_start = time.perf_counter()
    val_pred, val_true, _ = predict(model, loaders["val"], data["scale"])
    synchronize(device)
    inference_seconds = time.perf_counter() - inference_start

    results = {
        "run_name": run_name,
        "model": cfg["model"],
        "data": data_cfg,
        "training": train_cfg,
        "seed": seed,
        "device": str(device),
        "n_series": int(len(data["scale"])),
        "n_windows": {split: int(len(index)) for split, index in indices.items()},
        "parameters": count_parameters(model),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "inference_ms_per_1000_windows": inference_seconds / len(val_true) * 1e6,
        "val": regression_metrics(val_true, val_pred),
    }

    output_dir = Path(output_dir)
    for folder in ("metrics", "checkpoints", "predictions"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    if evaluate_test:
        test_pred, test_true, test_rows = predict(model, loaders["test"], data["scale"])
        results["test"] = regression_metrics(test_true, test_pred)
        results["test_by_horizon"] = horizon_metrics(test_true, test_pred)
        np.savez(
            output_dir / "predictions" / f"{run_name}_test.npz",
            y_pred=test_pred, y_true=test_true, rows=test_rows,
        )

    torch.save(best_state, output_dir / "checkpoints" / f"{run_name}.pt")
    pd.DataFrame(history).to_csv(
        output_dir / "metrics" / f"{run_name}_history.csv", index=False
    )
    with open(output_dir / "metrics" / f"{run_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    return model, results, history


def main():
    parser = argparse.ArgumentParser(description="Train one forecasting model.")
    parser.add_argument("--config", default=str(BASE / "configs" / "gru.yaml"))
    parser.add_argument("--data-path", default=str(BASE / "data" / "m5" / "extracted"))
    parser.add_argument("--cache", default=str(BASE / "data" / "m5" / "processed" / "features.npz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="Also evaluate on the test period.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["training"]["device"] = args.device
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.max_train_batches:
        cfg["training"]["max_train_batches"] = args.max_train_batches

    data = prepare_dataset(args.data_path, cfg["data"], cache_path=args.cache)
    _, results, _ = train_model(cfg, data, seed=args.seed, evaluate_test=args.test)

    print(json.dumps({key: results[key] for key in ("parameters", "train_seconds", "val")}, indent=2))
    if args.test:
        print(json.dumps({"test": results["test"]}, indent=2))


if __name__ == "__main__":
    main()
