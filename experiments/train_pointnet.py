"""
Trains the project's PointNet classifier on ModelNet40 (2048-point HDF5 release).

    python -m experiments.train_pointnet --epochs 80

Outputs
    models/weights/pointnet_modelnet40.pth            state_dict loaded by PointNetInferenceEngine
    outputs/experiments/pointnet/training_history.json
    outputs/experiments/pointnet/training_curves.png

The final-epoch weights are saved. The test split is only used to draw the
accuracy curve and to report the final number; it is never used to pick a checkpoint.
"""

import argparse
import json
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiments.common import (
    MODELNET_DIR, POINTNET_WEIGHTS, POINTNET_CLASSES, out_dir, save_json, style_axes, C_A, C_B,
)
from src.classification_3d.pointnet import PointNetClassifier, feature_transform_regularizer


def load_split(split: str):
    files = sorted(MODELNET_DIR.glob(f"ply_data_{split}*.h5"))
    if not files:
        raise FileNotFoundError(f"No ModelNet40 '{split}' HDF5 files in {MODELNET_DIR}")
    data, labels = [], []
    for fp in files:
        with h5py.File(fp, "r") as f:
            data.append(f["data"][:].astype(np.float32))
            labels.append(f["label"][:].astype(np.int64).reshape(-1))
    return np.concatenate(data), np.concatenate(labels)


def check_class_order():
    """The config class list must be in the same order as the dataset's label ids."""
    names = (MODELNET_DIR / "shape_names.txt").read_text().split()
    with open(POINTNET_CLASSES, "r") as f:
        cfg = json.load(f)
    if names != cfg:
        raise RuntimeError("config/pointnet_classes.json does not match ModelNet40 shape_names.txt order")
    return names


def augment(batch: torch.Tensor) -> torch.Tensor:
    """batch: (B, N, 3) on device. Rotation about the up (y) axis, anisotropic scale, shift, jitter."""
    b = batch.size(0)
    theta = torch.rand(b, device=batch.device) * 2 * np.pi
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.zeros(b, 3, 3, device=batch.device)
    rot[:, 0, 0], rot[:, 0, 2] = c, s
    rot[:, 1, 1] = 1.0
    rot[:, 2, 0], rot[:, 2, 2] = -s, c
    out = torch.bmm(batch, rot.transpose(1, 2))
    out = out * (torch.rand(b, 1, 3, device=batch.device) * 0.5 + 0.75)
    out = out + (torch.rand(b, 1, 3, device=batch.device) * 0.2 - 0.1)
    out = out + torch.clamp(torch.randn_like(out) * 0.01, -0.05, 0.05)
    return out


@torch.no_grad()
def evaluate(model, x, y, num_points, batch_size=128):
    model.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size, :num_points].transpose(1, 2)
        yb = y[i:i + batch_size]
        log_probs, _, _ = model(xb)
        loss_sum += F.nll_loss(log_probs, yb, reduction="sum").item()
        correct += (log_probs.argmax(1) == yb).sum().item()
    return correct / len(x), loss_sum / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-points", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    classes = check_class_order()
    xtr, ytr = load_split("train")
    xte, yte = load_split("test")
    print(f"[INFO] ModelNet40: {len(xtr)} train / {len(xte)} test shapes, {len(classes)} classes")

    xtr_t, ytr_t = torch.from_numpy(xtr).to(device), torch.from_numpy(ytr).to(device)
    xte_t, yte_t = torch.from_numpy(xte).to(device), torch.from_numpy(yte).to(device)

    model = PointNetClassifier(num_classes=len(classes), feature_transform=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "seconds": []}
    t_start = time.time()
    n = len(xtr_t)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, device=device)
        loss_sum, correct, seen = 0.0, 0, 0
        for i in range(0, n - args.batch_size + 1, args.batch_size):  # drop last partial batch (BatchNorm)
            idx = perm[i:i + args.batch_size]
            pts_idx = torch.randperm(xtr_t.size(1), device=device)[:args.num_points]
            xb = augment(xtr_t[idx][:, pts_idx]).transpose(1, 2)
            yb = ytr_t[idx]
            log_probs, _, trans_feat = model(xb)
            loss = F.nll_loss(log_probs, yb) + 0.001 * feature_transform_regularizer(trans_feat)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
            correct += (log_probs.argmax(1) == yb).sum().item()
            seen += len(idx)
        sched.step()

        test_acc, test_loss = evaluate(model, xte_t, yte_t, args.num_points)
        history["epoch"].append(epoch)
        history["train_loss"].append(loss_sum / seen)
        history["train_acc"].append(correct / seen)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["seconds"].append(time.time() - t0)
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {loss_sum / seen:.4f} acc {correct / seen:.4f} "
              f"| test loss {test_loss:.4f} acc {test_acc:.4f} | {time.time() - t0:.1f}s", flush=True)

    total_time = time.time() - t_start
    POINTNET_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, POINTNET_WEIGHTS)
    print(f"[INFO] Saved weights: {POINTNET_WEIGHTS}")

    out = out_dir("pointnet")
    history["config"] = vars(args)
    history["device"] = str(device)
    history["total_training_seconds"] = round(total_time, 1)
    history["final_test_accuracy"] = history["test_acc"][-1]
    history["num_train_shapes"] = int(len(xtr))
    history["num_test_shapes"] = int(len(xte))
    save_json(history, out / "training_history.json")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(history["epoch"], history["train_loss"], color=C_A, lw=2, label="Train")
    axs[0].plot(history["epoch"], history["test_loss"], color=C_B, lw=2, label="Test")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("NLL loss"); axs[0].set_title("PointNet loss on ModelNet40")
    axs[0].legend(frameon=False); style_axes(axs[0])
    axs[1].plot(history["epoch"], np.array(history["train_acc"]) * 100, color=C_A, lw=2, label="Train (augmented)")
    axs[1].plot(history["epoch"], np.array(history["test_acc"]) * 100, color=C_B, lw=2, label="Test")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_title(f"PointNet accuracy (final test: {history['test_acc'][-1] * 100:.1f}%)")
    axs[1].legend(frameon=False); style_axes(axs[1])
    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=150)
    plt.close()
    print(f"[INFO] Final test accuracy: {history['test_acc'][-1] * 100:.2f}%  (training time {total_time / 60:.1f} min)")


if __name__ == "__main__":
    main()
