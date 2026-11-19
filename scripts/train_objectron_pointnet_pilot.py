"""Train an explicitly experimental PointNet on accepted Objectron SfM crops.

The official Objectron test split is read only for final evaluation and never
participates in optimizer updates or checkpoint selection. Validation videos
are selected by source-video hash within the official train split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import open3d as o3d
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classification_3d.pointnet import PointNetClassifier, feature_transform_regularizer  # noqa: E402
from src.classification_3d.point_cloud_sampler import (  # noqa: E402
    augment_partial_real_scan, normalize_point_cloud, sample_point_cloud,
)
from src.classification_3d.pointnet_classifier import PointNetInferenceEngine  # noqa: E402

CLASSES = ["bicycle", "book", "bottle", "camera", "cereal_box", "chair", "cup", "laptop", "shoe"]


def read_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in data.get("samples", []):
        if row.get("label") not in CLASSES:
            continue
        cloud = Path(row.get("point_cloud", ""))
        if not cloud.is_file():
            continue
        records.append({**row, "point_cloud": cloud.resolve(), "label_id": CLASSES.index(row["label"])})
    return records


def read_points(record: dict) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(record["point_cloud"]))
    pts = np.asarray(cloud.points, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        raise ValueError(f"Too few finite points: {record['point_cloud']}")
    pts = PointNetInferenceEngine.canonicalize_to_modelnet_frame(pts)
    norm, _, _ = normalize_point_cloud(pts)
    return norm.astype(np.float32)


def stable_validation_groups(records: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_class: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_class[record["label_id"]].append(record)
    train, val = [], []
    for label, rows in by_class.items():
        rows = sorted(rows, key=lambda r: hashlib.sha256(
            f"{seed}:{r['group_id']}".encode("utf-8")).hexdigest())
        n_val = max(1, int(round(len(rows) * fraction))) if len(rows) >= 4 else 1
        n_val = min(n_val, len(rows) - 1)
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    return train, val


def sampled(points: np.ndarray, train: bool, n_points: int) -> np.ndarray:
    pts = points
    if train:
        pts = augment_partial_real_scan(pts, dropout_ratio=0.12, outlier_ratio=0.01, partial_view=True)
        pts, _, _ = normalize_point_cloud(pts)
        return sample_point_cloud(pts, target_num=n_points, method="random", jitter_upsample=True)
    return sample_point_cloud(pts, target_num=n_points, method="fps", jitter_upsample=False)


def batch_arrays(rows: list[dict], point_cache: dict[str, np.ndarray], train: bool, n_points: int):
    x = np.stack([sampled(point_cache[str(r["point_cloud"])], train, n_points) for r in rows])
    y = np.asarray([r["label_id"] for r in rows], dtype=np.int64)
    return torch.from_numpy(x.transpose(0, 2, 1).copy()), torch.from_numpy(y)


def evaluate(model, rows, point_cache, device, n_points, batch_size=8):
    model.eval()
    all_truth, all_pred, all_conf = [], [], []
    loss_fn = nn.NLLLoss()
    losses = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            x, y = batch_arrays(batch, point_cache, False, n_points)
            x, y = x.to(device), y.to(device)
            logp, _, _ = model(x)
            losses.append(float(loss_fn(logp, y).item()))
            probs = logp.exp()
            conf, pred = probs.max(dim=1)
            all_truth.extend(y.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
            all_conf.extend(conf.cpu().tolist())
    correct = sum(t == p for t, p in zip(all_truth, all_pred))
    per_class = {}
    for label_id, label in enumerate(CLASSES):
        indices = [i for i, t in enumerate(all_truth) if t == label_id]
        per_class[label] = {
            "support": len(indices),
            "correct": sum(all_truth[i] == all_pred[i] for i in indices),
            "accuracy": (sum(all_truth[i] == all_pred[i] for i in indices) / len(indices)) if indices else None,
        }
    confusion = [[0 for _ in CLASSES] for _ in CLASSES]
    for t, p in zip(all_truth, all_pred):
        confusion[t][p] += 1
    return {
        "samples": len(rows), "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "mean_nll": float(np.mean(losses)) if losses else None,
        "mean_max_softmax_uncalibrated": float(np.mean(all_conf)) if all_conf else None,
        "per_class": per_class, "confusion_matrix": confusion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "data/objectron_box_dataset/manifest.json")
    parser.add_argument("--test-manifest", type=Path, default=ROOT / "data/objectron_box_test_dataset/manifest.json")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "models/weights/pointnet_modelnet40.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/objectron_pointnet_pilot")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_all = read_manifest(args.train_manifest.resolve())
    test = read_manifest(args.test_manifest.resolve())
    if not train_all or not test:
        raise SystemExit("Train and official test manifests must each contain accepted cloud files.")
    train_rows, val_rows = stable_validation_groups(train_all, 0.2, args.seed)
    if set(r["group_id"] for r in train_rows) & set(r["group_id"] for r in val_rows):
        raise SystemExit("Source-video group leaked between training and validation.")
    train_hashes = {r["group_id"] for r in train_all}
    test_hashes = {r["group_id"] for r in test}
    if train_hashes & test_hashes:
        raise SystemExit("Official train/test source-video hashes overlap; refusing to train.")

    point_cache: dict[str, np.ndarray] = {}
    for row in [*train_all, *test]:
        key = str(row["point_cloud"])
        if key not in point_cache:
            point_cache[key] = read_points(row)

    model = PointNetClassifier(num_classes=len(CLASSES), feature_transform=True).to(device)
    transfer = {k: v for k, v in torch.load(args.pretrained, map_location="cpu", weights_only=True).items()
                if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    load_result = model.load_state_dict(transfer, strict=False)
    class_counts = Counter(r["label_id"] for r in train_rows)
    weights = torch.tensor([len(train_rows) / max(1, len(CLASSES) * class_counts[i]) for i in range(len(CLASSES))],
                           dtype=torch.float32, device=device)
    criterion = nn.NLLLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_val = float("inf"); best_epoch = 0; stale = 0; history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_state_dict.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(train_rows); random.shuffle(order)
        epoch_losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            if len(batch) < 2:
                continue
            x, y = batch_arrays(batch, point_cache, True, args.num_points)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logp, _, trans_feat = model(x)
            loss = criterion(logp, y) + 1e-3 * feature_transform_regularizer(trans_feat)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate(model, val_rows, point_cache, device, args.num_points, args.batch_size)
        train_metrics = evaluate(model, train_rows, point_cache, device, args.num_points, args.batch_size)
        val_loss = val_metrics["mean_nll"] if val_metrics["mean_nll"] is not None else float("inf")
        row = {"epoch": epoch, "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
               "train_accuracy": train_metrics["accuracy"], "validation_loss": val_loss,
               "validation_accuracy": val_metrics["accuracy"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_val:
            best_val, best_epoch, stale = val_loss, epoch, 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    val_report = evaluate(model, val_rows, point_cache, device, args.num_points, args.batch_size)
    test_report = evaluate(model, test, point_cache, device, args.num_points, args.batch_size)
    classes_path = args.output_dir / "classes.json"
    classes_path.write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    checkpoint_path = args.output_dir / "pointnet_objectron_pilot.pth"
    torch.save(model.state_dict(), checkpoint_path)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "EXPERIMENTAL_BASELINE_NO_GENERALIZATION_CLAIM",
        "device": str(device), "architecture": "PointNet feature-transform classifier",
        "initialization": str(args.pretrained.resolve()),
        "transferred_matching_parameter_tensors": len(transfer),
        "ignored_pretrained_keys": list(load_result.unexpected_keys),
        "class_order": CLASSES,
        "train_manifest": str(args.train_manifest.resolve()),
        "test_manifest": str(args.test_manifest.resolve()),
        "train_video_groups": len(train_hashes),
        "test_video_groups": len(test_hashes),
        "train_samples": len(train_rows), "validation_samples": len(val_rows), "test_samples": len(test),
        "train_per_class": {CLASSES[i]: class_counts[i] for i in range(len(CLASSES))},
        "validation_per_class": dict(Counter(CLASSES[r["label_id"]] for r in val_rows)),
        "test_per_class": dict(Counter(CLASSES[r["label_id"]] for r in test)),
        "best_epoch_selected_by_validation_loss": best_epoch,
        "validation": val_report, "official_test": test_report,
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()), "classes": str(classes_path.resolve()),
        "limitations": [
            "Training and validation groups are separated by source-video hash; physical-object identity across recordings is not known.",
            f"Only {len(train) + len(val_rows)} sparse SfM clouds are available for fitting and validation ({len(train)} fit, {len(val_rows)} validation), with uneven class counts.",
            "Official test results are one small Objectron benchmark and do not establish performance on the user's phone videos.",
            "Maximum softmax values are uncalibrated; this baseline does not calibrate confidence or implement unknown detection.",
        ],
    }
    report_path = args.output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "validation_accuracy": val_report["accuracy"],
                      "test_accuracy": test_report["accuracy"], "test_per_class": test_report["per_class"],
                      "report": str(report_path.resolve()), "checkpoint": str(checkpoint_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
