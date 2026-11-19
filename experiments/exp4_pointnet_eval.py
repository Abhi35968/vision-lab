"""
Experiment 4 - 3D recognition with PointNet: accuracy, noise robustness, and the real object.

  A. ModelNet40 test accuracy (overall and class-averaged), per-class accuracy, confusion matrix.
  B. Robustness on the test set to three corruptions that mimic SfM output:
       Gaussian jitter on point positions, fewer points, and a share of outlier points.
  C. The trained network applied to the real bottle: the sparse SfM object points and a dense
     sample of the fitted bottle mesh (both rotated so that "up" matches ModelNet's +y).

    python -m experiments.exp4_pointnet_eval
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import open3d as o3d
except (ImportError, Exception):
    o3d = None
import pandas as pd
import torch

from experiments.common import (
    PROJECT_ROOT, POINTNET_WEIGHTS, POINTNET_CLASSES, REFINED_CSV, BOTTLE_MESH,
    out_dir, save_json, style_axes, C_A, C_B, C_C, C_GREY,
)
from experiments.train_pointnet import load_split
from src.classification_3d.pointnet import PointNetClassifier
from src.classification_3d.pointnet_classifier import PointNetInferenceEngine

NUM_POINTS = 1024
JITTER_SIGMAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
POINT_COUNTS = [1024, 512, 256, 128, 64, 32, 16]
OUTLIER_RATIOS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


@torch.no_grad()
def predict(model, x, batch_size=128):
    preds = []
    for i in range(0, len(x), batch_size):
        log_probs, _, _ = model(x[i:i + batch_size].transpose(1, 2))
        preds.append(log_probs.argmax(1))
    return torch.cat(preds)


def renormalise(x):
    """Centre and scale each cloud to the unit sphere, as the inference engine does."""
    x = x - x.mean(dim=1, keepdim=True)
    return x / x.norm(dim=2).max(dim=1, keepdim=True)[0].clamp_min(1e-8).unsqueeze(-1)


def main():
    out = out_dir("exp4_pointnet")
    with open(POINTNET_CLASSES, "r") as f:
        classes = json.load(f)
    if not POINTNET_WEIGHTS.exists():
        raise FileNotFoundError(f"Train first: python -m experiments.train_pointnet  ({POINTNET_WEIGHTS} missing)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetClassifier(num_classes=len(classes), feature_transform=True).to(device)
    model.load_state_dict(torch.load(POINTNET_WEIGHTS, map_location=device))
    model.eval()

    xte, yte = load_split("test")
    x = torch.from_numpy(xte[:, :NUM_POINTS]).to(device)
    y = torch.from_numpy(yte).to(device)
    gen = torch.Generator(device=device).manual_seed(0)

    # ---------------- A. clean accuracy
    pred = predict(model, x)
    overall = (pred == y).float().mean().item()
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
        cm[t, p] += 1
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    mean_class = float(per_class.mean())
    pd.DataFrame({"class": classes, "test_shapes": cm.sum(axis=1), "accuracy": per_class}).to_csv(
        out / "per_class_accuracy.csv", index=False)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(11, 9.5))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"PointNet on ModelNet40 test set: overall {overall * 100:.1f}%, class-average {mean_class * 100:.1f}%")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Share of the true class")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()

    order = np.argsort(per_class)
    fig, ax = plt.subplots(figsize=(8, 9.5))
    cols = [C_B if classes[i] == "bottle" else C_A for i in order]
    ax.barh([classes[i] for i in order], per_class[order] * 100, color=cols)
    for k, i in enumerate(order):
        ax.text(per_class[i] * 100 + 0.8, k, f"{per_class[i] * 100:.0f}", va="center", fontsize=8)
    ax.set_xlim(0, 108); ax.set_xlabel("Test accuracy (%)")
    ax.set_title("Per-class accuracy (bottle highlighted)"); style_axes(ax, "x")
    plt.tight_layout(); plt.savefig(out / "per_class_accuracy.png", dpi=150); plt.close()

    # ---------------- B. robustness
    def acc(xc):
        return (predict(model, xc) == y).float().mean().item()

    jitter = []
    for s in JITTER_SIGMAS:
        noise = torch.randn(x.shape, device=device, generator=gen) * s
        jitter.append(acc(x + noise))
    fewer = []
    for n in POINT_COUNTS:
        idx = torch.randperm(NUM_POINTS, device=device, generator=gen)[:n]
        fewer.append(acc(renormalise(x[:, idx])))
    outl = []
    for r in OUTLIER_RATIOS:
        xc = x.clone()
        k = int(round(r * NUM_POINTS))
        if k:
            idx = torch.randperm(NUM_POINTS, device=device, generator=gen)[:k]
            xc[:, idx] = torch.rand((len(x), k, 3), device=device, generator=gen) * 2 - 1
        outl.append(acc(xc))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, xs, ys, xl, title in (
        (axs[0], JITTER_SIGMAS, jitter, "Gaussian jitter sigma (unit-sphere radius = 1)", "Position noise"),
        (axs[1], POINT_COUNTS, fewer, "Points per object", "Fewer points"),
        (axs[2], [r * 100 for r in OUTLIER_RATIOS], outl, "Outlier points (% of 1,024, uniform in the unit cube)", "Outlier points"),
    ):
        ax.plot(xs, np.array(ys) * 100, "-o", color=C_A, lw=2, ms=7)
        for xi, yi in zip(xs, ys):
            ax.annotate(f"{yi * 100:.0f}", (xi, yi * 100), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
        ax.axhline(100.0 / len(classes), color=C_GREY, ls="--", lw=1.2, label="Chance (2.5%)")
        ax.set_ylim(0, 105); ax.set_xlabel(xl); ax.set_ylabel("Test accuracy (%)"); ax.set_title(title)
        ax.legend(frameon=False, fontsize=8); style_axes(ax)
    axs[1].set_xscale("log", base=2); axs[1].set_xticks(POINT_COUNTS); axs[1].set_xticklabels(POINT_COUNTS)
    axs[1].invert_xaxis()
    axs[1].axvline(29, color=C_B, ls=":", lw=1.5); axs[1].text(29, 50, " real bottle:\n 29 points", color=C_B, fontsize=8)
    plt.suptitle("PointNet robustness on the ModelNet40 test set (no retraining)", fontsize=12)
    plt.tight_layout(); plt.savefig(out / "noise_robustness.png", dpi=150); plt.close()

    # ---------------- C. the real object
    engine = PointNetInferenceEngine(device="cpu")
    assert engine.has_pretrained_weights, "inference engine did not load the trained weights"
    df = pd.read_csv(REFINED_CSV)
    xyz = df[["x_world", "y_world", "z_world"]].values
    valid = df["refined_status"].isin(["HIGH_GEOMETRIC_CONFIDENCE", "MODERATE_GEOMETRIC_CONFIDENCE"]).values
    flip = np.diag([1.0, -1.0, -1.0])  # OpenCV world (y down) -> y up, keeps a right-handed frame
    mesh = o3d.io.read_triangle_mesh(str(BOTTLE_MESH))
    o3d.utility.random.seed(0)
    mesh_pts = np.asarray(mesh.sample_points_uniformly(number_of_points=2048).points)
    inputs = {
        "sparse_sfm_validated_29": (xyz[valid] @ flip.T, "29 validated SfM points (measured)"),
        "sparse_sfm_candidates_47": (xyz @ flip.T, "All 47 candidate SfM points (measured)"),
        "fitted_mesh_surface_2048": (mesh_pts @ flip.T, "2,048 points sampled on the fitted bottle mesh (model, not measured)"),
    }
    np.random.seed(0)
    real = {}
    for key, (pts, desc) in inputs.items():
        res = engine.classify_point_cloud(pts, top_k=5).to_dict()
        probs_bottle = next((p["confidence"] for p in res["top_predictions"] if p["class_name"] == "bottle"), None)
        real[key] = {"description": desc, "num_points": int(len(pts)), "predicted_class": res["predicted_class"],
                     "confidence": res["predicted_confidence"], "top5": res["top_predictions"],
                     "bottle_in_top5_confidence": probs_bottle,
                     "correct": res["predicted_class"] == "bottle"}
        print(f"[INFO] {key}: predicted {res['predicted_class']} ({res['predicted_confidence'] * 100:.1f}%), "
              f"top5 = {[(p['class_name'], round(p['confidence'], 3)) for p in res['top_predictions']]}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (key, r) in zip(axs, real.items()):
        names = [p["class_name"] for p in r["top5"]][::-1]
        vals = [p["confidence"] * 100 for p in r["top5"]][::-1]
        ax.barh(names, vals, color=[C_C if n == "bottle" else C_A for n in names])
        for k, v in enumerate(vals):
            ax.text(v + 1, k, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, 115); ax.set_xlabel("Softmax confidence (%)")
        ax.set_title(r["description"].split(" (")[0], fontsize=10); style_axes(ax, "x")
    plt.suptitle("PointNet (trained on ModelNet40) applied to the real bottle - true class: bottle (green)", fontsize=12)
    plt.tight_layout(); plt.savefig(out / "real_object_classification.png", dpi=150); plt.close()

    # keep the pipeline's own result file in step with the trained model
    cls_dir = PROJECT_ROOT / "outputs" / "classification_3d"
    cls_dir.mkdir(parents=True, exist_ok=True)
    save_json(real, cls_dir / "real_object_classification.json")

    summary = {
        "test_shapes": int(len(yte)), "num_points": NUM_POINTS,
        "overall_accuracy": overall, "mean_class_accuracy": mean_class,
        "bottle_class_accuracy": float(per_class[classes.index("bottle")]),
        "lowest_classes": [(classes[i], float(per_class[i])) for i in order[:5]],
        "jitter": dict(zip(map(str, JITTER_SIGMAS), jitter)),
        "point_count": dict(zip(map(str, POINT_COUNTS), fewer)),
        "outlier_ratio": dict(zip(map(str, OUTLIER_RATIOS), outl)),
        "real_object": real,
    }
    save_json(summary, out / "summary.json")
    print(f"[INFO] Overall accuracy {overall * 100:.2f}%, class-average {mean_class * 100:.2f}%")
    print("[INFO] jitter:", [round(a * 100, 1) for a in jitter])
    print("[INFO] fewer points:", [round(a * 100, 1) for a in fewer])
    print("[INFO] outliers:", [round(a * 100, 1) for a in outl])
    print(f"[INFO] Results written to {out}")


if __name__ == "__main__":
    main()
