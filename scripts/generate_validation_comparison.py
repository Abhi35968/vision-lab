"""Generate side-by-side, run-scoped validation figures without mixing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="outputs/runs")
    parser.add_argument("--output", default="outputs/validation_comparison")
    args = parser.parse_args()
    root, output = Path(args.runs_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(root.glob("*/manifest.json")):
        metrics = p.parent / "reconstruction" / "metrics.json"
        if not metrics.exists():
            continue
        m = json.loads(metrics.read_text(encoding="utf-8"))
        rows.append((p.parent.name, p.parent, m))
    rows = [r for r in rows if r[2].get("status") == "ok"]
    if not rows:
        raise SystemExit("No completed isolated runs found")

    names = [r[0][:18] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(names, [r[2].get("colmap_registered", 0) for r in rows])
    axes[0].set(title="Registered cameras", ylabel="count")
    axes[1].bar(names, [r[2].get("colmap_reproj_mean_px", np.nan) for r in rows])
    axes[1].set(title="COLMAP mean reprojection", ylabel="pixels")
    axes[2].bar(names, [r[2].get("object_points", 0) for r in rows])
    axes[2].set(title="Measured object points", ylabel="count")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=30)
    fig.suptitle("Isolated-run comparison (each bar has a separate manifest and input hash)")
    fig.tight_layout()
    fig.savefig(output / "run_metrics_comparison.png", dpi=150)
    plt.close(fig)

    # Do not combine clouds.  The side-by-side figure keeps each run's own
    # held-out overlay in its own labelled panel.
    tiles = []
    for run_id, run_root, _ in rows:
        image = cv2.imread(str(run_root / "reconstruction" / "overlay.jpg"))
        if image is None:
            continue
        image = cv2.resize(image, (720, 240))
        cv2.putText(image, run_id, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(image, run_id, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(image)
    if tiles:
        cv2.imwrite(str(output / "held_out_overlay_comparison.jpg"), np.vstack(tiles))

    (output / "comparison_manifest.json").write_text(json.dumps({
        "runs": [{"run_id": run_id, "metrics": str((run_root / 'reconstruction' / 'metrics.json'))}
                 for run_id, run_root, _ in rows],
        "warning": "Panels are side-by-side only. No point clouds or outputs were merged."
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
