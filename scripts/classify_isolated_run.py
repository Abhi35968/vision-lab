"""Classify only the measured object cloud belonging to one isolated run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification_3d.yolo_pointnet_fusion import YOLOPointNetFusionEngine  # noqa: E402
from src.runtime.run_manager import RunContext  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to one outputs/runs/<run-id> directory")
    args = parser.parse_args()
    root = Path(args.run).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ctx = RunContext(manifest["run_id"], root, root / manifest["input"]["archived_path"], root / "manifest.json")
    cloud_path = root / "reconstruction" / "object_points.ply"
    seg_path = root / "reconstruction" / "segmentation.csv"
    if not cloud_path.exists() or not seg_path.exists():
        raise FileNotFoundError("Run has no measured object cloud and segmentation evidence")
    points = np.asarray(o3d.io.read_point_cloud(str(cloud_path)).points)
    df = pd.read_csv(seg_path)
    detections = [{"class": row.label, "confidence": float(row.confidence)}
                  for row in df[df.detected].itertuples()]
    ctx.stage_started("classification")
    try:
        result = YOLOPointNetFusionEngine().fuse(points, detections).to_dict()
        out = root / "reconstruction" / "classification.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        ctx.register_artifact("classification", out)
        state = "partial" if result["is_uncertain"] else "passed"
        ctx.stage_finished("classification", status=state, details={"status": result["status"]})
        # This command can be rerun after classification logic changes. Remove
        # an earlier rejection warning when the refreshed result is decisive.
        if not result["is_uncertain"]:
            data = ctx.read_manifest()
            data["warnings"] = [
                warning for warning in data.get("warnings", [])
                if warning != "Classification evidence was insufficient; final class is unknown."
            ]
            ctx._write_manifest(data)
        ctx.finish("completed_with_warnings" if result["is_uncertain"] else "completed",
                   warning="Classification evidence was insufficient; final class is unknown." if result["is_uncertain"] else None)
        print(json.dumps(result, indent=2))
        return 0
    except Exception:
        ctx.stage_finished("classification", status="failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
