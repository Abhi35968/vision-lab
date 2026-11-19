"""Run one video in a fresh, provenance-recorded output directory.

Example:
    python scripts/run_isolated_reconstruction.py --video path/to/video.mp4 --category bottle

This is the only supported batch entrypoint for new real-video results.  It
never reads a point cloud, pose, mask, depth map, or mesh from another run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.objectron_batch import process  # noqa: E402
from src.classification_3d.pointnet_classifier import PointNetInferenceEngine  # noqa: E402
from src.runtime.run_manager import RunContext, RunManifestError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one isolated 3D reconstruction.")
    parser.add_argument("--video", required=True, help="Exactly one input video.")
    parser.add_argument("--category", default="auto", help="YOLO target class, or auto.")
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--calibration", default=None, help="Validated OpenCV calibration JSON.")
    parser.add_argument("--skip-own-sfm", action="store_true")
    parser.add_argument("--max-keyframes", type=int, default=40,
                        help="Maximum sharp keyframes sent to SfM (default: 40).")
    args = parser.parse_args()
    if args.max_keyframes < 8:
        parser.error("--max-keyframes must be at least 8")

    calibration = {"source": "auto_estimated"}
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        calibration["source"] = str(Path(args.calibration).resolve())

    try:
        ctx = RunContext.create(
            args.video, output_root=args.output_root, category=args.category,
            configuration={"entrypoint": "scripts/run_isolated_reconstruction.py",
                            "skip_own_sfm": args.skip_own_sfm,
                            "max_keyframes": args.max_keyframes},
            camera_parameters=calibration,
        )
    except RunManifestError as exc:
        print(f"[ERROR] {exc}")
        return 2

    print(f"[INFO] run_id={ctx.run_id}\n[INFO] output={ctx.root}")
    ctx.stage_started("reconstruction")
    try:
        from ultralytics import YOLO
        model = YOLO(str(PROJECT_ROOT / "yolo11n-seg.pt"))
        engine = PointNetInferenceEngine(device="cpu")
        result = process(
            ctx.video_path, model, engine, category=args.category,
            out_root=ctx.root, name="reconstruction", run_own=not args.skip_own_sfm,
            calibration_path=args.calibration, max_keyframes=args.max_keyframes,
        )
        metrics_path = ctx.path("reconstruction", "metrics.json")
        ctx.register_artifact("metrics", metrics_path)
        diag = ctx.path("reconstruction", "diagnostics", "camera_pose_validation.json")
        if diag.exists():
            ctx.register_artifact("camera_pose_validation", diag)
        stage_status = "passed" if result.get("status") == "ok" else "partial"
        ctx.stage_finished("reconstruction", status=stage_status, details={"pipeline_status": result.get("status")})
        ctx.finish("completed" if stage_status == "passed" else "completed_with_warnings",
                   warning=None if stage_status == "passed" else "Pipeline did not meet all reconstruction gates.")
        print(json.dumps({"run_id": ctx.run_id, "result": result, "manifest": str(ctx.manifest_path)}, indent=2, default=str))
        return 0 if stage_status == "passed" else 1
    except Exception as exc:
        ctx.stage_finished("reconstruction", status="failed", details={"error": f"{type(exc).__name__}: {exc}"})
        ctx.finish("failed", warning="See failed reconstruction stage for the exception.")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
