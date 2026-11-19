"""Conservative validation for a COLMAP reconstruction before dense fusion.

Camera registration alone is deliberately not a pass condition.  This module
records the measurements used to decide whether a reconstruction is suitable
for object-level dense processing and writes a trajectory/diagnostic figure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def camera_matrix(image: Any) -> np.ndarray:
    cfw = image.cam_from_world
    cfw = cfw() if callable(cfw) else cfw
    return np.asarray(cfw.matrix(), dtype=np.float64)


def camera_centers(reconstruction: Any) -> np.ndarray:
    mats = [camera_matrix(i) for i in reconstruction.images.values() if i.has_pose]
    if not mats:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray([-(m[:, :3].T @ m[:, 3]) for m in mats], dtype=np.float64)


def _point_track_stats(reconstruction: Any, max_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    points = list(reconstruction.points3D.values())[:max_points]
    errors = np.asarray([float(getattr(p, "error", np.nan)) for p in points], dtype=float)
    tracks = np.asarray([len(p.track.elements) for p in points], dtype=int)
    return errors[np.isfinite(errors)], tracks


def validate_reconstruction(
    reconstruction: Any,
    *,
    total_images: int,
    output_dir: str | Path,
    calibration_source: str,
    min_registered_ratio: float = 0.70,
    max_median_reprojection_px: float = 2.0,
    min_median_track_length: float = 3.0,
) -> Dict[str, Any]:
    """Validate pose conventions and reconstruction health without fabricating scale."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    centers = camera_centers(reconstruction)
    errors, tracks = _point_track_stats(reconstruction)
    registered = int(reconstruction.num_reg_images())
    ratio = registered / max(total_images, 1)

    rotation_errors = []
    for image in reconstruction.images.values():
        if image.has_pose:
            R = camera_matrix(image)[:, :3]
            rotation_errors.append(float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro")))
    median_error = float(np.median(errors)) if len(errors) else float("inf")
    median_track = float(np.median(tracks)) if len(tracks) else 0.0
    step_lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.array([])
    nonzero_steps = int(np.count_nonzero(step_lengths > 1e-8))

    checks = {
        "registered_camera_ratio": ratio >= min_registered_ratio,
        "median_reprojection_error": median_error <= max_median_reprojection_px,
        "median_track_length": median_track >= min_median_track_length,
        "rotation_matrices": bool(rotation_errors) and max(rotation_errors) < 1e-5,
        "non_degenerate_trajectory": len(centers) >= 3 and nonzero_steps >= 2,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "calibration_source": calibration_source,
        "coordinate_convention": "COLMAP camera-from-world: X_camera = R @ X_world + t; C_world = -R.T @ t",
        "scale": "arbitrary unless a metric reference is separately recorded",
        "registered_cameras": registered,
        "total_images": total_images,
        "registered_ratio": ratio,
        "point_count_sampled": int(len(tracks)),
        "median_reprojection_error_px": median_error if np.isfinite(median_error) else None,
        "p90_reprojection_error_px": float(np.percentile(errors, 90)) if len(errors) else None,
        "median_track_length": median_track,
        "max_rotation_orthonormality_error": max(rotation_errors) if rotation_errors else None,
        "trajectory_step_median": float(np.median(step_lengths)) if len(step_lengths) else None,
        "checks": checks,
        "thresholds": {
            "min_registered_ratio": min_registered_ratio,
            "max_median_reprojection_px": max_median_reprojection_px,
            "min_median_track_length": min_median_track_length,
        },
    }
    (out / "camera_pose_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if len(centers):
        axes[0].plot(centers[:, 0], centers[:, 2], "o-", ms=3)
        axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set(title="Camera trajectory (X–Z, arbitrary units)", xlabel="X", ylabel="Z")
    if len(errors):
        axes[1].hist(errors, bins=min(50, max(10, len(errors) // 20)), color="#2a78d6")
        axes[1].axvline(max_median_reprojection_px, color="#c33", ls="--", label="median threshold")
        axes[1].legend()
    axes[1].set(title="Sparse-point reprojection error", xlabel="pixels", ylabel="points")
    fig.suptitle(f"Pose validation: {status}")
    fig.tight_layout()
    fig.savefig(out / "camera_pose_diagnostics.png", dpi=150)
    plt.close(fig)
    return report


def reliable_point_ids(
    reconstruction: Any,
    *,
    max_reprojection_px: float = 3.0,
    min_track_length: int = 3,
) -> set[int]:
    """Return only points with explicit COLMAP quality evidence.

    A point's triangulation angle is handled by COLMAP during mapping; this
    gate avoids promoting short, high-error tracks into object geometry.
    """
    return {
        int(point_id)
        for point_id, point in reconstruction.points3D.items()
        if len(point.track.elements) >= min_track_length
        and np.isfinite(float(getattr(point, "error", np.inf)))
        and float(point.error) <= max_reprojection_px
    }
