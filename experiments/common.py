"""
Shared paths and helpers for the course-report experiments.

Every experiment script in this package is run from the project root, e.g.
    python -m experiments.exp1_feature_matching
and writes its tables, JSON and figures under outputs/experiments/<name>/.
"""

from pathlib import Path
import json
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KEYFRAMES_DIR = PROJECT_ROOT / "outputs" / "real_video_validation" / "keyframes"
SFM_DIR = PROJECT_ROOT / "outputs" / "real_video_sfm" / "focal_1300"
SEG_DIR = PROJECT_ROOT / "outputs" / "object_segmentation"
REFINED_CSV = PROJECT_ROOT / "outputs" / "object_refinement" / "refined_object_points.csv"
BOTTLE_MESH = PROJECT_ROOT / "outputs" / "object_reconstruction" / "bottle_mesh.ply"
EXPERIMENTS_OUT = PROJECT_ROOT / "outputs" / "experiments"
MODELNET_DIR = PROJECT_ROOT / "data" / "modelnet40" / "modelnet40_ply_hdf5_2048"
POINTNET_WEIGHTS = PROJECT_ROOT / "models" / "weights" / "pointnet_modelnet40.pth"
POINTNET_CLASSES = PROJECT_ROOT / "config" / "pointnet_classes.json"

# Intrinsics used by the selected reconstruction (estimated, not calibrated).
IMAGE_W, IMAGE_H = 1080, 1920
FOCAL_PX = 1300.0
K = np.array([[FOCAL_PX, 0.0, IMAGE_W / 2.0],
              [0.0, FOCAL_PX, IMAGE_H / 2.0],
              [0.0, 0.0, 1.0]], dtype=np.float64)

# One colour per series, reused across all figures.
C_SIFT = "#2a78d6"
C_ORB = "#eb6834"
C_A = "#2a78d6"
C_B = "#eb6834"
C_C = "#1baf7f"
C_GREY = "#8a8a8a"


def out_dir(name: str) -> Path:
    d = EXPERIMENTS_OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def keyframe_paths():
    paths = sorted(KEYFRAMES_DIR.glob("*.png")) + sorted(KEYFRAMES_DIR.glob("*.jpg"))
    if len(paths) < 2:
        raise FileNotFoundError(f"No keyframes found in {KEYFRAMES_DIR}")
    return paths


def save_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Not JSON serialisable: {type(o)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def run_sfm(paths, run_dir, detector: str = "SIFT", seed: int = 0, quiet: bool = True,
            k_mat=None, image_size=None, **overrides) -> dict:
    """
    Runs the project's incremental SfM once with the settings of the main reconstruction and
    returns a flat metrics row. `seed` sets OpenCV's RNG, which drives every RANSAC step.
    """
    import contextlib
    import io
    import time
    import cv2
    from src.sfm.incremental_sfm import IncrementalSfM, IncrementalSfMConfig

    params = dict(
        max_keyframes=len(paths), detector_type=detector, max_features=2000, matching_window=3,
        min_inliers_pair=25, min_track_length=2, min_pnp_inliers=15, min_triangulation_angle_deg=1.5,
        max_reprojection_error_px=4.0, enable_bundle_adjustment=True, ba_max_iterations=50,
        output_dir=str(run_dir),
    )
    params.update(overrides)
    cv2.setRNGSeed(seed)
    t0 = time.time()
    status, cams, diag = "ok", {}, None
    try:
        with contextlib.redirect_stdout(io.StringIO() if quiet else sys.stdout):
            cams, _, diag, _ = IncrementalSfM(IncrementalSfMConfig(**params)).run(
                image_paths=paths, camera_matrix_k=K if k_mat is None else k_mat,
                image_size=image_size or (IMAGE_W, IMAGE_H), calibration_status="estimated",
            )
    except Exception as e:  # a failed initialisation is a result, not a crash
        status = f"failed: {type(e).__name__}: {e}"
    elapsed = time.time() - t0
    rep = {}
    rp = Path(run_dir) / "reconstruction_report.json"
    if status == "ok" and rp.exists():
        rep = json.loads(rp.read_text(encoding="utf-8"))
    err = rep.get("reprojection_error_percentiles_px", {})
    reg = sorted(i for i, c in cams.items() if c.is_registered)
    return {
        "detector": detector, "seed": seed, "status": status, "frames_used": len(paths),
        "registered_cameras": len(reg),
        "registration_rate_pct": round(100.0 * len(reg) / len(paths), 1),
        "first_registered": reg[0] if reg else -1, "last_registered": reg[-1] if reg else -1,
        "points_3d": diag.num_sparse_3d_points if diag is not None else 0,
        "observations": diag.num_total_observations if diag is not None else 0,
        "mean_track_length": rep.get("track_statistics", {}).get("mean_track_length", 0.0),
        "reproj_mean_px": err.get("mean", 0.0), "reproj_median_px": err.get("median", 0.0),
        "reproj_p90_px": err.get("p90", 0.0),
        "runtime_s": round(elapsed, 1),
    }


def load_registered_cameras(sfm_dir: Path = SFM_DIR):
    """Returns {image_id: dict(R, t, C, name)} for registered cameras (world -> camera: x = R X + t)."""
    with open(sfm_dir / "cameras.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    cams = {}
    for v in raw.values():
        if not v.get("is_registered"):
            continue
        cams[int(v["image_id"])] = {
            "R": np.asarray(v["rotation_matrix"], dtype=np.float64),
            "t": np.asarray(v["translation_vector"], dtype=np.float64).reshape(3),
            "C": np.asarray(v["camera_center"], dtype=np.float64).reshape(3),
            "name": v["image_name"],
        }
    return dict(sorted(cams.items()))


def project(points_world: np.ndarray, R: np.ndarray, t: np.ndarray, K_mat: np.ndarray = K):
    """Pinhole projection. Returns (N,2) pixels and (N,) depths."""
    pc = points_world @ R.T + t
    z = pc[:, 2]
    uv = (pc @ K_mat.T)[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]
    return uv, z
