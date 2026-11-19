"""Crop a COLMAP sparse cloud with Objectron's 3D box annotations.

Example:
  python scripts/extract_objectron_box_cloud.py --run-dir outputs/runs/<run> \
      --annotation data/objectron_subset_pilot/media/bike/batch-4/47/annotation.pbdata

COLMAP's arbitrary Sim(3) frame is aligned to Objectron's world frame using
matched per-frame camera centers. The cropped cloud is therefore in the
annotation coordinate system; point density remains sparse SfM, not a mesh.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import open3d as o3d
import pycolmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_LABEL = {"bike": "bicycle", "motobike": "bicycle",
                    "cerealboxes": "cereal_box", "shoes": "shoe"}


def umeyama(source: np.ndarray, target: np.ndarray):
    """Similarity transform x_target = scale * R @ x_source + t."""
    src_mean, dst_mean = source.mean(0), target.mean(0)
    xs, ys = source - src_mean, target - dst_mean
    u, singular, vt = np.linalg.svd((ys.T @ xs) / len(source))
    fix = np.eye(3)
    fix[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ fix @ vt
    variance = np.mean(np.sum(xs * xs, axis=1))
    scale = float(np.sum(singular * np.diag(fix)) / variance)
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def camera_center(frame_annotation) -> np.ndarray:
    return np.asarray(frame_annotation.camera.transform, dtype=np.float64).reshape(4, 4)[:3, 3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--min-track", type=int, default=3)
    parser.add_argument("--max-reprojection-error", type=float, default=3.0)
    parser.add_argument("--max-pose-rmse", type=float, default=0.12,
                        help="Reject alignment above this Objectron-world RMSE (nominal meters).")
    args = parser.parse_args()
    run_dir, annotation_path = args.run_dir.resolve(), args.annotation.resolve()
    sparse_root = run_dir / "reconstruction" / "colmap" / "sparse"
    if not sparse_root.is_dir() or not annotation_path.is_file():
        parser.error("run sparse model or annotation file is missing")

    # Official generated Objectron proto is compatible with the pure Python backend.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    sys.path.insert(0, str(ROOT / "src"))
    from objectron.schema.annotation_data_pb2 import Sequence

    sequence = Sequence()
    sequence.ParseFromString(annotation_path.read_bytes())
    if not sequence.objects or not sequence.frame_annotations:
        raise SystemExit("Expected at least one annotated object and frame annotations")
    # Object-centric samples can annotate multiple same-category instances.
    # This pilot exports the first sequence object deterministically; the
    # selected index and all other object labels are retained in provenance.
    obj = sequence.objects[0]
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    requested_category = run_manifest.get("category_request", "")
    annotation_category = ANNOTATION_LABEL.get(obj.category, obj.category)
    output = run_dir / "reconstruction" / "objectron_box"
    output.mkdir(parents=True, exist_ok=True)
    if requested_category and annotation_category != requested_category:
        old_cloud = output / "object_points_objectron_box.ply"
        if old_cloud.exists():
            old_cloud.replace(output / "rejected_category_mismatch.ply")
        rejection = {"status": "rejected_category_mismatch", "source_run_id": run_dir.name,
                     "source_annotation": str(annotation_path), "requested_category": requested_category,
                     "annotation_category": obj.category, "canonical_annotation_category": annotation_category,
                     "reason": "Run label and Objectron annotation label disagree; no training cloud accepted."}
        (output / "report.json").write_text(json.dumps(rejection, indent=2), encoding="utf-8")
        raise SystemExit(f"Annotation category mismatch: run expects {requested_category!r}, annotation says {obj.category!r}; crop rejected")
    model_dirs = [p for p in sparse_root.iterdir() if p.is_dir() and (p / "images.bin").is_file()]
    if not model_dirs:
        raise SystemExit(f"No COLMAP binary models found under {sparse_root}")
    reconstructions = [(p, pycolmap.Reconstruction(str(p))) for p in model_dirs]
    model_dir, rec = max(reconstructions, key=lambda item: (item[1].num_reg_images(), item[1].num_points3D()))
    # Keyframe filenames carry the original decoded video frame number.
    frame_to_image = {}
    for image in rec.images.values():
        match = re.search(r"_f(\d+)\.", image.name)
        if image.has_pose and match:
            frame_to_image[int(match.group(1))] = image
    paired = [(frame, img, sequence.frame_annotations[frame])
              for frame, img in frame_to_image.items() if frame < len(sequence.frame_annotations)]
    if len(paired) < 8:
        rejection = {"status": "rejected_insufficient_pose_pairs", "source_run_id": run_dir.name,
                     "source_annotation": str(annotation_path), "matched_pose_pairs": len(paired),
                     "required_pose_pairs": 8,
                     "reason": "COLMAP reconstruction does not provide enough corresponding posed frames."}
        (output / "report.json").write_text(json.dumps(rejection, indent=2), encoding="utf-8")
        raise SystemExit(f"Only {len(paired)} matched posed frames; need at least 8")

    colmap_centers = np.asarray([img.projection_center() for _, img, _ in paired])
    objectron_centers = np.asarray([camera_center(ann) for _, _, ann in paired])
    scale, rotation, translation = umeyama(colmap_centers, objectron_centers)
    aligned_centers = scale * (colmap_centers @ rotation.T) + translation
    residuals = np.linalg.norm(aligned_centers - objectron_centers, axis=1)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    if not np.isfinite(rmse) or rmse > args.max_pose_rmse:
        rejection = {"status": "rejected_pose_alignment", "source_run_id": run_dir.name,
                     "source_annotation": str(annotation_path), "matched_pose_pairs": len(paired),
                     "pose_alignment_rmse": rmse, "max_pose_rmse": args.max_pose_rmse,
                     "reason": "COLMAP-to-Objectron camera-center alignment failed its residual threshold."}
        (output / "report.json").write_text(json.dumps(rejection, indent=2), encoding="utf-8")
        raise SystemExit(f"Pose Sim(3) alignment rejected: RMSE={rmse:.4f}m exceeds {args.max_pose_rmse:.3f}m")

    obj_rotation = np.asarray(obj.rotation, dtype=np.float64).reshape(3, 3)
    obj_translation = np.asarray(obj.translation, dtype=np.float64)
    obj_scale = np.asarray(obj.scale, dtype=np.float64)
    pids, xyz, colors, reproj, tracks = [], [], [], [], []
    for pid, point in rec.points3D.items():
        if len(point.track.elements) < args.min_track or float(point.error) > args.max_reprojection_error:
            continue
        pids.append(pid)
        xyz.append(point.xyz)
        colors.append(np.asarray(point.color, dtype=np.float64) / 255.0)
        reproj.append(float(point.error))
        tracks.append(len(point.track.elements))
    xyz = np.asarray(xyz, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    if len(xyz) == 0:
        raise SystemExit("No COLMAP points passed track and reprojection filters")
    xyz_world = scale * (xyz @ rotation.T) + translation

    # Objectron box is center + rotation * diag(scale) * normalized corners.
    local = (xyz_world - obj_translation) @ obj_rotation
    inside = np.all(np.abs(local) <= obj_scale[None, :] / 2.0, axis=1)
    crop = xyz_world[inside]
    crop_colors = colors[inside]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(crop)
    cloud.colors = o3d.utility.Vector3dVector(crop_colors)
    cloud_path = output / "object_points_objectron_box.ply"
    o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=True)
    mask_cloud_path = run_dir / "reconstruction" / "object_points.ply"
    comparison_path = None
    mask_cloud_count = 0
    mask_inside_fraction = None
    if mask_cloud_path.is_file():
        mask_cloud = o3d.io.read_point_cloud(str(mask_cloud_path))
        mask_xyz = np.asarray(mask_cloud.points, dtype=np.float64)
        if len(mask_xyz):
            mask_world = scale * (mask_xyz @ rotation.T) + translation
            mask_local = (mask_world - obj_translation) @ obj_rotation
            mask_inside = np.all(np.abs(mask_local) <= obj_scale[None, :] / 2.0, axis=1)
            mask_cloud_count = int(len(mask_xyz))
            mask_inside_fraction = float(mask_inside.mean())
            figure = plt.figure(figsize=(12, 5.5))
            axes = [figure.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
            groups = [(axes[0], mask_world, "Pipeline mask cloud", "#2878b5"),
                      (axes[1], crop, "Objectron 3D-box crop", "#d9534f")]
            if len(mask_world):
                combined = np.vstack([mask_world, crop]) if len(crop) else mask_world
                center = combined.mean(axis=0)
                radius = max(float(np.max(np.ptp(combined, axis=0))) / 2.0, 0.02)
            else:
                center, radius = obj_translation, 0.5
            for ax, points, title, color in groups:
                if len(points):
                    step = max(1, len(points) // 10000)
                    ax.scatter(points[::step, 0], points[::step, 1], points[::step, 2],
                               s=2, alpha=0.6, c=color)
                ax.set_title(title)
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
                ax.set_zlim(center[2] - radius, center[2] + radius)
                ax.set_box_aspect((1, 1, 1))
                ax.set_xlabel("Objectron X (nominal m)")
                ax.set_ylabel("Objectron Y (nominal m)")
                ax.set_zlabel("Objectron Z (nominal m)")
            figure.suptitle("Same COLMAP run: YOLO mask selection vs annotated 3D box")
            figure.tight_layout()
            comparison_path = output / "mask_vs_annotated_box.png"
            figure.savefig(comparison_path, dpi=160)
            plt.close(figure)
    report = {
        "status": "ok" if len(crop) >= 100 else "insufficient_object_points",
        "source_run_id": run_dir.name,
        "colmap_model": str(model_dir),
        "colmap_models_found": [{"path": str(path), "registered_images": int(model.num_reg_images()),
                                 "points3d": int(model.num_points3D())}
                                for path, model in reconstructions],
        "source_annotation": str(annotation_path),
        "objectron_category": obj.category,
        "selected_object_annotation_index": 0,
        "annotated_object_count": len(sequence.objects),
        "annotated_object_categories": [item.category for item in sequence.objects],
        "coordinate_frame": "Objectron world (camera pose alignment; nominal meters)",
        "pose_alignment": {"paired_frames": len(paired), "scale": scale,
                           "rmse": rmse, "median_error": float(np.median(residuals)),
                           "p90_error": float(np.percentile(residuals, 90)),
                           "max_error": float(np.max(residuals))},
        "point_filter": {"min_track_length": args.min_track,
                         "max_reprojection_error": args.max_reprojection_error,
                         "sfm_points_before_filter": int(rec.num_points3D()),
                         "points_after_filter": int(len(xyz)),
                         "points_inside_annotated_box": int(len(crop)),
                         "inside_fraction": float(inside.mean())},
        "comparison": {"pipeline_mask_cloud_points": mask_cloud_count,
                       "pipeline_mask_cloud_inside_box_fraction": mask_inside_fraction,
                       "side_by_side_plot": str(comparison_path) if comparison_path else None},
        "annotated_box": {"center": obj_translation.tolist(), "rotation_row_major": obj_rotation.tolist(),
                          "extent": obj_scale.tolist()},
        "output_cloud": str(cloud_path),
        "limitations": ["Sparse COLMAP points only; not dense MVS or a watertight surface.",
                        "The annotated box can include background/empty volume and cannot encode hidden concavities."],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if len(crop) >= 100 else 2


if __name__ == "__main__":
    raise SystemExit(main())
