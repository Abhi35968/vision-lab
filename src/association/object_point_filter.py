"""
Object Point Filter Module
Partitions reconstructed 3D points based on multi-view mask evidence,
computes bounding box extents and spatial centroids for object clusters,
and exports categorized point clouds to .PLY and .NPY formats with RGB vertex color coding.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.association.multiview_mask_associator import PointMaskAssociation


class ObjectPointFilter:
    """Filters, partitions, and exports 3D points based on multi-view mask association."""

    # Vertex RGB colors for PLY visualization
    COLOR_STRONG = (50, 205, 50)       # Lime Green
    COLOR_WEAK = (0, 210, 255)         # Cyan / Sky Blue
    COLOR_AMBIGUOUS = (255, 140, 0)    # Dark Orange
    COLOR_BACKGROUND = (160, 160, 160) # Neutral Grey
    COLOR_INSUFFICIENT = (186, 85, 211) # Medium Orchid / Purple

    @classmethod
    def partition_points(
        cls,
        associations: List[PointMaskAssociation],
    ) -> Dict[str, List[PointMaskAssociation]]:
        """
        Groups point associations by classification category.

        Returns:
            Dict mapping category name -> list of PointMaskAssociation records.
        """
        partitions: Dict[str, List[PointMaskAssociation]] = {
            "STRONG_OBJECT_CANDIDATE": [],
            "WEAK_OBJECT_CANDIDATE": [],
            "AMBIGUOUS": [],
            "BACKGROUND_CANDIDATE": [],
            "INSUFFICIENT_EVIDENCE": [],
        }

        for assoc in associations:
            cat = assoc.classification
            if cat in partitions:
                partitions[cat].append(assoc)
            else:
                partitions.setdefault(cat, []).append(assoc)

        return partitions

    @classmethod
    def compute_spatial_statistics(
        cls,
        points: Union[List[PointMaskAssociation], np.ndarray],
    ) -> Dict[str, Any]:
        """
        Computes 3D spatial extents, centroid, and bounding box dimensions in arbitrary units.
        """
        if isinstance(points, list):
            if not points:
                return {}
            pts = np.array([p.point_3d for p in points], dtype=np.float64)
        else:
            pts = points

        if pts.shape[0] == 0:
            return {}

        centroid = np.mean(pts, axis=0)
        std_dev = np.std(pts, axis=0)
        p_min = np.min(pts, axis=0)
        p_max = np.max(pts, axis=0)
        extents = p_max - p_min

        return {
            "count": int(pts.shape[0]),
            "centroid": [round(float(c), 4) for c in centroid],
            "std_dev": [round(float(s), 4) for s in std_dev],
            "bbox_min": [round(float(c), 4) for c in p_min],
            "bbox_max": [round(float(c), 4) for c in p_max],
            "bbox_extents": [round(float(e), 4) for e in extents],
            "bounding_diameter": round(float(np.linalg.norm(extents)), 4),
            "unit": "arbitrary_units",
        }

    @classmethod
    def write_ply(
        cls,
        output_path: Union[str, Path],
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ):
        """
        Exports 3D coordinates and optional RGB colors to standard ASCII PLY format.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        n_pts = len(points)
        has_color = (colors is not None and len(colors) == n_pts)

        with open(out_p, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n_pts}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_color:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")

            for i in range(n_pts):
                x, y, z = points[i]
                if has_color:
                    r, g, b = colors[i]
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    @classmethod
    def export_all_point_clouds(
        cls,
        associations: List[PointMaskAssociation],
        output_dir: Union[str, Path],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Exports all partitioned and combined point clouds (.PLY and .NPY).
        """
        out_path = Path(output_dir)
        pc_dir = out_path / "point_clouds"
        pc_dir.mkdir(parents=True, exist_ok=True)

        partitions = cls.partition_points(associations)
        stats_summary: Dict[str, Dict[str, Any]] = {}

        # 1. Original All Points
        all_pts = np.array([a.point_3d for a in associations], dtype=np.float64)
        np.save(pc_dir / "original_sparse_points.npy", all_pts)
        cls.write_ply(pc_dir / "original_sparse_points.ply", all_pts)
        stats_summary["original_sparse_points"] = cls.compute_spatial_statistics(all_pts)

        # 2. Strong Object Candidates
        strong_pts = np.array([a.point_3d for a in partitions["STRONG_OBJECT_CANDIDATE"]], dtype=np.float64)
        np.save(pc_dir / "strong_candidate_points.npy", strong_pts)
        strong_colors = np.tile(cls.COLOR_STRONG, (len(strong_pts), 1)) if len(strong_pts) > 0 else None
        cls.write_ply(pc_dir / "strong_candidate_points.ply", strong_pts, strong_colors)
        stats_summary["strong_candidate_points"] = cls.compute_spatial_statistics(strong_pts)

        # 3. Weak & Ambiguous Candidates Combined
        weak_ambig_pts = np.array(
            [a.point_3d for a in partitions["WEAK_OBJECT_CANDIDATE"] + partitions["AMBIGUOUS"]],
            dtype=np.float64,
        )
        np.save(pc_dir / "weak_ambiguous_points.npy", weak_ambig_pts)
        weak_colors = np.tile(cls.COLOR_WEAK, (len(weak_ambig_pts), 1)) if len(weak_ambig_pts) > 0 else None
        cls.write_ply(pc_dir / "weak_ambiguous_points.ply", weak_ambig_pts, weak_colors)
        stats_summary["weak_ambiguous_points"] = cls.compute_spatial_statistics(weak_ambig_pts)

        # 4. Background Candidates
        bg_pts = np.array([a.point_3d for a in partitions["BACKGROUND_CANDIDATE"]], dtype=np.float64)
        np.save(pc_dir / "background_points.npy", bg_pts)
        bg_colors = np.tile(cls.COLOR_BACKGROUND, (len(bg_pts), 1)) if len(bg_pts) > 0 else None
        cls.write_ply(pc_dir / "background_points.ply", bg_pts, bg_colors)
        stats_summary["background_points"] = cls.compute_spatial_statistics(bg_pts)

        # 5. Combined Color-Coded Point Cloud
        combined_colors = np.zeros((len(associations), 3), dtype=np.uint8)
        for i, a in enumerate(associations):
            if a.classification == "STRONG_OBJECT_CANDIDATE":
                combined_colors[i] = cls.COLOR_STRONG
            elif a.classification == "WEAK_OBJECT_CANDIDATE":
                combined_colors[i] = cls.COLOR_WEAK
            elif a.classification == "AMBIGUOUS":
                combined_colors[i] = cls.COLOR_AMBIGUOUS
            elif a.classification == "BACKGROUND_CANDIDATE":
                combined_colors[i] = cls.COLOR_BACKGROUND
            else:
                combined_colors[i] = cls.COLOR_INSUFFICIENT

        cls.write_ply(pc_dir / "combined_colored_points.ply", all_pts, combined_colors)
        print(f"[INFO] Exported 5 categorized point clouds to: {pc_dir}/")

        return stats_summary

