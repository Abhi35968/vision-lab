"""
Point-to-Mesh Evaluator (Phase 5.6).
Computes exact 3D Euclidean distances between sparse reconstructed object points and the parametric mesh surface.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import open3d as o3d


@dataclass
class PointTierSummary:
    """Summary statistics for point-to-mesh Euclidean distance within a confidence tier."""
    tier_name: str
    count: int
    mean_distance: float
    median_distance: float
    std_distance: float
    p90_distance: float
    min_distance: float
    max_distance: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PointMeshEvaluator:
    """Evaluates Euclidean proximity of triangulated sparse points to reconstructed mesh."""

    def __init__(self, mesh_path: Path):
        self.mesh_path = Path(mesh_path)
        if not self.mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found at {self.mesh_path}")

        # Ingest mesh into Open3D raycasting scene for exact distance query
        legacy_mesh = o3d.io.read_triangle_mesh(str(self.mesh_path))
        self.t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(self.t_mesh)

    def evaluate_points(self, refinement_csv: Path) -> Tuple[pd.DataFrame, Dict[str, PointTierSummary]]:
        """Computes point-to-surface distances and computes tier-wise summary statistics."""
        ref_df = pd.read_csv(refinement_csv)
        required_cols = ["point_id", "track_id", "x_world", "y_world", "z_world", "refined_status"]
        for col in required_cols:
            if col not in ref_df.columns:
                raise ValueError(f"Missing required column '{col}' in {refinement_csv}")

        pts = ref_df[["x_world", "y_world", "z_world"]].values.astype(np.float32)

        # 1. Exact Distance Computation
        query_pts = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
        distances = self.scene.compute_distance(query_pts).numpy().flatten()
        closest_res = self.scene.compute_closest_points(query_pts)
        closest_pts = closest_res["points"].numpy()
        closest_faces = closest_res["primitive_ids"].numpy()

        results_df = ref_df.copy()
        results_df["distance_to_mesh"] = distances
        results_df["closest_mesh_x"] = closest_pts[:, 0]
        results_df["closest_mesh_y"] = closest_pts[:, 1]
        results_df["closest_mesh_z"] = closest_pts[:, 2]
        results_df["closest_face_id"] = closest_faces

        # 2. Tier-wise Summary Calculation
        summaries: Dict[str, PointTierSummary] = {}

        tiers = [
            ("HIGH_GEOMETRIC_CONFIDENCE", "high_confidence"),
            ("MODERATE_GEOMETRIC_CONFIDENCE", "moderate_confidence"),
            ("LOW_GEOMETRIC_CONFIDENCE", "low_confidence"),
            ("GEOMETRIC_OUTLIER", "geometric_outliers"),
        ]

        for status_val, key_name in tiers:
            sub = results_df[results_df["refined_status"] == status_val]["distance_to_mesh"].values
            summaries[key_name] = self._compute_tier_summary(key_name, sub)

        # Combined validated points (high + moderate)
        val_sub = results_df[
            results_df["refined_status"].isin(["HIGH_GEOMETRIC_CONFIDENCE", "MODERATE_GEOMETRIC_CONFIDENCE"])
        ]["distance_to_mesh"].values
        summaries["combined_validated"] = self._compute_tier_summary("combined_validated", val_sub)

        # All 47 candidates
        all_sub = results_df["distance_to_mesh"].values
        summaries["all_candidates"] = self._compute_tier_summary("all_candidates", all_sub)

        return results_df, summaries

    @staticmethod
    def _compute_tier_summary(name: str, values: np.ndarray) -> PointTierSummary:
        if len(values) == 0:
            return PointTierSummary(
                tier_name=name,
                count=0,
                mean_distance=0.0,
                median_distance=0.0,
                std_distance=0.0,
                p90_distance=0.0,
                min_distance=0.0,
                max_distance=0.0,
            )
        return PointTierSummary(
            tier_name=name,
            count=len(values),
            mean_distance=float(np.mean(values)),
            median_distance=float(np.median(values)),
            std_distance=float(np.std(values)),
            p90_distance=float(np.percentile(values, 90)),
            min_distance=float(np.min(values)),
            max_distance=float(np.max(values)),
        )

