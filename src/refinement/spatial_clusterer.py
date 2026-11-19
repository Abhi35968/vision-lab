"""
Spatial Coherence Analysis and Clustering Module (Phase 5.3.3)
Performs rigorous 3D spatial analysis on candidate points using k-NN distance distributions,
DBSCAN multi-radius sensitivity analysis, and Statistical Outlier Removal (SOR).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from src.refinement.refinement_models import RefinementConfig


class SpatialClusterer:
    """Analyzes 3D point cloud spatial distributions, nearest-neighbor metrics, and clusters."""

    def __init__(self, config: Optional[RefinementConfig] = None):
        self.config = config or RefinementConfig()

    @staticmethod
    def compute_knn_distance_distribution(
        points: np.ndarray,
        k_values: Tuple[int, ...] = (1, 3, 5),
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        """
        Computes k-th nearest neighbor distance arrays and comprehensive percentiles.
        """
        n_pts = len(points)
        if n_pts == 0:
            return {}, {}

        tree = cKDTree(points)
        # Note: tree.query returns point itself as 1st neighbor (dist=0), so k=1 is index 1
        max_k = min(n_pts - 1, max(k_values))
        if max_k < 1:
            # Only 1 point
            knn_dict = {k: np.zeros(n_pts) for k in k_values}
            return knn_dict, {"median": 0.0, "p25": 0.0, "p75": 0.0}

        query_indices = [min(n_pts - 1, k) for k in k_values]
        dists, _ = tree.query(points, k=query_indices + [k + 1 for k in query_indices])

        # Query all up to max_k + 1
        all_dists, _ = tree.query(points, k=max_k + 1)
        knn_dict = {}
        for k in k_values:
            idx = min(k, max_k)
            knn_dict[k] = all_dists[:, idx]

        primary_k = 3 if 3 in knn_dict else list(knn_dict.keys())[0]
        ref_dists = knn_dict[primary_k]

        stats = {
            "mean": float(np.mean(ref_dists)),
            "std": float(np.std(ref_dists)),
            "min": float(np.min(ref_dists)),
            "p10": float(np.percentile(ref_dists, 10)),
            "p25": float(np.percentile(ref_dists, 25)),
            "median": float(np.median(ref_dists)),
            "p75": float(np.percentile(ref_dists, 75)),
            "p90": float(np.percentile(ref_dists, 90)),
            "max": float(np.max(ref_dists)),
        }

        return knn_dict, stats

    @staticmethod
    def run_statistical_outlier_removal(
        points: np.ndarray,
        k: int = 5,
        std_ratio: float = 2.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Filters points using Statistical Outlier Removal (SOR).
        A point is an outlier if its mean distance to k nearest neighbors exceeds mu + std_ratio * sigma.
        """
        n_pts = len(points)
        if n_pts <= k:
            return np.zeros(n_pts, dtype=bool), {"outlier_count": 0, "threshold": 0.0}

        tree = cKDTree(points)
        dists, _ = tree.query(points, k=k + 1)
        mean_knn_dists = np.mean(dists[:, 1:], axis=1)

        mu = float(np.mean(mean_knn_dists))
        sigma = float(np.std(mean_knn_dists))
        threshold = mu + std_ratio * sigma

        is_outlier = mean_knn_dists > threshold
        return is_outlier, {
            "mean_neighbor_dist": mu,
            "std_neighbor_dist": sigma,
            "threshold": threshold,
            "outlier_count": int(np.sum(is_outlier)),
        }

    def evaluate_dbscan_sensitivity(
        self,
        points: np.ndarray,
        candidate_eps: List[float],
        min_samples: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Tests clustering sensitivity across multiple candidate epsilon radii.
        """
        n_pts = len(points)
        results = []

        for eps in candidate_eps:
            if n_pts < min_samples:
                results.append({
                    "eps": round(eps, 3),
                    "num_clusters": 0,
                    "num_noise": n_pts,
                    "cluster_sizes": {-1: n_pts},
                    "largest_cluster_size": 0,
                })
                continue

            db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            labels = db.labels_

            unique_labels = set(labels)
            n_clusters = len(unique_labels - {-1})
            n_noise = int(np.sum(labels == -1))

            sizes = {}
            for lbl in unique_labels:
                sizes[int(lbl)] = int(np.sum(labels == lbl))

            largest_size = max([s for lbl, s in sizes.items() if lbl != -1], default=0)

            results.append({
                "eps": round(eps, 3),
                "num_clusters": n_clusters,
                "num_noise": n_noise,
                "cluster_sizes": sizes,
                "largest_cluster_size": largest_size,
            })

        return results

    def cluster_points(
        self,
        points: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Performs end-to-end spatial clustering and outlier identification.
        """
        n_pts = len(points)
        if n_pts == 0:
            return {
                "cluster_labels": np.array([]),
                "is_spatial_outlier": np.array([]),
                "nn_dists_1": np.array([]),
                "nn_dists_3": np.array([]),
                "sensitivity_results": [],
                "selected_eps": 0.0,
                "cluster_statistics": {},
            }

        # 1. k-NN distance distribution
        knn_dict, knn_stats = self.compute_knn_distance_distribution(points, k_values=(1, 3, 5))
        d1 = knn_dict.get(1, np.zeros(n_pts))
        d3 = knn_dict.get(3, np.zeros(n_pts))

        # 2. Select candidate epsilons from percentiles of 3-NN distances
        p25 = max(0.1, knn_stats.get("p25", 0.2))
        p50 = max(0.15, knn_stats.get("median", 0.3))
        p75 = max(0.2, knn_stats.get("p75", 0.6))
        p90 = max(0.25, knn_stats.get("p90", 1.0))
        candidate_eps = sorted(list({round(p25, 2), round(p50, 2), round(p75, 2), round(p90, 2), 1.0, 1.2}))

        sensitivity_results = self.evaluate_dbscan_sensitivity(
            points=points,
            candidate_eps=candidate_eps,
            min_samples=self.config.dbscan_min_samples,
        )

        # 3. Determine operational epsilon
        if self.config.dbscan_eps is not None:
            chosen_eps = float(self.config.dbscan_eps)
        else:
            # Use eps = 1.0 or p75 that unites the main bottle body while rejecting isolated points
            chosen_eps = float(min(1.2, max(0.8, p75)))

        db = DBSCAN(eps=chosen_eps, min_samples=self.config.dbscan_min_samples).fit(points)
        labels = db.labels_

        # 4. Statistical outlier removal (SOR)
        sor_outliers, sor_meta = self.run_statistical_outlier_removal(
            points=points,
            k=self.config.spatial_sor_k,
            std_ratio=self.config.spatial_sor_alpha,
        )

        # 5. Determine spatial outlier status:
        # A point is a spatial outlier if:
        # - It is noise in DBSCAN (label == -1) AND flagged by SOR, OR
        # - Distance to 3-NN is > 2.0 arbitrary units (clearly isolated)
        is_spatial_outlier = np.zeros(n_pts, dtype=bool)
        for i in range(n_pts):
            if labels[i] == -1 or d3[i] > 2.0 or sor_outliers[i]:
                is_spatial_outlier[i] = True

        # Compute cluster statistics
        cluster_stats = {}
        unique_labels = set(labels)
        for lbl in unique_labels:
            mask = (labels == lbl)
            pts_cluster = points[mask]
            centroid = np.mean(pts_cluster, axis=0) if len(pts_cluster) > 0 else np.zeros(3)
            p_min = np.min(pts_cluster, axis=0) if len(pts_cluster) > 0 else np.zeros(3)
            p_max = np.max(pts_cluster, axis=0) if len(pts_cluster) > 0 else np.zeros(3)
            extents = p_max - p_min
            cluster_stats[int(lbl)] = {
                "count": int(np.sum(mask)),
                "centroid": [round(float(c), 4) for c in centroid],
                "bbox_min": [round(float(c), 4) for c in p_min],
                "bbox_max": [round(float(c), 4) for c in p_max],
                "bbox_extents": [round(float(e), 4) for e in extents],
                "bounding_diameter": round(float(np.linalg.norm(extents)), 4),
            }

        return {
            "cluster_labels": labels,
            "is_spatial_outlier": is_spatial_outlier,
            "nn_dists_1": d1,
            "nn_dists_3": d3,
            "knn_stats": knn_stats,
            "sensitivity_results": sensitivity_results,
            "selected_eps": chosen_eps,
            "cluster_statistics": cluster_stats,
            "sor_meta": sor_meta,
        }

