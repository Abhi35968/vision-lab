"""
Real Video Object Refinement Pipeline Orchestrator (Phase 5.3.3)
Coordinates the geometric refinement of candidate object points for the tabletop water bottle.
Fuses reprojection error filtering, ray triangulation geometry, and spatial clustering.
Generates comprehensive diagnostics, point cloud exports, and statistical plots.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd

from src.refinement.refinement_models import PointGeometricProfile, RefinementConfig
from src.refinement.point_refiner import PointRefiner
from src.refinement.refinement_visualizer import RefinementVisualizer


class RealVideoRefinementPipeline:
    """Master orchestrator for Phase 5.3.3 Object-Specific Sparse Point Cloud Refinement."""

    def __init__(
        self,
        sfm_dir: Union[str, Path] = "outputs/real_video_sfm/focal_1300",
        assoc_dir: Union[str, Path] = "outputs/object_association",
        seg_dir: Union[str, Path] = "outputs/object_segmentation",
        output_dir: Union[str, Path] = "outputs/object_refinement",
        config: Optional[RefinementConfig] = None,
    ):
        self.sfm_dir = Path(sfm_dir)
        self.assoc_dir = Path(assoc_dir)
        self.seg_dir = Path(seg_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or RefinementConfig()
        self.refiner = PointRefiner(config=self.config)

    def load_inputs(self) -> Tuple[List[Dict[str, Any]], Dict[int, Any], Dict[int, Any], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Loads candidate associations, feature tracks, camera poses, intrinsics, and masks.
        """
        # 1. Candidate associations
        csv_path = self.assoc_dir / "point_mask_associations.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Association CSV not found at '{csv_path}'.")

        df = pd.read_csv(csv_path)
        cands_df = df[df["classification"].isin(["STRONG_OBJECT_CANDIDATE", "WEAK_OBJECT_CANDIDATE", "AMBIGUOUS"])].copy()
        candidate_records = cands_df.to_dict(orient="records")

        # Save Step 1 baseline CSV
        baseline_path = self.output_dir / "candidate_baseline_47.csv"
        cands_df.to_csv(baseline_path, index=False)
        print(f"[INFO] Step 1 baseline for {len(candidate_records)} candidates saved to: {baseline_path}")

        # 2. Camera poses
        cams_path = self.sfm_dir / "cameras.json"
        with open(cams_path, "r", encoding="utf-8") as f:
            cams_raw = json.load(f)
        reg_cams = {int(k): v for k, v in cams_raw.items() if v.get("is_registered", False)}

        # 3. Feature tracks
        tracks_path = self.sfm_dir / "feature_tracks.json"
        with open(tracks_path, "r", encoding="utf-8") as f:
            tracks_raw = json.load(f)
        tracks = tracks_raw.get("tracks", [])
        tracks_by_id = {t["track_id"]: t for t in tracks}

        # 4. Intrinsics (focal 1300)
        K = np.array([
            [1300.0, 0.0, 540.0],
            [0.0, 1300.0, 960.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # 5. Masks
        masks_dir = self.seg_dir / "masks"
        masks = {}
        if masks_dir.exists():
            for p in masks_dir.glob("*.png"):
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.ndim == 3:
                        m = m[:, :, 0]
                    masks[p.name] = m

        # 6. All SfM points for background visualization
        all_pts = np.array([t.get("point_3d", [0, 0, 0]) for t in tracks], dtype=np.float64)

        return candidate_records, tracks_by_id, reg_cams, K, masks, all_pts

    def run(self) -> Dict[str, Any]:
        """Executes the complete geometric refinement pipeline."""
        print("\n" + "=" * 80)
        print(" PHASE 5.3.3: OBJECT-SPECIFIC SPARSE POINT CLOUD REFINEMENT")
        print("=" * 80)
        print(f"SfM Directory        : {self.sfm_dir}")
        print(f"Association Dir      : {self.assoc_dir}")
        print(f"Segmentation Dir     : {self.seg_dir}")
        print(f"Output Directory     : {self.output_dir}")
        print(f"High Conf Max Med Err: {self.config.high_conf_max_median_err_px} px")
        print(f"High Conf Min Parall : {self.config.high_conf_min_triangulation_angle_deg}°")
        print("=" * 80)

        # 1. Load inputs
        candidates, tracks_by_id, reg_cams, K, masks, all_sfm_pts = self.load_inputs()
        print(f"[INFO] Loaded {len(candidates)} object candidate points (22 Strong, 20 Weak, 5 Ambiguous).")
        print(f"[INFO] Loaded {len(reg_cams)} registered camera poses from SfM.")
        print(f"[INFO] Loaded {len(masks)} binary segmentation masks.")

        # 2. Extract camera centers for 3D visualization
        camera_centers = {}
        for cid, cinfo in reg_cams.items():
            C = np.array(cinfo.get("camera_center", [0, 0, 0]), dtype=np.float64)
            camera_centers[cid] = C

        # 3. Execute refinement
        t0 = time.perf_counter()
        profiles = self.refiner.refine_candidate_points(
            candidate_associations=candidates,
            tracks_by_id=tracks_by_id,
            registered_cameras=reg_cams,
            camera_intrinsics_k=K,
            masks=masks,
        )
        elapsed = time.perf_counter() - t0
        print(f"[INFO] Geometric refinement completed in {elapsed:.3f}s ({elapsed/max(1, len(profiles))*1000.0:.2f} ms/point).")

        # 4. Export artifacts
        summary = self.refiner.export_refinement_artifacts(
            profiles=profiles,
            output_dir=self.output_dir,
        )

        # 5. Extract spatial cluster metadata for visualization
        coords = np.array([p.point_3d for p in profiles], dtype=np.float64)
        cluster_res = self.refiner.spatial_clusterer.cluster_points(coords)

        # 6. Render diagnostic visualizations
        print("\n[INFO] Rendering multi-panel diagnostic visualizations...")
        RefinementVisualizer.plot_3d_scene_comparison(
            all_sfm_points=all_sfm_pts,
            camera_centers=camera_centers,
            profiles=profiles,
            output_path=self.output_dir / "3d_scene_refinement.png",
        )

        RefinementVisualizer.plot_reprojection_error_distributions(
            profiles=profiles,
            output_path=self.output_dir / "reprojection_error_distributions.png",
        )

        RefinementVisualizer.plot_triangulation_and_depth(
            profiles=profiles,
            output_path=self.output_dir / "triangulation_angle_distribution.png",
        )

        RefinementVisualizer.plot_spatial_clustering_analysis(
            profiles=profiles,
            sensitivity_results=cluster_res["sensitivity_results"],
            knn_stats=cluster_res["knn_stats"],
            output_path=self.output_dir / "spatial_clustering_analysis.png",
        )

        # 7. Print console telemetry
        counts = summary["refined_classification_counts"]
        print("\n" + "=" * 80)
        print(" PHASE 5.3.3 REFINEMENT SUMMARY")
        print("=" * 80)
        print(f"Total Candidate Points Evaluated : {len(profiles)}")
        print(f"  HIGH_GEOMETRIC_CONFIDENCE      : {counts.get('HIGH_GEOMETRIC_CONFIDENCE', 0)} ({counts.get('HIGH_GEOMETRIC_CONFIDENCE', 0)/len(profiles)*100:.1f}%)")
        print(f"  MODERATE_GEOMETRIC_CONFIDENCE  : {counts.get('MODERATE_GEOMETRIC_CONFIDENCE', 0)} ({counts.get('MODERATE_GEOMETRIC_CONFIDENCE', 0)/len(profiles)*100:.1f}%)")
        print(f"  LOW_GEOMETRIC_CONFIDENCE       : {counts.get('LOW_GEOMETRIC_CONFIDENCE', 0)} ({counts.get('LOW_GEOMETRIC_CONFIDENCE', 0)/len(profiles)*100:.1f}%)")
        print(f"  GEOMETRIC_OUTLIER              : {counts.get('GEOMETRIC_OUTLIER', 0)} ({counts.get('GEOMETRIC_OUTLIER', 0)/len(profiles)*100:.1f}%)")
        print(f"  INSUFFICIENT_GEOMETRIC_EVIDENCE: {counts.get('INSUFFICIENT_GEOMETRIC_EVIDENCE', 0)} ({counts.get('INSUFFICIENT_GEOMETRIC_EVIDENCE', 0)/len(profiles)*100:.1f}%)")
        print("-" * 80)
        print(f"Artifacts saved to: {self.output_dir}/")
        print("  - candidate_baseline_47.csv")
        print("  - refined_object_points.csv")
        print("  - refinement_summary.json")
        print("  - refined_object_points.ply, high_confidence_points.ply, moderate_confidence_points.ply, ...")
        print("  - 3d_scene_refinement.png, reprojection_error_distributions.png, ...")
        print("=" * 80 + "\n")

        return summary

