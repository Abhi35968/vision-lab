"""
Real Video Object Association Pipeline
Orchestrates Phase 5.3.2: Multi-View 2D-to-3D Object Association using Segmentation Masks.
Connects verified SfM 3D points and camera poses with 2D instance segmentation masks,
performs projection and mask testing, partitions point clouds, and generates comprehensive diagnostics.
"""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np

from src.association.mask_projection import CameraProjector
from src.association.multiview_mask_associator import (
    MultiViewMaskAssociator,
    MultiViewAssociatorConfig,
    PointMaskAssociation,
)
from src.association.object_point_filter import ObjectPointFilter
from src.association.object_association_metrics import (
    ObjectAssociationSummary,
    ObjectAssociationMetricsCalculator,
)
from src.association.object_association_visualizer import ObjectAssociationVisualizer


class RealVideoObjectAssociationPipeline:
    """Master orchestrator for Phase 5.3.2 2D-to-3D mask association."""

    def __init__(
        self,
        sfm_dir: Union[str, Path] = "outputs/real_video_sfm/focal_1300",
        seg_dir: Union[str, Path] = "outputs/object_segmentation",
        keyframes_dir: Union[str, Path] = "outputs/real_video_validation/keyframes",
        output_dir: Union[str, Path] = "outputs/object_association",
        min_masked_observations: int = 3,
        min_inside_observations: int = 3,
        min_mask_support_ratio: float = 0.60,
    ):
        self.sfm_dir = Path(sfm_dir)
        self.seg_dir = Path(seg_dir)
        self.keyframes_dir = Path(keyframes_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = MultiViewAssociatorConfig(
            min_masked_observations=min_masked_observations,
            min_inside_observations=min_inside_observations,
            min_mask_support_ratio=min_mask_support_ratio,
            image_size=(1080, 1920),
        )
        self.associator = MultiViewMaskAssociator(config=self.config)

    def load_sfm_reconstruction(self) -> Tuple[Dict[int, Any], List[Dict[str, Any]], np.ndarray, Dict[int, int]]:
        """
        Loads camera poses, feature tracks, intrinsics, and point-to-track mappings.
        """
        cams_path = self.sfm_dir / "cameras.json"
        tracks_path = self.sfm_dir / "feature_tracks.json"
        pt_errors_path = self.sfm_dir / "point_errors.csv"

        if not cams_path.exists():
            raise FileNotFoundError(f"Cameras file not found at '{cams_path}'.")
        if not tracks_path.exists():
            raise FileNotFoundError(f"Feature tracks file not found at '{tracks_path}'.")

        with open(cams_path, "r", encoding="utf-8") as f:
            cams_raw = json.load(f)

        registered_cams = {
            int(cid): cinfo for cid, cinfo in cams_raw.items() if cinfo.get("is_registered", False)
        }

        with open(tracks_path, "r", encoding="utf-8") as f:
            tracks_raw = json.load(f)

        tracks_list = tracks_raw.get("tracks", [])

        # Intrinsic matrix K for focal_1300
        K = np.array([
            [1300.0, 0.0, 540.0],
            [0.0, 1300.0, 960.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Build point_id -> track_id mapping if available
        track_to_pt: Dict[int, int] = {}
        if pt_errors_path.exists():
            import pandas as pd
            df = pd.read_csv(pt_errors_path)
            for _, r in df.iterrows():
                track_to_pt[int(r["track_id"])] = int(r["point_id"])

        return registered_cams, tracks_list, K, track_to_pt

    def load_segmentation_masks(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """
        Loads binary masks and per-frame tracking status.
        """
        masks_dir = self.seg_dir / "masks"
        meta_path = self.seg_dir / "segmentation_metadata.json"

        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found at '{masks_dir}'.")

        masks: Dict[str, np.ndarray] = {}
        for p in masks_dir.glob("*.png"):
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.ndim == 3:
                    m = m[:, :, 0]
                masks[p.name] = m

        frame_statuses: Dict[str, str] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for f_info in meta.get("frames", []):
                frame_statuses[f_info["image_name"]] = f_info.get("target_status", "UNKNOWN")

        return masks, frame_statuses

    def run(self) -> Dict[str, Any]:
        """Executes the complete multi-view 2D-to-3D object association pipeline."""
        print("\n" + "=" * 80)
        print(" PHASE 5.3.2: MULTI-VIEW 2D-TO-3D OBJECT ASSOCIATION")
        print("=" * 80)
        print(f"SfM Directory        : {self.sfm_dir}")
        print(f"Segmentation Dir     : {self.seg_dir}")
        print(f"Keyframes Directory  : {self.keyframes_dir}")
        print(f"Output Directory     : {self.output_dir}")
        print(f"Min Masked Views     : {self.config.min_masked_observations}")
        print(f"Min Inside Views     : {self.config.min_inside_observations}")
        print(f"Min Support Ratio    : {self.config.min_mask_support_ratio}")
        print("=" * 80)

        # 1. Load inputs
        reg_cams, tracks, K, track_to_pt = self.load_sfm_reconstruction()
        masks, frame_statuses = self.load_segmentation_masks()

        print(f"[INFO] Loaded {len(reg_cams)} registered camera poses from SfM.")
        print(f"[INFO] Loaded {len(tracks)} multi-view feature tracks.")
        print(f"[INFO] Loaded {len(masks)} binary segmentation masks from Phase 5.3.1.")

        # 2. Extract camera centers for 3D visualization
        camera_centers: Dict[int, np.ndarray] = {}
        for cid, cinfo in reg_cams.items():
            C = np.array(cinfo.get("camera_center", [0, 0, 0]), dtype=np.float64)
            camera_centers[cid] = C

        # 3. Associate all 3D points
        print("\n[INFO] Evaluating 3D points against multi-view segmentation masks...")
        t0 = time.perf_counter()
        associations: List[PointMaskAssociation] = []

        for idx, tr in enumerate(tracks):
            tid = tr.get("track_id", idx)
            pid = track_to_pt.get(tid, idx)
            p3d = tuple(tr.get("point_3d", (0.0, 0.0, 0.0)))
            obs_list = tr.get("observations", [])

            assoc = self.associator.evaluate_point(
                point_id=pid,
                track_id=tid,
                point_3d=p3d,
                observations=obs_list,
                registered_cameras=reg_cams,
                camera_intrinsics_k=K,
                masks=masks,
                frame_statuses=frame_statuses,
            )
            associations.append(assoc)

        elapsed = time.perf_counter() - t0
        print(f"[INFO] Association completed in {elapsed:.2f}s ({elapsed/max(1, len(tracks))*1000.0:.2f} ms/point).")

        # 4. Partition points and export categorized point clouds
        print("\n[INFO] Partitioning point clouds and computing spatial extents...")
        spatial_extents = ObjectPointFilter.export_all_point_clouds(
            associations=associations,
            output_dir=self.output_dir,
        )

        # 5. Calculate summary metrics
        summary = ObjectAssociationMetricsCalculator.calculate_summary(
            associations=associations,
            spatial_extents=spatial_extents,
        )

        # 6. Render 3D and 2D visual diagnostics
        print("\n[INFO] Rendering 3D and 2D visual diagnostics...")
        ObjectAssociationVisualizer.plot_3d_scene(
            associations=associations,
            camera_centers=camera_centers,
            output_path=self.output_dir / "3d_scene_association_cloud.png",
        )

        ObjectAssociationVisualizer.plot_diagnostic_distributions(
            associations=associations,
            output_dir=self.output_dir,
        )

        ObjectAssociationVisualizer.render_2d_diagnostic_keyframes(
            associations=associations,
            keyframes_dir=self.keyframes_dir,
            masks_dir=self.seg_dir / "masks",
            output_dir=self.output_dir,
            target_camera_ids=[0, 12, 24, 25],
        )

        # 7. Export tabular CSV
        csv_path = self.output_dir / "point_mask_associations.csv"
        csv_rows = [a.to_csv_row() for a in associations]
        if csv_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"[INFO] Tabular associations CSV saved to: {csv_path}")

        # 8. Export master JSON metadata
        json_path = self.output_dir / "association_metadata.json"
        metadata = {
            "summary": summary.to_dict(),
            "points": [a.to_dict() for a in associations],
        }
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(metadata, f_json, indent=2)
        print(f"[INFO] Master association JSON saved to: {json_path}")

        # Print final console summary
        print("\n" + "=" * 80)
        print(" PHASE 5.3.2 2D-TO-3D OBJECT ASSOCIATION SUMMARY")
        print("=" * 80)
        print(f"Total Reconstructed Points   : {summary.total_reconstructed_points}")
        print(f"Points with Valid Tracks     : {summary.points_with_valid_tracks}")
        print(f"Points with >=3 Masked Views : {summary.points_with_sufficient_masked_views} ({summary.to_dict()['reconstruction_counts']['percent_with_sufficient_masked_views']}%)")
        print("-" * 80)
        print(f"Strong Object Candidates     : {summary.strong_candidate_count:>4} ({summary.to_dict()['classification_breakdown']['strong_object_candidates']['percent']}%)")
        print(f"Weak Object Candidates       : {summary.weak_candidate_count:>4} ({summary.to_dict()['classification_breakdown']['weak_object_candidates']['percent']}%)")
        print(f"Ambiguous Evidence Points    : {summary.ambiguous_count:>4} ({summary.to_dict()['classification_breakdown']['ambiguous_points']['percent']}%)")
        print(f"Background Scene Candidates  : {summary.background_candidate_count:>4} ({summary.to_dict()['classification_breakdown']['background_candidates']['percent']}%)")
        print(f"Insufficient Evidence Points : {summary.insufficient_evidence_count:>4} ({summary.to_dict()['classification_breakdown']['insufficient_evidence']['percent']}%)")
        print("-" * 80)
        print(f"Total Observations Evaluated : {summary.total_observations_evaluated}")
        print(f"Valid Projections            : {summary.total_valid_projections} ({summary.to_dict()['projections_and_observations']['valid_projection_rate_percent']}%)")
        print(f"Invalid Projections          : {summary.total_invalid_projections}")
        print(f"Mean Reprojection Error (All): {summary.reprojection_error_mean_all_px:.3f} px")
        print(f"Mean Reproj Error (Strong)   : {summary.reprojection_error_mean_strong_px:.3f} px")
        print(f"Median Reproj Error (Strong) : {summary.reprojection_error_median_strong_px:.3f} px")
        if "strong_candidate_points" in spatial_extents and "centroid" in spatial_extents["strong_candidate_points"]:
            c_strong = spatial_extents["strong_candidate_points"]["centroid"]
            e_strong = spatial_extents["strong_candidate_points"]["bbox_extents"]
            print(f"Strong Candidate Centroid    : [{c_strong[0]:.3f}, {c_strong[1]:.3f}, {c_strong[2]:.3f}] (arbitrary units)")
            print(f"Strong Candidate Extents     : dx={e_strong[0]:.3f}, dy={e_strong[1]:.3f}, dz={e_strong[2]:.3f}")
        print("=" * 80 + "\n")

        return metadata
