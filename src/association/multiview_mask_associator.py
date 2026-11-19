"""
Multi-View Mask Associator Module
Associates 3D reconstructed SfM points with 2D instance segmentation masks.
Computes multi-view support ratios, validates geometric projections against observations,
and classifies points into evidence categories (STRONG, WEAK, AMBIGUOUS, BACKGROUND, INSUFFICIENT).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.association.mask_projection import CameraProjector, ProjectionResult, MaskSampleResult


@dataclass
class ObservationMaskEvidence:
    """Observation-level evidence connecting a 2D feature and projection to a mask."""
    camera_id: int
    image_name: str
    obs_coords: Tuple[float, float]
    proj_coords: Optional[Tuple[float, float]] = None
    reprojection_error_px: Optional[float] = None
    camera_depth: Optional[float] = None
    is_valid_projection: bool = False
    has_mask: bool = False
    mask_status: str = "MISSING"  # "VALID", "NO_TARGET", "MISSING", "EXCLUDED"
    is_obs_inside_mask: bool = False
    is_proj_inside_mask: bool = False
    distance_to_boundary_px: float = 0.0
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes observation evidence to dictionary."""
        return {
            "camera_id": self.camera_id,
            "image_name": self.image_name,
            "obs_coords": [round(c, 2) for c in self.obs_coords],
            "proj_coords": [round(c, 2) for c in self.proj_coords] if self.proj_coords else None,
            "reprojection_error_px": round(self.reprojection_error_px, 3) if self.reprojection_error_px is not None else None,
            "camera_depth": round(self.camera_depth, 3) if self.camera_depth is not None else None,
            "is_valid_projection": self.is_valid_projection,
            "has_mask": self.has_mask,
            "mask_status": self.mask_status,
            "is_obs_inside_mask": self.is_obs_inside_mask,
            "is_proj_inside_mask": self.is_proj_inside_mask,
            "distance_to_boundary_px": self.distance_to_boundary_px,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class PointMaskAssociation:
    """Aggregated multi-view mask evidence and classification for a single 3D point."""
    point_id: int
    track_id: int
    point_3d: Tuple[float, float, float]
    total_observations: int
    valid_projections: int
    masked_observations: int
    inside_mask_observations: int
    outside_mask_observations: int
    proj_inside_mask_observations: int
    mask_support_ratio: float
    mean_reprojection_error_px: float
    median_reprojection_error_px: float
    min_depth: float
    max_depth: float
    classification: str  # STRONG_OBJECT_CANDIDATE, WEAK_OBJECT_CANDIDATE, AMBIGUOUS, BACKGROUND_CANDIDATE, INSUFFICIENT_EVIDENCE
    classification_reason: str
    supporting_camera_ids: List[int] = field(default_factory=list)
    observations: List[ObservationMaskEvidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes point association metadata to dictionary."""
        return {
            "point_id": self.point_id,
            "track_id": self.track_id,
            "point_3d": [round(c, 4) for c in self.point_3d],
            "total_observations": self.total_observations,
            "valid_projections": self.valid_projections,
            "masked_observations": self.masked_observations,
            "inside_mask_observations": self.inside_mask_observations,
            "outside_mask_observations": self.outside_mask_observations,
            "proj_inside_mask_observations": self.proj_inside_mask_observations,
            "mask_support_ratio": round(self.mask_support_ratio, 4),
            "mean_reprojection_error_px": round(self.mean_reprojection_error_px, 3),
            "median_reprojection_error_px": round(self.median_reprojection_error_px, 3),
            "depth_range": [round(self.min_depth, 3), round(self.max_depth, 3)],
            "classification": self.classification,
            "classification_reason": self.classification_reason,
            "supporting_camera_ids": self.supporting_camera_ids,
            "num_supporting_views": len(self.supporting_camera_ids),
        }

    def to_csv_row(self) -> Dict[str, Any]:
        """Formats tabular record for CSV export."""
        return {
            "point_id": self.point_id,
            "track_id": self.track_id,
            "x_world": round(self.point_3d[0], 4),
            "y_world": round(self.point_3d[1], 4),
            "z_world": round(self.point_3d[2], 4),
            "total_observations": self.total_observations,
            "valid_projections": self.valid_projections,
            "masked_observations": self.masked_observations,
            "inside_mask_observations": self.inside_mask_observations,
            "outside_mask_observations": self.outside_mask_observations,
            "mask_support_ratio": round(self.mask_support_ratio, 3),
            "mean_reprojection_error_px": round(self.mean_reprojection_error_px, 3),
            "median_reprojection_error_px": round(self.median_reprojection_error_px, 3),
            "min_depth": round(self.min_depth, 3),
            "max_depth": round(self.max_depth, 3),
            "num_supporting_views": len(self.supporting_camera_ids),
            "supporting_cameras": ";".join(str(c) for c in self.supporting_camera_ids),
            "classification": self.classification,
            "classification_reason": self.classification_reason,
        }


@dataclass
class MultiViewAssociatorConfig:
    """Thresholds and criteria for multi-view mask association."""
    min_masked_observations: int = 3
    min_inside_observations: int = 3
    min_mask_support_ratio: float = 0.60
    weak_inside_observations: int = 2
    weak_min_ratio: float = 0.35
    ambiguous_min_ratio: float = 0.25
    ambiguous_max_ratio: float = 0.60
    max_reprojection_error_px: float = 4.0
    image_size: Tuple[int, int] = (1080, 1920)  # (width, height)


class MultiViewMaskAssociator:
    """
    Evaluates 3D SfM points against multi-view 2D binary segmentation masks.
    Calculates observation-level mask containment, geometric projection error,
    and multi-view consensus classification.
    """

    def __init__(self, config: Optional[MultiViewAssociatorConfig] = None):
        self.config = config or MultiViewAssociatorConfig()

    def evaluate_point(
        self,
        point_id: int,
        track_id: int,
        point_3d: Tuple[float, float, float],
        observations: List[Dict[str, Any]],
        registered_cameras: Dict[int, Any],
        camera_intrinsics_k: np.ndarray,
        masks: Dict[str, np.ndarray],
        frame_statuses: Dict[str, str],
    ) -> PointMaskAssociation:
        """
        Evaluates a single 3D point track across all its multi-view observations.

        Args:
            point_id: Sequential 3D point ID.
            track_id: SfM feature track ID.
            point_3d: (X, Y, Z) in world space.
            observations: List of observation dicts from feature_tracks.json.
            registered_cameras: Dict mapping camera_id -> camera pose object/dict.
            camera_intrinsics_k: 3x3 camera matrix K.
            masks: Dict mapping mask_filename (or image_name) -> 2D binary mask.
            frame_statuses: Dict mapping image_name -> target status string.

        Returns:
            PointMaskAssociation record.
        """
        obs_evidence_list: List[ObservationMaskEvidence] = []
        valid_projs = 0
        masked_obs_count = 0
        inside_obs_count = 0
        outside_obs_count = 0
        proj_inside_count = 0
        reproj_errors: List[float] = []
        depths: List[float] = []
        supporting_cams: List[int] = []

        X_w = np.array(point_3d, dtype=np.float64)
        seen_camera_ids: Set[int] = set()

        for obs in observations:
            cid = obs.get("image_id", obs.get("camera_id"))
            coords = tuple(obs.get("coords", (0.0, 0.0)))

            # Check if camera is registered
            cam_info = registered_cameras.get(cid)
            cam_img_name = None
            if cam_info is not None:
                if isinstance(cam_info, dict):
                    cam_img_name = cam_info.get("image_name") or cam_info.get("filename")
                else:
                    cam_img_name = getattr(cam_info, "image_name", getattr(cam_info, "filename", None))

            img_name = obs.get("image_name") or cam_img_name or f"frame_{cid:04d}.png"

            # Prevent duplicate observations from the same camera view inflating support
            if cid in seen_camera_ids:
                obs_ev = ObservationMaskEvidence(
                    camera_id=cid,
                    image_name=img_name,
                    obs_coords=coords,
                    is_valid_projection=False,
                    has_mask=False,
                    mask_status="DUPLICATE",
                    rejection_reason=f"Duplicate observation for camera {cid} ignored to prevent support inflation",
                )
                obs_evidence_list.append(obs_ev)
                continue
            seen_camera_ids.add(cid)
            if cam_info is None:
                # Camera not registered in SfM
                obs_ev = ObservationMaskEvidence(
                    camera_id=cid,
                    image_name=img_name,
                    obs_coords=coords,
                    is_valid_projection=False,
                    has_mask=False,
                    mask_status="UNREGISTERED",
                    rejection_reason="Camera view is not registered in SfM",
                )
                obs_evidence_list.append(obs_ev)
                continue

            # Extract rotation and translation
            if isinstance(cam_info, dict):
                R = np.array(cam_info["rotation_matrix"], dtype=np.float64)
                t = np.array(cam_info["translation_vector"], dtype=np.float64).reshape(3, 1)
            else:
                R = np.array(cam_info.rotation_matrix, dtype=np.float64)
                t = np.array(cam_info.translation_vector, dtype=np.float64).reshape(3, 1)

            # Project 3D point
            proj_res = CameraProjector.project_point(
                point_3d_world=X_w,
                rotation_matrix=R,
                translation_vector=t,
                camera_matrix_k=camera_intrinsics_k,
                image_size=self.config.image_size,
            )

            if not proj_res.is_valid:
                obs_ev = ObservationMaskEvidence(
                    camera_id=cid,
                    image_name=img_name,
                    obs_coords=coords,
                    camera_depth=proj_res.depth,
                    is_valid_projection=False,
                    has_mask=False,
                    mask_status="INVALID_PROJECTION",
                    rejection_reason=proj_res.rejection_reason,
                )
                obs_evidence_list.append(obs_ev)
                continue

            valid_projs += 1
            depths.append(proj_res.depth)
            reproj_err = CameraProjector.compute_reprojection_error(proj_res.coords, coords)
            reproj_errors.append(reproj_err)

            # Check mask availability
            mask_name = Path(img_name).stem + "_mask.png"
            mask = masks.get(mask_name, masks.get(img_name))
            status = frame_statuses.get(img_name, "VALID" if (mask is not None and np.any(mask > 0)) else "NO_TARGET")

            if mask is None:
                obs_ev = ObservationMaskEvidence(
                    camera_id=cid,
                    image_name=img_name,
                    obs_coords=coords,
                    proj_coords=proj_res.coords,
                    reprojection_error_px=reproj_err,
                    camera_depth=proj_res.depth,
                    is_valid_projection=True,
                    has_mask=False,
                    mask_status="MISSING",
                    rejection_reason="Segmentation mask file not found",
                )
                obs_evidence_list.append(obs_ev)
                continue

            # Only count as masked observation if the frame had a VALID target bottle detection
            # Frames marked NO_TARGET (e.g. out-of-view / motion blur) are not counted against the point!
            if status == "VALID":
                masked_obs_count += 1

                # Sample observed coordinate
                obs_sample = CameraProjector.sample_mask(coords, mask)
                # Sample projected coordinate
                proj_sample = CameraProjector.sample_mask(proj_res.coords, mask)

                is_inside = obs_sample.is_inside
                if is_inside:
                    inside_obs_count += 1
                    supporting_cams.append(cid)
                else:
                    outside_obs_count += 1

                if proj_sample.is_inside:
                    proj_inside_count += 1

                obs_ev = ObservationMaskEvidence(
                    camera_id=cid,
                    image_name=img_name,
                    obs_coords=coords,
                    proj_coords=proj_res.coords,
                    reprojection_error_px=reproj_err,
                    camera_depth=proj_res.depth,
                    is_valid_projection=True,
                    has_mask=True,
                    mask_status=status,
                    is_obs_inside_mask=is_inside,
                    is_proj_inside_mask=proj_sample.is_inside,
                    distance_to_boundary_px=obs_sample.distance_to_boundary_px,
                )
            else:
                # Frame had no target detected (e.g., camera pointed at books)
                obs_ev = ObservationMaskEvidence(
                    camera_id=cid,
                    image_name=img_name,
                    obs_coords=coords,
                    proj_coords=proj_res.coords,
                    reprojection_error_px=reproj_err,
                    camera_depth=proj_res.depth,
                    is_valid_projection=True,
                    has_mask=True,
                    mask_status=status,
                    is_obs_inside_mask=False,
                    is_proj_inside_mask=False,
                    rejection_reason=f"Frame status is {status} (target not visible/untracked)",
                )

            obs_evidence_list.append(obs_ev)

        # Multi-view aggregation
        mask_support_ratio = (inside_obs_count / float(masked_obs_count)) if masked_obs_count > 0 else 0.0
        mean_err = float(np.mean(reproj_errors)) if reproj_errors else 0.0
        median_err = float(np.median(reproj_errors)) if reproj_errors else 0.0
        min_d = float(np.min(depths)) if depths else 0.0
        max_d = float(np.max(depths)) if depths else 0.0

        # Classification Logic
        classification, reason = self._classify_point(
            masked_obs=masked_obs_count,
            inside_obs=inside_obs_count,
            support_ratio=mask_support_ratio,
            mean_reproj_err=mean_err,
            total_obs=len(observations),
        )

        return PointMaskAssociation(
            point_id=point_id,
            track_id=track_id,
            point_3d=point_3d,
            total_observations=len(observations),
            valid_projections=valid_projs,
            masked_observations=masked_obs_count,
            inside_mask_observations=inside_obs_count,
            outside_mask_observations=outside_obs_count,
            proj_inside_mask_observations=proj_inside_count,
            mask_support_ratio=round(mask_support_ratio, 4),
            mean_reprojection_error_px=round(mean_err, 3),
            median_reprojection_error_px=round(median_err, 3),
            min_depth=round(min_d, 3),
            max_depth=round(max_d, 3),
            classification=classification,
            classification_reason=reason,
            supporting_camera_ids=supporting_cams,
            observations=obs_evidence_list,
        )

    def _classify_point(
        self,
        masked_obs: int,
        inside_obs: int,
        support_ratio: float,
        mean_reproj_err: float,
        total_obs: int,
    ) -> Tuple[str, str]:
        """
        Classifies the point based on multi-view evidence thresholds.
        """
        # Case 1: Insufficient valid masked observations
        if masked_obs < 2:
            return (
                "INSUFFICIENT_EVIDENCE",
                f"Only {masked_obs} masked observation(s) available (minimum 2 required for multi-view consensus)",
            )

        # Case 2: Strong object candidate
        # >= 3 masked obs, >= 3 inside obs, support_ratio >= 0.60
        if (
            masked_obs >= self.config.min_masked_observations
            and inside_obs >= self.config.min_inside_observations
            and support_ratio >= self.config.min_mask_support_ratio
        ):
            return (
                "STRONG_OBJECT_CANDIDATE",
                f"Strong multi-view support: {inside_obs}/{masked_obs} views inside mask "
                f"(ratio={support_ratio:.2f} >= {self.config.min_mask_support_ratio}, mean_err={mean_reproj_err:.2f}px)",
            )

        # Case 3: Weak object candidate
        # >= 2 inside obs with ratio >= 0.35, or perfect 2/2 views
        if (
            inside_obs >= self.config.weak_inside_observations
            and support_ratio >= self.config.weak_min_ratio
        ) or (inside_obs >= 1 and masked_obs == 2 and support_ratio >= 0.50):
            return (
                "WEAK_OBJECT_CANDIDATE",
                f"Moderate multi-view support: {inside_obs}/{masked_obs} views inside mask "
                f"(ratio={support_ratio:.2f}, >= {self.config.weak_min_ratio})",
            )

        # Case 4: Ambiguous
        # Inside observations exist but ratio is split (between 0.25 and 0.60)
        if inside_obs > 0 and (
            self.config.ambiguous_min_ratio <= support_ratio < self.config.min_mask_support_ratio
        ):
            return (
                "AMBIGUOUS",
                f"Conflicting multi-view observations: {inside_obs}/{masked_obs} inside mask "
                f"(ratio={support_ratio:.2f} in ambiguous band [{self.config.ambiguous_min_ratio}, {self.config.min_mask_support_ratio}))",
            )

        # Case 5: Background candidate
        # Dominantly or strictly outside mask
        if inside_obs == 0 or support_ratio < self.config.ambiguous_min_ratio:
            return (
                "BACKGROUND_CANDIDATE",
                f"Background consensus: {inside_obs}/{masked_obs} inside mask (ratio={support_ratio:.2f} < {self.config.ambiguous_min_ratio})",
            )

        return ("AMBIGUOUS", f"Unresolved evidence distribution ({inside_obs}/{masked_obs} inside)")

