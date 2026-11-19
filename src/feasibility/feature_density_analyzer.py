"""
Feature Density Analyzer Module
Analyzes SIFT keypoint distribution, spatial density inside vs. outside object segmentation masks,
silhouette vertical coverage, and multi-view feature track longevity.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class SilhouetteDistribution:
    """Breakdown of keypoints across vertical silhouette zones of the object."""
    cap_count: int         # Normalized y in [0.00, 0.20)
    neck_count: int        # Normalized y in [0.20, 0.35)
    body_count: int        # Normalized y in [0.35, 0.80)
    base_count: int        # Normalized y in [0.80, 1.00]
    total_inside: int


@dataclass
class FrameFeatureDensityRecord:
    """Per-frame keypoint extraction and density metrics."""
    frame_idx: int
    image_name: str
    is_registered: bool
    total_keypoints: int
    mask_present: bool
    mask_area_px: int
    mask_area_pct: float
    inside_mask_keypoints: int
    outside_mask_keypoints: int
    density_inside_per_10k_px2: float
    density_outside_per_10k_px2: float
    density_ratio_inside_to_outside: float
    cap_keypoints: int
    neck_keypoints: int
    body_keypoints: int
    base_keypoints: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackSurvivalSummary:
    """Multi-view track length and survival rate breakdown."""
    total_tracks: int
    pure_object_tracks: int
    pure_background_tracks: int
    mixed_tracks: int
    object_track_length_distribution: Dict[int, int]
    background_track_length_distribution: Dict[int, int]
    mixed_track_length_distribution: Dict[int, int]
    object_track_mean_length: float
    background_track_mean_length: float
    object_survival_rate_ge_3_views: float
    background_survival_rate_ge_3_views: float


class FeatureDensityAnalyzer:
    """
    Evaluates local feature repeatability and spatial density inside object masks
    to determine whether an object has sufficient texture for dense multi-view reconstruction.
    """

    def __init__(
        self,
        n_features: int = 4000,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
    ):
        self.n_features = n_features
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Extracts SIFT keypoints and descriptors from an RGB/BGR or grayscale image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        kps, descs = self.sift.detectAndCompute(gray, None)
        if descs is None:
            descs = np.zeros((0, 128), dtype=np.float32)
        return kps, descs

    def partition_keypoints_by_mask(
        self,
        keypoints: List[cv2.KeyPoint],
        mask: np.ndarray,
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint]]:
        """
        Partitions keypoints into inside-mask and outside-mask subsets.
        
        Args:
            keypoints: List of cv2.KeyPoint detected in the image.
            mask: Binary or grayscale mask (H, W) where >0 represents object.
        """
        h, w = mask.shape[:2]
        inside = []
        outside = []
        for kp in keypoints:
            x = int(round(kp.pt[0]))
            y = int(round(kp.pt[1]))
            # Safe boundary clamp
            x_c = min(max(0, x), w - 1)
            y_c = min(max(0, y), h - 1)
            if mask[y_c, x_c] > 0:
                inside.append(kp)
            else:
                outside.append(kp)
        return inside, outside

    def compute_silhouette_distribution(
        self,
        inside_keypoints: List[cv2.KeyPoint],
        mask: np.ndarray,
    ) -> SilhouetteDistribution:
        """
        Categorizes object keypoints into vertical silhouette compartments:
        - Cap: [0.00, 0.20) normalized height of mask bounding box
        - Neck: [0.20, 0.35)
        - Body / Label: [0.35, 0.80)
        - Base: [0.80, 1.00]
        """
        if len(inside_keypoints) == 0 or np.count_nonzero(mask > 0) == 0:
            return SilhouetteDistribution(0, 0, 0, 0, 0)

        ys, xs = np.where(mask > 0)
        ymin, ymax = int(ys.min()), int(ys.max())
        bbox_height = max(1, ymax - ymin)

        cap, neck, body, base = 0, 0, 0, 0
        for kp in inside_keypoints:
            rel_y = (kp.pt[1] - ymin) / float(bbox_height)
            if rel_y < 0.20:
                cap += 1
            elif rel_y < 0.35:
                neck += 1
            elif rel_y < 0.80:
                body += 1
            else:
                base += 1

        return SilhouetteDistribution(
            cap_count=cap,
            neck_count=neck,
            body_count=body,
            base_count=base,
            total_inside=len(inside_keypoints),
        )

    def analyze_frame(
        self,
        frame_idx: int,
        image_name: str,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        is_registered: bool = True,
    ) -> FrameFeatureDensityRecord:
        """Analyzes a single frame's keypoint count and density."""
        kps, _ = self.extract_features(image)
        h, w = image.shape[:2]
        total_px = h * w

        if mask is not None and np.count_nonzero(mask > 0) > 0:
            mask_px = int(np.count_nonzero(mask > 0))
            mask_pct = (mask_px / total_px) * 100.0
            mask_present = True
            inside_kps, outside_kps = self.partition_keypoints_by_mask(kps, mask)
            sil_dist = self.compute_silhouette_distribution(inside_kps, mask)
        else:
            mask_px = 0
            mask_pct = 0.0
            mask_present = False
            inside_kps = []
            outside_kps = kps
            sil_dist = SilhouetteDistribution(0, 0, 0, 0, 0)

        outside_px = total_px - mask_px

        # Density per 10,000 pixels (100x100 pixel patch)
        density_inside = (len(inside_kps) / (mask_px / 10000.0)) if mask_px > 0 else 0.0
        density_outside = (len(outside_kps) / (outside_px / 10000.0)) if outside_px > 0 else 0.0
        density_ratio = (density_inside / density_outside) if density_outside > 0 else 0.0

        return FrameFeatureDensityRecord(
            frame_idx=frame_idx,
            image_name=image_name,
            is_registered=is_registered,
            total_keypoints=len(kps),
            mask_present=mask_present,
            mask_area_px=mask_px,
            mask_area_pct=round(mask_pct, 2),
            inside_mask_keypoints=len(inside_kps),
            outside_mask_keypoints=len(outside_kps),
            density_inside_per_10k_px2=round(density_inside, 4),
            density_outside_per_10k_px2=round(density_outside, 4),
            density_ratio_inside_to_outside=round(density_ratio, 4),
            cap_keypoints=sil_dist.cap_count,
            neck_keypoints=sil_dist.neck_count,
            body_keypoints=sil_dist.body_count,
            base_keypoints=sil_dist.base_count,
        )

    def analyze_track_survival(
        self,
        tracks: List[dict],
        masks: List[Optional[np.ndarray]],
    ) -> TrackSurvivalSummary:
        """
        Evaluates track length distributions and survival rates for:
        - Pure object tracks (all observations inside mask)
        - Pure background tracks (all observations outside mask)
        - Mixed tracks (observations span both)
        """
        pure_obj = []
        pure_bg = []
        mixed = []

        for t in tracks:
            obs = t.get("observations", [])
            if not obs:
                continue
            length = len(obs)
            in_cnt = 0
            valid_mask_cnt = 0

            for o in obs:
                img_id = o.get("image_id", 0)
                if img_id < len(masks) and masks[img_id] is not None:
                    m = masks[img_id]
                    if np.count_nonzero(m > 0) > 0:
                        valid_mask_cnt += 1
                        x, y = int(round(o["coords"][0])), int(round(o["coords"][1]))
                        y_c = min(max(0, y), m.shape[0] - 1)
                        x_c = min(max(0, x), m.shape[1] - 1)
                        if m[y_c, x_c] > 0:
                            in_cnt += 1

            if in_cnt == 0:
                pure_bg.append(length)
            elif in_cnt == valid_mask_cnt and valid_mask_cnt > 0:
                pure_obj.append(length)
            else:
                mixed.append(length)

        def make_dist(lengths: List[int]) -> Dict[int, int]:
            d: Dict[int, int] = {}
            for l in lengths:
                d[l] = d.get(l, 0) + 1
            return dict(sorted(d.items()))

        obj_dist = make_dist(pure_obj)
        bg_dist = make_dist(pure_bg)
        mixed_dist = make_dist(mixed)

        mean_obj = float(np.mean(pure_obj)) if pure_obj else 0.0
        mean_bg = float(np.mean(pure_bg)) if pure_bg else 0.0

        surv_obj_ge_3 = (sum(cnt for l, cnt in obj_dist.items() if l >= 3) / len(pure_obj)) if pure_obj else 0.0
        surv_bg_ge_3 = (sum(cnt for l, cnt in bg_dist.items() if l >= 3) / len(pure_bg)) if pure_bg else 0.0

        return TrackSurvivalSummary(
            total_tracks=len(pure_obj) + len(pure_bg) + len(mixed),
            pure_object_tracks=len(pure_obj),
            pure_background_tracks=len(pure_bg),
            mixed_tracks=len(mixed),
            object_track_length_distribution=obj_dist,
            background_track_length_distribution=bg_dist,
            mixed_track_length_distribution=mixed_dist,
            object_track_mean_length=round(mean_obj, 2),
            background_track_mean_length=round(mean_bg, 2),
            object_survival_rate_ge_3_views=round(surv_obj_ge_3, 4),
            background_survival_rate_ge_3_views=round(surv_bg_ge_3, 4),
        )

