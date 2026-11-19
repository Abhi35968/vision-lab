"""
Bottle Fitter Module (Phase 5.5).
Optimizes parametric bottle geometry (height, radius, axis orientation, cap position)
against multi-view 2D segmentation masks with a strict train/validation split.
"""

from dataclasses import dataclass, asdict
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.optimize import minimize

from src.object_reconstruction.bottle_models import (
    BottleParameters,
    Mesh3D,
    SilhouetteViewMetric,
)
from src.object_reconstruction.parametric_bottle import ParametricBottle
from src.object_reconstruction.silhouette_projector import SilhouetteProjector


@dataclass
class FittingSummary:
    """Summary of the silhouette-based parametric optimization."""
    total_valid_views: int
    train_views_count: int
    val_views_count: int
    optimization_time_sec: float
    train_mean_iou: float
    train_median_iou: float
    train_max_iou: float
    train_min_iou: float
    val_mean_iou: float
    val_median_iou: float
    val_max_iou: float
    val_min_iou: float
    overall_mean_iou: float
    overall_median_iou: float
    overall_mean_precision: float
    overall_mean_recall: float
    estimated_height: float
    estimated_radius: float
    estimated_tilt_x_deg: float
    estimated_tilt_z_deg: float
    estimated_cap_center: List[float]

    def to_dict(self) -> dict:
        return asdict(self)


class BottleFitter:
    """
    Fits parametric bottle geometry to multi-view 2D binary segmentation masks.
    Uses bounded numerical optimization on a training split, validated on a held-out split.
    """

    def __init__(
        self,
        projector: SilhouetteProjector,
        downsample_factor: int = 4,
    ):
        self.projector = projector
        self.downsample_factor = downsample_factor

    @staticmethod
    def partition_train_val(
        views: List[dict],
    ) -> Tuple[List[dict], List[dict]]:
        """
        Splits views alternately by camera viewpoint into train and validation subsets.
        Ensures both sets cover the entire available viewing arc.
        """
        train_views = views[::2]
        val_views = views[1::2]
        return train_views, val_views

    def fit(
        self,
        cap_centroid: np.ndarray,
        views: List[dict],
        initial_height: float = 7.0,
        initial_radius: float = 1.15,
    ) -> Tuple[BottleParameters, List[SilhouetteViewMetric], FittingSummary]:
        """
        Performs bounded optimization of bottle parameters on training views,
        evaluates generalization on validation views, and computes full-res metrics.
        """
        t0 = time.perf_counter()
        train_views, val_views = self.partition_train_val(views)

        # Prepare downsampled observed masks for fast training evaluation
        H_orig, W_orig = self.projector.image_shape
        H_ds = H_orig // self.downsample_factor
        W_ds = W_orig // self.downsample_factor
        train_masks_ds = [
            cv2.resize(v["mask"], (W_ds, H_ds), interpolation=cv2.INTER_NEAREST)
            for v in train_views
        ]

        base_cap = np.asarray(cap_centroid, dtype=np.float64)

        # Cost function: Negative mean IoU on training set
        def cost_fn(x: np.ndarray) -> float:
            H, R, tx, tz, dx, dy = x
            c_cap = [base_cap[0] + dx, base_cap[1] + dy, base_cap[2]]
            params = BottleParameters(
                cap_center_world=c_cap,
                total_height=float(H),
                body_radius=float(R),
                axis_tilt_x_deg=float(tx),
                axis_tilt_z_deg=float(tz),
                axial_resolution=24,
                angular_resolution=18,
            )
            bottle = ParametricBottle(params)
            mesh = bottle.build_mesh()
            ious = []
            for idx, v in enumerate(train_views):
                rend, is_ok = self.projector.project_and_rasterize(
                    mesh=mesh,
                    R=v["R"],
                    t=v["t"],
                    downsample_factor=self.downsample_factor,
                )
                if not is_ok:
                    return 1.0  # Heavy penalty for behind-camera
                m_obs = train_masks_ds[idx]
                inter = np.count_nonzero((rend > 0) & (m_obs > 0))
                union = np.count_nonzero((rend > 0) | (m_obs > 0))
                ious.append(inter / union if union > 0 else 0.0)
            return -float(np.mean(ious))

        # Initial parameter vector: [H, R, tilt_x, tilt_z, dx, dy]
        x0 = [initial_height, initial_radius, 0.0, 0.0, 0.0, 0.0]
        bounds = [
            (5.0, 8.5),     # Height in SfM units
            (0.85, 1.45),   # Body radius
            (-8.0, 8.0),    # Tilt X deg
            (-8.0, 8.0),    # Tilt Z deg
            (-0.20, 0.20),  # Delta X cap position
            (-0.20, 0.20),  # Delta Y cap position
        ]

        # Bounded Nelder-Mead optimization
        res = minimize(
            cost_fn,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 80, "disp": False},
        )
        opt_time = time.perf_counter() - t0

        # Construct final high-resolution optimal parameter set
        opt_H, opt_R, opt_tx, opt_tz, opt_dx, opt_dy = res.x
        final_cap = [
            round(float(base_cap[0] + opt_dx), 5),
            round(float(base_cap[1] + opt_dy), 5),
            round(float(base_cap[2]), 5),
        ]

        optimal_params = BottleParameters(
            cap_center_world=final_cap,
            total_height=round(float(opt_H), 4),
            body_radius=round(float(opt_R), 4),
            axis_tilt_x_deg=round(float(opt_tx), 3),
            axis_tilt_z_deg=round(float(opt_tz), 3),
            axial_resolution=60,
            angular_resolution=36,
        )

        # Full-resolution evaluation across all views (train + val)
        opt_bottle = ParametricBottle(optimal_params)
        opt_mesh = opt_bottle.build_mesh()

        all_metrics: List[SilhouetteViewMetric] = []
        train_ious: List[float] = []
        val_ious: List[float] = []

        for v in views:
            is_train = v in train_views
            metric = self.projector.evaluate_view(
                camera_id=v["cid"],
                image_name=v["name"],
                is_train_split=is_train,
                mesh=opt_mesh,
                R=v["R"],
                t=v["t"],
                observed_mask=v["mask"],
            )
            if metric is not None:
                all_metrics.append(metric)
                if is_train:
                    train_ious.append(metric.iou)
                else:
                    val_ious.append(metric.iou)

        overall_ious = [m.iou for m in all_metrics]
        overall_prec = [m.precision for m in all_metrics]
        overall_rec = [m.recall for m in all_metrics]

        summary = FittingSummary(
            total_valid_views=len(all_metrics),
            train_views_count=len(train_ious),
            val_views_count=len(val_ious),
            optimization_time_sec=round(opt_time, 2),
            train_mean_iou=round(float(np.mean(train_ious)), 4),
            train_median_iou=round(float(np.median(train_ious)), 4),
            train_max_iou=round(float(np.max(train_ious)), 4),
            train_min_iou=round(float(np.min(train_ious)), 4),
            val_mean_iou=round(float(np.mean(val_ious)), 4),
            val_median_iou=round(float(np.median(val_ious)), 4),
            val_max_iou=round(float(np.max(val_ious)), 4),
            val_min_iou=round(float(np.min(val_ious)), 4),
            overall_mean_iou=round(float(np.mean(overall_ious)), 4),
            overall_median_iou=round(float(np.median(overall_ious)), 4),
            overall_mean_precision=round(float(np.mean(overall_prec)), 4),
            overall_mean_recall=round(float(np.mean(overall_rec)), 4),
            estimated_height=optimal_params.total_height,
            estimated_radius=optimal_params.body_radius,
            estimated_tilt_x_deg=optimal_params.axis_tilt_x_deg,
            estimated_tilt_z_deg=optimal_params.axis_tilt_z_deg,
            estimated_cap_center=final_cap,
        )

        return optimal_params, all_metrics, summary

