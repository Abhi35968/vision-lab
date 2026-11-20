"""
Depth Anything V2 Small Experimental Integration Module (Phase 6)
Investigates CPU-based monocular depth estimation and geometric alignment with SfM:
1. Uses Depth Anything V2 Small (or lightweight fallback) strictly on CPU.
2. Processes selected keyframes only (never full video frames) to conserve memory.
3. Caches dense depth arrays (.npy) and normalized colorized visual maps (.png).
4. Solves affine scale and shift (s * d_rel + b) using sparse SfM point correspondences.
5. Unprojects masked object pixels into an experimental dense 3D point cloud.
6. Isolates experimental depth outputs from core SfM pipeline.
"""

from dataclasses import dataclass, asdict, field
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("DepthAnythingEstimator")


@dataclass
class DepthEstimationResult:
    """Telemetry and aligned point cloud from monocular depth estimation."""
    frame_name: str
    depth_array_path: str               # Path to cached raw float32 .npy
    depth_vis_path: str                 # Path to normalized colormapped .png
    dense_ply_path: Optional[str]       # Path to unprojected dense point cloud .ply
    num_dense_points: int
    alignment_scale: float              # Scale factor s in s * d + b
    alignment_shift: float              # Shift factor b in s * d + b
    alignment_rmse_units: float         # Residual root-mean-square error against SfM points
    inference_time_ms: float
    model_name: str
    is_aligned_with_sfm: bool
    notes: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


class DepthAnythingEstimator:
    """
    Experimental monocular depth estimation and SfM geometric alignment engine.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        cache_dir: Union[str, Path] = "outputs/dense_depth",
        target_resolution: int = 518,    # Standard input size for Depth Anything V2
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.target_resolution = target_resolution
        self._pipe = None
        self._initialized = False

    def _init_pipeline(self) -> bool:
        """Lazily initializes the depth estimation pipeline on CPU."""
        if self._initialized:
            return self._pipe is not None

        try:
            import torch
            from transformers import pipeline
            torch.set_num_threads(4)
            logger.info(f"Loading {self.model_name} on CPU...")
            self._pipe = pipeline(
                task="depth-estimation",
                model=self.model_name,
                device="cpu",
            )
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"Could not load HuggingFace {self.model_name}: {e}. Enabling algorithmic fallback.")
            self._pipe = None
            self._initialized = True
            return False

    def estimate_relative_depth(
        self,
        image_bgr: np.ndarray,
        cache_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Computes normalized relative monocular depth map in [0.0, 1.0].
        
        Returns:
            depth_map: (H, W) float32 array where higher values = further depth
            inference_ms: Runtime in milliseconds
        """
        t0 = time.time()
        h, w = image_bgr.shape[:2]

        # Check cache if key provided
        if cache_key:
            cached_npy = self.cache_dir / f"{cache_key}_depth.npy"
            if cached_npy.exists():
                try:
                    loaded = np.load(str(cached_npy))
                    if loaded.shape == (h, w):
                        return loaded, 0.0
                except Exception:
                    pass

        has_model = self._init_pipeline()
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if has_model and self._pipe is not None:
            try:
                pil_img = Image.fromarray(img_rgb)
                # Resize to target resolution for fast CPU inference
                scale = float(self.target_resolution) / max(h, w)
                eval_w, eval_h = int(w * scale), int(h * scale)
                small_pil = pil_img.resize((eval_w, eval_h), Image.Resampling.BILINEAR)

                out = self._pipe(small_pil)
                pred_depth = np.array(out["predicted_depth"]).astype(np.float32)

                # Resize back to original image dimensions
                depth_map = cv2.resize(pred_depth, (w, h), interpolation=cv2.INTER_CUBIC)

                # Normalize to relative range [0, 1]
                d_min, d_max = np.min(depth_map), np.max(depth_map)
                if d_max > d_min:
                    depth_map = (depth_map - d_min) / (d_max - d_min)
            except Exception as e:
                logger.warning(f"Depth pipeline inference error: {e}. Falling back to structural depth estimation.")
                depth_map = self._structural_depth_fallback(image_bgr)
        else:
            depth_map = self._structural_depth_fallback(image_bgr)

        t_ms = (time.time() - t0) * 1000.0

        if cache_key:
            np.save(str(self.cache_dir / f"{cache_key}_depth.npy"), depth_map)
            vis_img = (depth_map * 255.0).astype(np.uint8)
            vis_color = cv2.applyColorMap(vis_img, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(self.cache_dir / f"{cache_key}_depth_vis.png"), vis_color)

        return depth_map, round(t_ms, 2)

    @staticmethod
    def _structural_depth_fallback(img_bgr: np.ndarray) -> np.ndarray:
        """
        Lightweight CPU structural depth gradient prior based on vertical perspective and blur.
        Guarantees functionality even in offline or memory-restricted environments.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Perspective vertical gradient: points lower in frame tend to be closer (tabletop prior)
        y_grad = np.linspace(0.2, 0.9, h)[:, np.newaxis]
        y_plane = np.tile(y_grad, (1, w)).astype(np.float32)
        # Add high-frequency texture gradient
        blur = cv2.GaussianBlur(gray, (15, 15), 0).astype(np.float32) / 255.0
        depth = 0.7 * y_plane + 0.3 * blur
        return np.clip(depth, 0.0, 1.0)

    def align_depth_with_sfm(
        self,
        depth_map: np.ndarray,
        sparse_points_3d: np.ndarray,      # (N, 3) SfM 3D points
        camera_R: np.ndarray,              # (3, 3)
        camera_t: np.ndarray,              # (3, 1)
        camera_K: np.ndarray,              # (3, 3)
        mode: str = "inverse",              # "inverse" (for Depth Anything disparity) or "linear"
        return_metadata: bool = False,
    ) -> Union[Tuple[float, float, float, np.ndarray], Tuple[float, float, float, np.ndarray, dict]]:
        """
        Solves alignment between monocular depth and sparse SfM points.

        For Depth Anything V2 (relative disparity / inverse depth):
            1 / Z_sfm = s * d_relative + b
            Z_metric = 1.0 / max(s * depth_map + b, 1e-4)
        Uses robust Huber regression to reject sparse triangulation outliers,
        rejects non-positive depths (Z <= 0.05), and guards against degenerate scale (s <= 0).
        """
        h, w = depth_map.shape[:2]
        pts = np.asarray(sparse_points_3d, dtype=np.float64)
        t_vec = camera_t.reshape(3, 1)

        # 1. Project SfM points into camera view
        pts_cam = (camera_R @ pts.T + t_vec).T
        depths_true = pts_cam[:, 2]

        proj = (camera_K @ pts_cam.T).T
        valid = depths_true > 0.05
        u = np.zeros(len(pts))
        v = np.zeros(len(pts))
        u[valid] = proj[valid, 0] / depths_true[valid]
        v[valid] = proj[valid, 1] / depths_true[valid]

        inside = (
            valid
            & (u >= 0) & (u < w)
            & (v >= 0) & (v < h)
        )

        meta = {
            "num_candidates": int(len(pts)),
            "num_in_view": int(inside.sum()),
            "mode": mode,
            "is_degenerate": False,
            "status": "success",
        }

        if inside.sum() < 4:
            # Not enough points for regression; use robust median fallback
            mean_z = float(np.median(depths_true[valid])) if valid.sum() > 0 else 2.5
            mean_z = max(0.2, mean_z)
            meta["is_degenerate"] = True
            meta["status"] = "insufficient_points_fallback"
            if mode == "inverse":
                aligned_depth = np.full_like(depth_map, mean_z, dtype=np.float32)
                res = (1.0, round(float(1.0 / mean_z), 4), 0.50, aligned_depth)
            else:
                aligned_depth = np.full_like(depth_map, mean_z, dtype=np.float32)
                res = (1.0, round(mean_z, 4), 0.50, aligned_depth)
            self.last_alignment_metadata = meta
            return (*res, meta) if return_metadata else res

        u_in = np.clip(np.round(u[inside]).astype(int), 0, w - 1)
        v_in = np.clip(np.round(v[inside]).astype(int), 0, h - 1)
        d_rel = depth_map[v_in, u_in].astype(np.float64)
        z_sfm = depths_true[inside]

        if mode == "inverse":
            # Depth Anything disparity: larger values = closer (1/Z)
            inv_z = 1.0 / np.clip(z_sfm, 0.05, 50.0)
            try:
                from sklearn.linear_model import HuberRegressor
                huber = HuberRegressor(epsilon=1.35, max_iter=200)
                huber.fit(d_rel[:, np.newaxis], inv_z)
                s = float(huber.coef_[0])
                b = float(huber.intercept_)
            except Exception:
                A = np.column_stack([d_rel, np.ones_like(d_rel)])
                sol, _, _, _ = np.linalg.lstsq(A, inv_z, rcond=None)
                s, b = float(sol[0]), float(sol[1])

            # Degenerate alignment detection: disparity must correlate positively with 1/Z
            if s <= 1e-4:
                meta["is_degenerate"] = True
                meta["status"] = "degenerate_scale_fallback"
                s = 0.05
                b = float(np.median(inv_z))

            pred_inv_z = np.maximum(s * d_rel + b, 1e-4)
            pred_z = 1.0 / pred_inv_z
            rmse = float(np.sqrt(np.mean((pred_z - z_sfm) ** 2)))

            inv_aligned = np.maximum(s * depth_map.astype(np.float64) + b, 1e-4)
            aligned_depth = np.clip(1.0 / inv_aligned, 0.05, 50.0).astype(np.float32)
        else:
            # Linear depth fit: Z = s * d + b
            A = np.column_stack([d_rel, np.ones_like(d_rel)])
            try:
                sol, _, _, _ = np.linalg.lstsq(A, z_sfm, rcond=None)
                s, b = float(sol[0]), float(sol[1])
                if s <= 0.01:
                    s = 1.0
                    b = float(np.median(z_sfm))
                pred = s * d_rel + b
                rmse = float(np.sqrt(np.mean((pred - z_sfm) ** 2)))
            except Exception:
                s = 1.0
                b = float(np.median(z_sfm))
                rmse = 0.50
            aligned_depth = np.clip(s * depth_map + b, 0.1, 50.0).astype(np.float32)

        meta["scale_s"] = round(s, 6)
        meta["offset_b"] = round(b, 6)
        meta["rmse_px"] = round(rmse, 4)
        self.last_alignment_metadata = meta

        result = (round(s, 4), round(b, 4), round(rmse, 4), aligned_depth)
        return (*result, meta) if return_metadata else result

    def unproject_dense_cloud(
        self,
        aligned_depth: np.ndarray,
        camera_R: np.ndarray,
        camera_t: np.ndarray,
        camera_K: np.ndarray,
        object_mask: Optional[np.ndarray] = None,
        stride: int = 4,                   # Subsample pixels for CPU performance
        max_points: int = 15000,
        max_depth_gradient: Optional[float] = None,
    ) -> np.ndarray:
        """
        Unprojects aligned depth pixels into a dense 3D point cloud in world coordinates.
        Optional max_depth_gradient filters out boundary edge ramps (flying pixels).
        """
        h, w = aligned_depth.shape[:2]
        K_inv = np.linalg.inv(camera_K)
        R_inv = camera_R.T
        t_vec = camera_t.reshape(3, 1)

        # Meshgrid of pixel coordinates
        yy, xx = np.mgrid[0:h:stride, 0:w:stride]
        valid_mask = np.ones_like(xx, dtype=bool)

        if object_mask is not None:
            small_mask = object_mask[::stride, ::stride]
            valid_mask = valid_mask & (small_mask > 0)

        # Optional Depth Gradient Filtering: reject flying pixels and boundary edge ramps
        if max_depth_gradient is not None and max_depth_gradient > 0:
            z_smooth = cv2.GaussianBlur(aligned_depth.astype(np.float32), (3, 3), 0)
            gx = cv2.Sobel(z_smooth, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(z_smooth, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2)
            rel_grad = grad_mag / np.maximum(z_smooth, 1e-4)
            small_grad_mask = (rel_grad <= max_depth_gradient)[::stride, ::stride]
            valid_mask = valid_mask & small_grad_mask

        xx_v = xx[valid_mask]
        yy_v = yy[valid_mask]
        z_v = aligned_depth[0:h:stride, 0:w:stride][valid_mask]

        # Filter invalid depths
        good = z_v > 0.05
        xx_v = xx_v[good]
        yy_v = yy_v[good]
        z_v = z_v[good]

        # Cap points
        if len(z_v) > max_points:
            sub = np.random.choice(len(z_v), max_points, replace=False)
            xx_v = xx_v[sub]
            yy_v = yy_v[sub]
            z_v = z_v[sub]

        if len(z_v) == 0:
            return np.empty((0, 3), dtype=np.float64)

        # Pixel rays: [u, v, 1]^T
        pixels_homo = np.vstack([xx_v, yy_v, np.ones_like(xx_v)])
        rays_cam = K_inv @ pixels_homo
        # Scale rays by depth: X_c = z * ray
        pts_cam = rays_cam * z_v

        # Transform to world frame: X_world = R^T * (X_c - t)
        pts_world = (R_inv @ (pts_cam - t_vec)).T
        return pts_world.astype(np.float64)
