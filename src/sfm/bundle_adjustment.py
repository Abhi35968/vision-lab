"""
Bundle Adjustment Module
Jointly refines camera poses and 3D point positions via Levenberg-Marquardt / TRF with sparse Jacobian.
Enforces gauge freedom constraints (anchored Camera 0) and seed baseline scale invariance.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.sfm.camera_registration import CameraPose
from src.sfm.feature_tracks import FeatureTrack, FeatureTrackSystem


@dataclass
class BundleAdjustmentResult:
    """Telemetry, error percentiles, scale constraints, and convergence metrics from nonlinear bundle adjustment."""
    # Reprojection RMSE (Root Mean Square of 2D Euclidean errors)
    initial_rmse_px: float
    final_rmse_px: float
    error_reduction_percent: float

    # Mean Euclidean Reprojection Error (Average of 2D pixel distances)
    initial_mean_px: float
    final_mean_px: float
    mean_reduction_percent: float

    # Error percentiles (evaluated on exact same observation set)
    initial_median_px: float
    final_median_px: float
    initial_p90_px: float
    final_p90_px: float
    initial_max_px: float
    final_max_px: float

    # Observation and geometry counts
    num_cameras: int
    num_points: int
    num_observations_before: int
    num_observations_after: int

    # Gauge & Scale constraint verification
    anchor_camera_id: int
    scale_camera_id: int
    seed_baseline_before: float
    seed_baseline_after: float
    seed_baseline_preserved: bool
    scale_constraint_enforced: bool

    # Solver diagnostics
    num_iterations: int
    optimization_status: int
    termination_message: str
    cost_initial: float
    cost_final: float
    cost_reduction: float
    optimization_time_ms: float
    is_success: bool
    rmse_vs_mean_explanation: str

    def to_dict(self) -> dict:
        return {
            "reprojection_rmse_px": {
                "initial": round(self.initial_rmse_px, 3),
                "final": round(self.final_rmse_px, 3),
                "reduction_percent": round(self.error_reduction_percent, 2),
            },
            "reprojection_mean_px": {
                "initial": round(self.initial_mean_px, 3),
                "final": round(self.final_mean_px, 3),
                "reduction_percent": round(self.mean_reduction_percent, 2),
            },
            "reprojection_percentiles_px": {
                "initial_median": round(self.initial_median_px, 3),
                "final_median": round(self.final_median_px, 3),
                "initial_p90": round(self.initial_p90_px, 3),
                "final_p90": round(self.final_p90_px, 3),
                "initial_max": round(self.initial_max_px, 3),
                "final_max": round(self.final_max_px, 3),
            },
            "observations": {
                "num_cameras": self.num_cameras,
                "num_points": self.num_points,
                "num_observations_before": self.num_observations_before,
                "num_observations_after": self.num_observations_after,
            },
            "gauge_and_scale_constraints": {
                "anchor_camera_id": self.anchor_camera_id,
                "scale_camera_id": self.scale_camera_id,
                "seed_baseline_before": round(self.seed_baseline_before, 6),
                "seed_baseline_after": round(self.seed_baseline_after, 6),
                "seed_baseline_difference": round(abs(self.seed_baseline_after - self.seed_baseline_before), 8),
                "seed_baseline_preserved": self.seed_baseline_preserved,
                "scale_constraint_enforced": self.scale_constraint_enforced,
                "unit": "arbitrary_units",
            },
            "solver_diagnostics": {
                "num_iterations": self.num_iterations,
                "optimization_status": self.optimization_status,
                "termination_message": self.termination_message,
                "cost_initial": round(self.cost_initial, 2),
                "cost_final": round(self.cost_final, 2),
                "cost_reduction": round(self.cost_reduction, 2),
                "optimization_time_ms": round(self.optimization_time_ms, 2),
                "is_success": self.is_success,
            },
            "mathematical_note": self.rmse_vs_mean_explanation,
        }


class BundleAdjuster:
    """
    Executes nonlinear least-squares optimization to minimize total reprojection error.
    
    Mathematical Formulation:
    --------------------------
        min_{r_i, t_i, X_j}  Σ_{(i, j) ∈ O} ρ( || x_ij - π(K, R(r_i), t_i, X_j) ||² )
    
    where:
        x_ij: observed 2D subpixel coordinate in camera i
        X_j:  3D coordinate in world frame
        r_i:  3D Rodrigues rotation vector of camera i (R = exp([r_i]x))
        t_i:  3D translation vector of camera i
        K:    Camera intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        ρ:    Huber robust loss function (dampens outlier matches)
    
    Gauge Freedom & Scale Invariance Constraints:
    ----------------------------------------------
    Monocular SfM has 7 degrees of gauge freedom: 3 translation, 3 rotation, 1 global scale.
    Without constraints, the optimization matrix is rank-deficient and scale can drift freely.
    
    1. Origin & Rotation Gauge (6 DoF):
       Camera anchor_id is strictly anchored at the world origin:
           R_anchor = I,  t_anchor = [0, 0, 0]^T
       Its 6 parameters are omitted from optimization vector θ, locking world position and orientation.
       
    2. Global Scale Gauge (1 DoF):
       The baseline distance between the seed camera pair (anchor_id <-> scale_camera_id) is
       constrained to its initial length d_0 = ||t_scale0||.
       Camera scale_camera_id translation is parameterized by its 2D direction on the sphere of radius d_0:
           t_scale(α_1, α_2) = d_0 * (v / ||v||)
       where v = t_hat_0 + α_1 u_1 + α_2 u_2, with u_1, u_2 ⊥ t_hat_0.
       This preserves the initial baseline length identically to machine precision at every iteration,
       completely eliminating global scale drift while permitting valid rotation and directional refinement.
    """

    def __init__(
        self,
        loss: str = "huber",
        f_scale: float = 1.0,
        max_nfev: int = 100,
        ftol: float = 1e-4,
        xtol: float = 1e-4,
    ):
        self.loss = loss
        self.f_scale = f_scale
        self.max_nfev = max_nfev
        self.ftol = ftol
        self.xtol = xtol

    @staticmethod
    def _construct_orthogonal_basis(hat_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs two orthonormal vectors u1, u2 perpendicular to unit vector hat_t."""
        ref_axis = np.array([1.0, 0.0, 0.0]) if abs(hat_t[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        u1 = ref_axis - np.dot(ref_axis, hat_t) * hat_t
        u1_norm = np.linalg.norm(u1)
        u1 = u1 / (u1_norm if u1_norm > 1e-9 else 1.0)
        u2 = np.cross(hat_t, u1)
        u2_norm = np.linalg.norm(u2)
        u2 = u2 / (u2_norm if u2_norm > 1e-9 else 1.0)
        return u1, u2

    def optimize(
        self,
        registered_cameras: Dict[int, CameraPose],
        track_system: FeatureTrackSystem,
        camera_matrix_k: np.ndarray,
        fix_camera_id: Optional[int] = None,
        scale_camera_id: Optional[int] = None,
    ) -> BundleAdjustmentResult:
        """
        Runs sparse bundle adjustment with exact gauge and seed baseline scale constraints.
        """
        start_time = time.time()

        # 1. Gather registered cameras and triangulated tracks
        sorted_cam_ids = sorted([cid for cid, cam in registered_cameras.items() if cam.is_registered])
        if len(sorted_cam_ids) < 2:
            return BundleAdjustmentResult(
                initial_rmse_px=0.0, final_rmse_px=0.0, error_reduction_percent=0.0,
                initial_mean_px=0.0, final_mean_px=0.0, mean_reduction_percent=0.0,
                initial_median_px=0.0, final_median_px=0.0,
                initial_p90_px=0.0, final_p90_px=0.0,
                initial_max_px=0.0, final_max_px=0.0,
                num_cameras=len(sorted_cam_ids), num_points=0,
                num_observations_before=0, num_observations_after=0,
                anchor_camera_id=-1, scale_camera_id=-1,
                seed_baseline_before=0.0, seed_baseline_after=0.0,
                seed_baseline_preserved=False, scale_constraint_enforced=False,
                num_iterations=0, optimization_status=-1,
                termination_message="At least 2 registered cameras required for bundle adjustment.",
                cost_initial=0.0, cost_final=0.0, cost_reduction=0.0,
                optimization_time_ms=0.0, is_success=False,
                rmse_vs_mean_explanation="",
            )

        # Anchor camera (rigid gauge: fixed at origin)
        anchor_id = fix_camera_id if fix_camera_id is not None and fix_camera_id in sorted_cam_ids else sorted_cam_ids[0]
        
        # Scale camera (scale gauge: baseline magnitude fixed)
        if scale_camera_id is not None and scale_camera_id in sorted_cam_ids and scale_camera_id != anchor_id:
            scale_id = scale_camera_id
        else:
            candidates = [c for c in sorted_cam_ids if c != anchor_id]
            scale_id = candidates[0]

        valid_tracks = [t for t in track_system.tracks.values() if t.is_triangulated and t.point_3d is not None]
        if not valid_tracks:
            return BundleAdjustmentResult(
                initial_rmse_px=0.0, final_rmse_px=0.0, error_reduction_percent=0.0,
                initial_mean_px=0.0, final_mean_px=0.0, mean_reduction_percent=0.0,
                initial_median_px=0.0, final_median_px=0.0,
                initial_p90_px=0.0, final_p90_px=0.0,
                initial_max_px=0.0, final_max_px=0.0,
                num_cameras=len(sorted_cam_ids), num_points=0,
                num_observations_before=0, num_observations_after=0,
                anchor_camera_id=anchor_id, scale_camera_id=scale_id,
                seed_baseline_before=0.0, seed_baseline_after=0.0,
                seed_baseline_preserved=False, scale_constraint_enforced=False,
                num_iterations=0, optimization_status=-1,
                termination_message="No triangulated 3D points available for optimization.",
                cost_initial=0.0, cost_final=0.0, cost_reduction=0.0,
                optimization_time_ms=0.0, is_success=False,
                rmse_vs_mean_explanation="",
            )

        # Map camera IDs and track IDs to compact indices
        cam_id_to_idx = {cid: idx for idx, cid in enumerate(sorted_cam_ids)}
        track_id_to_idx = {t.track_id: idx for idx, t in enumerate(valid_tracks)}

        # Build observations list: (cam_idx, point_idx, obs_u, obs_v)
        observations = []
        for p_idx, track in enumerate(valid_tracks):
            # Verify finite 3D coordinates
            if not np.all(np.isfinite(track.point_3d)):
                continue
            for cid, obs in track.observations.items():
                if cid in cam_id_to_idx:
                    c_idx = cam_id_to_idx[cid]
                    u, v = obs.coords
                    if np.isfinite(u) and np.isfinite(v):
                        observations.append((c_idx, p_idx, float(u), float(v)))

        num_obs = len(observations)
        if num_obs < 10:
            return BundleAdjustmentResult(
                initial_rmse_px=0.0, final_rmse_px=0.0, error_reduction_percent=0.0,
                initial_mean_px=0.0, final_mean_px=0.0, mean_reduction_percent=0.0,
                initial_median_px=0.0, final_median_px=0.0,
                initial_p90_px=0.0, final_p90_px=0.0,
                initial_max_px=0.0, final_max_px=0.0,
                num_cameras=len(sorted_cam_ids), num_points=len(valid_tracks),
                num_observations_before=num_obs, num_observations_after=num_obs,
                anchor_camera_id=anchor_id, scale_camera_id=scale_id,
                seed_baseline_before=0.0, seed_baseline_after=0.0,
                seed_baseline_preserved=False, scale_constraint_enforced=False,
                num_iterations=0, optimization_status=-1,
                termination_message="Insufficient valid observations for bundle adjustment (< 10).",
                cost_initial=0.0, cost_final=0.0, cost_reduction=0.0,
                optimization_time_ms=0.0, is_success=False,
                rmse_vs_mean_explanation="",
            )

        obs_array = np.array(observations, dtype=np.float64)
        obs_cam_indices = obs_array[:, 0].astype(int)
        obs_pt_indices = obs_array[:, 1].astype(int)
        obs_pixels = obs_array[:, 2:4]

        anchor_idx = cam_id_to_idx[anchor_id]
        scale_idx = cam_id_to_idx[scale_id]

        # 2. Setup Baseline Scale Constraint
        scale_cam = registered_cameras[scale_id]
        t_scale0 = scale_cam.tvec.ravel().copy()
        d0 = float(np.linalg.norm(t_scale0))
        if d0 < 1e-7:
            d0 = 1.0
        hat_t0 = t_scale0 / d0
        u1, u2 = self._construct_orthogonal_basis(hat_t0)

        # 3. Setup Camera Parameter Layout:
        # - Anchor Camera (anchor_idx): 0 parameters (fixed gauge)
        # - Scale Camera (scale_idx): 5 parameters (3 rot rvec, 2 trans dir alpha)
        # - Other Cameras (c): 6 parameters (3 rot rvec, 3 trans tvec)
        cam_param_slices: Dict[int, Tuple[int, int]] = {}
        curr_offset = 0

        for c in range(len(sorted_cam_ids)):
            if c == anchor_idx:
                continue
            elif c == scale_idx:
                cam_param_slices[c] = (curr_offset, 5)
                curr_offset += 5
            else:
                cam_param_slices[c] = (curr_offset, 6)
                curr_offset += 6

        num_cam_params = curr_offset
        num_points = len(valid_tracks)
        point_param_offset = num_cam_params
        num_total_params = num_cam_params + num_points * 3

        # Pack Initial Parameter Vector x0
        x0 = np.zeros(num_total_params, dtype=np.float64)

        for c, (start, count) in cam_param_slices.items():
            cid = sorted_cam_ids[c]
            cam = registered_cameras[cid]
            if c == scale_idx:
                x0[start : start + 3] = cam.rvec.ravel()
                x0[start + 3 : start + 5] = [0.0, 0.0]  # alpha = (0, 0)
            else:
                x0[start : start + 3] = cam.rvec.ravel()
                x0[start + 3 : start + 6] = cam.tvec.ravel()

        for j, track in enumerate(valid_tracks):
            p_start = point_param_offset + j * 3
            x0[p_start : p_start + 3] = track.point_3d

        # 4. Construct Sparse Jacobian Sparsity Pattern
        A_sparsity = lil_matrix((2 * num_obs, num_total_params), dtype=int)
        for k in range(num_obs):
            c = obs_cam_indices[k]
            p = obs_pt_indices[k]
            row_u = 2 * k
            row_v = 2 * k + 1

            if c in cam_param_slices:
                start, count = cam_param_slices[c]
                A_sparsity[row_u, start : start + count] = 1
                A_sparsity[row_v, start : start + count] = 1

            p_start = point_param_offset + p * 3
            A_sparsity[row_u, p_start : p_start + 3] = 1
            A_sparsity[row_v, p_start : p_start + 3] = 1

        # Anchor camera fixed poses
        anchor_cam = registered_cameras[anchor_id]
        anchor_rvec = anchor_cam.rvec.ravel().copy()
        anchor_tvec = anchor_cam.tvec.ravel().copy()

        fx, fy = camera_matrix_k[0, 0], camera_matrix_k[1, 1]
        cx, cy = camera_matrix_k[0, 2], camera_matrix_k[1, 2]

        def unpack_cameras(params: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
            """Unpacks camera rotation and translation vectors respecting scale constraints."""
            cams_r = {}
            cams_t = {}
            cams_r[anchor_idx] = anchor_rvec
            cams_t[anchor_idx] = anchor_tvec

            for c, (start, count) in cam_param_slices.items():
                if c == scale_idx:
                    cams_r[c] = params[start : start + 3]
                    alpha = params[start + 3 : start + 5]
                    v = hat_t0 + alpha[0] * u1 + alpha[1] * u2
                    v_norm = np.linalg.norm(v)
                    # Enforce exact baseline magnitude d0
                    cams_t[c] = d0 * (v / (v_norm if v_norm > 1e-9 else 1.0))
                else:
                    cams_r[c] = params[start : start + 3]
                    cams_t[c] = params[start + 3 : start + 6]

            return cams_r, cams_t

        def residual_fun(params: np.ndarray) -> np.ndarray:
            cams_r, cams_t = unpack_cameras(params)
            pts_3d = params[point_param_offset:].reshape((num_points, 3))
            residuals = np.empty(2 * num_obs, dtype=np.float64)

            for c in range(len(sorted_cam_ids)):
                mask = (obs_cam_indices == c)
                if not np.any(mask):
                    continue
                obs_k_indices = np.where(mask)[0]
                pt_idxs = obs_pt_indices[mask]
                sub_pts = pts_3d[pt_idxs]

                rvec_c = cams_r[c]
                tvec_c = cams_t[c]

                R_c, _ = cv2.Rodrigues(rvec_c)
                cam_coords = (R_c @ sub_pts.T).T + tvec_c.ravel()
                z = cam_coords[:, 2]
                z_safe = np.where(z > 1e-4, z, 1e-4)

                u_est = fx * (cam_coords[:, 0] / z_safe) + cx
                v_est = fy * (cam_coords[:, 1] / z_safe) + cy

                # Penalize non-finite projections and points behind camera
                invalid = (z <= 1e-4) | (~np.isfinite(u_est)) | (~np.isfinite(v_est))
                u_est[invalid] = 1e4
                v_est[invalid] = 1e4

                residuals[2 * obs_k_indices]     = u_est - obs_pixels[mask, 0]
                residuals[2 * obs_k_indices + 1] = v_est - obs_pixels[mask, 1]

            return residuals

        def compute_euclidean_errors(residuals_2m: np.ndarray) -> np.ndarray:
            """Computes per-observation 2D Euclidean pixel error e_k = sqrt(du^2 + dv^2)."""
            du = residuals_2m[0::2]
            dv = residuals_2m[1::2]
            return np.sqrt(du ** 2 + dv ** 2)

        # 5. Evaluate Initial Reprojection Metrics (Before Optimization)
        init_res = residual_fun(x0)
        init_e = compute_euclidean_errors(init_res)
        init_rmse = float(np.sqrt(np.mean(init_e ** 2)))
        init_mean = float(np.mean(init_e))
        init_median = float(np.median(init_e))
        init_p90 = float(np.percentile(init_e, 90))
        init_max = float(np.max(init_e))
        cost_initial = float(0.5 * np.sum(init_res ** 2))

        # 6. Execute Nonlinear Optimization (TRF with Huber loss and sparse Jacobian)
        opt_res = least_squares(
            fun=residual_fun,
            x0=x0,
            jac_sparsity=A_sparsity,
            method="trf",
            loss=self.loss,
            f_scale=self.f_scale,
            max_nfev=self.max_nfev,
            ftol=self.ftol,
            xtol=self.xtol,
            verbose=0,
        )

        # 7. Evaluate Final Reprojection Metrics (After Optimization) on exact same observations
        final_res = opt_res.fun
        final_e = compute_euclidean_errors(final_res)
        final_rmse = float(np.sqrt(np.mean(final_e ** 2)))
        final_mean = float(np.mean(final_e))
        final_median = float(np.median(final_e))
        final_p90 = float(np.percentile(final_e, 90))
        final_max = float(np.max(final_e))
        cost_final = float(0.5 * np.sum(final_res ** 2))

        rmse_reduction = float(((init_rmse - final_rmse) / max(1e-6, init_rmse)) * 100)
        mean_reduction = float(((init_mean - final_mean) / max(1e-6, init_mean)) * 100)
        cost_reduction = cost_initial - cost_final
        elapsed_ms = (time.time() - start_time) * 1000.0

        # 8. Unpack Optimized Variables and Update State
        opt_params = opt_res.x
        cams_r_opt, cams_t_opt = unpack_cameras(opt_params)

        for c, cid in enumerate(sorted_cam_ids):
            if c == anchor_idx:
                continue
            rvec_c = cams_r_opt[c].reshape(3, 1)
            tvec_c = cams_t_opt[c].reshape(3, 1)
            R_c, _ = cv2.Rodrigues(rvec_c)
            C_c = (-R_c.T @ tvec_c).ravel()

            cam = registered_cameras[cid]
            cam.rvec = rvec_c
            cam.tvec = tvec_c
            cam.rotation_matrix = R_c
            cam.translation_vector = tvec_c
            cam.camera_center = C_c

        # Unpack 3D points
        opt_pts = opt_params[point_param_offset:].reshape((num_points, 3))
        for j, track in enumerate(valid_tracks):
            track.point_3d = opt_pts[j]

        # 9. Verify Seed Baseline Scale Invariance
        t_scale_final = registered_cameras[scale_id].tvec.ravel()
        d_final = float(np.linalg.norm(t_scale_final))
        baseline_preserved = bool(abs(d_final - d0) < 1e-6)

        explanation = (
            "RMSE (Root Mean Square Error) weights large pixel deviations quadratically, making it sensitive to outliers. "
            "Mean Euclidean Error computes the arithmetic average distance in pixels (always <= RMSE by Jensen's inequality). "
            "A Huber robust loss was used during optimization to dampen outlier influence."
        )

        return BundleAdjustmentResult(
            initial_rmse_px=init_rmse,
            final_rmse_px=final_rmse,
            error_reduction_percent=rmse_reduction,
            initial_mean_px=init_mean,
            final_mean_px=final_mean,
            mean_reduction_percent=mean_reduction,
            initial_median_px=init_median,
            final_median_px=final_median,
            initial_p90_px=init_p90,
            final_p90_px=final_p90,
            initial_max_px=init_max,
            final_max_px=final_max,
            num_cameras=len(sorted_cam_ids),
            num_points=num_points,
            num_observations_before=num_obs,
            num_observations_after=num_obs,
            anchor_camera_id=anchor_id,
            scale_camera_id=scale_id,
            seed_baseline_before=d0,
            seed_baseline_after=d_final,
            seed_baseline_preserved=baseline_preserved,
            scale_constraint_enforced=True,
            num_iterations=int(opt_res.nfev),
            optimization_status=int(opt_res.status),
            termination_message=str(opt_res.message),
            cost_initial=cost_initial,
            cost_final=cost_final,
            cost_reduction=cost_reduction,
            optimization_time_ms=elapsed_ms,
            is_success=bool(opt_res.success),
            rmse_vs_mean_explanation=explanation,
        )
