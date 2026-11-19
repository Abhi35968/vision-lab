"""
ArUco-Based Metric Scale Recovery Module
Estimates absolute metric scale for Structure from Motion reconstructions:
1. Detects standard ArUco planar fiducial markers across keyframes.
2. Estimates 6-DoF marker pose using camera intrinsics K and lens distortion coefficients.
3. Computes the metric scale factor lambda (metric_units / sfm_units).
4. Scales 3D point clouds, camera trajectories, and meshes consistently.
5. Calculates real-world object extents (Height, Width, Depth in cm) and volumetric bounds.
6. Assesses calibration limitations, reprojection quality, and measurement confidence.
"""

from dataclasses import dataclass, asdict, field
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger("ArucoScaleCalibrator")


@dataclass
class MarkerObservation:
    """Detection and pose record of an ArUco marker in a single keyframe."""
    frame_name: str
    marker_id: int
    corners_2d: np.ndarray              # (4, 2) pixel coordinates
    rvec: np.ndarray                    # (3, 1) Rodrigues rotation vector
    tvec: np.ndarray                    # (3, 1) translation vector in camera frame (cm)
    camera_distance_cm: float           # ||tvec||
    reprojection_error_px: float        # Reprojection residual of marker corners
    camera_center_sfm: Optional[np.ndarray] = None  # (3,) camera position in SfM units

    def to_dict(self) -> dict:
        return {
            "frame_name": self.frame_name,
            "marker_id": self.marker_id,
            "corners_2d": self.corners_2d.tolist(),
            "rvec": self.rvec.flatten().tolist(),
            "tvec": self.tvec.flatten().tolist(),
            "camera_distance_cm": round(float(self.camera_distance_cm), 2),
            "reprojection_error_px": round(float(self.reprojection_error_px), 3),
        }


@dataclass
class MetricDimensions:
    """Physical bounding dimensions of an object in metric centimetres."""
    width_x_cm: float
    height_y_cm: float
    depth_z_cm: float
    bounding_box_min_cm: List[float]
    bounding_box_max_cm: List[float]
    estimated_volume_cm3: float
    estimated_volume_litres: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetricScaleResult:
    """Encapsulates scale calibration outputs, scaled geometry, and limitations."""
    scale_factor: float                 # Multiplier: 1 SfM unit = scale_factor cm
    marker_size_cm: float               # Physical side length of square ArUco marker
    num_marker_observations: int
    marker_id_used: int
    marker_observations: List[MarkerObservation]
    object_dimensions: Optional[MetricDimensions]
    confidence_score: float             # In [0.0, 1.0]
    confidence_tier: str                # "HIGH", "MODERATE", "LOW", "UNVERIFIED"
    calibration_status: str             # "SUCCESS", "APPROXIMATED", "FAILED"
    limitations: List[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["marker_observations"] = [o.to_dict() for o in self.marker_observations]
        if self.object_dimensions:
            d["object_dimensions"] = self.object_dimensions.to_dict()
        return d


class ArucoScaleCalibrator:
    """
    Recovers metric scale and converts arbitrary SfM units into physical centimetres.
    """

    SUPPORTED_DICTIONARIES = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }

    def __init__(
        self,
        dictionary_name: str = "DICT_4X4_50",
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ):
        self.dict_name = dictionary_name
        dict_id = self.SUPPORTED_DICTIONARIES.get(dictionary_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

        self.K = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((5, 1), dtype=np.float64)

    def detect_markers(self, image_bgr: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Detects planar ArUco marker corners in a BGR image.
        Returns:
            corners: List of (1, 4, 2) arrays
            ids: List of integer marker IDs
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        id_list = [int(i) for i in np.ravel(ids)] if ids is not None and len(ids) > 0 else []
        return (list(corners) if corners else []), id_list

    def estimate_marker_pose(
        self,
        corners: np.ndarray,
        marker_size_cm: float,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> Tuple[bool, np.ndarray, np.ndarray, float]:
        """
        Estimates 6-DoF pose of a square planar marker via solvePnP.
        Marker origin (0,0,0) is placed at the center of the marker plane.
        """
        half_s = marker_size_cm / 2.0
        # 3D object coordinates in counter-clockwise order matching OpenCV corner convention
        obj_points = np.array([
            [-half_s,  half_s, 0.0],
            [ half_s,  half_s, 0.0],
            [ half_s, -half_s, 0.0],
            [-half_s, -half_s, 0.0],
        ], dtype=np.float64)

        img_pts = corners.reshape(-1, 2).astype(np.float64)
        dist = dist_coeffs if dist_coeffs is not None else np.zeros((5, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )

        if not success:
            return False, np.zeros((3, 1)), np.zeros((3, 1)), 999.0

        # Calculate corner reprojection error
        proj_pts, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist)
        proj_pts = proj_pts.reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(img_pts - proj_pts, axis=1)))

        return True, rvec, tvec, err

    def compute_metric_scale(
        self,
        keyframes: Dict[str, np.ndarray],          # {frame_name: bgr_image}
        camera_poses_sfm: Dict[str, dict],         # {frame_name: {"camera_center": [x,y,z], ...}}
        marker_size_cm: float = 5.0,
        target_marker_id: Optional[int] = None,
        assumed_focal_px: Optional[float] = None,
        object_points_sfm: Optional[np.ndarray] = None,
    ) -> MetricScaleResult:
        """
        Computes the global metric scale factor lambda relating SfM units to centimetres.
        """
        limitations: List[str] = []

        if marker_size_cm <= 0:
            raise ValueError(f"Marker size must be positive, got: {marker_size_cm} cm")

        # Resolve camera intrinsics K
        if self.K is not None:
            K = self.K
            calib_status = "SUCCESS"
        else:
            first_img = next(iter(keyframes.values())) if keyframes else None
            if first_img is not None:
                h, w = first_img.shape[:2]
                f = assumed_focal_px or (1.2 * max(h, w))
                K = np.array([
                    [f, 0.0, w / 2.0],
                    [0.0, f, h / 2.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
                calib_status = "APPROXIMATED"
                limitations.append("Camera intrinsics approximated based on image resolution; precise metric scale requires checkerboard calibration.")
            else:
                return MetricScaleResult(
                    scale_factor=1.0,
                    marker_size_cm=marker_size_cm,
                    num_marker_observations=0,
                    marker_id_used=-1,
                    marker_observations=[],
                    object_dimensions=None,
                    confidence_score=0.0,
                    confidence_tier="UNVERIFIED",
                    calibration_status="FAILED",
                    limitations=["No keyframe images provided for ArUco scale detection."],
                )

        observations: List[MarkerObservation] = []

        # Build pose lookup table
        pose_lookup: Dict[str, dict] = {}
        for k, c in camera_poses_sfm.items():
            if isinstance(c, dict):
                if "image_name" in c:
                    pose_lookup[c["image_name"]] = c
                pose_lookup[k] = c

        # 1. Detect marker across keyframes
        for fname, img in keyframes.items():
            corners_list, id_list = self.detect_markers(img)
            for corners, mid in zip(corners_list, id_list):
                if target_marker_id is not None and mid != target_marker_id:
                    continue
                ok, rvec, tvec, r_err = self.estimate_marker_pose(corners, marker_size_cm, K, self.dist_coeffs)
                if ok and r_err < 8.0:
                    dist_cm = float(np.linalg.norm(tvec))
                    c_sfm = None
                    if fname in pose_lookup:
                        c_sfm = np.array(pose_lookup[fname].get("camera_center", [0, 0, 0]))

                    observations.append(
                        MarkerObservation(
                            frame_name=fname,
                            marker_id=mid,
                            corners_2d=corners.reshape(-1, 2),
                            rvec=rvec,
                            tvec=tvec,
                            camera_distance_cm=dist_cm,
                            reprojection_error_px=r_err,
                            camera_center_sfm=c_sfm,
                        )
                    )

        if not observations:
            # Fallback: estimate reasonable default scale if no physical marker was visible in the scene
            limitations.append("No ArUco marker detected in video keyframes. Defaulting to nominal tabletop baseline scale (1.0).")
            dims = self.measure_object_dimensions(object_points_sfm, scale_factor=1.0) if object_points_sfm is not None else None
            return MetricScaleResult(
                scale_factor=1.0,
                marker_size_cm=marker_size_cm,
                num_marker_observations=0,
                marker_id_used=-1,
                marker_observations=[],
                object_dimensions=dims,
                confidence_score=0.10,
                confidence_tier="UNVERIFIED",
                calibration_status="APPROXIMATED",
                limitations=limitations,
            )

        # 2. Select primary marker ID (most frequently observed)
        marker_counts: Dict[int, int] = {}
        for o in observations:
            marker_counts[o.marker_id] = marker_counts.get(o.marker_id, 0) + 1
        best_mid = max(marker_counts, key=marker_counts.get)
        filtered_obs = [o for o in observations if o.marker_id == best_mid]

        # 3. Calculate Scale Factor from Multi-View Baselines
        scale_candidates: List[float] = []

        # Method A: Pairwise inter-camera baseline ratio (metric vs SfM)
        for i in range(len(filtered_obs)):
            for j in range(i + 1, len(filtered_obs)):
                o1 = filtered_obs[i]
                o2 = filtered_obs[j]
                if o1.camera_center_sfm is not None and o2.camera_center_sfm is not None:
                    # SfM baseline distance between camera centers
                    dist_sfm = float(np.linalg.norm(o1.camera_center_sfm - o2.camera_center_sfm))
                    # Metric camera centers in marker reference frame
                    R1, _ = cv2.Rodrigues(o1.rvec)
                    R2, _ = cv2.Rodrigues(o2.rvec)
                    c_marker1 = -R1.T @ o1.tvec
                    c_marker2 = -R2.T @ o2.tvec
                    dist_metric_cm = float(np.linalg.norm(c_marker1 - c_marker2))

                    if dist_sfm > 0.05 and dist_metric_cm > 0.5:
                        cand_scale = dist_metric_cm / dist_sfm
                        # Filter out wild outliers
                        if 0.1 <= cand_scale <= 200.0:
                            scale_candidates.append(cand_scale)

        # Method B: Marker distance ratio fallback if camera baseline is small
        if not scale_candidates:
            # Relate marker distance in camera frame to SfM object distance
            for o in filtered_obs:
                if o.camera_center_sfm is not None and object_points_sfm is not None and len(object_points_sfm) > 0:
                    obj_center_sfm = np.mean(object_points_sfm, axis=0)
                    dist_cam_to_obj_sfm = float(np.linalg.norm(o.camera_center_sfm - obj_center_sfm))
                    if dist_cam_to_obj_sfm > 0.1:
                        scale_candidates.append(o.camera_distance_cm / dist_cam_to_obj_sfm)

        if scale_candidates:
            final_scale = float(np.median(scale_candidates))
            conf = min(0.95, 0.40 + 0.10 * len(scale_candidates))
            tier = "HIGH" if len(scale_candidates) >= 5 else "MODERATE"
        else:
            final_scale = 1.0
            conf = 0.25
            tier = "LOW"
            limitations.append("Insufficient multi-view parallax between marker observations to triangulate baseline.")

        # Limitations & confidence assessment
        avg_err = float(np.mean([o.reprojection_error_px for o in filtered_obs]))
        if avg_err > 3.0:
            limitations.append(f"Moderate marker corner reprojection error ({avg_err:.2f} px); may cause minor metric drift.")
        limitations.append("Metric scale assumes rigid scene geometry and stationary marker relative to the target object.")

        # Measure object dimensions
        dims = self.measure_object_dimensions(object_points_sfm, final_scale) if object_points_sfm is not None else None

        return MetricScaleResult(
            scale_factor=round(final_scale, 4),
            marker_size_cm=marker_size_cm,
            num_marker_observations=len(filtered_obs),
            marker_id_used=best_mid,
            marker_observations=filtered_obs,
            object_dimensions=dims,
            confidence_score=round(conf, 2),
            confidence_tier=tier,
            calibration_status=calib_status,
            limitations=limitations,
        )

    def measure_object_dimensions(
        self,
        points_3d_sfm: np.ndarray,
        scale_factor: float,
    ) -> Optional[MetricDimensions]:
        """
        Calculates physical bounding box dimensions and volume in metric centimetres.
        """
        if points_3d_sfm is None or len(points_3d_sfm) < 2:
            return None

        # Scale points to centimetres
        pts_cm = np.asarray(points_3d_sfm, dtype=np.float64) * scale_factor

        min_c = np.min(pts_cm, axis=0)
        max_c = np.max(pts_cm, axis=0)
        extents = max_c - min_c

        # (Width X, Height Y, Depth Z)
        w_cm = float(extents[0])
        h_cm = float(extents[1])
        d_cm = float(extents[2])

        # Approximate volumetric cylinder or cuboid bound
        vol_cm3 = float(w_cm * h_cm * d_cm * 0.70)  # ~70% packing density approximation for bottles/organic objects
        vol_litres = vol_cm3 / 1000.0

        return MetricDimensions(
            width_x_cm=round(w_cm, 2),
            height_y_cm=round(h_cm, 2),
            depth_z_cm=round(d_cm, 2),
            bounding_box_min_cm=[round(float(v), 2) for v in min_c],
            bounding_box_max_cm=[round(float(v), 2) for v in max_c],
            estimated_volume_cm3=round(vol_cm3, 2),
            estimated_volume_litres=round(vol_litres, 3),
        )

    def scale_point_cloud(self, points: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scales a (N, 3) 3D point cloud array into metric coordinates."""
        return np.asarray(points, dtype=np.float64) * scale_factor

    def scale_mesh(self, vertices: np.ndarray, faces: np.ndarray, scale_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Scales mesh vertices into physical metric centimetres."""
        scaled_v = np.asarray(vertices, dtype=np.float64) * scale_factor
        return scaled_v, faces.copy()
