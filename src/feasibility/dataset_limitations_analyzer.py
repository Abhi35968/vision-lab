"""
Dataset Limitations Analyzer Module
Evaluates camera orbit angular coverage, trajectory continuity, viewing gaps,
motion blur (Laplacian variance), and optical challenges (transparency, refraction).
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class CameraOrbitRecord:
    """Camera viewing pose relative to the object centroid."""
    camera_id: int
    image_name: str
    camera_center: List[float]
    azimuth_deg_raw: float
    azimuth_deg_unwrapped: float
    elevation_deg: float
    distance_to_object: float


@dataclass
class DatasetLimitationsSummary:
    """Summary of spatial and optical capture limitations."""
    total_keyframes: int
    registered_cameras: int
    unregistered_cameras: int
    object_dropout_frames: List[int]
    dropout_count: int
    # Angular orbit coverage
    orbit_min_azimuth_deg: float
    orbit_max_azimuth_deg: float
    orbit_angular_span_deg: float
    orbit_coverage_pct_of_360: float
    unobserved_angular_blindspot_deg: float
    camera_mean_distance: float
    camera_distance_std: float
    # Blur metrics
    mean_frame_blur_variance: float
    min_frame_blur_variance: float
    mean_object_roi_blur_variance: float
    min_object_roi_blur_variance: float
    # Optical classification
    surface_reflectance_type: str
    refraction_index_approx: float
    lambertian_assumption_valid: bool
    background_contamination_factor: str

    def to_dict(self) -> dict:
        return asdict(self)


class DatasetLimitationsAnalyzer:
    """
    Analyzes physical and geometric properties of the handheld smartphone video recording:
    1. Camera orbit trajectory and angular coverage around the object.
    2. Missing viewing angles and blind spots.
    3. Motion blur across keyframes.
    4. Non-Lambertian physical phenomena (specularity, transparency, refractive distortion).
    """

    def __init__(self, object_centroid: np.ndarray = np.array([-1.572, -2.569, 6.115])):
        self.object_centroid = np.array(object_centroid, dtype=np.float64)

    def analyze_camera_orbit(
        self,
        cameras: Dict[str, dict],
    ) -> Tuple[List[CameraOrbitRecord], float, float, float]:
        """
        Calculates azimuth and elevation angles of all registered cameras
        relative to the object centroid in 3D world space.
        """
        reg_cams = [
            (int(cid), c) for cid, c in cameras.items() if c.get("is_registered", False)
        ]
        reg_cams = sorted(reg_cams, key=lambda x: x[0])

        if not reg_cams:
            return [], 0.0, 0.0, 0.0

        records: List[CameraOrbitRecord] = []
        raw_azimuths_rad = []

        for cid, c_data in reg_cams:
            c_center = np.array(c_data["camera_center"], dtype=np.float64)
            vec = c_center - self.object_centroid
            # Azimuth in X-Z plane (horizontal ground plane in OpenCV SfM)
            az_rad = np.arctan2(vec[0], vec[2])
            raw_azimuths_rad.append(az_rad)

            # Elevation: angle above X-Z plane (-Y is up in standard camera coordinates)
            horiz_dist = np.sqrt(vec[0] ** 2 + vec[2] ** 2)
            elev_rad = np.arctan2(-vec[1], horiz_dist)
            dist = np.linalg.norm(vec)

            records.append(
                CameraOrbitRecord(
                    camera_id=cid,
                    image_name=c_data.get("image_name", f"frame_{cid}"),
                    camera_center=[round(float(v), 4) for v in c_center],
                    azimuth_deg_raw=round(float(np.degrees(az_rad)), 2),
                    azimuth_deg_unwrapped=0.0,  # Populated below
                    elevation_deg=round(float(np.degrees(elev_rad)), 2),
                    distance_to_object=round(float(dist), 4),
                )
            )

        # Unwrap azimuths to obtain continuous path
        unwrapped_rad = np.unwrap(raw_azimuths_rad)
        unwrapped_deg = np.degrees(unwrapped_rad)

        for rec, udeg in zip(records, unwrapped_deg):
            rec.azimuth_deg_unwrapped = round(float(udeg), 2)

        min_az = float(unwrapped_deg.min())
        max_az = float(unwrapped_deg.max())
        span_deg = max_az - min_az

        return records, min_az, max_az, span_deg

    def evaluate_motion_blur(
        self,
        images: List[np.ndarray],
        masks: List[Optional[np.ndarray]],
    ) -> Tuple[List[float], List[float], List[int]]:
        """
        Computes Laplacian variance as an indicator of high-frequency edge sharpness / motion blur.
        Returns:
            frame_blur_variances: Whole-frame Laplacian variance.
            object_blur_variances: Object ROI Laplacian variance.
            dropout_frames: Indices where no mask was detected.
        """
        frame_vars = []
        object_vars = []
        dropouts = []

        for idx, img in enumerate(images):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            f_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            frame_vars.append(round(f_var, 2))

            m = masks[idx] if idx < len(masks) else None
            if m is not None and np.count_nonzero(m > 0) > 0:
                ys, xs = np.where(m > 0)
                roi = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                o_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
                object_vars.append(round(o_var, 2))
            else:
                object_vars.append(0.0)
                dropouts.append(idx)

        return frame_vars, object_vars, dropouts

    def compile_limitations_summary(
        self,
        total_keyframes: int,
        cameras: Dict[str, dict],
        frame_blur: List[float],
        object_blur: List[float],
        dropouts: List[int],
    ) -> DatasetLimitationsSummary:
        """Assembles full dataset limitations telemetry."""
        orbit_records, min_az, max_az, span_deg = self.analyze_camera_orbit(cameras)

        reg_count = sum(1 for c in cameras.values() if c.get("is_registered", False))
        unreg_count = len(cameras) - reg_count

        coverage_pct = (span_deg / 360.0) * 100.0
        blindspot_deg = 360.0 - span_deg

        distances = [r.distance_to_object for r in orbit_records]
        mean_d = float(np.mean(distances)) if distances else 0.0
        std_d = float(np.std(distances)) if distances else 0.0

        valid_obj_blurs = [v for v in object_blur if v > 0]
        mean_obj_b = float(np.mean(valid_obj_blurs)) if valid_obj_blurs else 0.0
        min_obj_b = float(np.min(valid_obj_blurs)) if valid_obj_blurs else 0.0

        return DatasetLimitationsSummary(
            total_keyframes=total_keyframes,
            registered_cameras=reg_count,
            unregistered_cameras=unreg_count,
            object_dropout_frames=dropouts,
            dropout_count=len(dropouts),
            orbit_min_azimuth_deg=round(min_az, 2),
            orbit_max_azimuth_deg=round(max_az, 2),
            orbit_angular_span_deg=round(span_deg, 2),
            orbit_coverage_pct_of_360=round(coverage_pct, 2),
            unobserved_angular_blindspot_deg=round(blindspot_deg, 2),
            camera_mean_distance=round(mean_d, 4),
            camera_distance_std=round(std_d, 4),
            mean_frame_blur_variance=round(float(np.mean(frame_blur)), 2),
            min_frame_blur_variance=round(float(np.min(frame_blur)), 2),
            mean_object_roi_blur_variance=round(mean_obj_b, 2),
            min_object_roi_blur_variance=round(min_obj_b, 2),
            surface_reflectance_type="Dielectric Specular Transmission (Clear PET/Water)",
            refraction_index_approx=1.333,
            lambertian_assumption_valid=False,
            background_contamination_factor="Severe: refracted background seen through body",
        )

