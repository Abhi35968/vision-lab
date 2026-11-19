"""
Feature Extractor Module
Extracts scale- and rotation-invariant keypoints and local descriptors using SIFT or ORB.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class ExtractedFeatures:
    """Container for detected keypoints, feature descriptors, and extraction telemetry."""
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]       # (N, 128) float32 for SIFT, (N, 32) uint8 for ORB
    keypoint_coords: np.ndarray            # (N, 2) float32 coordinates [x, y] in pixel space
    num_keypoints: int
    detector_name: str                     # "SIFT" or "ORB"
    descriptor_dim: int                    # 128 (SIFT) or 32 (ORB)
    descriptor_dtype: str                  # "float32" or "uint8"
    image_path: Optional[str]
    extraction_time_ms: float
    image_size: Tuple[int, int]            # (width, height)

    def to_dict(self) -> dict:
        return {
            "num_keypoints": self.num_keypoints,
            "detector_name": self.detector_name,
            "descriptor_dim": self.descriptor_dim,
            "descriptor_dtype": self.descriptor_dtype,
            "image_path": self.image_path,
            "extraction_time_ms": round(self.extraction_time_ms, 2),
            "image_size": self.image_size,
        }


class FeatureExtractor:
    """
    Extracts local geometric keypoints and descriptors from images.
    
    Technical Difference: Keypoints vs. Descriptors:
    -------------------------------------------------
    1. Keypoint: A specific (x, y) 2D pixel location in the image plane that exhibits distinct
       spatial gradient variance (e.g., corners, blobs, edge intersections). It is characterized
       by its coordinate position, scale (octave radius), orientation angle, and response strength.
    
    2. Descriptor: A numerical feature vector (e.g., 128-D float vector for SIFT, 256-bit binary
       string for ORB) summarizing the local appearance, gradient directions, or intensity
       comparisons within a normalized patch around the keypoint. Descriptors are engineered to be
       invariant to illumination, 2D rotation, and affine scale transformations.
    """

    def __init__(
        self,
        detector: str = "SIFT",
        n_features: int = 8000,
        contrast_threshold: float = 0.03,
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
        orb_scale_factor: float = 1.2,
        orb_n_levels: int = 8,
        use_root_sift: bool = True,
    ):
        self.detector_name = detector.upper()
        self.n_features = n_features
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        self.orb_scale_factor = orb_scale_factor
        self.orb_n_levels = orb_n_levels
        self.use_root_sift = use_root_sift

        self._init_detector()

    def _init_detector(self) -> None:
        """Initializes the underlying OpenCV feature detector instance."""
        if self.detector_name == "SIFT":
            self.detector = cv2.SIFT_create(
                nfeatures=self.n_features,
                contrastThreshold=self.contrast_threshold,
                edgeThreshold=self.edge_threshold,
                sigma=self.sigma,
            )
            self.descriptor_dim = 128
            self.descriptor_dtype = "float32"
            # Root-SIFT flag stored; actual conversion in extract()
        elif self.detector_name == "ORB":
            self.detector = cv2.ORB_create(
                nfeatures=self.n_features,
                scaleFactor=self.orb_scale_factor,
                nlevels=self.orb_n_levels,
                edgeThreshold=int(self.edge_threshold),
            )
            self.descriptor_dim = 32  # 32 bytes = 256 bits
            self.descriptor_dtype = "uint8"
        else:
            raise ValueError(f"Unsupported detector: {self.detector_name}. Supported: 'SIFT', 'ORB'.")

    def extract(self, image_or_path: Union[np.ndarray, str, Path]) -> ExtractedFeatures:
        """
        Detects keypoints and computes descriptors for a given image.
        
        Handles image loading, grayscale conversion, and zero-feature edge cases.
        """
        img_path_str: Optional[str] = None
        if isinstance(image_or_path, (str, Path)):
            img_path = Path(image_or_path).resolve()
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise ValueError(f"Failed to decode image from: {img_path}")
            img_path_str = str(img_path)
        elif isinstance(image_or_path, np.ndarray):
            bgr = image_or_path
        else:
            raise TypeError(f"Expected image path or np.ndarray, got {type(image_or_path)}")

        h, w = bgr.shape[:2]
        if len(bgr.shape) == 3:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr

        # Measure extraction latency
        t0 = time.perf_counter()
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Handle edge cases: No keypoints or no descriptors found
        if keypoints is None or len(keypoints) == 0:
            keypoints = []
            descriptors = None
            coords = np.empty((0, 2), dtype=np.float32)
        else:
            coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        if descriptors is not None:
            # Ensure correct dtype for downstream matchers
            if self.detector_name == "SIFT" and descriptors.dtype != np.float32:
                descriptors = descriptors.astype(np.float32)
            elif self.detector_name == "ORB" and descriptors.dtype != np.uint8:
                descriptors = descriptors.astype(np.uint8)

            # Root-SIFT: L1-normalize then sqrt → better matching on smooth surfaces
            if self.detector_name == "SIFT" and self.use_root_sift:
                eps = 1e-7
                descriptors = descriptors.astype(np.float32)
                norms = np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True)
                descriptors = descriptors / (norms + eps)
                descriptors = np.sqrt(descriptors)

        return ExtractedFeatures(
            keypoints=keypoints,
            descriptors=descriptors,
            keypoint_coords=coords,
            num_keypoints=len(keypoints),
            detector_name=self.detector_name,
            descriptor_dim=self.descriptor_dim,
            descriptor_dtype=self.descriptor_dtype,
            image_path=img_path_str,
            extraction_time_ms=elapsed_ms,
            image_size=(w, h),
        )

    # Alias for convenience
    extract_from_path = extract

    @staticmethod
    def visualize_keypoints(
        image: np.ndarray,
        features: ExtractedFeatures,
        max_draw: int = 500,
        flags: int = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    ) -> np.ndarray:
        """Draws keypoints with scale circles and orientation vectors."""
        kps = features.keypoints
        if len(kps) > max_draw:
            # Sort by response strength and draw top max_draw
            kps = sorted(kps, key=lambda k: k.response, reverse=True)[:max_draw]

        vis = cv2.drawKeypoints(image, kps, None, color=(0, 255, 0), flags=flags)
        # Overlay metadata header
        text = f"{features.detector_name}: {features.num_keypoints} kps ({features.extraction_time_ms:.1f}ms)"
        cv2.putText(vis, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis

