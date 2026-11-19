"""
Image Quality Assessment Module
Computes blur, exposure, contrast, and resolution metrics to qualify frames for SfM.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class ImageQualityMetrics:
    """Quantitative quality measurements and validation flags for a single image."""
    blur_score: float              # Variance of the Laplacian
    is_blurry: bool
    brightness_mean: float         # Mean grayscale intensity in [0, 255]
    is_underexposed: bool
    is_overexposed: bool
    contrast_std: float            # Standard deviation of grayscale intensities
    is_low_contrast: bool
    width: int
    height: int
    is_sufficient_resolution: bool
    quality_passed: bool           # True if all quality checks pass
    rejection_reasons: list        # Explanations if frame failed any check

    def to_dict(self) -> dict:
        return asdict(self)


class ImageQualityAssessor:
    """
    Evaluates photographic and photometric quality for Structure from Motion keypoints.
    
    Mathematical Foundations:
    1. Blur Detection (Variance of the Laplacian):
       The continuous Laplacian operator is defined as:
           ∇²I = ∂²I/∂x² + ∂²I/∂y²
       Convolving the grayscale image with a discrete Laplacian kernel highlights rapid
       spatial intensity changes (edges and textures). The variance across the convolved
       image measures edge sharpness:
           Var(∇²I) = E[(∇²I - E[∇²I])²]
       A sharp image with crisp high-frequency textures produces a large variance.
       A blurred image (motion blur or defocus) acts as a low-pass filter, depressing
       edge gradients and resulting in a low Laplacian variance.

    2. Brightness & Contrast:
       - Brightness μ = (1/N) * Σ I(x, y)
       - Contrast σ = sqrt((1/N) * Σ (I(x, y) - μ)²)
       Extreme brightness values indicate sensor saturation (overexposure) or lack of
       light (underexposure). Very low contrast indicates washed-out or feature-deficient images.
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_contrast: float = 20.0,
        min_dimension: int = 480,
    ):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_dimension = min_dimension

    def compute_blur_score(self, gray: np.ndarray) -> float:
        """Computes the continuous variance of the Laplacian sharpness score."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score = float(laplacian.var())
        return round(score, 3)

    def compute_brightness_and_contrast(self, gray: np.ndarray) -> Tuple[float, float]:
        """Calculates the mean (brightness) and standard deviation (contrast)."""
        mean, std = cv2.meanStdDev(gray)
        return round(float(mean[0, 0]), 2), round(float(std[0, 0]), 2)

    def evaluate(self, image_or_path: Union[np.ndarray, str]) -> ImageQualityMetrics:
        """
        Assesses image quality from a loaded NumPy array (BGR) or image file path.
        """
        if isinstance(image_or_path, (str, bytes)):
            bgr = cv2.imread(str(image_or_path))
            if bgr is None:
                raise ValueError(f"Failed to read image from path: {image_or_path}")
        else:
            bgr = image_or_path

        h, w = bgr.shape[:2]

        # Convert to grayscale for metric calculations
        if len(bgr.shape) == 3 and bgr.shape[2] == 3:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        elif len(bgr.shape) == 2:
            gray = bgr
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGRA2GRAY)

        # 1. Blur
        blur_score = self.compute_blur_score(gray)
        is_blurry = blur_score < self.blur_threshold

        # 2. Brightness & Contrast
        brightness, contrast = self.compute_brightness_and_contrast(gray)
        is_underexposed = brightness < self.min_brightness
        is_overexposed = brightness > self.max_brightness
        is_low_contrast = contrast < self.min_contrast

        # 3. Resolution
        is_sufficient_resolution = min(h, w) >= self.min_dimension

        # Compile rejection reasons
        reasons = []
        if is_blurry:
            reasons.append(f"Blur score {blur_score} < threshold {self.blur_threshold}")
        if is_underexposed:
            reasons.append(f"Brightness {brightness} < minimum {self.min_brightness}")
        if is_overexposed:
            reasons.append(f"Brightness {brightness} > maximum {self.max_brightness}")
        if is_low_contrast:
            reasons.append(f"Contrast {contrast} < minimum {self.min_contrast}")
        if not is_sufficient_resolution:
            reasons.append(f"Small resolution ({w}x{h}) < min dimension {self.min_dimension}")

        quality_passed = len(reasons) == 0

        return ImageQualityMetrics(
            blur_score=blur_score,
            is_blurry=is_blurry,
            brightness_mean=brightness,
            is_underexposed=is_underexposed,
            is_overexposed=is_overexposed,
            contrast_std=contrast,
            is_low_contrast=is_low_contrast,
            width=w,
            height=h,
            is_sufficient_resolution=is_sufficient_resolution,
            quality_passed=quality_passed,
            rejection_reasons=reasons,
        )

