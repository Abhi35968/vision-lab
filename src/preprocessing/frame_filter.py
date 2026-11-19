"""
Frame Filter Module
Identifies and filters exact duplicates and near-duplicate redundant frames using hashing.
"""

from dataclasses import dataclass
import hashlib
from typing import Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class SimilarityComparison:
    """Detailed comparison between two sequential frames."""
    exact_duplicate: bool
    perceptual_distance: int        # Hamming distance in [0, 64]
    near_duplicate: bool
    similarity_score: float         # Normalized similarity in [0.0, 1.0]


class FrameFilter:
    """
    Detects exact and near-duplicate frames using cryptographic and perceptual hashing.
    
    Technical Rationale:
    1. Exact Duplicates:
       Identified via SHA-256 / MD5 hashing of normalized pixel arrays. These arise from
       repeating frames during video container decoding or camera sensor lag.
    
    2. Near-Duplicates (Difference Hash - dHash):
       Downscales the image to 9x8 grayscale and computes relative horizontal gradient:
           bit(x, y) = 1 if I(x+1, y) > I(x, y) else 0
       This creates a 64-bit integer fingerprint robust to illumination shifts and compression artifacts.
       The Hamming distance (bit count of hash_A XOR hash_B) measures perceptual difference:
           - distance <= max_hamming_distance: near-duplicate (camera was stationary).
    
    SfM Consideration:
    Structure from Motion requires baseline disparity (t != 0) between camera views to
    triangulate 3D points. When a camera is static, frames have zero baseline, creating
    ill-conditioned triangulation without contributing new geometry. Filtering near-duplicates
    eliminates zero-baseline frames while preserving necessary multi-view overlap.
    """

    def __init__(
        self,
        max_hamming_distance: int = 3,       # dHash bit difference threshold (<= 3 implies near-identical)
        min_hist_correlation: float = 0.96,  # Color histogram correlation threshold
    ):
        self.max_hamming_distance = max_hamming_distance
        self.min_hist_correlation = min_hist_correlation

    @staticmethod
    def compute_exact_hash(image: np.ndarray) -> str:
        """Computes cryptographic SHA-256 hash of image array bytes."""
        return hashlib.sha256(image.tobytes()).hexdigest()

    @staticmethod
    def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
        """
        Computes 64-bit Difference Hash (dHash) from image gradients.
        
        Resizes to (hash_size + 1, hash_size), computes horizontal gradient, and packs into an integer.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize to (width=9, height=8) for 8x8 gradient comparisons
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)

        # Compare adjacent columns: diff has shape (8, 8)
        diff = resized[:, 1:] > resized[:, :-1]

        # Convert 64 boolean array to an integer
        hash_val = 0
        for bit in diff.flatten():
            hash_val = (hash_val << 1) | int(bit)

        return hash_val

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Computes bitwise Hamming distance between two 64-bit integer hashes."""
        return bin(hash1 ^ hash2).count("1")

    def compare_hashes(self, hash1: int, hash2: int) -> SimilarityComparison:
        """Compares two 64-bit integer perceptual hashes directly without requiring image arrays."""
        dist = self.hamming_distance(hash1, hash2)
        similarity = 1.0 - (dist / 64.0)
        is_exact = (dist == 0)
        is_near = dist <= self.max_hamming_distance

        return SimilarityComparison(
            exact_duplicate=is_exact,
            perceptual_distance=dist,
            near_duplicate=is_near,
            similarity_score=round(similarity, 4),
        )

    def compare_frames(
        self,
        frame1: Optional[np.ndarray],
        frame2: Optional[np.ndarray],
        hash1: Optional[int] = None,
        hash2: Optional[int] = None,
    ) -> SimilarityComparison:
        """
        Compares two frames for exact identity and perceptual near-identity.
        """
        h1 = hash1 if hash1 is not None else (self.compute_dhash(frame1) if frame1 is not None else 0)
        h2 = hash2 if hash2 is not None else (self.compute_dhash(frame2) if frame2 is not None else 0)

        # 1. Exact duplicate check
        if frame1 is not None and frame2 is not None and frame1 is not frame2:
            is_exact = np.array_equal(frame1, frame2)
        else:
            is_exact = (h1 == h2)

        # 2. Perceptual Difference Hash distance
        dist = self.hamming_distance(h1, h2)
        similarity = 1.0 - (dist / 64.0)
        is_near = is_exact or (dist <= self.max_hamming_distance)

        return SimilarityComparison(
            exact_duplicate=is_exact,
            perceptual_distance=dist,
            near_duplicate=is_near,
            similarity_score=round(similarity, 4),
        )
