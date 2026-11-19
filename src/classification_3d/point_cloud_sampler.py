"""
Point Cloud Normalization, Sampling, and Augmentation Module.
Prepares sparse, irregular SfM point clouds for 3D deep learning architectures (e.g. PointNet):
- Zero-mean centering and unit sphere normalization
- Farthest Point Sampling (FPS) and upsampling with jitter
- Point coordinate augmentation (rotation, jitter)
"""

from typing import Optional, Tuple
import numpy as np


def normalize_point_cloud(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalizes a point cloud to fit within a unit sphere centered at origin.
    Returns:
        normalized_points: (N, 3) array with centroid at [0, 0, 0] and max norm 1.0.
        centroid: (3,) original centroid
        max_distance: scalar maximum distance from centroid
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N, 3), got {pts.shape}")

    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    distances = np.linalg.norm(pts_centered, axis=1)
    max_dist = float(np.max(distances)) if len(distances) > 0 and np.max(distances) > 1e-8 else 1.0

    normalized = pts_centered / max_dist
    return normalized, centroid, max_dist


def farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Selects n_samples from points using Farthest Point Sampling (FPS).
    Ensures maximal geometric coverage of the underlying 3D structure.
    """
    pts = np.asarray(points, dtype=np.float32)
    n_points = len(pts)

    if n_samples >= n_points:
        return pts

    sampled_indices = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(n_points, fill_value=np.inf, dtype=np.float32)

    # Pick the first point arbitrarily
    farthest_idx = 0
    sampled_indices[0] = farthest_idx

    for i in range(1, n_samples):
        current_pt = pts[farthest_idx]
        dist_to_current = np.sum((pts - current_pt) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_current)
        farthest_idx = int(np.argmax(distances))
        sampled_indices[i] = farthest_idx

    return pts[sampled_indices]


def sample_point_cloud(
    points: np.ndarray,
    target_num: int = 1024,
    method: str = "fps",
    jitter_upsample: bool = True,
) -> np.ndarray:
    """
    Samples or upsamples a point cloud to exactly target_num points.
    If the point cloud has fewer points than target_num (typical in sparse SfM),
    it repeats existing points and optionally adds small Gaussian jitter.
    """
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)

    if n == 0:
        # Return zeros if completely empty
        return np.zeros((target_num, 3), dtype=np.float32)

    if n == target_num:
        return pts.copy()

    if n > target_num:
        if method.lower() == "fps":
            return farthest_point_sampling(pts, target_num)
        else:
            indices = np.random.choice(n, size=target_num, replace=False)
            return pts[indices]

    # n < target_num: Upsampling with repetition and jitter
    repeats = int(np.ceil(target_num / n))
    expanded = np.tile(pts, (repeats, 1))[:target_num]

    if jitter_upsample and n > 1:
        # Add slight jitter to duplicated points to form a small local neighborhood
        std = float(np.mean(np.std(pts, axis=0))) * 0.05
        std = max(std, 0.01)
        noise = np.random.normal(0, std, size=expanded.shape).astype(np.float32)
        # Keep original points untouched for first n entries
        expanded[n:] += noise[n:]

    return expanded


def augment_point_cloud(
    points: np.ndarray,
    rotation_axis: str = "y",
    max_angle_deg: float = 360.0,
    jitter_std: float = 0.01,
    jitter_clip: float = 0.05,
) -> np.ndarray:
    """
    Applies 3D data augmentation to point cloud:
    - Random rotation around specified axis (x, y, or z)
    - Coordinate jittering with clipped Gaussian noise
    """
    pts = np.asarray(points, dtype=np.float32).copy()

    # 1. Random rotation
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    if rotation_axis.lower() == "x":
        rot_mat = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]], dtype=np.float32)
    elif rotation_axis.lower() == "z":
        rot_mat = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
    else:  # 'y'
        rot_mat = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32)

    pts = pts @ rot_mat.T

    # 2. Add clipped jitter
    if jitter_std > 0:
        noise = np.clip(np.random.normal(0, jitter_std, size=pts.shape), -jitter_clip, jitter_clip).astype(np.float32)
        pts += noise

    return pts


def augment_partial_real_scan(
    points: np.ndarray,
    *,
    dropout_ratio: float = 0.15,
    outlier_ratio: float = 0.01,
    partial_view: bool = True,
) -> np.ndarray:
    """Augment a measured object cloud without pretending it is a complete CAD mesh.

    The transform uses a uniformly sampled 3-D rotation, point dropout,
    optional half-space occlusion, scale/translation perturbation, and a small
    number of outliers.  It is intended for a future project-specific dataset,
    not for converting ModelNet40 accuracy into a real-world claim.
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 8:
        return pts.copy()
    # Uniform SO(3) quaternion (Shoemake method).
    u1, u2, u3 = np.random.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ], dtype=np.float32)
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    out = pts @ R.T
    if partial_view:
        normal = np.random.normal(size=3).astype(np.float32)
        normal /= max(np.linalg.norm(normal), 1e-8)
        keep = (out @ normal) >= np.percentile(out @ normal, 30)
        out = out[keep]
    keep = np.random.random(len(out)) >= dropout_ratio
    out = out[keep] if keep.any() else out[:1]
    out *= np.random.uniform(0.8, 1.2)
    out += np.random.uniform(-0.1, 0.1, size=(1, 3)).astype(np.float32)
    n_out = int(round(len(out) * outlier_ratio))
    if n_out:
        lo, hi = out.min(axis=0), out.max(axis=0)
        out = np.vstack([out, np.random.uniform(lo - 0.2, hi + 0.2, size=(n_out, 3))])
    return out.astype(np.float32)

