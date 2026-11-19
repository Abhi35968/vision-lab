"""
PointNet 3D Object Classifier Inference Engine.
Coordinates preprocessing, normalization, sampling, and classification of 3D object point clouds.
Evaluates both real-world sparse SfM clusters and synthetic CAD geometries.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch

from src.classification_3d.pointnet import PointNetClassifier
from src.classification_3d.point_cloud_sampler import (
    normalize_point_cloud,
    sample_point_cloud,
    augment_point_cloud,
)


@dataclass
class ClassificationPrediction:
    """Telemetry for a single class prediction."""
    class_id: int
    class_name: str
    confidence: float  # [0.0, 1.0]
    log_prob: float


@dataclass
class PointNetInferenceResult:
    """Consolidated 3D PointNet classification output."""
    top_predictions: List[ClassificationPrediction]
    predicted_class: str
    predicted_confidence: float
    num_input_points: int
    num_sampled_points: int
    is_sparse_sfm_cloud: bool
    domain_gap_analysis: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "predicted_confidence": round(self.predicted_confidence, 4),
            "top_predictions": [
                {
                    "class_id": p.class_id,
                    "class_name": p.class_name,
                    "confidence": round(p.confidence, 4),
                    "log_prob": round(p.log_prob, 4),
                }
                for p in self.top_predictions
            ],
            "num_input_points": self.num_input_points,
            "num_sampled_points": self.num_sampled_points,
            "is_sparse_sfm_cloud": self.is_sparse_sfm_cloud,
            "domain_gap_analysis": self.domain_gap_analysis,
        }


class PointNetInferenceEngine:
    """High-level inference and evaluation pipeline for PointNet."""

    DEFAULT_CLASSES_FILE = Path("config/pointnet_classes.json")
    DEFAULT_WEIGHTS_FILE = Path("models/weights/pointnet_modelnet40.pth")

    def __init__(
        self,
        classes_path: Optional[Union[str, Path]] = None,
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        num_points: int = 1024,
    ):
        self.device = torch.device(device)
        self.num_points = num_points

        # 1. Load class labels
        c_path = Path(classes_path) if classes_path else self.DEFAULT_CLASSES_FILE
        if c_path.exists():
            with open(c_path, "r") as f:
                self.classes = json.load(f)
        else:
            self.classes = [f"class_{i}" for i in range(40)]

        self.num_classes = len(self.classes)

        # 2. Build model
        self.model = PointNetClassifier(num_classes=self.num_classes, feature_transform=True)
        self.model.to(self.device)

        # 3. Load weights if present
        w_path = Path(weights_path) if weights_path else self.DEFAULT_WEIGHTS_FILE
        self.has_pretrained_weights = False
        if w_path.exists():
            try:
                state_dict = torch.load(str(w_path), map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.has_pretrained_weights = True
            except Exception as e:
                # Log and fallback to initialized weights
                self.has_pretrained_weights = False

        self.model.eval()

    @staticmethod
    def canonicalize_to_modelnet_frame(points: np.ndarray) -> np.ndarray:
        """
        Transforms point clouds into ModelNet canonical reference coordinates:
        1. Centers points at the origin (0, 0, 0).
        2. Applies Principal Component Analysis (PCA) to align the major elongation axis with the +Y upright axis.
        3. Resolves 180-degree sign ambiguity by measuring radial spread: orienting the narrower neck towards +Y.
        """
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) < 4:
            return pts
        centered = pts - np.mean(pts, axis=0)
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        primary = evecs[:, np.argmax(evals)]

        # Align primary axis with target upright [0, 1, 0]
        target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = np.cross(primary, target)
        s = np.linalg.norm(v)
        c = float(np.dot(primary, target))
        if s < 1e-6:
            R_align = np.eye(3, dtype=np.float32) if c > 0 else -np.eye(3, dtype=np.float32)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)
            R_align = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))

        aligned = (R_align @ centered.T).T

        # Resolve upright flip: check radial spread in upper half vs lower half
        upper_mask = aligned[:, 1] > 0
        lower_mask = aligned[:, 1] < 0
        r_upper = np.mean(np.linalg.norm(aligned[upper_mask, :][:, [0, 2]], axis=1)) if np.any(upper_mask) else 1.0
        r_lower = np.mean(np.linalg.norm(aligned[lower_mask, :][:, [0, 2]], axis=1)) if np.any(lower_mask) else 1.0

        if r_upper > r_lower:
            aligned[:, 1] = -aligned[:, 1]

        return aligned

    def classify_point_cloud(
        self,
        points: np.ndarray,
        top_k: int = 5,
        sampling_method: str = "fps",
        canonicalize_upright: bool = True,
        tta_rotations: int = 10,
    ) -> PointNetInferenceResult:
        """
        Classifies an input 3D point cloud:
        1. Removes floor/table plane (bottom 10% Y) to avoid staircase artifacts.
        2. Applies Statistical Outlier Removal via open3d to clean the cloud.
        3. Canonicalizes orientation to ModelNet upright (+Y) reference frame.
        4. Normalizes points to unit sphere centered at origin.
        5. Samples or upsamples to target point count (e.g. 1024).
        6. Test-time augmentation: runs tta_rotations Y-axis rotations, votes on majority class.
        """
        import open3d as o3d

        pts = np.asarray(points, dtype=np.float32)
        n_orig = len(pts)

        if n_orig == 0:
            raise ValueError("Input point cloud is empty (0 points).")

        # --- Step 0: Statistical Outlier Removal ---
        if n_orig >= 20:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                nb = min(30, n_orig - 1)
                pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=1.5)
                cleaned = np.asarray(pcd_clean.points, dtype=np.float32)
                if len(cleaned) >= 10:
                    pts = cleaned
            except Exception:
                pass  # fallback to raw points

        # --- Step 1: Remove table plane (bottom 10% Y = floor artifacts) ---
        if len(pts) >= 20:
            y_vals = pts[:, 1]
            y_thresh = float(np.percentile(y_vals, 10))
            above_floor = pts[y_vals > y_thresh]
            if len(above_floor) >= 10:
                pts = above_floor

        # --- Step 2: Canonicalize upright ---
        if canonicalize_upright:
            pts = self.canonicalize_to_modelnet_frame(pts)

        # --- Step 3: Unit sphere normalization ---
        norm_pts, _, _ = normalize_point_cloud(pts)

        # --- Step 4: Sampling to 1024 ---
        sampled_pts = sample_point_cloud(
            norm_pts,
            target_num=self.num_points,
            method=sampling_method,
            jitter_upsample=False,
        )

        # --- Step 5: Test-Time Augmentation (Y-axis rotation voting) ---
        if tta_rotations > 1:
            vote_counts = np.zeros(self.num_classes, dtype=np.float64)
            angles = np.linspace(0, 2 * np.pi, tta_rotations, endpoint=False)
            for angle in angles:
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                rot_y = np.array(
                    [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32
                )
                pts_rot = (sampled_pts @ rot_y.T).astype(np.float32)
                inp = torch.from_numpy(pts_rot.T).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    log_probs, _, _ = self.model(inp)
                    probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
                vote_counts += probs
            # Average across rotations
            probs_avg = vote_counts / tta_rotations
            probs = probs_avg.astype(np.float32)
            log_probs_np = np.log(np.clip(probs, 1e-10, 1.0))
        else:
            # --- Single-pass inference ---
            input_tensor = torch.from_numpy(sampled_pts.T).unsqueeze(0).to(self.device)
            with torch.no_grad():
                log_probs, _, _ = self.model(input_tensor)
                probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
                log_probs_np = log_probs.squeeze(0).cpu().numpy()

        # --- Step 6: Rank top predictions ---
        top_indices = np.argsort(probs)[::-1][:top_k]
        predictions = []
        for idx in top_indices:
            cls_name = self.classes[idx] if idx < len(self.classes) else f"class_{idx}"
            predictions.append(
                ClassificationPrediction(
                    class_id=int(idx),
                    class_name=cls_name,
                    confidence=float(probs[idx]),
                    log_prob=float(log_probs_np[idx]),
                )
            )

        best_pred = predictions[0]

        # 5. Domain gap analysis
        is_sparse = bool(n_orig < 100)
        domain_notes = (
            f"Evaluated on {n_orig} input points (TTA×{tta_rotations}, resampled to {self.num_points}). "
            + ("Real-world sparse SfM point clouds lack dense CAD surface topology, representing an experimental "
               "domain shift relative to ModelNet CAD training distributions." if is_sparse else
               "Dense input point cloud provides strong structural coverage.")
        )

        return PointNetInferenceResult(
            top_predictions=predictions,
            predicted_class=best_pred.class_name,
            predicted_confidence=best_pred.confidence,
            num_input_points=n_orig,
            num_sampled_points=self.num_points,
            is_sparse_sfm_cloud=is_sparse,
            domain_gap_analysis=domain_notes,
        )

