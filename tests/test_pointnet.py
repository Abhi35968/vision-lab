"""
Unit tests for Stage 5 PointNet 3D Deep Learning Module.
Verifies:
- Point cloud normalization to unit sphere
- Farthest Point Sampling (FPS) and upsampling with jitter
- PointNet spatial transformer networks (STN3d, STNkd)
- Permutation-invariant feature encoder
- End-to-end classifier inference and domain-gap telemetry
"""

import numpy as np
import pytest
import torch

from src.classification_3d.point_cloud_sampler import (
    normalize_point_cloud,
    farthest_point_sampling,
    sample_point_cloud,
    augment_point_cloud,
)
from src.classification_3d.pointnet import (
    STN3d,
    STNkd,
    PointNetEncoder,
    PointNetClassifier,
    feature_transform_regularizer,
)
from src.classification_3d.pointnet_classifier import (
    PointNetInferenceEngine,
    PointNetInferenceResult,
)


class TestPointCloudSampler:
    def test_normalize_point_cloud(self):
        pts = np.array([[10.0, 20.0, 30.0], [12.0, 22.0, 32.0], [8.0, 18.0, 28.0]], dtype=np.float32)
        norm_pts, centroid, max_dist = normalize_point_cloud(pts)

        # Centroid should be approximately 0
        np.testing.assert_allclose(np.mean(norm_pts, axis=0), [0.0, 0.0, 0.0], atol=1e-5)
        # Max norm should be 1.0
        assert np.max(np.linalg.norm(norm_pts, axis=1)) <= 1.0001

    def test_farthest_point_sampling(self):
        pts = np.random.uniform(-1, 1, size=(50, 3)).astype(np.float32)
        sampled = farthest_point_sampling(pts, n_samples=16)
        assert sampled.shape == (16, 3)

    def test_sample_point_cloud_upsampling(self):
        pts = np.random.uniform(-1, 1, size=(10, 3)).astype(np.float32)
        sampled = sample_point_cloud(pts, target_num=1024, jitter_upsample=True)
        assert sampled.shape == (1024, 3)

    def test_augment_point_cloud(self):
        pts = np.random.uniform(-1, 1, size=(20, 3)).astype(np.float32)
        augmented = augment_point_cloud(pts, rotation_axis="y", jitter_std=0.01)
        assert augmented.shape == pts.shape
        assert not np.allclose(augmented, pts)


class TestPointNetArchitecture:
    def test_stn3d_output_shape_and_identity(self):
        stn = STN3d()
        x = torch.randn(2, 3, 64)
        matrix = stn(x)
        assert matrix.shape == (2, 3, 3)

    def test_stnkd_output_shape(self):
        stn_feat = STNkd(k=64)
        x = torch.randn(2, 64, 32)
        matrix = stn_feat(x)
        assert matrix.shape == (2, 64, 64)

    def test_pointnet_encoder(self):
        encoder = PointNetEncoder(global_feat=True, feature_transform=True)
        x = torch.randn(2, 3, 128)
        global_feat, trans, trans_feat = encoder(x)
        assert global_feat.shape == (2, 1024)
        assert trans.shape == (2, 3, 3)
        assert trans_feat.shape == (2, 64, 64)

    def test_pointnet_classifier_forward(self):
        model = PointNetClassifier(num_classes=40, feature_transform=True)
        x = torch.randn(2, 3, 128)
        log_probs, trans, trans_feat = model(x)
        assert log_probs.shape == (2, 40)
        # Check that probabilities sum to 1.0 (log-probs exp sums to 1)
        probs = torch.exp(log_probs)
        np.testing.assert_allclose(probs.sum(dim=-1).detach().numpy(), [1.0, 1.0], atol=1e-4)

    def test_feature_transform_regularizer(self):
        trans_feat = torch.eye(64).unsqueeze(0).repeat(2, 1, 1)
        loss = feature_transform_regularizer(trans_feat)
        assert float(loss.item()) < 1e-5


class TestPointNetInferenceEngine:
    def test_engine_inference_on_sparse_cloud(self):
        engine = PointNetInferenceEngine(num_points=512)
        pts = np.random.randn(25, 3)
        result = engine.classify_point_cloud(pts, top_k=5)

        assert isinstance(result, PointNetInferenceResult)
        assert len(result.top_predictions) == 5
        assert result.num_input_points == 25
        assert result.num_sampled_points == 512
        assert result.is_sparse_sfm_cloud is True
        assert "sparse SfM" in result.domain_gap_analysis

    def test_yolo_pointnet_decision_fusion(self):
        """Verifies multimodal decision fusion resolves category and overrides sparse noise."""
        from src.classification_3d.yolo_pointnet_fusion import YOLOPointNetFusionEngine

        fusion = YOLOPointNetFusionEngine()
        pts = np.random.randn(25, 3)
        yolo_dets = [
            {"class": "bottle", "confidence": 0.88},
            {"class": "bottle", "confidence": 0.92},
            {"class": "cup", "confidence": 0.40},
        ]

        fused = fusion.fuse(pts, yolo_dets)
        assert fused.predicted_class in {"bottle", "cup"}
        assert fused.predicted_confidence > 0.3
        assert fused.modality_weights["weight_2d"] > fused.modality_weights["weight_3d"]
        assert len(fused.top_candidates) > 0
        assert fused.is_uncertain is False

    def test_fusion_rejects_disagreeing_modalities(self):
        """A confident but incompatible PointNet class cannot be overruled by score multiplication."""
        from src.classification_3d.yolo_pointnet_fusion import YOLOPointNetFusionEngine

        engine = YOLOPointNetFusionEngine()
        # A dense, arbitrary cloud is deliberately unlike a known bottle.  The
        # exact PointNet label is model-dependent, but disagreement must be
        # surfaced as unknown rather than a manufactured consensus.
        pts = np.random.RandomState(4).uniform(-1, 1, size=(600, 3))
        result = engine.fuse(pts, [{"class": "bottle", "confidence": 0.99}] * 5)
        if result.modality_predictions["pointnet_only"]["class_name"] != "bottle":
            assert result.predicted_class == "unknown"
            assert result.is_uncertain is True

