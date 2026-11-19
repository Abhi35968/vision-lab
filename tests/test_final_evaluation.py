"""
Unit Tests for Phase 5.6: Final Geometric Evaluation and Reconstruction Validation.
"""

from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd

from src.final_evaluation.mesh_auditor import MeshAuditor, MeshAuditResult
from src.final_evaluation.point_mesh_evaluator import PointMeshEvaluator, PointTierSummary
from src.final_evaluation.viewpoint_validator import ViewpointValidator, SilhouetteAuditResult
from src.final_evaluation.evaluation_pipeline import FinalEvaluationPipeline


@pytest.fixture
def reconstruction_paths():
    return {
        "ply": Path("outputs/object_reconstruction/bottle_mesh.ply"),
        "obj": Path("outputs/object_reconstruction/bottle_mesh.obj"),
        "params": Path("outputs/object_reconstruction/reconstruction_parameters.json"),
        "metrics": Path("outputs/object_reconstruction/per_view_silhouette_metrics.csv"),
        "refinement": Path("outputs/object_refinement/refined_object_points.csv"),
        "cameras": Path("outputs/real_video_sfm/focal_1300/cameras.json"),
        "eval_dir": Path("outputs/final_evaluation"),
    }


def test_mesh_auditor_ply_topology(reconstruction_paths):
    """Verifies that the reconstructed mesh is a watertight 2-manifold disk/sphere."""
    auditor = MeshAuditor()
    res = auditor.audit(reconstruction_paths["ply"], reconstruction_paths["obj"])

    assert isinstance(res, MeshAuditResult)
    assert res.num_vertices == 2162
    assert res.num_faces == 4320
    assert res.num_edges_unique == 6480
    assert res.num_connected_components == 1
    assert res.is_watertight is True
    assert res.is_edge_manifold is True
    assert res.num_boundary_edges == 0
    assert res.num_non_manifold_edges == 0
    assert res.num_degenerate_faces == 0
    assert res.min_face_area > 0.001
    assert res.num_duplicate_vertices == 0
    assert res.num_duplicate_faces == 0
    assert res.euler_characteristic == 2
    assert res.genus == 0
    assert res.enclosed_volume > 20.0


def test_mesh_format_equivalence(reconstruction_paths):
    """Verifies numerical equivalence between Stanford PLY and Wavefront OBJ exports."""
    auditor = MeshAuditor(tolerance=1e-5)
    res = auditor.audit(reconstruction_paths["ply"], reconstruction_paths["obj"])

    assert res.ply_obj_equivalent is True
    assert res.ply_obj_max_vertex_diff < 1e-5


def test_point_to_mesh_evaluator_statistics(reconstruction_paths):
    """Verifies point-to-surface distance computation and monotonic tier ordering."""
    evaluator = PointMeshEvaluator(reconstruction_paths["ply"])
    df, summaries = evaluator.evaluate_points(reconstruction_paths["refinement"])

    assert len(df) == 47
    assert "distance_to_mesh" in df.columns
    assert "closest_mesh_x" in df.columns

    # Verify counts
    assert summaries["high_confidence"].count == 14
    assert summaries["moderate_confidence"].count == 15
    assert summaries["low_confidence"].count == 13
    assert summaries["geometric_outliers"].count == 5
    assert summaries["combined_validated"].count == 29

    high = summaries["high_confidence"]
    mod = summaries["moderate_confidence"]
    low = summaries["low_confidence"]
    out = summaries["geometric_outliers"]

    # Monotonic ordering of mean distances: High < Mod < Low < Outliers
    assert high.mean_distance < mod.mean_distance
    assert mod.mean_distance < low.mean_distance
    assert low.mean_distance < out.mean_distance

    # Monotonic ordering of median distances
    assert high.median_distance < mod.median_distance
    assert mod.median_distance < low.median_distance
    assert low.median_distance < out.median_distance

    # Cap-anchored high-confidence points must have tight median distance
    assert high.median_distance < 0.20
    assert high.mean_distance < 0.40


def test_viewpoint_validator_splits(reconstruction_paths):
    """Verifies silhouette validation audit and interleaved vs contiguous splits."""
    validator = ViewpointValidator(
        cameras_json_path=reconstruction_paths["cameras"],
        silhouette_metrics_csv=reconstruction_paths["metrics"],
        params_json_path=reconstruction_paths["params"],
    )
    audit = validator.audit()

    assert isinstance(audit, SilhouetteAuditResult)
    assert audit.total_keyframes == 47
    assert audit.registered_cameras_count == 32
    assert audit.unregistered_keyframes_count == 15
    assert audit.empty_mask_cameras_count == 4
    assert audit.valid_reconstruction_views_count == 28

    # Angular span should be approximately 57 degrees
    assert 56.0 <= audit.total_angular_span_deg <= 58.0

    # Interleaved Split (50/50)
    assert audit.interleaved_train.num_views == 14
    assert audit.interleaved_val.num_views == 14
    assert audit.interleaved_train.mean_iou > 0.55
    assert audit.interleaved_val.mean_iou > 0.50
    assert 0.02 <= audit.interleaved_gap_iou <= 0.06

    # Contiguous Split (Arc A vs Arc B)
    assert audit.contiguous_arc_a.num_views == 14
    assert audit.contiguous_arc_b.num_views == 14
    assert audit.contiguous_arc_a.mean_iou > 0.55
    assert audit.contiguous_arc_b.mean_iou > 0.50
    assert audit.contiguous_gap_iou > audit.interleaved_gap_iou


def test_evaluation_pipeline_execution(reconstruction_paths):
    """Verifies end-to-end execution of Phase 5.6 pipeline and deliverable persistence."""
    pipeline = FinalEvaluationPipeline(
        reconstruction_dir=reconstruction_paths["ply"].parent,
        refinement_csv=reconstruction_paths["refinement"],
        sfm_dir=reconstruction_paths["cameras"].parent,
        output_dir=reconstruction_paths["eval_dir"],
    )
    result = pipeline.run()

    eval_dir = reconstruction_paths["eval_dir"]
    assert (eval_dir / "mesh_integrity_audit.json").exists()
    assert (eval_dir / "point_to_mesh_distances.csv").exists()
    assert (eval_dir / "point_to_mesh_summary.json").exists()
    assert (eval_dir / "silhouette_viewpoint_audit.json").exists()
    assert (eval_dir / "phase_5_6_evaluation_report.md").exists()

    expected_figures = [
        "mesh_integrity_visualization.png",
        "point_to_mesh_distance_visualization.png",
        "camera_viewpoint_validation.png",
        "silhouette_validation_comparison.png",
        "final_reconstruction_overview.png",
    ]
    for fig_name in expected_figures:
        fig_p = eval_dir / fig_name
        assert fig_p.exists(), f"Figure {fig_name} missing"
        assert fig_p.stat().st_size > 50000, f"Figure {fig_name} is too small (<50KB)"

