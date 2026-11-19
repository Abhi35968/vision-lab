"""
Final Geometric Evaluation and Reconstruction Validation Module (Phase 5.6).
Performs quantitative mesh integrity auditing, sparse point-to-mesh distance evaluation,
viewpoint validation auditing, and generates final synthesis diagnostics and reports.
"""

from src.final_evaluation.mesh_auditor import MeshAuditor, MeshAuditResult
from src.final_evaluation.point_mesh_evaluator import PointMeshEvaluator, PointTierSummary
from src.final_evaluation.viewpoint_validator import ViewpointValidator, SilhouetteAuditResult
from src.final_evaluation.evaluation_visualizer import EvaluationVisualizer
from src.final_evaluation.evaluation_pipeline import FinalEvaluationPipeline

__all__ = [
    "MeshAuditor",
    "MeshAuditResult",
    "PointMeshEvaluator",
    "PointTierSummary",
    "ViewpointValidator",
    "SilhouetteAuditResult",
    "EvaluationVisualizer",
    "FinalEvaluationPipeline",
]

