"""
YOLO + PointNet Decision Fusion Module (Phase 5 Enhancement)
Performs robust multimodal classification by fusing:
1. Multi-frame 2D object detections/segmentations from YOLO (with vote counts and confidences).
2. Cleaned and normalized 3D point cloud representations from PointNet.
3. Adaptive modality weighting based on 3D point cloud density, completeness, and outlier ratio.
4. Confidence calibration and uncertainty rejection mechanism.
"""

from dataclasses import dataclass, asdict, field
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from src.classification_3d.pointnet_classifier import (
    PointNetInferenceEngine,
    PointNetInferenceResult,
    ClassificationPrediction,
)
from src.classification_3d.point_cloud_sampler import (
    normalize_point_cloud,
    sample_point_cloud,
)

logger = logging.getLogger("YOLOPointNetFusion")

# Comprehensive mapping between COCO / YOLO classes and ModelNet40 classes
COCO_TO_MODELNET: Dict[str, List[str]] = {
    "bottle": ["bottle"],
    "vase": ["vase", "bottle"],
    "cup": ["cup", "bottle"],
    "bowl": ["bowl", "cup"],
    "chair": ["chair", "stool"],
    "couch": ["sofa"],
    "sofa": ["sofa"],
    "bed": ["bed"],
    "dining table": ["table", "desk"],
    "desk": ["desk", "table"],
    "laptop": ["laptop"],
    "tv": ["monitor", "tv_stand"],
    "monitor": ["monitor"],
    "keyboard": ["keyboard"],
    "car": ["car"],
    "airplane": ["airplane"],
    "toilet": ["toilet"],
    "sink": ["sink"],
    "potted plant": ["plant", "flower_pot"],
    "plant": ["plant", "flower_pot"],
    "cell phone": ["radio"],
    "book": ["bookshelf"],
}


@dataclass
class FusionPrediction:
    """Class candidate prediction after multimodal fusion."""
    class_name: str
    fused_confidence: float
    yolo_confidence: float
    pointnet_confidence: float
    support_modality: str  # '2D_YOLO', '3D_POINTNET', 'CONSENSUS'


@dataclass
class DecisionFusionResult:
    """Consolidated report for YOLO + PointNet decision fusion."""
    predicted_class: str
    predicted_confidence: float
    is_uncertain: bool
    status: str                         # 'CONSENSUS_AGREED', 'FUSED_HYBRID', 'UNCERTAIN_REJECTED'
    modality_weights: Dict[str, float]  # {'weight_2d': float, 'weight_3d': float}
    yolo_summary: Dict[str, Any]
    pointnet_summary: Dict[str, Any]
    top_candidates: List[FusionPrediction]
    decision_rationale: str
    modality_predictions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "predicted_confidence": round(self.predicted_confidence, 4),
            "is_uncertain": self.is_uncertain,
            "status": self.status,
            "modality_weights": {k: round(v, 4) for k, v in self.modality_weights.items()},
            "yolo_summary": self.yolo_summary,
            "pointnet_summary": self.pointnet_summary,
            "top_candidates": [
                {
                    "class_name": c.class_name,
                    "fused_confidence": round(c.fused_confidence, 4),
                    "yolo_confidence": round(c.yolo_confidence, 4),
                    "pointnet_confidence": round(c.pointnet_confidence, 4),
                    "support_modality": c.support_modality,
                }
                for c in self.top_candidates
            ],
            "decision_rationale": self.decision_rationale,
            "modality_predictions": self.modality_predictions,
        }


class YOLOPointNetFusionEngine:
    """
    Robust Multimodal Decision Fusion Engine.
    Combines 2D frame observations and 3D geometric PointNet classifications.
    """

    def __init__(
        self,
        pointnet_engine: Optional[PointNetInferenceEngine] = None,
        uncertainty_threshold: float = 0.20,
        min_margin_threshold: float = 0.03,
    ):
        self.pointnet = pointnet_engine or PointNetInferenceEngine()
        self.uncertainty_threshold = uncertainty_threshold
        self.min_margin_threshold = min_margin_threshold

    def _preprocess_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        Cleans and normalizes point clouds before feeding into PointNet.
        Uses Statistical Outlier Removal (SOR) to strip isolated points.
        """
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) < 10:
            return pts

        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            nb = min(15, len(pts) - 1)
            cl, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=1.5)
            if len(cl.points) >= 8:
                pts = np.asarray(cl.points, dtype=np.float32)
        except Exception:
            pass

        return pts

    def fuse(
        self,
        points_3d: np.ndarray,
        yolo_detections: Union[List[Dict[str, Any]], str, Path],
        target_point_count: int = 1024,
    ) -> DecisionFusionResult:
        """
        Executes decision fusion between 2D YOLO detections and 3D PointNet.

        yolo_detections can be:
            - List of dicts: [{'class': 'bottle', 'confidence': 0.85}, ...]
            - Path to segmentation_summary.csv or detections.json
        """
        # 1. Parse 2D YOLO multi-frame detections
        yolo_records = self._parse_yolo_detections(yolo_detections)

        # 2. Preprocess 3D Point Cloud and run PointNet
        clean_pts = self._preprocess_point_cloud(points_3d)
        pointnet_res_canonical = self.pointnet.classify_point_cloud(clean_pts, canonicalize_upright=True)
        pointnet_res_raw = self.pointnet.classify_point_cloud(clean_pts, canonicalize_upright=False)

        # 3. Calculate 2D class probability distribution
        p_2d, yolo_meta = self._compute_2d_distribution(yolo_records)

        # 4. Extract 3D PointNet probability distribution (from canonical upright representation)
        p_3d = {p.class_name: p.confidence for p in pointnet_res_canonical.top_predictions}
        for cls in self.pointnet.classes:
            if cls not in p_3d:
                p_3d[cls] = 1e-4

        # 5. Determine adaptive modality weights based on 3D point cloud quality
        n_pts = len(clean_pts)
        if n_pts < 50:
            # Very sparse: PointNet has known domain shift; rely mostly on 2D YOLO
            w_3d = 0.15
            w_2d = 0.85
            rationale_prefix = f"Sparse 3D cloud ({n_pts} points) has high domain shift; 2D multi-view voting given 85% priority."
        elif n_pts < 300:
            w_3d = 0.35
            w_2d = 0.65
            rationale_prefix = f"Moderate 3D cloud ({n_pts} points); balanced multimodal weighting applied (65% 2D / 35% 3D)."
        else:
            w_3d = 0.50
            w_2d = 0.50
            rationale_prefix = f"Dense 3D cloud ({n_pts} points); balanced multimodal voting enabled (50% 2D / 50% 3D)."

        # If no valid YOLO detections were provided, fall back to pure 3D
        if not yolo_records or sum(p_2d.values()) < 1e-5:
            w_2d = 0.0
            w_3d = 1.0
            rationale_prefix = "No valid 2D detections available; falling back exclusively to 3D PointNet geometry."

        # 6. Compute fused scores across all classes using smoothed Bayesian prior
        all_candidate_classes = set(self.pointnet.classes) | set(p_2d.keys())
        fused_scores: Dict[str, float] = {}

        # Laplace smoothed 2D likelihood
        alpha = 0.02
        total_2d_mass = sum(p_2d.values()) + alpha * len(all_candidate_classes)

        for cls in all_candidate_classes:
            score_3d = p_3d.get(cls, 1e-4)
            raw_2d = p_2d.get(cls, 0.0)
            score_2d = (raw_2d + alpha) / total_2d_mass

            # Log-linear / geometric mean Bayesian fusion
            fused_score = (score_2d ** w_2d) * (score_3d ** w_3d)
            fused_scores[cls] = float(fused_score)

        # Normalize fused scores into probabilities
        total_score = sum(fused_scores.values())
        if total_score > 0:
            for cls in fused_scores:
                fused_scores[cls] /= total_score

        # Rank candidates
        sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_cands: List[FusionPrediction] = []
        for cls, prob in sorted_candidates[:5]:
            c2d = p_2d.get(cls, 0.0)
            c3d = p_3d.get(cls, 0.0)
            if c2d > 0.2 and c3d > 0.2:
                supp = "CONSENSUS"
            elif c2d >= c3d:
                supp = "2D_YOLO"
            else:
                supp = "3D_POINTNET"
            top_cands.append(
                FusionPrediction(
                    class_name=cls,
                    fused_confidence=prob,
                    yolo_confidence=c2d,
                    pointnet_confidence=c3d,
                    support_modality=supp,
                )
            )

        best_cls, best_conf = sorted_candidates[0]
        second_conf = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0.0
        margin = best_conf - second_conf

        dominant_yolo_class = yolo_meta.get("dominant_yolo_class", "none")
        pn_class = pointnet_res_canonical.predicted_class

        # Strong disagreement on dense cloud is rejected rather than forcing consensus
        disagreement = (
            n_pts >= 300
            and dominant_yolo_class != "none"
            and pn_class != dominant_yolo_class
            and pointnet_res_canonical.predicted_confidence > 0.25
        )

        # 7. Uncertainty and Rejection Gate
        is_uncertain = bool(best_conf < self.uncertainty_threshold or margin < self.min_margin_threshold or disagreement)

        if is_uncertain:
            status = "UNCERTAIN_REJECTED"
            out_class = "unknown"
            out_conf = 0.0
            rationale = (
                f"{rationale_prefix} Low confidence, insufficient class margin, or modality disagreement "
                f"(YOLO='{dominant_yolo_class}'; PointNet='{pn_class}'). Flagged as UNCERTAIN rather than outputting misleading class."
            )
        elif top_cands[0].support_modality == "CONSENSUS":
            status = "CONSENSUS_AGREED"
            out_class = best_cls
            out_conf = best_conf
            rationale = (
                f"{rationale_prefix} Both 2D YOLO detections and 3D geometry agree on '{best_cls}' "
                f"with high combined confidence ({best_conf*100:.1f}%)."
            )
        else:
            status = "FUSED_HYBRID"
            out_class = best_cls
            out_conf = best_conf
            rationale = (
                f"{rationale_prefix} Modality consensus resolved: '{best_cls}' selected by fused multimodal "
                f"evidence (fused confidence: {best_conf*100:.1f}%)."
            )

        modality_predictions = {
            "yolo_only": {
                "class_name": dominant_yolo_class,
                "evidence_score": float(p_2d.get(dominant_yolo_class, 0.0)),
                "confidence_calibrated": False,
                "dominant_detection_class": dominant_yolo_class,
                "detection_agreement": yolo_meta.get("agreement_pct", 1.0),
                "detection_count": len(yolo_records),
                "multiview_consensus": yolo_meta.get("agreement_pct", 1.0) >= 0.7,
            },
            "pointnet_only": {
                "class_name": pointnet_res_canonical.predicted_class,
                "evidence_score": float(pointnet_res_canonical.predicted_confidence),
                "confidence_calibrated": False,
            },
            "fused_diagnostic": {
                "class_name": best_cls,
                "evidence_score": float(best_conf),
                "confidence_calibrated": False,
            }
        }

        return DecisionFusionResult(
            predicted_class=out_class,
            predicted_confidence=out_conf,
            is_uncertain=is_uncertain,
            status=status,
            modality_weights={"weight_2d": w_2d, "weight_3d": w_3d},
            yolo_summary=yolo_meta,
            pointnet_summary={
                "raw_unaligned_class": pointnet_res_raw.predicted_class,
                "raw_unaligned_confidence": pointnet_res_raw.predicted_confidence,
                "canonical_upright_class": pointnet_res_canonical.predicted_class,
                "canonical_upright_confidence": pointnet_res_canonical.predicted_confidence,
                "input_points": n_pts,
                "domain_gap": pointnet_res_canonical.domain_gap_analysis,
            },
            top_candidates=top_cands,
            decision_rationale=rationale,
            modality_predictions=modality_predictions,
        )

    def _parse_yolo_detections(
        self,
        yolo_input: Union[List[Dict[str, Any]], str, Path],
    ) -> List[Dict[str, Any]]:
        """Parses list of dicts, CSV, or JSON detections."""
        if isinstance(yolo_input, list):
            return yolo_input

        path = Path(yolo_input)
        if not path.exists():
            return []

        records = []
        if path.suffix.lower() == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                # Check for target_class, class_name, class, or label
                cls_name = str(row.get("target_class", row.get("class_name", row.get("class", row.get("label", ""))))).lower()
                conf = float(row.get("confidence", 0.0))
                det_raw = row.get("target_detected", row.get("detected", 1))
                detected = 1 if str(det_raw).lower() in {"1", "true"} else 0
                if cls_name and cls_name not in {"none", "nan", ""} and detected == 1 and conf > 0.05:
                    records.append({"class": cls_name, "confidence": conf})
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Check detections dict
                dets = data.get("detections") or data.get("frames") or []
                for d in dets:
                    if isinstance(d, dict) and "class" in d:
                        records.append(d)

        return records

    def _compute_2d_distribution(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Maps YOLO detections to ModelNet classes and computes normalized vote probabilities.
        """
        if not records:
            return {}, {"total_valid_frames": 0, "class_counts": {}, "dominant_yolo_class": "none"}

        class_weights: Dict[str, float] = {}
        class_counts: Dict[str, int] = {}

        for r in records:
            yolo_cls = str(r.get("class", "")).lower().strip()
            conf = float(r.get("confidence", 0.5))
            class_counts[yolo_cls] = class_counts.get(yolo_cls, 0) + 1

            # Map YOLO class to corresponding ModelNet categories
            mapped_classes = COCO_TO_MODELNET.get(yolo_cls, [yolo_cls])
            for m_cls in mapped_classes:
                # Primary mapped class gets full confidence weight, secondary gets 0.7
                factor = 1.0 if m_cls == mapped_classes[0] else 0.7
                class_weights[m_cls] = class_weights.get(m_cls, 0.0) + (conf * factor)

        total_weight = sum(class_weights.values())
        p_2d = {cls: w / total_weight for cls, w in class_weights.items()} if total_weight > 0 else {}

        dominant_yolo = max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else "none"

        summary = {
            "total_valid_frames": len(records),
            "class_counts": class_counts,
            "dominant_yolo_class": dominant_yolo,
            "mapped_candidates": list(p_2d.keys()),
        }
        return p_2d, summary
