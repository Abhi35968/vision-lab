"""
Viewpoint Validator (Phase 5.6).
Audits multi-view silhouette reprojection metrics, investigates interleaved vs contiguous split generalization,
and analyzes camera angular coverage relative to the reconstructed object.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np
import pandas as pd


@dataclass
class ViewpointGroupMetrics:
    """Quantitative performance metrics for a specific viewpoint split group."""
    group_name: str
    camera_ids: List[int]
    num_views: int
    mean_iou: float
    median_iou: float
    std_iou: float
    min_iou: float
    max_iou: float
    mean_precision: float
    mean_recall: float
    angular_span_deg: float


@dataclass
class SilhouetteAuditResult:
    """Comprehensive multi-view silhouette audit findings."""
    total_keyframes: int
    registered_cameras_count: int
    unregistered_keyframes_count: int
    empty_mask_cameras_count: int
    valid_reconstruction_views_count: int
    total_angular_span_deg: float
    interleaved_train: ViewpointGroupMetrics
    interleaved_val: ViewpointGroupMetrics
    interleaved_gap_iou: float
    contiguous_arc_a: ViewpointGroupMetrics
    contiguous_arc_b: ViewpointGroupMetrics
    contiguous_gap_iou: float
    per_view_records: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class ViewpointValidator:
    """Audits silhouette evaluations and conducts contiguous viewpoint partitioning."""

    def __init__(
        self,
        cameras_json_path: Path,
        silhouette_metrics_csv: Path,
        params_json_path: Path,
    ):
        self.cameras_json_path = Path(cameras_json_path)
        self.silhouette_metrics_csv = Path(silhouette_metrics_csv)
        self.params_json_path = Path(params_json_path)

        if not self.cameras_json_path.exists():
            raise FileNotFoundError(f"Cameras JSON not found: {self.cameras_json_path}")
        if not self.silhouette_metrics_csv.exists():
            raise FileNotFoundError(f"Silhouette metrics CSV not found: {self.silhouette_metrics_csv}")
        if not self.params_json_path.exists():
            raise FileNotFoundError(f"Parameters JSON not found: {self.params_json_path}")

    def audit(self) -> SilhouetteAuditResult:
        """Executes full silhouette and viewpoint audit."""
        with open(self.cameras_json_path, "r", encoding="utf-8") as f:
            cams_data = json.load(f)
        with open(self.params_json_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        metrics_df = pd.read_csv(self.silhouette_metrics_csv)

        # 1. Trajectory and Angle Computation
        cap_center = np.asarray(params["cap_center_world"], dtype=np.float64)
        total_height = float(params["total_height"])
        bottle_center = cap_center + np.array([0.0, total_height / 2.0, 0.0])

        reg_cams = {int(k): v for k, v in cams_data.items() if v.get("is_registered", False)}
        total_keyframes = 47
        registered_count = len(reg_cams)
        unregistered_count = total_keyframes - registered_count

        # Compute azimuths relative to bottle center
        cam_azimuths: Dict[int, float] = {}
        for cid, c_data in reg_cams.items():
            R = np.array(c_data["rotation_matrix"], dtype=np.float64)
            t = np.array(c_data["translation_vector"], dtype=np.float64).reshape(3, 1)
            C = -R.T @ t
            vec = (C.flatten() - bottle_center)
            azimuth = float(np.degrees(np.arctan2(vec[2], vec[0])))
            cam_azimuths[cid] = azimuth

        # 2. View Status Audit
        valid_cids = sorted(metrics_df["camera_id"].tolist())
        valid_count = len(valid_cids)
        empty_mask_cids = sorted([cid for cid in reg_cams.keys() if cid not in valid_cids])
        empty_mask_count = len(empty_mask_cids)

        valid_azimuths = [cam_azimuths[cid] for cid in valid_cids]
        total_span = float(np.ptp(valid_azimuths))

        # 3. Interleaved Split Audit (Original Reported Split)
        train_df = metrics_df[metrics_df["is_train_split"] == True]
        val_df = metrics_df[metrics_df["is_train_split"] == False]

        interleaved_train = self._compute_group_metrics(
            "Interleaved Train", train_df, cam_azimuths
        )
        interleaved_val = self._compute_group_metrics(
            "Interleaved Validation", val_df, cam_azimuths
        )
        interleaved_gap = float(interleaved_train.mean_iou - interleaved_val.mean_iou)

        # 4. Viewpoint-Grouped Contiguous Split (14 Early Views vs 14 Late Views)
        # Partition 28 valid views into 2 contiguous halves
        n_half = valid_count // 2
        arc_a_df = metrics_df.iloc[:n_half]
        arc_b_df = metrics_df.iloc[n_half:]

        contiguous_arc_a = self._compute_group_metrics(
            "Contiguous Arc A (Early Views 00-13)", arc_a_df, cam_azimuths
        )
        contiguous_arc_b = self._compute_group_metrics(
            "Contiguous Arc B (Late Views 14-31)", arc_b_df, cam_azimuths
        )
        contiguous_gap = float(contiguous_arc_a.mean_iou - contiguous_arc_b.mean_iou)

        # Per-view enriched records
        records: List[Dict[str, Any]] = []
        for _, row in metrics_df.iterrows():
            cid = int(row["camera_id"])
            is_arc_a = cid in contiguous_arc_a.camera_ids
            records.append({
                "camera_id": cid,
                "image_name": row["image_name"],
                "azimuth_deg": round(cam_azimuths.get(cid, 0.0), 2),
                "is_interleaved_train": bool(row["is_train_split"]),
                "is_contiguous_arc_a": bool(is_arc_a),
                "iou": float(row["iou"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "observed_mask_px": int(row["observed_mask_px"]),
                "rendered_mask_px": int(row["rendered_mask_px"]),
            })

        return SilhouetteAuditResult(
            total_keyframes=total_keyframes,
            registered_cameras_count=registered_count,
            unregistered_keyframes_count=unregistered_count,
            empty_mask_cameras_count=empty_mask_count,
            valid_reconstruction_views_count=valid_count,
            total_angular_span_deg=total_span,
            interleaved_train=interleaved_train,
            interleaved_val=interleaved_val,
            interleaved_gap_iou=interleaved_gap,
            contiguous_arc_a=contiguous_arc_a,
            contiguous_arc_b=contiguous_arc_b,
            contiguous_gap_iou=contiguous_gap,
            per_view_records=records,
        )

    @staticmethod
    def _compute_group_metrics(
        name: str, group_df: pd.DataFrame, azimuths: Dict[int, float]
    ) -> ViewpointGroupMetrics:
        cids = sorted(group_df["camera_id"].tolist())
        ious = group_df["iou"].values
        prec = group_df["precision"].values
        rec = group_df["recall"].values

        group_az = [azimuths[c] for c in cids if c in azimuths]
        span = float(np.ptp(group_az)) if len(group_az) > 1 else 0.0

        return ViewpointGroupMetrics(
            group_name=name,
            camera_ids=cids,
            num_views=len(group_df),
            mean_iou=float(np.mean(ious)),
            median_iou=float(np.median(ious)),
            std_iou=float(np.std(ious)),
            min_iou=float(np.min(ious)),
            max_iou=float(np.max(ious)),
            mean_precision=float(np.mean(prec)),
            mean_recall=float(np.mean(rec)),
            angular_span_deg=span,
        )

