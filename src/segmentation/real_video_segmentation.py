"""
Real Video Object Segmentation Pipeline
Orchestrates Phase 5.3.1: Pretrained instance segmentation on real smartphone keyframes,
tracking the tabletop water bottle, and exporting full-resolution binary masks,
overlays, contact sheets, and telemetry.
"""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Union
import cv2
import numpy as np

from src.segmentation.segmentation_models import FrameSegmentation, SegmentationSummary
from src.segmentation.segmenter import YOLOSegmenter, SegmenterConfig
from src.segmentation.segmentation_visualizer import SegmentationVisualizer


class RealVideoSegmentationPipeline:
    """End-to-end runner for Phase 5.3.1 real-world keyframe segmentation."""

    def __init__(
        self,
        keyframes_dir: Union[str, Path] = "outputs/real_video_validation/keyframes",
        sfm_dir: Union[str, Path] = "outputs/real_video_sfm/focal_1300",
        output_dir: Union[str, Path] = "outputs/object_segmentation",
        model_weights: str = "yolo11n-seg.pt",
        conf_threshold: float = 0.15,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        self.keyframes_dir = Path(keyframes_dir)
        self.sfm_dir = Path(sfm_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = SegmenterConfig(
            model_weights=model_weights,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            device=device,
            output_dir=str(self.output_dir),
        )
        self.segmenter = YOLOSegmenter(config=self.config)

    def load_registered_cameras(self) -> Set[str]:
        """Loads the registered camera image names from the validated SfM cameras.json."""
        cams_json = self.sfm_dir / "cameras.json"
        registered_names: Set[str] = set()

        if not cams_json.exists():
            print(f"[WARNING] SfM cameras.json not found at '{cams_json}'. Assuming all frames registered.")
            return registered_names

        try:
            with open(cams_json, "r", encoding="utf-8") as f:
                cams_data = json.load(f)

            for cam_id_str, cam_info in cams_data.items():
                if cam_info.get("is_registered", False):
                    img_name = cam_info.get("image_name")
                    if img_name:
                        registered_names.add(img_name)

            print(f"[INFO] Loaded {len(registered_names)} registered camera views from SfM: {cams_json}")
        except Exception as e:
            print(f"[WARNING] Error reading SfM cameras.json: {e}. Proceeding without SfM mask filtering.")

        return registered_names

    def run(self) -> Dict[str, Any]:
        """Executes the complete segmentation pipeline."""
        print("\n" + "=" * 75)
        print(" PHASE 5.3.1: PRETRAINED OBJECT SEGMENTATION ON REAL VIDEO KEYFRAMES")
        print("=" * 75)
        print(f"Keyframes Directory : {self.keyframes_dir}")
        print(f"SfM Reference Dir   : {self.sfm_dir}")
        print(f"Output Directory    : {self.output_dir}")
        print(f"Model Architecture  : {self.config.model_weights} (Ultralytics YOLO11n-Seg, CPU)")
        print(f"Confidence Threshold: {self.config.conf_threshold}")
        print("=" * 75)

        # 1. Discover keyframes
        keyframe_paths = sorted(
            list(self.keyframes_dir.glob("*.png")) + list(self.keyframes_dir.glob("*.jpg"))
        )
        if not keyframe_paths:
            raise FileNotFoundError(f"No keyframe images found in '{self.keyframes_dir}'.")

        print(f"[INFO] Found {len(keyframe_paths)} keyframes to process.")

        # 2. Load registered cameras
        registered_names = self.load_registered_cameras()

        # 3. Create subdirectories
        masks_dir = self.output_dir / "masks"
        overlays_dir = self.output_dir / "overlays"
        side_by_side_dir = self.output_dir / "side_by_side"
        plots_dir = self.output_dir / "plots"

        masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)
        side_by_side_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 4. Run Sequence Segmentation
        frames, summary, target_masks = self.segmenter.segment_sequence(
            image_paths=keyframe_paths,
            registered_camera_names=registered_names,
            output_dir=self.output_dir,
        )

        # 5. Render overlays and side-by-side comparisons
        print("\n[INFO] Rendering visual overlays and side-by-side comparisons...")
        for f in frames:
            img = cv2.imread(f.image_path)
            t_mask = target_masks.get(f.frame_idx)

            # Single overlay
            overlay = SegmentationVisualizer.render_overlay(img, t_mask, f)
            overlay_path = overlays_dir / f"{Path(f.image_name).stem}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)
            f.overlay_path = str(overlay_path)

            # 4-panel progression
            sbs = SegmentationVisualizer.render_side_by_side(img, t_mask, f, target_height=720)
            sbs_path = side_by_side_dir / f"{Path(f.image_name).stem}_side_by_side.png"
            cv2.imwrite(str(sbs_path), sbs)
            f.side_by_side_path = str(sbs_path)

        # 6. Render Contact Sheet
        contact_sheet_path = self.output_dir / "contact_sheet.png"
        SegmentationVisualizer.render_contact_sheet(
            frames=frames,
            target_masks=target_masks,
            output_path=contact_sheet_path,
            cols=8,
            thumb_height=240,
        )

        # 7. Generate Diagnostic Time-Series Plots
        print("[INFO] Generating diagnostic time-series plots...")
        SegmentationVisualizer.plot_time_series_metrics(
            frames=frames,
            output_dir=self.output_dir,
        )

        # 8. Export Tabular Summary CSV
        csv_path = self.output_dir / "segmentation_summary.csv"
        csv_rows = [f.to_csv_row() for f in frames]
        if csv_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"[INFO] Tabular summary CSV exported to: {csv_path}")

        # 9. Export Comprehensive JSON Telemetry
        json_path = self.output_dir / "segmentation_metadata.json"
        metadata = {
            "summary": summary.to_dict(),
            "frames": [f.to_dict() for f in frames],
        }
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(metadata, f_json, indent=2)
        print(f"[INFO] Master segmentation JSON exported to: {json_path}")

        # Print final console summary
        print("\n" + "=" * 75)
        print(" REAL-WORLD KEYFRAME OBJECT SEGMENTATION SUMMARY")
        print("=" * 75)
        print(f"Total Keyframes Evaluated     : {summary.total_frames}")
        print(f"SfM Registered Cameras        : {summary.sfm_registered_frames}")
        print(f"Valid Target Mask Frames      : {summary.valid_target_frames} / {summary.total_frames} ({summary.to_dict()['target_coverage_overall_pct']}%)")
        print(f"SfM Registered Target Frames  : {summary.sfm_valid_target_frames} / {summary.sfm_registered_frames} ({summary.to_dict()['target_coverage_sfm_pct']}%)")
        print(f"Ambiguous Target Frames       : {summary.ambiguous_frames}")
        print(f"No Target Detected Frames     : {summary.no_target_frames}")
        print(f"Border Touching Masks         : {summary.border_touching_count}")
        print(f"Mean Target Area Coverage     : {summary.mean_target_mask_area_pct:.2f}% of frame ({int(summary.mean_target_mask_area_pct/100*1080*1920):,} px)")
        print(f"Mean Target Confidence        : {summary.mean_target_confidence:.4f}")
        print(f"Mean Inference Speed (CPU)    : {summary.mean_inference_time_ms:.1f} ms/frame")
        print(f"Target Semantic Class Counts  : {summary.target_classes_detected}")
        print("-" * 75)
        print(f"Saved Binary Masks to         : {masks_dir}/")
        print(f"Saved Overlays to             : {overlays_dir}/")
        print(f"Saved 4-Panel Progressions to : {side_by_side_dir}/")
        print(f"Saved Contact Sheet to        : {contact_sheet_path}")
        print(f"Saved Diagnostic Plots to     : {plots_dir}/")
        print(f"Saved Tabular Summary to      : {csv_path}")
        print(f"Saved JSON Telemetry to       : {json_path}")
        print("=" * 75 + "\n")

        return metadata

