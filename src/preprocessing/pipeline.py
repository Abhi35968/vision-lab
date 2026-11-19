"""
Preprocessing Pipeline Orchestrator
Coordinates VideoReader, Quality Assessment, Frame Filtering, Keyframe Selection,
and Visual Reporting into a streamlined CPU-efficient workflow.
"""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml

from src.preprocessing.video_reader import VideoReader, VideoMetadata
from src.preprocessing.image_quality import ImageQualityAssessor, ImageQualityMetrics
from src.preprocessing.frame_filter import FrameFilter
from src.preprocessing.keyframe_selector import KeyframeSelector, KeyframeDecision


class PreprocessingPipeline:
    """
    End-to-end video ingestion and keyframe selection engine for Structure from Motion.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[dict] = None,
    ):
        self.config = self._load_config(config_path, config_dict)
        self._init_components()

    def _load_config(
        self,
        config_path: Optional[Union[str, Path]],
        config_dict: Optional[dict],
    ) -> dict:
        """Loads configuration from YAML file or dictionary with comprehensive defaults."""
        default_config = {
            "preprocessing": {
                "video_path": "data/raw_videos/sample_scene.mp4",
                "output_frames_dir": "data/extracted_frames",
                "frame_step": 4,
                "blur_threshold": 100.0,
                "min_brightness": 40.0,
                "max_brightness": 220.0,
                "min_contrast": 20.0,
                "max_image_dimension": 1920,
                "min_time_delta_seconds": 0.20,
                "max_time_gap_seconds": 2.0,
                "export_format": "png",
                "overwrite": True,
            },
            "visualization": {
                "output_dir": "outputs/visualizations",
                "generate_contact_sheet": True,
                "contact_sheet_cols": 5,
            },
            "reporting": {
                "report_path": "outputs/preprocessing_report.json",
            },
        }

        if config_dict:
            # Merge dictionary
            default_config.update(config_dict)
            return default_config

        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    # Update nested keys
                    for section, values in loaded.items():
                        if section in default_config and isinstance(values, dict):
                            default_config[section].update(values)
                        else:
                            default_config[section] = values

        return default_config

    def _init_components(self) -> None:
        """Initializes internal processing components based on active configuration."""
        p_cfg = self.config.get("preprocessing", {})
        self.assessor = ImageQualityAssessor(
            blur_threshold=float(p_cfg.get("blur_threshold", 100.0)),
            min_brightness=float(p_cfg.get("min_brightness", 40.0)),
            max_brightness=float(p_cfg.get("max_brightness", 220.0)),
            min_contrast=float(p_cfg.get("min_contrast", 20.0)),
            min_dimension=480,
        )
        self.frame_filter = FrameFilter(
            max_hamming_distance=3,
        )
        self.selector = KeyframeSelector(
            quality_assessor=self.assessor,
            frame_filter=self.frame_filter,
            min_time_delta_seconds=float(p_cfg.get("min_time_delta_seconds", 0.20)),
            max_time_gap_seconds=float(p_cfg.get("max_time_gap_seconds", 2.0)),
        )

    def _resize_if_needed(self, frame: np.ndarray, max_dim: Optional[int]) -> np.ndarray:
        """Resizes frame maintaining aspect ratio if largest dimension exceeds max_dim."""
        if not max_dim:
            return frame
        h, w = frame.shape[:2]
        if max(h, w) <= max_dim:
            return frame
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def run(
        self,
        video_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        frame_step_override: Optional[int] = None,
        blur_threshold_override: Optional[float] = None,
        max_dim_override: Optional[int] = None,
        show_progress: bool = True,
    ) -> dict:
        """
        Executes the end-to-end preprocessing pipeline.
        
        Returns:
            Dictionary containing comprehensive summary report, telemetry, and output paths.
        """
        p_cfg = self.config.get("preprocessing", {})
        v_path = Path(video_path or p_cfg.get("video_path")).resolve()
        out_frames_dir = Path(output_dir or p_cfg.get("output_frames_dir")).resolve()
        frame_step = frame_step_override or int(p_cfg.get("frame_step", 4))
        max_dim = max_dim_override or p_cfg.get("max_image_dimension", 1920)
        export_fmt = p_cfg.get("export_format", "png").lower().lstrip(".")
        report_path = Path(self.config.get("reporting", {}).get("report_path", "outputs/preprocessing_report.json")).resolve()
        vis_dir = Path(self.config.get("visualization", {}).get("output_dir", "outputs/visualizations")).resolve()

        if blur_threshold_override is not None:
            self.assessor.blur_threshold = blur_threshold_override

        out_frames_dir.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 1. Video Ingestion
        reader = VideoReader(v_path)
        metadata = reader.metadata

        # 2. Sequential Extraction & Keyframe Selection (Streamed)
        evaluated_records: List[KeyframeDecision] = []
        selected_frames_meta: List[dict] = []
        blur_scores_all: List[float] = []
        brightness_all: List[float] = []
        thumbnails_for_contact_sheet: List[Tuple[int, np.ndarray]] = []

        last_selected_time: Optional[float] = None
        last_selected_hash: Optional[int] = None
        keyframe_counter = 0

        total_frames = metadata.total_frames
        pbar = tqdm(
            desc="Preprocessing Video",
            total=total_frames if total_frames > 0 else None,
            disable=not show_progress,
            unit="frame",
        )

        for frame_idx, timestamp_sec, raw_frame in reader.read_frames():
            pbar.update(1)

            # Subsample by step
            if (frame_idx % frame_step) != 0:
                continue

            # Resize if required
            processed_frame = self._resize_if_needed(raw_frame, max_dim)
            h, w = processed_frame.shape[:2]

            # Quality assessment
            metrics = self.assessor.evaluate(processed_frame)
            blur_scores_all.append(metrics.blur_score)
            brightness_all.append(metrics.brightness_mean)

            # Decision logic
            if not metrics.quality_passed:
                reason = "; ".join(metrics.rejection_reasons)
                evaluated_records.append(
                    KeyframeDecision(
                        frame_index=frame_idx,
                        timestamp_seconds=round(timestamp_sec, 3),
                        quality_metrics=metrics.to_dict(),
                        is_selected=False,
                        rejection_reason=f"Quality check failed: {reason}",
                    )
                )
                continue

            current_hash = self.frame_filter.compute_dhash(processed_frame)

            # Anchor frame (unconditional accept for first valid frame)
            if last_selected_time is None or last_selected_hash is None:
                is_selected = True
                rejection_reason = None
                sim_score = 1.0
                hamm_dist = 0
            else:
                time_delta = timestamp_sec - last_selected_time
                if time_delta < self.selector.min_time_delta:
                    is_selected = False
                    rejection_reason = f"Temporal delta {time_delta:.2f}s < {self.selector.min_time_delta:.2f}s"
                    sim_score = None
                    hamm_dist = None
                else:
                    sim_comp = self.frame_filter.compare_hashes(last_selected_hash, current_hash)
                    sim_score = sim_comp.similarity_score
                    hamm_dist = sim_comp.perceptual_distance

                    if sim_comp.near_duplicate and time_delta < self.selector.max_time_gap:
                        is_selected = False
                        rejection_reason = f"Near-duplicate (Hamming dist {hamm_dist} <= {self.frame_filter.max_hamming_distance})"
                    else:
                        is_selected = True
                        rejection_reason = None

            decision = KeyframeDecision(
                frame_index=frame_idx,
                timestamp_seconds=round(timestamp_sec, 3),
                quality_metrics=metrics.to_dict(),
                is_selected=is_selected,
                rejection_reason=rejection_reason,
                similarity_with_last_keyframe=sim_score,
                hamming_distance=hamm_dist,
            )
            evaluated_records.append(decision)

            if is_selected:
                # Save keyframe image to disk
                kf_filename = f"keyframe_{keyframe_counter:04d}_orig_{frame_idx:05d}.{export_fmt}"
                kf_path = out_frames_dir / kf_filename
                
                if export_fmt in {"jpg", "jpeg"}:
                    cv2.imwrite(str(kf_path), processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    cv2.imwrite(str(kf_path), processed_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])

                selected_frames_meta.append({
                    "keyframe_id": keyframe_counter,
                    "original_frame_index": frame_idx,
                    "timestamp_seconds": round(timestamp_sec, 3),
                    "filename": kf_filename,
                    "path": str(kf_path),
                    "width": w,
                    "height": h,
                    "blur_score": metrics.blur_score,
                    "brightness_mean": metrics.brightness_mean,
                })

                # Thumbnail for contact sheet (keep up to 30 thumbnails in memory for visualizer)
                if len(thumbnails_for_contact_sheet) < 30:
                    thumb = cv2.resize(processed_frame, (240, 135), interpolation=cv2.INTER_AREA)
                    thumbnails_for_contact_sheet.append((keyframe_counter, thumb))

                last_selected_time = timestamp_sec
                last_selected_hash = current_hash
                keyframe_counter += 1

        pbar.close()
        reader.release()

        # 3. Generate Visual Analytics
        vis_paths = self._generate_visualizations(
            vis_dir=vis_dir,
            blur_scores=blur_scores_all,
            blur_threshold=self.assessor.blur_threshold,
            brightness_values=brightness_all,
            decisions=evaluated_records,
            thumbnails=thumbnails_for_contact_sheet,
        )

        # 4. Compile Comprehensive Report
        total_evaluated = len(evaluated_records)
        total_selected = len(selected_frames_meta)
        rejection_counts: Dict[str, int] = {}
        for d in evaluated_records:
            if not d.is_selected:
                cat = d.rejection_reason.split(":")[0] if d.rejection_reason else "Other"
                rejection_counts[cat] = rejection_counts.get(cat, 0) + 1

        report = {
            "session_timestamp": datetime.now().isoformat(),
            "source_video": metadata.to_dict(),
            "pipeline_parameters": {
                "frame_step": frame_step,
                "blur_threshold": self.assessor.blur_threshold,
                "min_brightness": self.assessor.min_brightness,
                "max_brightness": self.assessor.max_brightness,
                "min_contrast": self.assessor.min_contrast,
                "max_image_dimension": max_dim,
                "export_format": export_fmt,
            },
            "summary_metrics": {
                "total_video_frames": metadata.total_frames,
                "total_evaluated_frames": total_evaluated,
                "selected_keyframes": total_selected,
                "rejected_frames": total_evaluated - total_selected,
                "selection_rate_percent": round((total_selected / total_evaluated * 100), 2) if total_evaluated > 0 else 0.0,
                "mean_blur_score": round(float(np.mean(blur_scores_all)), 2) if blur_scores_all else 0.0,
                "mean_brightness": round(float(np.mean(brightness_all)), 2) if brightness_all else 0.0,
            },
            "rejection_breakdown": rejection_counts,
            "visualizations": vis_paths,
            "selected_keyframes": selected_frames_meta,
        }

        # Save JSON report
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def _generate_visualizations(
        self,
        vis_dir: Path,
        blur_scores: List[float],
        blur_threshold: float,
        brightness_values: List[float],
        decisions: List[KeyframeDecision],
        thumbnails: List[Tuple[int, np.ndarray]],
    ) -> Dict[str, str]:
        """Generates plots for blur distribution, brightness distribution, selection stats, and contact sheet."""
        paths = {}

        # 1. Blur Score Distribution Plot
        if blur_scores:
            plt.figure(figsize=(8, 4))
            plt.hist(blur_scores, bins=30, color="#2b5c8f", edgecolor="black", alpha=0.7)
            plt.axvline(blur_threshold, color="red", linestyle="--", linewidth=2, label=f"Blur Threshold ({blur_threshold})")
            plt.title("Laplacian Variance (Sharpness) Distribution")
            plt.xlabel("Blur Score (Variance of Laplacian)")
            plt.ylabel("Candidate Frame Count")
            plt.legend()
            plt.tight_layout()
            blur_plot_p = vis_dir / "blur_distribution.png"
            plt.savefig(str(blur_plot_p), dpi=150)
            plt.close()
            paths["blur_distribution"] = str(blur_plot_p)

        # 2. Brightness Distribution Plot
        if brightness_values:
            plt.figure(figsize=(8, 4))
            plt.hist(brightness_values, bins=30, color="#488f5c", edgecolor="black", alpha=0.7)
            plt.axvline(40, color="orange", linestyle="--", label="Underexposed (40)")
            plt.axvline(220, color="purple", linestyle="--", label="Overexposed (220)")
            plt.title("Grayscale Brightness Distribution")
            plt.xlabel("Mean Intensity (0 - 255)")
            plt.ylabel("Candidate Frame Count")
            plt.legend()
            plt.tight_layout()
            bright_plot_p = vis_dir / "brightness_distribution.png"
            plt.savefig(str(bright_plot_p), dpi=150)
            plt.close()
            paths["brightness_distribution"] = str(bright_plot_p)

        # 3. Selection vs Rejection Counts Breakdown
        if decisions:
            selected_count = sum(1 for d in decisions if d.is_selected)
            rejected_count = len(decisions) - selected_count
            plt.figure(figsize=(6, 4))
            bars = plt.bar(["Selected", "Rejected"], [selected_count, rejected_count], color=["#2ca02c", "#d62728"])
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, str(yval), ha="center", va="bottom", fontweight="bold")
            plt.title("Keyframe Selection Outcome")
            plt.ylabel("Count")
            plt.tight_layout()
            sel_plot_p = vis_dir / "selection_summary.png"
            plt.savefig(str(sel_plot_p), dpi=150)
            plt.close()
            paths["selection_summary"] = str(sel_plot_p)

        # 4. Contact Sheet (Grid of Keyframe Thumbnails)
        if thumbnails:
            cols = 5
            rows = int(np.ceil(len(thumbnails) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
            axes = np.array(axes).reshape(-1)

            for i in range(len(axes)):
                if i < len(thumbnails):
                    kf_id, thumb_bgr = thumbnails[i]
                    thumb_rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(thumb_rgb)
                    axes[i].set_title(f"Keyframe {kf_id:03d}", fontsize=9)
                    axes[i].axis("off")
                else:
                    axes[i].axis("off")

            plt.suptitle("Selected Keyframes Contact Sheet", fontsize=14, fontweight="bold")
            plt.tight_layout()
            sheet_p = vis_dir / "keyframes_contact_sheet.png"
            plt.savefig(str(sheet_p), dpi=150)
            plt.close()
            paths["contact_sheet"] = str(sheet_p)

        return paths
