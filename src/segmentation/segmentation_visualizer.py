"""
Segmentation Visualizer Module
Produces publication-grade visual diagnostics:
1. Single-frame alpha-blended mask overlays with contour lines and status badges.
2. 4-panel side-by-side progressions (Original, BBoxes, Mask Overlay, Binary Mask).
3. Sequence contact sheet showing temporal mask evolution.
4. Quantitative time-series diagnostic plots (Mask Area over Time, Confidence over Time).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.segmentation.segmentation_models import FrameSegmentation, InstanceMask


class SegmentationVisualizer:
    """Renderer for object segmentation visual inspection and time-series telemetry."""

    # Color palette (BGR for OpenCV)
    COLOR_TARGET = (255, 180, 0)      # Bright Cyan / Blue
    COLOR_CONTOUR = (255, 120, 0)     # Darker Blue
    COLOR_NON_TARGET = (128, 128, 128) # Grey
    COLOR_VALID = (50, 205, 50)       # Lime Green
    COLOR_AMBIGUOUS = (0, 165, 255)   # Orange
    COLOR_NO_TARGET = (0, 0, 220)     # Crimson Red
    COLOR_BORDER_WARN = (0, 69, 255)  # Red-Orange

    @classmethod
    def render_overlay(
        cls,
        image_bgr: np.ndarray,
        target_mask: Optional[np.ndarray],
        frame_seg: FrameSegmentation,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Renders a composite overlay on top of the original image.
        """
        canvas = image_bgr.copy()
        h, w = canvas.shape[:2]

        # Draw target mask overlay if present
        if target_mask is not None and np.any(target_mask > 0):
            # Mask region overlay
            colored_mask = np.zeros_like(canvas)
            colored_mask[target_mask > 0] = cls.COLOR_TARGET

            # Alpha blend inside mask
            mask_bool = target_mask > 0
            canvas[mask_bool] = cv2.addWeighted(
                canvas[mask_bool], 1.0 - alpha,
                colored_mask[mask_bool], alpha, 0
            )

            # Draw crisp contour boundary
            contours, _ = cv2.findContours(
                (target_mask > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(canvas, contours, -1, cls.COLOR_CONTOUR, thickness=3)

            # Draw centroid marker
            inst = frame_seg.target_instance
            if inst:
                cx, cy = int(inst.centroid[0]), int(inst.centroid[1])
                cv2.drawMarker(
                    canvas, (cx, cy), cls.COLOR_CONTOUR,
                    markerType=cv2.MARKER_CROSS, markerSize=25, thickness=3
                )

        # Draw target bounding box and label
        inst = frame_seg.target_instance
        if inst is not None:
            x1, y1, x2, y2 = [int(round(c)) for c in inst.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), cls.COLOR_TARGET, thickness=3)

            label = f"[TARGET] {inst.class_name} ({inst.confidence:.2f})"
            if inst.touches_border:
                label += " [BORDER TOUCH]"

            # Label banner
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            y_label = max(lh + 10, y1 - 10)
            cv2.rectangle(
                canvas,
                (x1, y_label - lh - 8),
                (x1 + lw + 10, y_label + 4),
                cls.COLOR_TARGET,
                cv2.FILLED,
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 5, y_label - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Draw top status banner
        status_color = (
            cls.COLOR_VALID if frame_seg.target_status == "VALID"
            else cls.COLOR_AMBIGUOUS if frame_seg.target_status == "AMBIGUOUS"
            else cls.COLOR_NO_TARGET
        )
        sfm_str = "SfM REG" if frame_seg.is_registered_sfm else "UNREGISTERED"
        status_text = f"Frame {frame_seg.frame_idx:02d} | Status: {frame_seg.target_status} | {sfm_str}"

        (sw, sh), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(canvas, (20, 20), (30 + sw, 35 + sh), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(canvas, (20, 20), (30 + sw, 35 + sh), status_color, 2)
        cv2.putText(
            canvas,
            status_text,
            (25, 25 + sh),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2,
            cv2.LINE_AA,
        )

        return canvas

    @classmethod
    def render_bbox_view(
        cls,
        image_bgr: np.ndarray,
        frame_seg: FrameSegmentation,
    ) -> np.ndarray:
        """Renders bounding boxes for all detected instances."""
        canvas = image_bgr.copy()
        for inst in frame_seg.all_instances:
            x1, y1, x2, y2 = [int(round(c)) for c in inst.bbox]
            is_tgt = (frame_seg.target_instance and inst.instance_id == frame_seg.target_instance.instance_id)
            color = cls.COLOR_TARGET if is_tgt else cls.COLOR_NON_TARGET
            thick = 3 if is_tgt else 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=thick)

            prefix = "[TARGET] " if is_tgt else ""
            txt = f"{prefix}{inst.class_name}: {inst.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, cv2.FILLED)
            cv2.putText(
                canvas, txt, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
            )
        return canvas

    @classmethod
    def render_side_by_side(
        cls,
        image_bgr: np.ndarray,
        target_mask: Optional[np.ndarray],
        frame_seg: FrameSegmentation,
        target_height: int = 720,
    ) -> np.ndarray:
        """
        Creates a 4-panel progression:
        [1: Original] [2: Bounding Boxes] [3: Segmentation Overlay] [4: Binary Mask]
        """
        h_orig, w_orig = image_bgr.shape[:2]
        scale = target_height / float(h_orig)
        target_width = int(round(w_orig * scale))

        # Panel 1: Original
        p1 = cv2.resize(image_bgr, (target_width, target_height))

        # Panel 2: BBoxes
        bbox_view = cls.render_bbox_view(image_bgr, frame_seg)
        p2 = cv2.resize(bbox_view, (target_width, target_height))

        # Panel 3: Mask Overlay
        overlay_view = cls.render_overlay(image_bgr, target_mask, frame_seg)
        p3 = cv2.resize(overlay_view, (target_width, target_height))

        # Panel 4: Binary Mask
        if target_mask is not None and np.any(target_mask > 0):
            p4_gray = target_mask
        else:
            p4_gray = np.zeros((h_orig, w_orig), dtype=np.uint8)
        p4_bgr = cv2.cvtColor(p4_gray, cv2.COLOR_GRAY2BGR)
        p4 = cv2.resize(p4_bgr, (target_width, target_height))

        # Add top title banners to each panel
        panels = [
            (p1, "1. Original Frame"),
            (p2, "2. Bounding Boxes"),
            (p3, "3. Mask Overlay"),
            (p4, "4. Binary Mask (1080x1920)"),
        ]
        titled_panels = []
        for panel, title in panels:
            p_annot = panel.copy()
            cv2.rectangle(p_annot, (0, 0), (target_width, 36), (20, 20, 20), cv2.FILLED)
            cv2.putText(
                p_annot, title, (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )
            titled_panels.append(p_annot)

        # Concatenate horizontally
        side_by_side = np.hstack(titled_panels)
        return side_by_side

    @classmethod
    def render_contact_sheet(
        cls,
        frames: List[FrameSegmentation],
        target_masks: Dict[int, np.ndarray],
        output_path: Union[str, Path],
        cols: int = 8,
        thumb_height: int = 240,
    ):
        """
        Renders a grid overview of all keyframe mask overlays.
        """
        n_frames = len(frames)
        if n_frames == 0:
            return

        rows = int(np.ceil(n_frames / float(cols)))
        sample_img = cv2.imread(frames[0].image_path)
        h_orig, w_orig = sample_img.shape[:2] if sample_img is not None else (1920, 1080)
        scale = thumb_height / float(h_orig)
        thumb_width = int(round(w_orig * scale))

        sheet = np.zeros((rows * thumb_height, cols * thumb_width, 3), dtype=np.uint8)

        for i, f in enumerate(frames):
            r = i // cols
            c = i % cols
            y0 = r * thumb_height
            x0 = c * thumb_width

            img = cv2.imread(f.image_path)
            if img is None:
                img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)

            t_mask = target_masks.get(f.frame_idx)
            overlay = cls.render_overlay(img, t_mask, f)
            thumb = cv2.resize(overlay, (thumb_width, thumb_height))

            # Border indicating status
            border_color = (
                cls.COLOR_VALID if f.target_status == "VALID"
                else cls.COLOR_AMBIGUOUS if f.target_status == "AMBIGUOUS"
                else cls.COLOR_NO_TARGET
            )
            cv2.rectangle(thumb, (0, 0), (thumb_width - 1, thumb_height - 1), border_color, thickness=2)

            # Mini label
            cv2.rectangle(thumb, (0, thumb_height - 24), (thumb_width, thumb_height), (0, 0, 0), cv2.FILLED)
            lbl = f"KF {f.frame_idx:02d} | {f.target_status}"
            cv2.putText(
                thumb, lbl, (5, thumb_height - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1, cv2.LINE_AA
            )

            sheet[y0:y0 + thumb_height, x0:x0 + thumb_width] = thumb

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), sheet)
        print(f"[INFO] Contact sheet saved to: {out_p}")

    @classmethod
    def plot_time_series_metrics(
        cls,
        frames: List[FrameSegmentation],
        output_dir: Union[str, Path],
    ):
        """
        Generates publication-quality diagnostic plots:
        1. mask_area_over_time.png
        2. confidence_over_time.png
        """
        out_dir = Path(output_dir)
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        indices = [f.frame_idx for f in frames]
        is_reg = [f.is_registered_sfm for f in frames]
        statuses = [f.target_status for f in frames]
        areas_px = [f.target_instance.mask_area_px if f.target_instance else 0 for f in frames]
        areas_pct = [f.target_instance.mask_area_pct if f.target_instance else 0.0 for f in frames]
        confs = [f.target_instance.confidence if f.target_instance else 0.0 for f in frames]
        classes = [f.target_instance.class_name if f.target_instance else "none" for f in frames]

        # -------------------------------------------------------------
        # Plot 1: Mask Area Over Time
        # -------------------------------------------------------------
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Shading for SfM registered range
        reg_indices = [idx for idx, reg in zip(indices, is_reg) if reg]
        if reg_indices:
            ax1.axvspan(
                min(reg_indices) - 0.5, max(reg_indices) + 0.5,
                color="lavender", alpha=0.5, label="SfM Registered Range (0-31)"
            )

        color_px = "tab:blue"
        ax1.set_xlabel("Keyframe Index", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Target Mask Area (Pixels)", color=color_px, fontsize=12, fontweight="bold")
        line1 = ax1.plot(indices, areas_px, color=color_px, linestyle="--", alpha=0.6, label="Area (px)")

        # Scatter points colored by status
        for idx, area, status, reg in zip(indices, areas_px, statuses, is_reg):
            if status == "VALID":
                c = "forestgreen" if reg else "royalblue"
                m = "o" if reg else "s"
            elif status == "AMBIGUOUS":
                c = "darkorange"
                m = "^"
            else:
                c = "crimson"
                m = "x"
            ax1.scatter(idx, area, color=c, marker=m, s=60, zorder=5)

        ax2 = ax1.twinx()
        color_pct = "tab:purple"
        ax2.set_ylabel("Frame Area Coverage (%)", color=color_pct, fontsize=12, fontweight="bold")
        line2 = ax2.plot(indices, areas_pct, color=color_pct, linewidth=2, label="Area (%)")

        # Custom legend elements
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="tab:blue", linestyle="--", label="Area (px)"),
            Line2D([0], [0], color="tab:purple", linewidth=2, label="Area (%)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="forestgreen", markersize=9, label="Valid & SfM Registered"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="royalblue", markersize=8, label="Valid (Unregistered)"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="darkorange", markersize=9, label="Ambiguous"),
            Line2D([0], [0], marker="x", color="crimson", markersize=9, label="No Target / Lost"),
        ]
        ax1.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

        plt.title("Target Bottle 2D Mask Area Across Keyframe Sequence", fontsize=14, fontweight="bold", pad=15)
        ax1.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        area_plot_path = plots_dir / "mask_area_over_time.png"
        plt.savefig(str(area_plot_path), dpi=300)
        plt.close()
        print(f"[INFO] Mask area time-series plot saved: {area_plot_path}")

        # -------------------------------------------------------------
        # Plot 2: Detection Confidence and Class Classification Over Time
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 6))

        if reg_indices:
            ax.axvspan(
                min(reg_indices) - 0.5, max(reg_indices) + 0.5,
                color="lavender", alpha=0.5, label="SfM Registered Range (0-31)"
            )

        ax.axhline(0.25, color="gray", linestyle=":", label="Default YOLO Conf (0.25)")
        ax.plot(indices, confs, color="black", linestyle="-", linewidth=1.5, alpha=0.5)

        # Plot points grouped by semantic class
        class_colors = {"bottle": "blue", "vase": "green", "cup": "orange", "none": "red"}
        for idx, conf, cls_name, status in zip(indices, confs, classes, statuses):
            c = class_colors.get(cls_name, "purple")
            m = "o" if status == "VALID" else "x"
            ax.scatter(idx, conf, color=c, marker=m, s=65, zorder=5)

        ax.set_xlabel("Keyframe Index", fontsize=12, fontweight="bold")
        ax.set_ylabel("Detection Confidence Score", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)

        cls_legend = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=9, label="COCO: bottle"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=9, label="COCO: vase (Container Alias)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=9, label="COCO: cup (Container Alias)"),
            Line2D([0], [0], marker="x", color="red", markersize=9, label="No Detection / Lost"),
        ]
        ax.legend(handles=cls_legend, loc="upper right", framealpha=0.9)

        plt.title("Instance Detection Confidence & Class Aliasing Across Keyframe Sequence", fontsize=14, fontweight="bold", pad=15)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        conf_plot_path = plots_dir / "confidence_over_time.png"
        plt.savefig(str(conf_plot_path), dpi=300)
        plt.close()
        print(f"[INFO] Confidence time-series plot saved: {conf_plot_path}")

