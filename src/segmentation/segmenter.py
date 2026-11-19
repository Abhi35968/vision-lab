"""
Pretrained YOLO Instance Segmenter Module
Provides CPU inference using Ultralytics YOLO segmentation models,
extracts pixel-accurate masks at native resolution (1080 x 1920),
and integrates with InstanceTracker for temporal multi-view target tracking.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO

from src.segmentation.segmentation_models import (
    InstanceMask,
    FrameSegmentation,
    SegmentationSummary,
)
from src.segmentation.mask_processor import MaskProcessor
from src.segmentation.instance_tracker import InstanceTracker, InstanceTrackerConfig


@dataclass
class SegmenterConfig:
    """Configuration settings for the instance segmentation engine."""
    model_weights: str = "yolo11n-seg.pt"
    conf_threshold: float = 0.15
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str = "cpu"
    retina_masks: bool = True
    output_dir: str = "outputs/object_segmentation"
    tracker_config: InstanceTrackerConfig = field(default_factory=InstanceTrackerConfig)


class YOLOSegmenter:
    """
    High-level instance segmentation engine wrapping Ultralytics YOLO-seg on CPU.
    Produces binary target masks and frame-level tracking diagnostics.
    """

    def __init__(self, config: Optional[SegmenterConfig] = None):
        self.config = config or SegmenterConfig()
        self.tracker = InstanceTracker(config=self.config.tracker_config)
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """Loads and prepares the pretrained YOLO segmentation model on CPU."""
        weights_path = Path(self.config.model_weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{weights_path}'. Please ensure the pretrained weights are present."
            )

        print(f"[INFO] Loading YOLO segmentation model: {weights_path} on device='{self.config.device}'...")
        model = YOLO(str(weights_path))

        # Perform a warm-up dry run on a dummy canvas to eliminate JIT / initial allocation overhead
        dummy_img = np.zeros((384, 640, 3), dtype=np.uint8)
        _ = model.predict(
            source=dummy_img,
            device=self.config.device,
            verbose=False,
            imgsz=self.config.imgsz,
        )
        print("[INFO] Segmentation model successfully loaded and warmed up.")
        return model

    def segment_frame(
        self,
        image_input: Union[str, Path, np.ndarray],
        camera_id: int,
        frame_idx: int,
        image_name: Optional[str] = None,
        is_registered_sfm: bool = True,
        is_excluded: bool = False,
    ) -> Tuple[FrameSegmentation, Optional[np.ndarray]]:
        """
        Runs instance segmentation and target tracking on a single image.

        Args:
            image_input: File path or loaded BGR numpy array.
            camera_id: Sequential or SfM camera ID.
            frame_idx: Original sequence keyframe index (0..46).
            image_name: Filename string.
            is_registered_sfm: True if registered in Phase 5.2 SfM.
            is_excluded: True if frame is excluded from reconstruction.

        Returns:
            Tuple of:
              - FrameSegmentation record
              - target_mask_binary: uint8 array of shape (H, W) in {0, 255}, or None
        """
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            img_name = image_name or img_path.name
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to read image at '{img_path}'.")
        else:
            img = image_input
            img_name = image_name or f"frame_{frame_idx:04d}.png"
            img_path = Path(img_name)

        h_orig, w_orig = img.shape[:2]
        total_pixels = h_orig * w_orig

        # Inference timing
        t0 = time.perf_counter()
        results = self.model.predict(
            source=img,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.config.device,
            retina_masks=self.config.retina_masks,
            verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        all_instances: List[InstanceMask] = []
        instance_masks_raw: List[np.ndarray] = []

        if results.boxes is not None and len(results.boxes) > 0 and results.masks is not None:
            boxes = results.boxes
            masks_data = results.masks.data.cpu().numpy()  # (N, H, W)
            names = results.names

            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls[0].item())
                cls_name = str(names.get(cls_id, f"class_{cls_id}"))
                conf = float(box.conf[0].item())
                xyxy = [float(c.item()) for c in box.xyxy[0]]

                # Extract and process binary mask
                raw_m = masks_data[i]
                bin_mask = MaskProcessor.resize_to_target_resolution(
                    raw_mask=raw_m,
                    target_shape=(h_orig, w_orig),
                    threshold=0.5,
                )

                # Mask diagnostics
                mask_px = int(np.sum(bin_mask > 0))
                area_pct = (mask_px / total_pixels) * 100.0
                touches_border = MaskProcessor.check_touches_border(bin_mask, margin=2)
                num_comp, largest_ratio = MaskProcessor.analyze_connected_components(bin_mask)
                centroid = MaskProcessor.compute_mask_centroid(bin_mask)

                # Boundary erosion for 2D-to-3D association robustness
                eroded_m, erode_meta = MaskProcessor.erode_mask(
                    bin_mask, kernel_size=3, shape="ellipse", frame_id=frame_idx, object_id=i
                )

                inst = InstanceMask(
                    instance_id=i,
                    camera_id=camera_id,
                    image_filename=img_name,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    mask_area_px=mask_px,
                    mask_area_pct=round(area_pct, 3),
                    centroid=centroid,
                    touches_border=touches_border,
                    num_components=num_comp,
                    largest_component_ratio=largest_ratio,
                    is_target=False,
                    is_ambiguous=False,
                    erosion_kernel=erode_meta["erosion_kernel"],
                    eroded_mask_area_px=erode_meta["eroded_area_px"],
                    mask_metadata=erode_meta,
                )
                all_instances.append(inst)
                instance_masks_raw.append(eroded_m)

        # Spatiotemporal target tracking
        target_inst, status, updated_instances = self.tracker.track_frame(
            frame_idx=frame_idx,
            all_instances=all_instances,
            frame_shape=(h_orig, w_orig),
            is_registered_sfm=is_registered_sfm,
            is_excluded=is_excluded,
        )

        target_mask_binary: Optional[np.ndarray] = None
        if target_inst is not None and target_inst.instance_id < len(instance_masks_raw):
            target_mask_binary = instance_masks_raw[target_inst.instance_id]

        frame_seg = FrameSegmentation(
            frame_idx=frame_idx,
            image_name=img_name,
            image_path=str(img_path),
            image_width=w_orig,
            image_height=h_orig,
            is_registered_sfm=is_registered_sfm,
            target_detected=(target_inst is not None and status == "VALID"),
            target_status=status,
            target_instance=target_inst,
            all_instances=updated_instances,
            inference_time_ms=elapsed_ms,
        )

        return frame_seg, target_mask_binary

    def segment_sequence(
        self,
        image_paths: List[Union[str, Path]],
        registered_camera_names: Optional[Set[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[List[FrameSegmentation], SegmentationSummary, Dict[int, np.ndarray]]:
        """
        Runs instance segmentation on an ordered sequence of keyframes, tracking
        the target bottle and exporting binary masks.

        Args:
            image_paths: List of file paths to sequential keyframes.
            registered_camera_names: Optional set of filenames registered in Phase 5.2 SfM.
            output_dir: Directory where binary masks will be saved.

        Returns:
            Tuple of:
              - frame_segmentations: List[FrameSegmentation]
              - summary: SegmentationSummary
              - target_masks: Dict mapping frame_idx -> binary uint8 mask (1080x1920)
        """
        self.tracker.reset()
        out_path = Path(output_dir or self.config.output_dir)
        masks_dir = out_path / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        reg_names = registered_camera_names or set()
        frame_results: List[FrameSegmentation] = []
        target_masks: Dict[int, np.ndarray] = {}

        total_frames = len(image_paths)
        print(f"[INFO] Beginning sequence segmentation across {total_frames} keyframes...")
        t_seq_start = time.perf_counter()

        for idx, p in enumerate(image_paths):
            p_obj = Path(p)
            fname = p_obj.name
            is_reg = (fname in reg_names) if reg_names else True

            frame_seg, target_m = self.segment_frame(
                image_input=p_obj,
                camera_id=idx,
                frame_idx=idx,
                image_name=fname,
                is_registered_sfm=is_reg,
            )

            # Save full resolution 1080x1920 binary mask
            mask_filename = f"{p_obj.stem}_mask.png"
            mask_save_path = masks_dir / mask_filename

            if target_m is not None and frame_seg.target_status == "VALID":
                cv2.imwrite(str(mask_save_path), target_m)
                frame_seg.mask_path = str(mask_save_path)
                target_masks[idx] = target_m
            else:
                # Save empty black mask if target is not detected/ambiguous
                empty_mask = np.zeros((frame_seg.image_height, frame_seg.image_width), dtype=np.uint8)
                cv2.imwrite(str(mask_save_path), empty_mask)
                frame_seg.mask_path = str(mask_save_path)
                target_masks[idx] = empty_mask

            frame_results.append(frame_seg)

            status_str = f"[{frame_seg.target_status}]"
            class_str = frame_seg.target_instance.class_name if frame_seg.target_instance else "-"
            conf_str = f"{frame_seg.target_instance.confidence:.2f}" if frame_seg.target_instance else "-"
            area_str = f"{frame_seg.target_instance.mask_area_pct:.1f}%" if frame_seg.target_instance else "-"
            print(
                f"  [{idx+1:02d}/{total_frames:02d}] {fname} | Status: {status_str:<11} | "
                f"Class: {class_str:<7} (c={conf_str}) | Area: {area_str:<6} | {frame_seg.inference_time_ms:.1f}ms"
            )

        total_time_s = time.perf_counter() - t_seq_start
        print(f"[INFO] Sequence segmentation completed in {total_time_s:.2f}s ({total_time_s/total_frames*1000.0:.1f} ms/frame).")

        # Compile quantitative summary
        valid_frames = [f for f in frame_results if f.target_status == "VALID"]
        sfm_valid_frames = [f for f in valid_frames if f.is_registered_sfm]
        sfm_reg_frames = [f for f in frame_results if f.is_registered_sfm]
        ambig_frames = [f for f in frame_results if f.target_status == "AMBIGUOUS"]
        no_target_frames = [f for f in frame_results if f.target_status == "NO_TARGET"]
        excluded_frames = [f for f in frame_results if f.target_status == "EXCLUDED"]

        class_counts: Dict[str, int] = {}
        border_count = 0
        areas = []
        confs = []

        for vf in valid_frames:
            inst = vf.target_instance
            if inst:
                class_counts[inst.class_name] = class_counts.get(inst.class_name, 0) + 1
                if inst.touches_border:
                    border_count += 1
                areas.append(inst.mask_area_pct)
                confs.append(inst.confidence)

        mean_area = float(np.mean(areas)) if areas else 0.0
        mean_conf = float(np.mean(confs)) if confs else 0.0
        mean_inf_time = float(np.mean([f.inference_time_ms for f in frame_results])) if frame_results else 0.0

        summary = SegmentationSummary(
            total_frames=total_frames,
            sfm_registered_frames=len(sfm_reg_frames),
            valid_target_frames=len(valid_frames),
            sfm_valid_target_frames=len(sfm_valid_frames),
            ambiguous_frames=len(ambig_frames),
            no_target_frames=len(no_target_frames),
            excluded_frames=len(excluded_frames),
            mean_inference_time_ms=mean_inf_time,
            mean_target_mask_area_pct=mean_area,
            mean_target_confidence=mean_conf,
            border_touching_count=border_count,
            target_classes_detected=class_counts,
        )

        return frame_results, summary, target_masks

