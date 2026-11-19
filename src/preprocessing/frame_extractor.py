"""
Frame Extractor Module
Handles subsampling, resizing, and persistent storage of video frames with rich metadata.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union
import cv2
import numpy as np
from tqdm import tqdm

from src.preprocessing.video_reader import VideoReader


@dataclass
class ExtractedFrameInfo:
    """Metadata record for an extracted image frame."""
    extracted_index: int
    original_frame_index: int
    timestamp_seconds: float
    filename: str
    file_path: str
    width: int
    height: int
    original_width: int
    original_height: int
    downsampled: bool

    def to_dict(self) -> dict:
        return asdict(self)


class FrameExtractor:
    """
    Subsamples and saves video frames to disk with resolution capping and metadata tracking.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        method: str = "frame_step",  # "frame_step" or "time_interval"
        frame_step: int = 4,
        time_interval: float = 0.5,
        max_dimension: Optional[int] = 1920,
        image_format: str = "png",
        overwrite: bool = False,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.method = method
        self.frame_step = max(1, frame_step)
        self.time_interval = max(0.01, time_interval)
        self.max_dimension = max_dimension
        self.image_format = image_format.lower().lstrip(".")
        self.overwrite = overwrite

        if self.image_format not in {"png", "jpg", "jpeg"}:
            raise ValueError(f"Unsupported image format: {self.image_format}. Use 'png' or 'jpg'.")

    def _should_extract(self, frame_idx: int, timestamp_sec: float, last_extracted_time: float) -> bool:
        """Determines if the current frame matches the extraction schedule."""
        if self.method == "frame_step":
            return (frame_idx % self.frame_step) == 0
        elif self.method == "time_interval":
            return (timestamp_sec - last_extracted_time) >= self.time_interval
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")

    def _resize_if_needed(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Downscales frame if its largest dimension exceeds max_dimension."""
        if self.max_dimension is None:
            return frame, False

        h, w = frame.shape[:2]
        largest_dim = max(h, w)

        if largest_dim <= self.max_dimension:
            return frame, False

        scale = self.max_dimension / float(largest_dim)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # Use INTER_AREA for high-quality downsampling without aliasing
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, True

    def extract_from_reader(
        self,
        reader: VideoReader,
        show_progress: bool = True,
    ) -> List[ExtractedFrameInfo]:
        """
        Extracts frames sequentially from a VideoReader instance and writes them to output_dir.
        
        Returns:
            List of ExtractedFrameInfo records describing all saved frames.
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check existing files if overwrite is disabled
        existing_frames = list(self.output_dir.glob(f"*.{self.image_format}"))
        if existing_frames and not self.overwrite:
            raise FileExistsError(
                f"Output directory '{self.output_dir}' already contains {len(existing_frames)} "
                f"'.{self.image_format}' frames. Set overwrite=True to replace them."
            )

        metadata = reader.metadata
        total_expected = metadata.total_frames
        
        # Determine progress bar total
        pbar_total = total_expected if total_expected > 0 else None
        pbar = tqdm(
            desc=f"Extracting frames ({self.method})",
            total=pbar_total,
            disable=not show_progress,
            unit="frame",
        )

        extracted_records: List[ExtractedFrameInfo] = []
        extracted_count = 0
        last_extracted_time = -1.0

        for frame_idx, timestamp_sec, raw_frame in reader.read_frames():
            pbar.update(1)

            if not self._should_extract(frame_idx, timestamp_sec, last_extracted_time):
                continue

            orig_h, orig_w = raw_frame.shape[:2]
            processed_frame, is_downscaled = self._resize_if_needed(raw_frame)
            proc_h, proc_w = processed_frame.shape[:2]

            filename = f"frame_{extracted_count:04d}_orig_{frame_idx:05d}.{self.image_format}"
            file_path = self.output_dir / filename

            # Write frame to disk
            if self.image_format in {"jpg", "jpeg"}:
                cv2.imwrite(str(file_path), processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(file_path), processed_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            record = ExtractedFrameInfo(
                extracted_index=extracted_count,
                original_frame_index=frame_idx,
                timestamp_seconds=round(timestamp_sec, 4),
                filename=filename,
                file_path=str(file_path),
                width=proc_w,
                height=proc_h,
                original_width=orig_w,
                original_height=orig_h,
                downsampled=is_downscaled,
            )
            extracted_records.append(record)
            extracted_count += 1
            last_extracted_time = timestamp_sec

        pbar.close()

        # Save extraction manifest metadata
        manifest_path = self.output_dir / "extraction_metadata.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_video": metadata.to_dict(),
                    "extraction_config": {
                        "method": self.method,
                        "frame_step": self.frame_step,
                        "time_interval_seconds": self.time_interval,
                        "max_dimension": self.max_dimension,
                        "image_format": self.image_format,
                    },
                    "total_extracted": len(extracted_records),
                    "frames": [r.to_dict() for r in extracted_records],
                },
                f,
                indent=2,
            )

        return extracted_records

