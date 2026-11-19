"""
Video Reader Module
Handles robust video ingestion, metadata extraction, and sequential frame streaming.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Structured container for video file metadata."""
    file_path: str
    file_name: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    fourcc: str
    is_valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


class VideoReader:
    """
    A robust video stream reader optimized for sequential frame access.
    
    Supports context management, metadata extraction, corrupted frame recovery,
    and automatic resource cleanup.
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path).resolve()
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._validate_file()

    def _validate_file(self) -> None:
        """Verifies file existence and format before opening."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if not self.video_path.is_file():
            raise ValueError(f"Provided path is not a file: {self.video_path}")
        if self.video_path.stat().st_size == 0:
            raise ValueError(f"Video file is empty (0 bytes): {self.video_path}")
        if self.video_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video extension '{self.video_path.suffix}'. "
                f"Supported: {sorted(list(self.SUPPORTED_EXTENSIONS))}"
            )

    def open(self) -> "VideoReader":
        """Initializes the OpenCV VideoCapture stream."""
        if self._cap is not None and self._cap.isOpened():
            return self

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(
                f"OpenCV failed to open video stream: {self.video_path}. "
                "The file may be corrupted or missing appropriate codec support."
            )
        self._extract_metadata()
        return self

    def _extract_metadata(self) -> None:
        """Extracts geometric, temporal, and codec metadata from the video stream."""
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Cannot extract metadata: VideoCapture is not open.")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Handle potential zero or negative FPS / frame counts from unindexed streams
        if fps <= 0.0:
            fps = 30.0  # Safe fallback for standard phone video

        duration = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0

        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_chars = [chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]
        fourcc_str = "".join(c for c in fourcc_chars if c.isprintable()).strip() or "UNKNOWN"

        is_valid = width > 0 and height > 0 and total_frames >= 0

        self._metadata = VideoMetadata(
            file_path=str(self.video_path),
            file_name=self.video_path.name,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            fourcc=fourcc_str,
            is_valid=is_valid,
        )

    @property
    def metadata(self) -> VideoMetadata:
        """Returns the extracted video metadata. Opens stream if necessary."""
        if self._metadata is None:
            self.open()
        return self._metadata  # type: ignore

    def read_frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Sequentially reads frames from the video.
        
        Yields:
            Tuple of (frame_index: int, timestamp_seconds: float, frame: np.ndarray)
        """
        if self._cap is None or not self._cap.isOpened():
            self.open()

        assert self._cap is not None
        fps = self.metadata.fps
        frame_idx = 0

        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break

            # Calculate timestamp in seconds
            timestamp_sec = frame_idx / fps if fps > 0 else 0.0

            yield frame_idx, timestamp_sec, frame
            frame_idx += 1

    def release(self) -> None:
        """Releases the underlying OpenCV VideoCapture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

