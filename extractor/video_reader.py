from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
import cv2


@dataclass
class VideoFrame:
    """Represents a single video frame with metadata."""
    frame_index: int
    timestamp: float
    image: np.ndarray  # RGB


class BaseVideoReader(ABC):
    """Abstract base class for video readers."""

    @abstractmethod
    def read(self, video_path: str) -> List[VideoFrame]:
        """Read video and return list of VideoFrame objects."""
        pass
        

class OpenCVVideoReader(BaseVideoReader):
    """Video reader implementation using OpenCV."""

    def __init__(self, frame_stride: int = 10, resize_to: Optional[Tuple[int, int]] = None, max_frames: Optional[int] = None):
        """
        Initialize the OpenCV video reader.

        Args:
            frame_stride: Extract every Nth frame.
            resize_to: Resize frames to this size (width, height). None to keep original.
            max_frames: Maximum number of frames to extract. None for all.
        """
        self.frame_stride = frame_stride
        self.resize_to = resize_to
        self.max_frames = max_frames

    def read(self, video_path: str) -> List[VideoFrame]:
        """
        Read video frames using OpenCV.

        Args:
            video_path: Path to the video file.

        Returns:
            List of VideoFrame objects.

        Raises:
            ValueError: If video cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % self.frame_stride == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize if needed
                if self.resize_to:
                    rgb_frame = cv2.resize(rgb_frame, self.resize_to)
                # Compute timestamp
                timestamp = frame_index / fps
                vf = VideoFrame(frame_index=frame_index, timestamp=timestamp, image=rgb_frame)
                frames.append(vf)

                if self.max_frames and len(frames) >= self.max_frames:
                    break

            frame_index += 1

        cap.release()
        return frames