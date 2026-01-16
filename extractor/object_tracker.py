from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .sam_segmenter import FrameSegmentationResult, SegmentationMask


@dataclass
class TrackedObject:
    """Represents a tracked object across frames."""
    object_id: str
    start_time: float
    end_time: float
    masks: List[np.ndarray]
    bboxes: List[Tuple[int, int, int, int]]
    frame_indices: List[int]
    timestamps: List[float]

    def get_representative_index(self) -> int:
        """Returns the index of the frame with the largest mask area."""
        if not self.masks:
            return 0
        areas = [np.sum(mask) for mask in self.masks]
        return int(np.argmax(areas))


@dataclass
class ObjectTrackResult:
    """Result of object tracking."""
    objects: List[TrackedObject]


class ObjectTracker:
    """Tracks objects across frames using IoU-based matching."""

    def __init__(self, iou_match_threshold: float = 0.5, max_missing_frames: int = 3, min_frames_per_object: int = 5):
        """
        Initialize the object tracker.

        Args:
            iou_match_threshold: Minimum IoU to consider a match.
            max_missing_frames: Maximum frames an object can be missing before closing.
            min_frames_per_object: Minimum frames an object must appear in to be kept.
        """
        self.iou_match_threshold = iou_match_threshold
        self.max_missing_frames = max_missing_frames
        self.min_frames_per_object = min_frames_per_object

        # Active objects: list of dicts with 'object', 'last_mask', 'missing_count'
        self.active_objects = []
        self.next_object_id = 1

    def track(self, results: List[FrameSegmentationResult]) -> ObjectTrackResult:
        """
        Track objects across the sequence of frame results.

        Args:
            results: List of FrameSegmentationResult in chronological order.

        Returns:
            ObjectTrackResult with tracked objects.
        """
        # Sort results by frame_index to ensure order
        results = sorted(results, key=lambda r: r.frame_index)

        all_objects = []

        for frame_result in results:
            self._process_frame(frame_result)

        # Close all remaining active objects
        for active in self.active_objects:
            obj = active['object']
            obj.end_time = obj.timestamps[-1] if obj.timestamps else obj.start_time
            if len(obj.frame_indices) >= self.min_frames_per_object:
                all_objects.append(obj)

        return ObjectTrackResult(objects=all_objects)

    def _process_frame(self, frame_result: FrameSegmentationResult):
        """Process a single frame's segmentation results."""
        current_masks = frame_result.masks
        assigned = [False] * len(current_masks)

        # Try to match current masks to active objects
        for i, active in enumerate(self.active_objects):
            best_iou = -1
            best_mask_idx = -1
            for j, mask in enumerate(current_masks):
                if assigned[j]:
                    continue
                iou = self._compute_iou(active['last_mask'], mask.mask)
                if iou > best_iou:
                    best_iou = iou
                    best_mask_idx = j

            if best_iou >= self.iou_match_threshold:
                # Assign
                obj = active['object']
                mask = current_masks[best_mask_idx]
                obj.masks.append(mask.mask)
                obj.bboxes.append(mask.bbox)
                obj.frame_indices.append(frame_result.frame_index)
                obj.timestamps.append(frame_result.timestamp)
                obj.end_time = frame_result.timestamp
                active['last_mask'] = mask.mask
                active['missing_count'] = 0
                assigned[best_mask_idx] = True
            else:
                # No match, increment missing count
                active['missing_count'] += 1

        # Remove closed objects
        self.active_objects = [a for a in self.active_objects if a['missing_count'] <= self.max_missing_frames]

        # Create new objects for unassigned masks
        for j, mask in enumerate(current_masks):
            if not assigned[j]:
                obj_id = f"obj_{self.next_object_id:04d}"
                self.next_object_id += 1
                obj = TrackedObject(
                    object_id=obj_id,
                    start_time=frame_result.timestamp,
                    end_time=frame_result.timestamp,
                    masks=[mask.mask],
                    bboxes=[mask.bbox],
                    frame_indices=[frame_result.frame_index],
                    timestamps=[frame_result.timestamp]
                )
                self.active_objects.append({
                    'object': obj,
                    'last_mask': mask.mask,
                    'missing_count': 0
                })

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute IoU between two binary masks.

        Args:
            mask1: First binary mask.
            mask2: Second binary mask.

        Returns:
            IoU value between 0 and 1.
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union