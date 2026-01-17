from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
import cv2
import yaml
import torch
# import clip
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


@dataclass
class SegmentationMask:
    """Represents a single segmentation mask with metadata."""
    mask: np.ndarray  # Binary mask, HxW, dtype bool
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    area: int
    confidence: Optional[float] = None


@dataclass
class FrameSegmentationResult:
    """Represents segmentation results for a single frame."""
    frame_index: int
    timestamp: float
    masks: List[SegmentationMask]


class BaseSegmenter(ABC):
    """Abstract base class for segmenters."""

    @abstractmethod
    def segment(self, image: np.ndarray, frame_index: int, timestamp: float, prompt: Optional[str] = None) -> FrameSegmentationResult:
        """Segment the image and return results.

        Args:
            image: Input image as numpy array (HxWx3, RGB).
            frame_index: Index of the frame.
            timestamp: Timestamp of the frame.
            prompt: Optional natural-language prompt to guide segmentation.

        Returns:
            FrameSegmentationResult with masks.
        """
        pass


class SAMSegmenter(BaseSegmenter):
    """SAM-based segmenter using Segment Anything Model."""

    def __init__(self, model_type: str, checkpoint_path: str, device: str = "cuda", config_path: Optional[str] = None):
        """
        Initialize the SAM segmenter.

        Args:
            model_type: Type of SAM model (e.g., 'vit_h', 'vit_l', 'vit_b').
            checkpoint_path: Path to the SAM model checkpoint.
            device: Device to run the model on ('cuda' or 'cpu').
            config_path: Optional path to YAML config file with parameters.
        """
        self.device = device
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path

        # Load SAM model
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.sam = sam

        # Load config if provided
        config = {}
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        # Set default parameters
        self.min_mask_area = config.get('min_mask_area', 100)
        self.points_per_side = config.get('points_per_side', 32)
        self.pred_iou_thresh = config.get('pred_iou_thresh', 0.88)
        self.stability_score_thresh = config.get('stability_score_thresh', 0.95)

        # Initialize AutomaticMaskGenerator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_area,
        )

        # Load CLIP model for prompt handling
        # self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    def segment(self, image: np.ndarray, frame_index: int, timestamp: float, prompt: Optional[str] = None) -> FrameSegmentationResult:
        """
        Segment the image using SAM.

        If prompt is provided, use CLIP to filter masks that match the prompt intent.
        Otherwise, run full automatic segmentation.

        Args:
            image: Input image as numpy array (HxWx3, RGB).
            frame_index: Index of the frame.
            timestamp: Timestamp of the frame.
            prompt: Optional natural-language prompt to guide segmentation.

        Returns:
            FrameSegmentationResult with filtered masks.
        """
        # Run automatic segmentation
        masks = self.mask_generator.generate(image)

        segmentation_masks = []
        for mask_data in masks:
            mask = mask_data['segmentation'].astype(bool)  # Ensure binary
            area = int(np.sum(mask))
            if area < self.min_mask_area:
                continue

            # Compute bounding box
            coords = np.argwhere(mask)
            if len(coords) == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = (int(x_min), int(y_min), int(x_max), int(y_max))

            confidence = mask_data.get('predicted_iou', None)

            seg_mask = SegmentationMask(
                mask=mask.astype(bool),
                bbox=bbox,
                area=area,
                confidence=confidence
            )
            segmentation_masks.append(seg_mask)

       # Prompt-based filtering will be added later (CLIP)
# For now, return all SAM masks
        return FrameSegmentationResult(
            frame_index=frame_index,
            timestamp=timestamp,
            masks=segmentation_masks
        )

    # def _filter_masks_with_prompt(self, image: np.ndarray, masks: List[SegmentationMask], prompt: str) -> List[SegmentationMask]:
        """
        Filter masks based on the text prompt using CLIP similarity.

        For each mask, extract the masked region, compute CLIP embedding, and compare with prompt embedding.
        Keep masks with high similarity.

        Args:
            image: Original image.
            masks: List of SegmentationMask.
            prompt: Text prompt.

        Returns:
            Filtered list of SegmentationMask.
        """
        if not masks:
            return masks

        # Preprocess prompt
        text_input = clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        filtered_masks = []
        for seg_mask in masks:
            # Extract masked region
            masked_image = image * seg_mask.mask[:, :, np.newaxis]  # Apply mask to image

            # Preprocess for CLIP
            pil_image = Image.fromarray(masked_image.astype(np.uint8))
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

            # Compute image features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (text_features @ image_features.T).item()

            # Threshold for filtering (adjust as needed)
            if similarity > 0.2:  # Example threshold
                filtered_masks.append(seg_mask)

        return filtered_masks