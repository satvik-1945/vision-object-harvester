import cv2
from extractor.sam_segmenter import SAMSegmenter

image = cv2.imread("demo/sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

segmenter = SAMSegmenter(
    model_type="vit_b",
    checkpoint_path="checkpoints/sam_vit_b_01ec64.pth",
    device="cpu"
)

result = segmenter.segment(
    image=image,
    frame_index=0,
    timestamp=0.0,
    prompt="saree"
)

print(len(result.masks))