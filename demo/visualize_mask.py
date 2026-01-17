import cv2
import numpy as np
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
    timestamp=0.0
)

vis = image.copy()

for mask in result.masks:
    color = np.random.randint(0, 255, size=3)
    vis[mask.mask] = vis[mask.mask] * 0.4 + color * 0.6

vis = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite("demo/output.png", vis)

print("Saved demo/output.png")