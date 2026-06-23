"""Dump detection (EfficientViT) reference: pinned pixel_values + sigmoid heatmap.

Feeds one preprocessed 1200x1200 input through the model directly (no chunking) so the Swift
port can be compared at the model's native output resolution.
"""
import sys
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from surya.detection.loader import DetectionModelLoader

IMG, OUT = sys.argv[1], sys.argv[2]

loader = DetectionModelLoader()
model = loader.model(dtype=torch.float32)  # fp32 for an apples-to-apples parity vs the Swift port
processor = loader.processor()

img = Image.open(IMG).convert("RGB")
new_size = (processor.size["width"], processor.size["height"])  # 1200 x 1200
img.thumbnail(new_size, Image.Resampling.LANCZOS)
img = img.resize(new_size, Image.Resampling.LANCZOS)
arr = np.asarray(img, dtype=np.uint8)
px = processor(arr)["pixel_values"][0]  # [3,H,W] normalized
px_t = torch.from_numpy(np.asarray(px)).unsqueeze(0).to(model.dtype).to(model.device)

with torch.inference_mode():
    out = model(pixel_values=px_t)
heat = out.logits.float().cpu().contiguous()  # [1,2,H/4,W/4], sigmoided

save_file({"pixel_values": px_t.float().cpu().contiguous(), "heatmap": heat}, OUT)
print("pixel_values:", tuple(px_t.shape), "dtype", model.dtype)
print("heatmap:", tuple(heat.shape), "min/max", float(heat.min()), float(heat.max()))
print("wrote", OUT)
