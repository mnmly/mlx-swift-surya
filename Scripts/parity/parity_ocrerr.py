"""Dump OCR-error (DistilBERT) reference: pinned input_ids + logits for a fixed text."""
import sys
import torch
from safetensors.torch import save_file
from surya.ocr_error import OCRErrorPredictor

OUT = sys.argv[1]
TEXT = "The quick brown fox jumps over the lazy dog."

p = OCRErrorPredictor()
model, processor = p.model, p.processor
enc = processor([TEXT], padding="longest", truncation=True, return_tensors="pt")
ids, mask = enc.input_ids, enc.attention_mask
with torch.inference_mode():
    out = model(ids.to(model.device), attention_mask=mask.to(model.device))
logits = out.logits.float().cpu().contiguous()
save_file(
    {"input_ids": ids.to(torch.int64).cpu().contiguous(), "logits": logits}, OUT)
print("seq_len:", ids.shape[1], "| first ids:", ids[0, :10].tolist())
print("logits:", logits.tolist(), "| argmax:", int(logits.argmax(-1)))
print("wrote", OUT)
