# Numerical parity vs the Python reference

Verifies the Swift/MLX ports match upstream [surya](https://github.com/datalab-to/surya) within
tolerance. Two steps: dump pinned reference tensors from Python, then compare with the Swift CLI.

## 1. Dump references (Python, in a surya checkout)

Run inside a clone of `datalab-to/surya` (these import `surya` + torch via `uv`):

```bash
cd /path/to/python/surya
DUMP=/path/to/this/repo/Scripts/parity   # or any scratch dir
IMG=static/images/excerpt_text.png

uv run python "$DUMP/parity_ocrerr.py"      "$DUMP/ocrerr_parity.safetensors"
uv run python "$DUMP/parity_detect.py" "$IMG" "$DUMP/detect_parity.safetensors"
# VLM tokenizer/chat-template ref (uses the HF snapshot's tokenizer.json + chat_template.jinja):
SNAP=$(find ~/.cache/huggingface/hub/models--datalab-to--surya-ocr-2/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)
uv run python "$DUMP/parity_vlm_inputs.py" "$SNAP" "$DUMP/vlm_inputs.json"
```

## 2. Compare (Swift CLI, Release)

```bash
BIN=.xcdd-rel/Build/Products/Release/surya-cli
$BIN parity ocrerr     --ref Scripts/parity/ocrerr_parity.safetensors
$BIN parity detect     --ref Scripts/parity/detect_parity.safetensors
$BIN parity vlm-inputs --ref Scripts/parity/vlm_inputs.json
# bisection helper (per encoder stage), if a detection check ever regresses:
$BIN parity detect-stages --ref …/detect_parity.safetensors --stages …/detect_stages.safetensors
```

## Status (verified)

| Check | Result |
|---|---|
| OCR-error (DistilBERT) logits | max abs diff **0.0027** ✅ |
| Detection (EfficientViT) heatmap | max abs diff **0.0037** ✅ |
| VLM input_ids (tokenizer + chat template) | **identical** (exact) ✅ |

Detection compares fp32-vs-fp32; the others are near-lossless. The detection check originally
*failed* (max diff 0.84) and surfaced a real bug — the model wasn't in eval mode, so MLX
`BatchNorm` used per-input batch statistics instead of the loaded `running_mean`/`running_var`.
Fixed with `model.train(false)` in `DetectionEngine`. The VLM model itself runs through upstream
`MLXVLM`; here we verify only the input pipeline (tokenizer + chat template) that this port owns.
