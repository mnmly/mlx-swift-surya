# mlx-swift-surya — agent notes

Swift / MLX port of [datalab-to/surya](https://github.com/datalab-to/surya).
The Python reference lives at `~/Development-local/GitHub/python/surya`.

## Architecture (read before porting)

surya 0.20 is **three independent components**:

1. **`surya-ocr-2`** — a `qwen3_5` VLM doing layout + table + recognition +
   reading order. Upstream serves it via vLLM / llama.cpp / OpenAI; this port
   loads it through **`MLXVLM`** (same arch family as `mlx-swift-chandra` — use
   that port as the wiring template). Do **not** hand-port the architecture.
2. **Text detection** — native EfficientViT segmentation. Native MLX port
   (encoder + LiteMLA linear attention + decode head; heatmap→polygon postproc).
   `mlx-swift-PaddleOCR/Detection/*` is the structural template.
3. **OCR-error** — native DistilBERT classifier. Native MLX port; BertTokenizer
   from swift-transformers.

The single library-side driver is `SuryaSession` (the
`swift-cli-gui-shared-driver` pattern): all model/engine setup lives there, the
CLI and SwiftUI app own only their loop + presentation. Keep `SuryaSession`
actor-agnostic (`@unchecked Sendable`, single-caller contract) — never
`@MainActor`, never leak SwiftUI types into the library (`CGImage` is fine).

## Build / run

MLX needs Metal, so use the **Xcode toolchain**, not the bare `swift` CLI (a
swiftly 6.1.2 toolchain shadows it for SourceKit/`swift`):

```bash
xcodebuild -scheme mlx-swift-surya -destination 'platform=macOS' \
  -derivedDataPath .xcdd build
.xcdd/Build/Products/Debug/surya-cli info
xcodebuild -scheme mlx-swift-surya -destination 'platform=macOS' test
```

`swift build` / `swift test` are OK only for code that touches no MLX op
(the pure-Swift skeleton tests). Anything that creates an `MLXArray` needs Xcode.

## Porting conventions

- 1:1 file mapping from the Python source; keep weight keys aligned with
  `@ModuleInfo(key:)`. PyTorch Conv2d weights transpose NCHW→NHWC.
- Cast weights to the target dtype at load time (see `MLXSupport.defaultDType`).
- Every slice ships the three release deliverables: end-to-end parity test,
  benchmark vs Python, and the high-level `SuryaSession` API.

## Documentation

`MLXSurya` ships DocC-generated reference docs (see
`Sources/MLXSurya/Documentation.docc/` and `Scripts/build_docs.sh`).
**`///` doc comments on public/`open` symbols are published** to the static site
and (with `EMIT_LLMS_TXT=1`) into `docs/llms.txt`.

When you add or modify a `public` or `open` declaration:

- Write a `///` doc comment — one-sentence summary, then a paragraph if the *why*
  is non-obvious. Skip restating the signature.
- Document each parameter with `- Parameter name:` (use the **internal** name
  when there's an external label — DocC warns otherwise).
- Cross-reference with signature-sensitive double-backtick links, e.g.
  `` ``SuryaSession/ocr(page:blocks:)`` ``.
- File new top-level symbols under a `## Topics` group (by *user task*) in
  `Sources/MLXSurya/Documentation.docc/MLXSurya.md`.

Verify before declaring docs done: `Scripts/build_docs.sh` (expect exit 0, no new
"doesn't exist at" / "external name used to document parameter" warnings).
