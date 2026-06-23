# ``MLXSurya``

A Swift/MLX port of [datalab-to/surya](https://github.com/datalab-to/surya) —
OCR, layout analysis, table recognition, and reading order on Apple Silicon.

## Overview

surya is three independent components, ported with three strategies:

- **Foundation VLM (`surya-ocr-2`)** — a `qwen3_5` vision-language model that
  produces layout, table structure, recognized text, and reading order. It is
  loaded through `MLXVLM` (the same architecture family as `mlx-swift-chandra`),
  not hand-ported.
- **Text detection** — a native EfficientViT segmentation network ported
  directly to MLX, turning a page image into text-line polygons.
- **OCR-error detection** — a native DistilBERT classifier ported to MLX that
  flags low-quality recognized spans.

All non-presentation work lives behind a single library-side driver,
``SuryaSession``, consumed identically by the `surya-cli` executable and a
SwiftUI app (the shared-driver pattern). The CLI and GUI own only their loop,
presentation, and cadence.

```swift
import MLXSurya

let session = try await SuryaSession.load(
    SuryaSessionConfig(configuration: SuryaConfiguration()))
let pages = try session.loadPages(fileURL: pdfURL)
let layout = try await session.layout(page: pages[0])
let text = try await session.ocr(page: pages[0], blocks: layout.bboxes)

// Post-process recognized blocks into a reading-ordered document with whole
// paragraphs (stitched across page/column breaks) and sentence segmentation.
let doc = try await session.structure(pages: pages)
print(doc.markdown())
```

> Note: This package is mid-port. Each ``SuryaSession`` stage throws
> ``SuryaError/notImplemented(_:)`` until its slice lands.

## Topics

### Driving the pipeline

- ``SuryaSession``
- ``SuryaSessionConfig``
- ``SuryaConfiguration``

### The VLM engine

- ``SuryaModel``
- ``SuryaEngine``
- ``SuryaPipeline``
- ``SuryaPromptType``
- ``SuryaPrompts``

### Native text detection (EfficientViT)

- ``DetectionModel``
- ``DetectionEngine``
- ``EfficientViTForSemanticSegmentation``
- ``EfficientViTConfig``

### Native OCR-error detection (DistilBERT)

- ``OCRErrorModel``
- ``OCRErrorEngine``
- ``DistilBertForSequenceClassification``
- ``DistilBertConfig``

### Results

- ``DetectionResult``
- ``LayoutResult``
- ``OCRResult``
- ``TableResult``
- ``OCRErrorVerdict``

### Geometry & blocks

- ``PolygonBox``
- ``TextLine``
- ``LayoutBox``
- ``OCRBlock``
- ``TableRow``
- ``TableCol``
- ``TableCell``

### Parsing model output

- ``SuryaParsers``
- ``ParsedLayoutBlock``
- ``ParsedTableElement``
- ``ParsedFullPageBlock``

### Document structuring

- ``Structurer``
- ``StructuredDocument``
- ``DocElement``
- ``Paragraph``

### Images & utilities

- ``ImageOps``
- ``parsePageRange(_:)``
- ``SuryaError``
