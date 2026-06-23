import Foundation

// Result types returned by ``SuryaSession``. These mirror the upstream surya schemas
// (`surya/*/schema.py`). Bboxes are in pixel coordinates of the source page.

/// One detected text line (output of the native EfficientViT detection model).
public struct TextLine: Sendable, Codable, Equatable {
    /// The line region in image pixels.
    public var box: PolygonBox
    public init(box: PolygonBox) { self.box = box }
}

/// Result of running text detection on a single page.
public struct DetectionResult: Sendable, Codable, Equatable {
    /// Detected text-line regions.
    public var lines: [TextLine]
    /// Source image size in pixels `[width, height]`.
    public var imageSize: [Int]
    public init(lines: [TextLine], imageSize: [Int]) {
        self.lines = lines
        self.imageSize = imageSize
    }
}

/// One layout block (output of the VLM's layout prompt). Mirrors `surya.layout.schema.LayoutBox`.
public struct LayoutBox: Sendable, Codable, Equatable {
    /// The block region in image pixels.
    public var box: PolygonBox
    /// Canonical layout label (e.g. `"Text"`, `"Table"`, `"Picture"`) via `LAYOUT_PRED_RELABEL`.
    public var label: String
    /// The original model-emitted label before canonicalization.
    public var rawLabel: String
    /// Reading-order position (0-indexed).
    public var position: Int
    /// The model's token-count estimate for the block.
    public var count: Int
    public init(box: PolygonBox, label: String, rawLabel: String, position: Int, count: Int = 0) {
        self.box = box
        self.label = label
        self.rawLabel = rawLabel
        self.position = position
        self.count = count
    }
}

/// Result of running layout analysis on a single page.
/// Mirrors `surya.layout.schema.LayoutResult`.
public struct LayoutResult: Sendable, Codable, Equatable {
    public var bboxes: [LayoutBox]
    /// `[0, 0, width, height]` of the source page.
    public var imageBbox: [Double]
    /// The model's raw output (for debugging / parity).
    public var raw: String?
    public init(bboxes: [LayoutBox], imageBbox: [Double], raw: String? = nil) {
        self.bboxes = bboxes
        self.imageBbox = imageBbox
        self.raw = raw
    }
}

/// One OCR'd block. Mirrors `surya.recognition.schema.BlockOCRResult`.
public struct OCRBlock: Sendable, Codable, Equatable {
    public var box: PolygonBox
    /// Canonical layout label.
    public var label: String
    /// Original model-emitted label.
    public var rawLabel: String
    /// Position in reading order.
    public var readingOrder: Int
    /// Recognized content (HTML).
    public var html: String
    /// True if this block's label is in `SKIP_OCR_LABELS` (not OCR'd).
    public var skipped: Bool
    public init(
        box: PolygonBox, label: String, rawLabel: String = "", readingOrder: Int = 0,
        html: String = "", skipped: Bool = false
    ) {
        self.box = box
        self.label = label
        self.rawLabel = rawLabel
        self.readingOrder = readingOrder
        self.html = html
        self.skipped = skipped
    }
}

/// Result of running recognition (OCR) on a single page.
/// Mirrors `surya.recognition.schema.PageOCRResult`.
public struct OCRResult: Sendable, Codable, Equatable {
    public var blocks: [OCRBlock]
    /// `[0, 0, width, height]` of the source page.
    public var imageBbox: [Double]
    /// The model's raw output (for debugging / parity).
    public var raw: String?
    public init(blocks: [OCRBlock], imageBbox: [Double], raw: String? = nil) {
        self.blocks = blocks
        self.imageBbox = imageBbox
        self.raw = raw
    }
}

/// A detected table row. Mirrors `surya.table_rec.schema.TableRow`.
public struct TableRow: Sendable, Codable, Equatable {
    public var rowId: Int
    public var box: PolygonBox
    public init(rowId: Int, box: PolygonBox) {
        self.rowId = rowId
        self.box = box
    }
}

/// A detected table column. Mirrors `surya.table_rec.schema.TableCol`.
public struct TableCol: Sendable, Codable, Equatable {
    public var colId: Int
    public var box: PolygonBox
    public init(colId: Int, box: PolygonBox) {
        self.colId = colId
        self.box = box
    }
}

/// A table cell at a row/column intersection. Mirrors `surya.table_rec.schema.TableCell`.
public struct TableCell: Sendable, Codable, Equatable {
    public var rowId: Int
    public var colId: Int
    public var cellId: Int
    public var box: PolygonBox
    public init(rowId: Int, colId: Int, cellId: Int, box: PolygonBox) {
        self.rowId = rowId
        self.colId = colId
        self.cellId = cellId
        self.box = box
    }
}

/// Result of table-structure recognition. Mirrors `surya.table_rec.schema.TableResult`.
public struct TableResult: Sendable, Codable, Equatable {
    public var rows: [TableRow]
    public var cols: [TableCol]
    /// Cells derived geometrically from row ∩ col intersections.
    public var cells: [TableCell]
    /// `[0, 0, width, height]` of the source table image.
    public var imageBbox: [Double]
    /// The model's raw output (for debugging / parity).
    public var raw: String?
    public init(
        rows: [TableRow], cols: [TableCol], cells: [TableCell], imageBbox: [Double],
        raw: String? = nil
    ) {
        self.rows = rows
        self.cols = cols
        self.cells = cells
        self.imageBbox = imageBbox
        self.raw = raw
    }
}

/// Verdict from the DistilBERT OCR-error model for a span of recognized text.
public struct OCRErrorVerdict: Sendable, Codable, Equatable {
    /// `"good"` or `"bad"` (`config.id2label`).
    public var label: String
    /// Probability of the predicted label.
    public var confidence: Double
    public init(label: String, confidence: Double) {
        self.label = label
        self.confidence = confidence
    }
}
