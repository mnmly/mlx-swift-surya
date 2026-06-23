import CoreGraphics
import Foundation

/// Static pipeline + model tunables, ported from the relevant subset of
/// `surya.settings.Settings`. Backend/server knobs (vLLM, llama.cpp, OpenAI,
/// Docker) are intentionally dropped — this port runs the VLM natively through
/// MLX, so only the model-behavior knobs carry over.
///
/// Defaults match upstream surya so Swift output lines up with the reference
/// pipeline. These are consumed identically by the CLI and a SwiftUI frontend
/// via ``SuryaSessionConfig`` (the `swift-cli-gui-shared-driver` pattern).
public struct SuryaConfiguration: Sendable {
    // MARK: Foundation VLM (surya-ocr-2)

    /// Hugging Face checkpoint id for the `qwen3_5` OCR VLM (`SURYA_MODEL_CHECKPOINT`).
    public var modelCheckpoint: String

    // MARK: Native models

    /// Source checkpoint for the EfficientViT text-detection model
    /// (`DETECTOR_MODEL_CHECKPOINT`). The upstream value is an `s3://…` path on
    /// `models.datalab.to`; the MLX port resolves a converted-weights snapshot.
    public var detectorCheckpoint: String
    /// Source checkpoint for the DistilBERT OCR-error model (`OCR_ERROR_MODEL_CHECKPOINT`).
    public var ocrErrorCheckpoint: String

    // MARK: Rasterization

    /// DPI for layout + text detection (coarse structure) — `IMAGE_DPI`.
    public var imageDPI: CGFloat
    /// DPI for recognition + table recognition (fine glyphs) — `IMAGE_DPI_HIGHRES`.
    public var imageDPIHighres: CGFloat

    // MARK: Detection postprocessing

    /// Text-confidence threshold for the detection heatmap (`DETECTOR_TEXT_THRESHOLD`).
    public var detectorTextThreshold: CGFloat
    /// Blank-region threshold (`DETECTOR_BLANK_THRESHOLD`).
    public var detectorBlankThreshold: CGFloat
    /// Fractional vertical expansion applied to detected boxes (`DETECTOR_BOX_Y_EXPAND_MARGIN`).
    public var detectorBoxYExpandMargin: CGFloat

    // MARK: Token budgets (VLM generation caps)

    /// Max output tokens for layout (`SURYA_MAX_TOKENS_LAYOUT`).
    public var maxTokensLayout: Int
    /// Max output tokens for table recognition (`SURYA_MAX_TOKENS_TABLE_REC`).
    public var maxTokensTableRec: Int
    /// Ceiling on output tokens for per-block OCR (`SURYA_MAX_TOKENS_BLOCK_CEILING`).
    public var maxTokensBlockCeiling: Int
    /// Max output tokens for full-page OCR (`SURYA_MAX_TOKENS_FULL_PAGE`).
    public var maxTokensFullPage: Int

    /// Coordinate space the model emits boxes in; normalized 0..`bboxScale` (`BBOX_SCALE`).
    public var bboxScale: Int

    // MARK: VLM image-token budget (the OCR/layout/table speed lever)

    /// Max pixels fed to the VLM after `scale_to_fit` (default ~6.3 MP = surya parity). Lowering
    /// this cuts vision tokens → faster OCR/layout/table, at an accuracy cost on small/dense text.
    public var maxImagePixels: Int
    /// Minimum pixels floor for `scale_to_fit`.
    public var minImagePixels: Int

    public init(
        modelCheckpoint: String = "datalab-to/surya-ocr-2",
        detectorCheckpoint: String = "s3://text_detection/2025_05_07",
        ocrErrorCheckpoint: String = "s3://ocr_error_detection/2025_02_18",
        imageDPI: CGFloat = 96,
        imageDPIHighres: CGFloat = 192,
        detectorTextThreshold: CGFloat = 0.6,
        detectorBlankThreshold: CGFloat = 0.35,
        detectorBoxYExpandMargin: CGFloat = 0.05,
        maxTokensLayout: Int = 3072,
        maxTokensTableRec: Int = 3072,
        maxTokensBlockCeiling: Int = 8192,
        maxTokensFullPage: Int = 12288,
        bboxScale: Int = 1000,
        maxImagePixels: Int = 3072 * 2048,
        minImagePixels: Int = 1792 * 28
    ) {
        self.modelCheckpoint = modelCheckpoint
        self.detectorCheckpoint = detectorCheckpoint
        self.ocrErrorCheckpoint = ocrErrorCheckpoint
        self.imageDPI = imageDPI
        self.imageDPIHighres = imageDPIHighres
        self.detectorTextThreshold = detectorTextThreshold
        self.detectorBlankThreshold = detectorBlankThreshold
        self.detectorBoxYExpandMargin = detectorBoxYExpandMargin
        self.maxTokensLayout = maxTokensLayout
        self.maxTokensTableRec = maxTokensTableRec
        self.maxTokensBlockCeiling = maxTokensBlockCeiling
        self.maxTokensFullPage = maxTokensFullPage
        self.bboxScale = bboxScale
        self.maxImagePixels = maxImagePixels
        self.minImagePixels = minImagePixels
    }
}
