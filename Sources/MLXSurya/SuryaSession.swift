import CoreGraphics
import Foundation

/// Compute precision for the surya-ocr-2 VLM.
///
/// - `bf16` (default): full precision — the numerical-parity reference.
/// - `int8`: quantizes the **language model** in-memory at load (group 64, 8-bit; the vision
///   tower stays full-precision or image conditioning breaks). Near-lossless and ~40% lighter on
///   LM weight memory, but **not faster** for surya: the text model is small (1024-d) and per-page
///   cost is dominated by image prefill through the vision tower, so int8's matmul win is
///   negligible. Use it to save memory, not time.
public enum SuryaPrecision: String, Sendable, CaseIterable {
    case bf16
    case int8
}

/// Configuration for a ``SuryaSession``. All knobs that affect loading + the pipeline live here
/// with defaults that work for both the CLI and a SwiftUI frontend.
public struct SuryaSessionConfig: Sendable {
    /// Directory containing the surya-ocr-2 VLM snapshot (config.json + safetensors + tokenizer).
    /// When `nil`, the session resolves a cached snapshot or downloads one (see ``autoDownload``).
    public var modelDirectory: URL?
    /// Pipeline/model tunables ported from `surya.settings`.
    public var configuration: SuryaConfiguration
    /// Bound on MLX's reusable Metal buffer cache (bytes); `nil` leaves the MLX default.
    public var gpuCacheLimit: Int?
    /// When ``modelDirectory`` is `nil` and nothing is cached, download from Hugging Face.
    public var autoDownload: Bool
    /// VLM compute precision (default `.bf16`; `.int8` saves LM memory, not time — see ``SuryaPrecision``).
    public var precision: SuryaPrecision

    public init(
        modelDirectory: URL? = nil,
        configuration: SuryaConfiguration = SuryaConfiguration(),
        gpuCacheLimit: Int? = 512 * 1024 * 1024,
        autoDownload: Bool = true,
        precision: SuryaPrecision = .bf16
    ) {
        self.modelDirectory = modelDirectory
        self.configuration = configuration
        self.gpuCacheLimit = gpuCacheLimit
        self.autoDownload = autoDownload
        self.precision = precision
    }
}

/// The single library-side driver for the surya pipeline: detection → layout → recognition →
/// table recognition, plus the OCR-error check. The CLI and a SwiftUI app consume this
/// identically (the `swift-cli-gui-shared-driver` pattern).
///
/// Contract: one calling task at a time. The VLM (`surya-ocr-2`) is loaded **lazily** on the
/// first VLM-backed call, so ``load(_:)`` is cheap and never touches the network — model
/// acquisition happens on first use. UI isolation stays frontend-side, which is why the session
/// is `@unchecked Sendable` rather than an actor.
///
/// - Note: ``detectLines(page:)`` (detection model) and ``detectOCRErrors(text:)`` (OCR-error
///   model) are pending their native-MLX slices and throw ``SuryaError/notImplemented(_:)``.
public final class SuryaSession: @unchecked Sendable {
    /// The configuration this session was loaded with.
    public let config: SuryaSessionConfig

    private var _pipeline: SuryaPipeline?
    private var _detection: DetectionEngine?
    private var _ocrError: OCRErrorEngine?

    private init(config: SuryaSessionConfig) {
        self.config = config
    }

    /// Construct a session. Cheap and offline — the model loads on first VLM call.
    public static func load(_ config: SuryaSessionConfig) async throws -> SuryaSession {
        SuryaSession(config: config)
    }

    /// Resolve, load (once), and cache the VLM pipeline.
    private func pipeline() async throws -> SuryaPipeline {
        if let p = _pipeline { return p }
        let dir = try await resolveModelDirectory()
        let engine = try await SuryaEngine(
            modelDirectory: dir, gpuCacheLimit: config.gpuCacheLimit, precision: config.precision)
        let p = SuryaPipeline(engine: engine, configuration: config.configuration)
        _pipeline = p
        return p
    }

    private func resolveModelDirectory() async throws -> URL {
        if let dir = config.modelDirectory {
            let resolved = SuryaModel.resolveSnapshotDirectory(dir)
            try? SuryaModel.patchProcessorClass(in: resolved)
            return resolved
        }
        if let cached = SuryaModel.cachedSnapshotDirectory() {
            try? SuryaModel.patchProcessorClass(in: cached)
            return cached
        }
        guard config.autoDownload else {
            throw SuryaError.modelNotAvailable(
                "Set SuryaSessionConfig.modelDirectory or call SuryaModel.download().")
        }
        return try await SuryaModel.download()
    }

    // MARK: - Pipeline stages

    /// Resolve, load (once), and cache the native EfficientViT detection engine.
    private func detectionEngine() async throws -> DetectionEngine {
        if let e = _detection { return e }
        let dir: URL
        if let provided = config.modelDirectory.map({
            $0.deletingLastPathComponent().appendingPathComponent("text_detection")
        }), FileManager.default.fileExists(atPath: provided.appendingPathComponent("config.json").path) {
            dir = provided
        } else {
            dir = try await DetectionModel.download()
        }
        let engine = try DetectionEngine(modelDirectory: dir)
        _detection = engine
        return engine
    }

    /// Detect text-line regions on a page (native EfficientViT model).
    public func detectLines(page: CGImage) async throws -> DetectionResult {
        let engine = try await detectionEngine()
        let c = config.configuration
        return engine.detect(
            page,
            textThreshold: Float(c.detectorTextThreshold),
            lowText: Float(c.detectorBlankThreshold),
            yExpandMargin: Float(c.detectorBoxYExpandMargin))
    }

    /// Analyze page layout into labeled blocks with reading order (surya-ocr-2 VLM).
    public func layout(page: CGImage, maxTokens: Int? = nil) async throws -> LayoutResult {
        try await pipeline().layout(page: page, maxTokens: maxTokens)
    }

    /// Recognize text on a page (surya-ocr-2 VLM). Full-page when `blocks` is `nil`, otherwise
    /// per-block OCR over the supplied layout regions.
    public func ocr(
        page: CGImage, blocks: [LayoutBox]? = nil, maxTokens: Int? = nil
    ) async throws -> OCRResult {
        let p = try await pipeline()
        if let blocks {
            return try await p.ocrBlocks(page: page, blocks: blocks, maxTokens: maxTokens)
        }
        return try await p.ocrFullPage(page: page, maxTokens: maxTokens)
    }

    /// Recognize table structure (rows/cols/cells) on a table image (surya-ocr-2 VLM).
    public func tableRecognition(page: CGImage, maxTokens: Int? = nil) async throws -> TableResult {
        try await pipeline().tableRecognition(page: page, maxTokens: maxTokens)
    }

    /// OCR each page (in order) and post-process the results into a reading-ordered
    /// ``StructuredDocument`` — paragraphs stitched whole across page/column breaks and
    /// segmented into sentences. The structuring step is model-free (see ``Structurer``);
    /// this convenience just chains it after recognition so a frontend can call one method.
    ///
    /// - Parameters:
    ///   - pages: The page images to recognize, in reading order.
    ///   - options: Structuring options (orthography normalization, de-hyphenation, …).
    public func structure(
        pages: [CGImage], options: Structurer.Options = .init()
    ) async throws -> StructuredDocument {
        var results: [OCRResult] = []
        results.reserveCapacity(pages.count)
        for page in pages {
            results.append(try await ocr(page: page))
        }
        return Structurer(options: options).structure(results)
    }

    /// Resolve, load (once), and cache the native DistilBERT OCR-error engine.
    private func ocrErrorEngine() async throws -> OCRErrorEngine {
        if let e = _ocrError { return e }
        let dir = try await OCRErrorModel.download()
        let engine = try await OCRErrorEngine(modelDirectory: dir)
        _ocrError = engine
        return engine
    }

    /// Classify a span of recognized text as good/bad (native DistilBERT model).
    public func detectOCRErrors(text: String) async throws -> OCRErrorVerdict {
        try await ocrErrorEngine().detect(text)
    }

    /// Run the VLM on a page and return its **raw** output (no parsing). For parity testing.
    public func rawGenerate(
        page: CGImage, promptType: SuryaPromptType = .highAccuracyBbox, maxTokens: Int? = nil
    ) async throws -> String {
        let p = try await pipeline()
        let modelImage = ImageOps.scaleToFit(
            page, maxPixels: config.configuration.maxImagePixels,
            minPixels: config.configuration.minImagePixels)
        return try await p.engine.generate(
            image: modelImage,
            prompt: SuryaPrompts.prompt(for: promptType),
            maxTokens: maxTokens ?? config.configuration.maxTokensFullPage
        ).raw
    }

    // MARK: - Input loading (weight-free; usable before models load)

    /// Load every requested page of a file as `CGImage`s. PDFs render per page at `dpi`
    /// (default: the high-res recognition DPI); raster images load directly.
    public func loadPages(
        fileURL: URL, pageRange: [Int]? = nil, dpi: CGFloat? = nil
    ) throws -> [CGImage] {
        try SuryaSession.loadPages(
            fileURL: fileURL, pageRange: pageRange,
            dpi: dpi ?? config.configuration.imageDPIHighres)
    }

    /// Render a single page of a PDF/image file to a `CGImage` (frontend preview helper).
    public func renderPage(fileURL: URL, page: Int = 0, dpi: CGFloat? = nil) throws -> CGImage {
        try SuryaSession.renderPage(
            fileURL: fileURL, page: page, dpi: dpi ?? config.configuration.imageDPIHighres)
    }

    /// Weight-free document loading. Static so a frontend (or a unit test) can rasterize inputs
    /// without first loading the model.
    public static func loadPages(fileURL: URL, pageRange: [Int]? = nil, dpi: CGFloat) throws
        -> [CGImage]
    {
        if fileURL.pathExtension.lowercased() == "pdf" {
            guard let count = ImageOps.pdfPageCount(fileURL) else { throw SuryaError.pdfUnsupported }
            var images: [CGImage] = []
            for page in 0..<count where pageRange == nil || pageRange!.contains(page) {
                images.append(try ImageOps.renderPDFPage(fileURL, page: page, imageDPI: dpi))
            }
            return images
        }
        return [try ImageOps.load(fileURL)]
    }

    /// Weight-free single-page render (see ``loadPages(fileURL:pageRange:dpi:)``).
    public static func renderPage(fileURL: URL, page: Int = 0, dpi: CGFloat) throws -> CGImage {
        if fileURL.pathExtension.lowercased() == "pdf" {
            return try ImageOps.renderPDFPage(fileURL, page: page, imageDPI: dpi)
        }
        return try ImageOps.load(fileURL)
    }

    /// Current MLX GPU memory snapshot (active flat across runs ⇒ no leak).
    public static func memorySnapshot() -> SuryaEngine.MemorySnapshot {
        SuryaEngine.memorySnapshot()
    }
}

/// Parse a page-range string like `"1-5,7,9-12"` into a sorted, de-duplicated 0-based list.
public func parsePageRange(_ rangeStr: String) -> [Int] {
    var pages: Set<Int> = []
    for part in rangeStr.split(separator: ",") {
        if part.contains("-") {
            let bounds = part.split(separator: "-")
            if bounds.count == 2, let start = Int(bounds[0]), let end = Int(bounds[1]) {
                for p in start...max(start, end) { pages.insert(p) }
            }
        } else if let p = Int(part) {
            pages.insert(p)
        }
    }
    return pages.sorted()
}
