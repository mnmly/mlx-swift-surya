import ArgumentParser
import CoreGraphics
import Foundation
import MLXSurya

@main
struct SuryaCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "surya-cli",
        abstract: "surya document AI (OCR, layout, table, reading order) on Apple Silicon (MLX).",
        discussion: """
            The VLM stages (layout / ocr / table) run the surya-ocr-2 qwen3_5 model via MLXVLM.
            With no --model, the model is resolved from the Hugging Face cache or downloaded.
            Detection and OCR-error are pending their native-MLX slices.
            """,
        subcommands: [
            Info.self, Detect.self, Layout.self, OCR.self, Structure.self, Table.self, Gen.self,
            QA.self, Bench.self, Parity.self,
        ],
        defaultSubcommand: Info.self
    )
}

/// Numerical parity against pinned Python reference tensors (see scripts that dump them).
struct Parity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "parity",
        abstract: "Numerical parity vs pinned Python references.",
        subcommands: [
            OCRErrParity.self, DetectParity.self, DetectStagesParity.self, VLMInputsParity.self,
        ])

    static func report(_ r: SuryaParity.Result) throws {
        print(
            String(
                format: "%@: max abs diff %.5f (tol %.5f) — %@", r.name, r.maxAbsDiff, r.tolerance,
                r.pass ? "PASS ✅" : "FAIL ❌"))
        print("  \(r.detail)")
        if !r.pass { throw ExitCode.failure }
    }

    struct OCRErrParity: AsyncParsableCommand {
        static let configuration = CommandConfiguration(commandName: "ocrerr")
        @Option(name: .long, help: "Pinned reference safetensors (input_ids + logits).")
        var ref: String
        func run() async throws {
            err("Loading OCR-error model…")
            let engine = try await OCRErrorEngine(modelDirectory: OCRErrorModel.download())
            try Parity.report(
                try SuryaParity.ocrError(refURL: URL(fileURLWithPath: ref), engine: engine))
        }
    }

    struct DetectParity: AsyncParsableCommand {
        static let configuration = CommandConfiguration(commandName: "detect")
        @Option(name: .long, help: "Pinned reference safetensors (pixel_values + heatmap).")
        var ref: String
        func run() async throws {
            err("Loading detection model…")
            let engine = try await DetectionEngine(modelDirectory: DetectionModel.download())
            try Parity.report(
                try SuryaParity.detection(refURL: URL(fileURLWithPath: ref), engine: engine))
        }
    }

    struct VLMInputsParity: AsyncParsableCommand {
        static let configuration = CommandConfiguration(commandName: "vlm-inputs")
        @Option(name: .long, help: "Reference JSON (rendered + input_ids) from parity_vlm_inputs.py.")
        var ref: String
        func run() async throws {
            err("Loading VLM tokenizer…")
            let dir = SuryaModel.resolveSnapshotDirectory(try await SuryaModel.download())
            let tokenizer = try SuryaWordLevelTokenizer(directory: dir)
            try Parity.report(
                try SuryaParity.vlmInputs(refURL: URL(fileURLWithPath: ref), tokenizer: tokenizer))
        }
    }

    struct DetectStagesParity: AsyncParsableCommand {
        static let configuration = CommandConfiguration(commandName: "detect-stages")
        @Option(name: .long, help: "Reference safetensors with pixel_values.")
        var ref: String
        @Option(name: .long, help: "Reference safetensors with stage0..3 + decode_logits.")
        var stages: String
        func run() async throws {
            err("Loading detection model…")
            let engine = try await DetectionEngine(modelDirectory: DetectionModel.download())
            let results = try SuryaParity.detectionStages(
                pixelValuesURL: URL(fileURLWithPath: ref),
                stagesURL: URL(fileURLWithPath: stages), engine: engine)
            for r in results {
                print(String(format: "%-14@ max=%.5f  %@", r.name as NSString, r.maxAbsDiff, r.detail))
            }
        }
    }
}

/// Benchmark + memory-leak harness: loops one stage on a warm session and reports MLX memory
/// each iteration. Flat `active` across iterations ⇒ no leak. Build Release for real numbers.
struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Benchmark a stage and check for memory leaks.")
    @OptionGroup var common: CommonOptions

    @Option(name: .long, help: "Stage: detect | layout | ocr | table.")
    var mode: String = "detect"

    @Option(name: .long, help: "Number of iterations (first is warm-up, excluded from average).")
    var iterations: Int = 4

    func run() async throws {
        err("Loading model…")
        let session = try await common.makeSession()
        let page = try common.loadPage(session)
        err("Image \(page.width)×\(page.height); benchmarking '\(mode)' ×\(iterations)…")

        func fmt(_ bytes: Int) -> String { String(format: "%.0f MB", Double(bytes) / 1_048_576) }
        func runOnce() async throws {
            switch mode {
            case "detect": _ = try await session.detectLines(page: page)
            case "layout": _ = try await session.layout(page: page)
            case "ocr": _ = try await session.ocr(page: page)
            case "table": _ = try await session.tableRecognition(page: page)
            default: throw ValidationError("Unknown --mode '\(mode)'.")
            }
        }

        var times: [Double] = []
        var firstActive = 0
        for i in 1...iterations {
            let start = Date()
            try await runOnce()
            times.append(Date().timeIntervalSince(start))
            let m = SuryaSession.memorySnapshot()
            if i == 1 { firstActive = m.active }
            err(
                String(
                    format: "iter %d: %6.2fs  active=%@ cache=%@ peak=%@  Δactive=%@",
                    i, times[i - 1], fmt(m.active), fmt(m.cache), fmt(m.peak),
                    fmt(m.active - firstActive)))
        }
        let warm = times.dropFirst()
        let avg = warm.isEmpty ? times[0] : warm.reduce(0, +) / Double(warm.count)
        err(String(format: "\nwarm avg (excl. first): %.2fs over %d run(s)", avg, warm.count))
        err("active-memory drift: \(fmt(SuryaSession.memorySnapshot().active - firstActive)) (≈0 ⇒ no leak)")
    }
}

/// OCR-error detection: classify a text span as good/bad (native DistilBERT, no VLM).
struct QA: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "qa", abstract: "Classify text quality as good/bad (DistilBERT).")

    @Option(name: .long, help: "Text span to classify.")
    var text: String

    func run() async throws {
        err("Loading OCR-error model…")
        let session = try await SuryaSession.load(SuryaSessionConfig())
        let v = try await session.detectOCRErrors(text: text)
        print(String(format: "%@  (confidence %.3f)", v.label, v.confidence))
    }
}

/// Native EfficientViT text-line detection (no VLM needed).
struct Detect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Detect text lines (EfficientViT).")
    @OptionGroup var common: CommonOptions

    @Option(name: .long, help: "Print at most this many boxes.")
    var limit: Int = 10

    func run() async throws {
        err("Loading detection model…")
        let session = try await common.makeSession()
        let img = try common.loadPage(session)
        err("Image \(img.width)×\(img.height); detecting…")
        let start = Date()
        let result = try await session.detectLines(page: img)
        err(String(format: "Done in %.1fs — %d line(s).", Date().timeIntervalSince(start), result.lines.count))
        for line in result.lines.prefix(limit) {
            let bb = line.box.bbox.map { Int($0.rounded()) }
            let conf = line.box.confidence ?? 0
            print(String(format: "bbox=%@ conf=%.3f", "\(bb)", conf))
        }
    }
}

private func err(_ s: String) {
    FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
}

/// Print the resolved configuration and porting status. Needs no model.
struct Info: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Show the resolved configuration and porting status.")

    func run() async throws {
        let cfg = SuryaConfiguration()
        let cached = SuryaModel.cachedSnapshotDirectory()?.path ?? "(not cached)"
        print("mlx-swift-surya")
        print("")
        print("Foundation VLM : \(cfg.modelCheckpoint)  (qwen3_5, via MLXVLM)")
        print("  cached at    : \(cached)")
        print("Detection      : \(cfg.detectorCheckpoint)  (EfficientViT, native MLX)")
        print("OCR-error      : \(cfg.ocrErrorCheckpoint)  (DistilBERT, native MLX)")
        print("bbox scale     : \(cfg.bboxScale)")
        print("")
        print("Pipeline stages (status):")
        print("  layout  → SuryaSession.layout            [ready: VLM]")
        print("  ocr     → SuryaSession.ocr               [ready: VLM, full-page + block]")
        print("  struct  → SuryaSession.structure         [ready: post-process, no model]")
        print("  table   → SuryaSession.tableRecognition  [ready: VLM]")
        print("  detect  → SuryaSession.detectLines       [pending: detection slice]")
        print("  qa      → SuryaSession.detectOCRErrors   [pending: ocr-error slice]")
    }
}

struct CommonOptions: ParsableArguments {
    @Option(name: .long, help: "Path to the surya-ocr-2 model snapshot (default: HF cache/download).")
    var model: String?

    @Option(name: .long, help: "Input PDF or image file.")
    var input: String

    @Option(name: .long, help: "0-based page index for PDF input.")
    var page: Int = 0

    @Option(name: .long, help: "Bound on MLX's Metal buffer cache, in MB (0 = MLX default).")
    var cacheLimitMB: Int = 512

    @Option(name: .long, help: "VLM precision: bf16 (default) or int8 (less LM memory, not faster).")
    var precision: String = "bf16"

    @Option(
        name: .long,
        help: "VLM image-token budget in megapixels (default 6.3 = surya parity; lower = faster OCR, less accurate).")
    var maxImageMP: Double?

    func makeSession() async throws -> SuryaSession {
        let limit = cacheLimitMB > 0 ? cacheLimitMB * 1024 * 1024 : nil
        let prec = SuryaPrecision(rawValue: precision) ?? .bf16
        var configuration = SuryaConfiguration()
        if let mp = maxImageMP {
            configuration.maxImagePixels = max(1, Int(mp * 1_000_000))
        }
        let cfg = SuryaSessionConfig(
            modelDirectory: model.map { URL(fileURLWithPath: $0) }, configuration: configuration,
            gpuCacheLimit: limit, precision: prec)
        return try await SuryaSession.load(cfg)
    }

    func loadPage(_ session: SuryaSession) throws -> CGImage {
        try session.renderPage(fileURL: URL(fileURLWithPath: input), page: page)
    }
}

/// Layout analysis → labeled blocks with reading order.
struct Layout: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Analyze page layout (VLM).")
    @OptionGroup var common: CommonOptions

    func run() async throws {
        err("Loading model…")
        let session = try await common.makeSession()
        let img = try common.loadPage(session)
        err("Image \(img.width)×\(img.height); running layout…")
        let result = try await session.layout(page: img)
        for b in result.bboxes {
            let bb = b.box.bbox.map { Int($0.rounded()) }
            print("[\(b.position)] \(b.label) (\(b.rawLabel)) bbox=\(bb) count=\(b.count)")
        }
        err("\(result.bboxes.count) block(s).")
    }
}

/// Full-page OCR → HTML blocks (or block-mode if a layout is run first).
struct OCR: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Full-page OCR to HTML (VLM).")
    @OptionGroup var common: CommonOptions

    @Option(name: .long, help: "Write combined HTML to this file (default: stdout).")
    var output: String?

    func run() async throws {
        err("Loading model…")
        let session = try await common.makeSession()
        let img = try common.loadPage(session)
        err("Image \(img.width)×\(img.height); running OCR…")
        let start = Date()
        let result = try await session.ocr(page: img)
        err(String(format: "Done in %.1fs (%d block(s)).", Date().timeIntervalSince(start), result.blocks.count))
        let html = result.blocks.map { b in
            "<!-- \(b.label) @ \(b.box.bbox.map { Int($0.rounded()) }) -->\n\(b.html)"
        }.joined(separator: "\n\n")
        if let output {
            try html.write(toFile: output, atomically: true, encoding: .utf8)
            err("Wrote \(output)")
        } else {
            print(html)
        }
    }
}

/// OCR → reading-ordered structured document (paragraphs stitched whole, segmented into
/// sentences). Demonstrates `SuryaSession.structure` / `Structurer`.
struct Structure: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "OCR then structure into paragraphs/sentences (md or json).")
    @OptionGroup var common: CommonOptions

    @Option(name: .long, help: "Output format: md (default) or json.")
    var format: String = "md"

    @Option(
        name: .long,
        help: "Page range like \"0-3,5\" (PDF). Overrides --page; demos cross-page stitching.")
    var pageRange: String?

    @Flag(
        name: .long,
        help: "Normalize historical orthography (long-s→s, ligatures) into a normalized track.")
    var normalize: Bool = false

    @Option(name: .long, help: "Write output to this file (default: stdout).")
    var output: String?

    func run() async throws {
        err("Loading model…")
        let session = try await common.makeSession()
        let url = URL(fileURLWithPath: common.input)
        let pages: [CGImage]
        if let pageRange {
            pages = try session.loadPages(fileURL: url, pageRange: parsePageRange(pageRange))
        } else {
            pages = [try common.loadPage(session)]
        }
        err("Loaded \(pages.count) page(s); running OCR + structuring…")
        let start = Date()
        let doc = try await session.structure(
            pages: pages, options: Structurer.Options(normalizeOrthography: normalize))
        err(
            String(
                format: "Done in %.1fs (%d element(s)).", Date().timeIntervalSince(start),
                doc.elements.count))

        let rendered: String
        switch format.lowercased() {
        case "json":
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            rendered = String(data: try encoder.encode(doc), encoding: .utf8) ?? ""
        case "md", "markdown":
            rendered = doc.markdown()
        default:
            throw ValidationError("Unknown --format '\(format)' (use md | json).")
        }
        if let output {
            try rendered.write(toFile: output, atomically: true, encoding: .utf8)
            err("Wrote \(output)")
        } else {
            print(rendered)
        }
    }
}

/// Table-structure recognition → rows / cols / cells.
struct Table: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Table structure recognition (VLM).")
    @OptionGroup var common: CommonOptions

    func run() async throws {
        err("Loading model…")
        let session = try await common.makeSession()
        let img = try common.loadPage(session)
        err("Image \(img.width)×\(img.height); running table rec…")
        let result = try await session.tableRecognition(page: img)
        print("rows=\(result.rows.count) cols=\(result.cols.count) cells=\(result.cells.count)")
        for r in result.rows { print("Row \(r.rowId): \(r.box.bbox.map { Int($0.rounded()) })") }
        for c in result.cols { print("Col \(c.colId): \(c.box.bbox.map { Int($0.rounded()) })") }
    }
}

/// Emit the model's raw output for one image (no parsing). For parity testing.
struct Gen: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Generate raw model output for one image (parity testing).")
    @OptionGroup var common: CommonOptions

    @Option(name: .long, help: "Prompt type: layout | block | table_rec | high_accuracy_bbox.")
    var promptType: String = "high_accuracy_bbox"

    @Option(name: .long, help: "Max output tokens.")
    var maxTokens: Int?

    @Option(name: .long, help: "Write raw output to this file (default: stdout).")
    var output: String?

    func run() async throws {
        guard let pt = SuryaPromptType(rawValue: promptType) else {
            throw ValidationError("Unknown --prompt-type '\(promptType)'.")
        }
        err("Loading model…")
        let session = try await common.makeSession()
        let img = try common.loadPage(session)
        err("Image \(img.width)×\(img.height); generating…")
        let start = Date()
        let raw = try await session.rawGenerate(page: img, promptType: pt, maxTokens: maxTokens)
        err(String(format: "Done in %.1fs (%d chars).", Date().timeIntervalSince(start), raw.count))
        if let output {
            try raw.write(toFile: output, atomically: true, encoding: .utf8)
            err("Wrote \(output)")
        } else {
            print(raw)
        }
    }
}
