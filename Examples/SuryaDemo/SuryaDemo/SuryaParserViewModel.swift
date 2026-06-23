import AppKit
import CoreGraphics
import Foundation
import MLXSurya
import Observation
import SwiftUI
import UniformTypeIdentifiers

/// The pipeline stage to run from the GUI.
enum DemoMode: String, CaseIterable, Identifiable {
    case detect = "Detect lines"
    case layout = "Layout"
    case ocr = "OCR (full page)"
    case structure = "Structure"
    case table = "Table"
    var id: String { rawValue }
}

/// VLM image-token budget preset — the OCR/layout/table speed lever (fewer pixels = faster, less
/// accurate on small text). `full` matches surya's default resolution.
enum ImageDetail: String, CaseIterable, Identifiable {
    case full = "Full"
    case balanced = "Balanced"
    case fast = "Fast"
    var id: String { rawValue }
    var maxPixels: Int {
        switch self {
        case .full: return 3072 * 2048  // ~6.3 MP (surya parity)
        case .balanced: return 3_000_000  // ~3 MP
        case .fast: return 1_500_000  // ~1.5 MP
        }
    }
}

/// One overlay box: a polygon in source-image pixels + optional label + reading-order rank.
struct DemoBox: Identifiable, Sendable {
    let id = UUID()
    let points: [CGPoint]
    let label: String?
    /// 0-indexed reading-order position (nil for stages without an order, e.g. detection/table).
    var order: Int? = nil

    /// Centroid of the box (source-image pixels), for drawing the reading-order flow line.
    var center: CGPoint {
        guard !points.isEmpty else { return .zero }
        let sx = points.reduce(0) { $0 + $1.x }, sy = points.reduce(0) { $0 + $1.y }
        return CGPoint(x: sx / CGFloat(points.count), y: sy / CGFloat(points.count))
    }
}

/// A completed page: its rendered image, detected boxes (with rectangles + reading order), and
/// text output. Kept per page so the user can flip back and forth. Not `Sendable` (holds a
/// `CGImage`); only assembled + read on the main actor.
struct PageResult: Identifiable {
    let id = UUID()
    let pageNumber: Int
    let image: CGImage
    let imageSize: CGSize
    let boxes: [DemoBox]
    let text: String
}

/// Drives ``SuryaSession`` from the GUI. Mirrors the CLI's flow, wrapped in a detached `Task`
/// with `MainActor` hops for observable updates. References the Session + frontend helpers only
/// (the `swift-cli-gui-shared-driver` pattern).
@MainActor
@Observable
final class SuryaParserViewModel {
    var inputPath: String = ""
    var mode: DemoMode = .detect
    /// VLM precision (bf16 = full precision/default; int8 = ~40% less LM memory, near-lossless,
    /// but not faster for surya).
    var precision: SuryaPrecision = .bf16
    /// Image-token budget — the OCR/layout/table speed lever.
    var imageDetail: ImageDetail = .full
    /// Structure mode: fold historical orthography (long-s → s, ligatures) into a normalized
    /// track. No effect on other modes.
    var normalizeOrthography: Bool = false

    var status: String = "Choose an input file, pick a mode, then Run."
    var isRunning: Bool = false
    var isDownloading: Bool = false
    var downloadProgress: Double = 0

    /// Per-page results, appended as each page lands. Flip through them with ``selectedPage``.
    var results: [PageResult] = []
    var selectedPage: Int = 0
    var totalPages: Int = 0

    /// Structure mode only: the whole-document Markdown, re-stitched across all pages processed
    /// so far (so paragraphs spanning a page break are joined). Updates as each page lands.
    var structuredMarkdown: String = ""

    @ObservationIgnored private var task: Task<Void, Never>?

    /// The page currently shown in the preview (nil before the first page completes).
    var current: PageResult? {
        results.indices.contains(selectedPage) ? results[selectedPage] : nil
    }
    var canPrev: Bool { selectedPage > 0 }
    var canNext: Bool { selectedPage < results.count - 1 }
    func goPrev() { if canPrev { selectedPage -= 1 } }
    func goNext() { if canNext { selectedPage += 1 } }

    var canRun: Bool { !inputPath.isEmpty && !isRunning && !isDownloading }

    /// Whether the surya-ocr-2 VLM is already in the Hugging Face cache (`~/.cache/huggingface`).
    var isModelCached: Bool { SuryaModel.cachedSnapshotDirectory() != nil }

    /// Ensure the VLM is present: no-op if already cached, else download `datalab-to/surya-ocr-2`
    /// (~1.4 GB) into the HF cache with progress. The detection + OCR-error models download
    /// on-demand from the datalab store when their stages first run.
    func downloadModel() {
        guard !isDownloading && !isRunning else { return }
        if isModelCached {
            status = "surya-ocr-2 already cached."
            return
        }
        isDownloading = true
        downloadProgress = 0
        status = "Downloading surya-ocr-2…"
        task = Task { [weak self] in
            do {
                _ = try await SuryaModel.download { progress in
                    self?.downloadProgress = progress.fractionCompleted
                    self?.status =
                        "Downloading surya-ocr-2… \(Int(progress.fractionCompleted * 100))%"
                }
                self?.isDownloading = false
                self?.status = "Model ready (surya-ocr-2 cached)."
            } catch {
                self?.isDownloading = false
                self?.status = "Download failed: \(error)"
            }
        }
    }

    func pickInput() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.pdf, .png, .jpeg, .tiff, .bmp, .gif, .webP]
        if panel.runModal() == .OK, let url = panel.url { inputPath = url.path }
    }

    func run() {
        guard canRun else { return }
        isRunning = true
        results = []
        selectedPage = 0
        totalPages = 0
        structuredMarkdown = ""
        status = "Loading…"

        let input = inputPath
        let mode = mode
        let precision = precision
        let normalize = normalizeOrthography
        var config = SuryaConfiguration()
        config.maxImagePixels = imageDetail.maxPixels

        task = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let pages = try SuryaSession.loadPages(
                    fileURL: URL(fileURLWithPath: input), dpi: config.imageDPIHighres)
                if pages.isEmpty {
                    await self?.set { $0.status = "No pages."; $0.isRunning = false }
                    return
                }
                await self?.set { $0.totalPages = pages.count }
                let session = try await SuryaSession.load(
                    SuryaSessionConfig(configuration: config, precision: precision))

                // Process every page, appending a PageResult as each lands. In Structure mode we
                // also accumulate the OCR results and re-structure the whole document so paragraphs
                // stitch across page boundaries, refreshing the combined output as each page lands.
                var ocrResults: [OCRResult] = []
                for (idx, page) in pages.enumerated() {
                    if Task.isCancelled { break }
                    await self?.set {
                        $0.status = "\(mode.rawValue) — page \(idx + 1)/\(pages.count)…"
                    }
                    let (boxes, text, ocr) = try await Self.processPage(
                        session: session, page: page, mode: mode)
                    if Task.isCancelled { break }
                    await self?.set {
                        // Auto-advance to the new page only if the user was viewing the latest.
                        let wasFollowing = $0.selectedPage >= $0.results.count - 1
                        $0.results.append(
                            PageResult(
                                pageNumber: idx + 1, image: page,
                                imageSize: CGSize(width: page.width, height: page.height),
                                boxes: boxes, text: text))
                        if wasFollowing { $0.selectedPage = $0.results.count - 1 }
                    }
                    // Structure mode: re-stitch across all pages seen so far (off the main actor),
                    // then hop the combined Markdown back for display.
                    if mode == .structure, let ocr {
                        ocrResults.append(ocr)
                        let markdown = Structurer(options: .init(normalizeOrthography: normalize))
                            .structure(ocrResults).markdown()
                        await self?.set { $0.structuredMarkdown = markdown }
                    }
                }

                await self?.set {
                    $0.status =
                        Task.isCancelled
                        ? "Cancelled (\($0.results.count) page(s) done)."
                        : "Done — \(pages.count) page(s)."
                    $0.isRunning = false
                }
            } catch {
                await self?.set { $0.status = "Error: \(error)"; $0.isRunning = false }
            }
        }
    }

    /// Run one page through the chosen stage. `nonisolated` so it runs off the main actor inside
    /// the detached task; returns Sendable values that are hopped back to the main actor.
    private nonisolated static func processPage(
        session: SuryaSession, page: CGImage, mode: DemoMode
    ) async throws -> (boxes: [DemoBox], text: String, ocr: OCRResult?) {
        switch mode {
        case .detect:
            let r = try await session.detectLines(page: page)
            return (r.lines.map { DemoBox(points: $0.box.cgPoints, label: nil) },
                "\(r.lines.count) text line(s) detected.", nil)
        case .layout:
            let r = try await session.layout(page: page)
            return (
                r.bboxes.map { DemoBox(points: $0.box.cgPoints, label: $0.label, order: $0.position) },
                r.bboxes.map { "[\($0.position)] \($0.label)" }.joined(separator: "\n"),
                nil
            )
        case .ocr:
            let r = try await session.ocr(page: page)
            return (
                r.blocks.map {
                    DemoBox(points: $0.box.cgPoints, label: $0.label, order: $0.readingOrder)
                },
                r.blocks.enumerated().map { "[\($0.offset)] \($0.element.html)" }
                    .joined(separator: "\n\n"),
                r
            )
        case .structure:
            // OCR the page; the whole-document structuring happens in `run()` so paragraphs can
            // stitch across pages. Boxes reuse the OCR blocks for the overlay.
            let r = try await session.ocr(page: page)
            return (
                r.blocks.map {
                    DemoBox(points: $0.box.cgPoints, label: $0.label, order: $0.readingOrder)
                },
                "",
                r
            )
        case .table:
            let r = try await session.tableRecognition(page: page)
            return (r.rows.map { DemoBox(points: $0.box.cgPoints, label: "R\($0.rowId)") }
                + r.cols.map { DemoBox(points: $0.box.cgPoints, label: "C\($0.colId)") },
                "rows=\(r.rows.count) cols=\(r.cols.count) cells=\(r.cells.count)", nil)
        }
    }

    func cancel() {
        task?.cancel()
        status = "Cancelling…"
    }

    private func set(_ mutate: @MainActor (SuryaParserViewModel) -> Void) async {
        await MainActor.run { mutate(self) }
    }
}

extension PolygonBox {
    /// Corner points as `CGPoint`s (source-image pixels).
    var cgPoints: [CGPoint] { polygon.map { CGPoint(x: $0[0], y: $0[1]) } }
}
