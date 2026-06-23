import CoreGraphics
import Foundation

/// Drives one page from image → parsed result, reusing a loaded ``SuryaEngine``. Ports the
/// orchestration in `surya.layout`, `surya.recognition`, and `surya.table_rec`.
///
/// Bboxes are computed against the **original** image dimensions; the model sees a
/// `scaleToFit`-resized copy (matching the Python pipeline, where the resize is local to the
/// model input while output bboxes are denormalized to the source size).
public struct SuryaPipeline: Sendable {
    let engine: SuryaEngine
    let configuration: SuryaConfiguration

    public init(engine: SuryaEngine, configuration: SuryaConfiguration) {
        self.engine = engine
        self.configuration = configuration
    }

    /// Apply surya's `scale_to_fit` using the configured image-token budget.
    private func scaled(_ image: CGImage) -> CGImage {
        ImageOps.scaleToFit(
            image, maxPixels: configuration.maxImagePixels,
            minPixels: configuration.minImagePixels)
    }

    /// Layout analysis: `PROMPT_TYPE_LAYOUT` → JSON → ``LayoutResult``.
    public func layout(page: CGImage, maxTokens: Int? = nil) async throws -> LayoutResult {
        let (w, h) = (page.width, page.height)
        let modelImage = scaled(page)
        let gen = try await engine.generate(
            image: modelImage,
            prompt: SuryaPrompts.prompt(for: .layout),
            maxTokens: maxTokens ?? configuration.maxTokensLayout)

        let parsed = SuryaParsers.parseLayout(
            gen.raw, imageWidth: w, imageHeight: h, scale: configuration.bboxScale)
        let boxes = parsed.enumerated().map { idx, b in
            LayoutBox(
                box: PolygonBox(bbox: b.bbox),
                label: SuryaPrompts.canonicalLabel(b.label),
                rawLabel: b.label,
                position: idx,
                count: b.count)
        }
        return LayoutResult(bboxes: boxes, imageBbox: [0, 0, Double(w), Double(h)], raw: gen.raw)
    }

    /// Full-page OCR: `PROMPT_TYPE_HIGH_ACCURACY_BBOX` → HTML divs → ``OCRResult``.
    public func ocrFullPage(page: CGImage, maxTokens: Int? = nil) async throws -> OCRResult {
        let (w, h) = (page.width, page.height)
        let modelImage = scaled(page)
        let gen = try await engine.generate(
            image: modelImage,
            prompt: SuryaPrompts.prompt(for: .highAccuracyBbox),
            maxTokens: maxTokens ?? configuration.maxTokensFullPage)

        let parsed = SuryaParsers.parseFullPageHTML(
            gen.raw, imageWidth: w, imageHeight: h, scale: configuration.bboxScale)
        let blocks = parsed.enumerated().map { idx, b -> OCRBlock in
            let canonical = SuryaPrompts.canonicalLabel(b.label)
            let skipped = SuryaPrompts.skipCanonLabels.contains(canonical)
            return OCRBlock(
                box: PolygonBox(bbox: b.bbox),
                label: canonical,
                rawLabel: b.label,
                readingOrder: idx,
                html: skipped ? "" : b.html,
                skipped: skipped)
        }
        return OCRResult(blocks: blocks, imageBbox: [0, 0, Double(w), Double(h)], raw: gen.raw)
    }

    /// Per-block OCR: crop each layout block and run `PROMPT_TYPE_BLOCK` → cleaned HTML. Ports
    /// the block-mode path of `surya.recognition`. Blocks whose canonical label is in
    /// `SKIP_OCR_LABELS` are returned as `skipped` with empty HTML and never sent to the model.
    public func ocrBlocks(
        page: CGImage, blocks: [LayoutBox], maxTokens: Int? = nil
    ) async throws -> OCRResult {
        let (w, h) = (page.width, page.height)
        var out: [OCRBlock] = []
        out.reserveCapacity(blocks.count)
        for block in blocks {
            let canonical = SuryaPrompts.canonicalLabel(block.rawLabel.isEmpty ? block.label : block.rawLabel)
            if SuryaPrompts.skipCanonLabels.contains(canonical) {
                out.append(
                    OCRBlock(
                        box: block.box, label: canonical, rawLabel: block.rawLabel,
                        readingOrder: block.position, html: "", skipped: true))
                continue
            }
            let bb = block.box.bbox
            let rect = CGRect(x: bb[0], y: bb[1], width: bb[2] - bb[0], height: bb[3] - bb[1])
            let html: String
            if let crop = ImageOps.cropPixels(page, rect) {
                // image_token_budget(count) is approximated by the configured ceiling; greedy
                // decode stops at the model's EOS well before the cap for normal blocks.
                let gen = try await engine.generate(
                    image: scaled(crop),
                    prompt: SuryaPrompts.prompt(for: .block),
                    maxTokens: maxTokens ?? configuration.maxTokensBlockCeiling)
                html = SuryaParsers.cleanBlockHTML(gen.raw)
            } else {
                html = ""
            }
            out.append(
                OCRBlock(
                    box: block.box, label: canonical, rawLabel: block.rawLabel,
                    readingOrder: block.position, html: html, skipped: false))
        }
        return OCRResult(blocks: out, imageBbox: [0, 0, Double(w), Double(h)])
    }

    /// Table-structure recognition (simple mode): `PROMPT_TYPE_TABLE_REC` → JSON rows/cols →
    /// geometric cells. Ports `surya.table_rec` simple path.
    public func tableRecognition(page: CGImage, maxTokens: Int? = nil) async throws -> TableResult {
        let (w, h) = (page.width, page.height)
        let modelImage = scaled(page)
        let gen = try await engine.generate(
            image: modelImage,
            prompt: SuryaPrompts.prompt(for: .tableRec),
            maxTokens: maxTokens ?? configuration.maxTokensTableRec)

        let parsed = SuryaParsers.parseTableRec(
            gen.raw, imageWidth: w, imageHeight: h, scale: configuration.bboxScale)
        var rows: [TableRow] = []
        var cols: [TableCol] = []
        for el in parsed {
            if el.label == "Row" {
                rows.append(TableRow(rowId: rows.count, box: PolygonBox(bbox: el.bbox)))
            } else {
                cols.append(TableCol(colId: cols.count, box: PolygonBox(bbox: el.bbox)))
            }
        }
        // Derive cells from row ∩ col intersections.
        var cells: [TableCell] = []
        var cellId = 0
        for row in rows {
            let rb = row.box.bbox
            for col in cols {
                let cb = col.box.bbox
                let x0 = max(rb[0], cb[0])
                let y0 = max(rb[1], cb[1])
                let x1 = min(rb[2], cb[2])
                let y1 = min(rb[3], cb[3])
                guard x1 > x0 && y1 > y0 else { continue }
                cells.append(
                    TableCell(
                        rowId: row.rowId, colId: col.colId, cellId: cellId,
                        box: PolygonBox(bbox: [x0, y0, x1, y1])))
                cellId += 1
            }
        }
        return TableResult(
            rows: rows, cols: cols, cells: cells,
            imageBbox: [0, 0, Double(w), Double(h)], raw: gen.raw)
    }
}
