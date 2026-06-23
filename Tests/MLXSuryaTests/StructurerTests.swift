import XCTest

@testable import MLXSurya

// Pure-Swift tests for `Structurer`. No model / Metal needed (structuring is post-processing
// on `OCRResult`), so these run under `swift test` as well as Xcode. They feed synthetic
// `OCRResult`s — reading-ordered, label-tagged blocks of HTML — and pin the stitching,
// routing, segmentation, and normalization behavior.
final class StructurerTests: XCTestCase {

    // MARK: - Fixtures

    private func block(
        _ label: String, _ html: String, order: Int, skipped: Bool = false
    ) -> OCRBlock {
        OCRBlock(
            box: PolygonBox(bbox: [0, 0, 100, 20]), label: label, rawLabel: label,
            readingOrder: order, html: html, skipped: skipped)
    }

    private func page(_ blocks: [OCRBlock]) -> OCRResult {
        OCRResult(blocks: blocks, imageBbox: [0, 0, 600, 800])
    }

    private func paragraphs(_ doc: StructuredDocument) -> [Paragraph] {
        doc.elements.compactMap { if case .paragraph(let p) = $0 { return p } else { return nil } }
    }

    // MARK: - Stitching

    func testStitchesParagraphSplitAcrossBlocks() {
        let doc = Structurer().structure(
            page([
                block("Text", "<p>The quick brown fox jumps over the lazy</p>", order: 0),
                block("Text", "<p>dog and keeps running through the field.</p>", order: 1),
            ]))
        XCTAssertEqual(doc.elements.count, 1)
        XCTAssertEqual(
            paragraphs(doc).first?.text,
            "The quick brown fox jumps over the lazy dog and keeps running through the field.")
    }

    func testStitchesAcrossPageBoundary() {
        let p1 = page([block("Text", "<p>An incomplete sentence that continues on the next</p>", order: 0)])
        let p2 = page([block("Text", "<p>page without any interruption at all.</p>", order: 0)])
        let doc = Structurer().structure([p1, p2])
        XCTAssertEqual(doc.elements.count, 1)
        XCTAssertEqual(
            paragraphs(doc).first?.text,
            "An incomplete sentence that continues on the next page without any interruption at all.")
    }

    func testDoesNotStitchWhenPreviousEndsInPunctuation() {
        let doc = Structurer().structure(
            page([
                block("Text", "<p>First paragraph is complete.</p>", order: 0),
                block("Text", "<p>Second paragraph is separate.</p>", order: 1),
            ]))
        XCTAssertEqual(paragraphs(doc).map(\.text), [
            "First paragraph is complete.", "Second paragraph is separate.",
        ])
    }

    func testDehyphenatesWordSplitAcrossBlocks() {
        let doc = Structurer().structure(
            page([
                block("Text", "<p>The treaty was an important docu-</p>", order: 0),
                block("Text", "<p>ment that ended the long war.</p>", order: 1),
            ]))
        XCTAssertEqual(
            paragraphs(doc).first?.text,
            "The treaty was an important document that ended the long war.")
    }

    func testHeadingActsAsStitchBarrier() {
        let doc = Structurer().structure(
            page([
                block("Text", "<p>A trailing line with no final period</p>", order: 0),
                block("SectionHeader", "<h2>A New Section</h2>", order: 1),
                block("Text", "<p>Body of the next section.</p>", order: 2),
            ]))
        XCTAssertEqual(doc.elements.count, 3)
        guard case .paragraph(let first) = doc.elements[0] else { return XCTFail("expected paragraph") }
        XCTAssertEqual(first.text, "A trailing line with no final period")
        guard case .heading(let level, let text) = doc.elements[1] else { return XCTFail("expected heading") }
        XCTAssertEqual(level, 2)
        XCTAssertEqual(text, "A New Section")
    }

    func testSortsByReadingOrder() {
        let doc = Structurer().structure(
            page([
                block("Text", "<p>Third.</p>", order: 2),
                block("Text", "<p>First.</p>", order: 0),
                block("Text", "<p>Second.</p>", order: 1),
            ]))
        XCTAssertEqual(paragraphs(doc).map(\.text), ["First.", "Second.", "Third."])
    }

    func testRunningFooterDroppedAndDoesNotBreakStitch() {
        let blocks = [
            block("Text", "<p>This sentence continues past the page</p>", order: 0),
            block("PageFooter", "<p>42</p>", order: 1),
            block("Text", "<p>footer and finishes here.</p>", order: 2),
        ]
        // Default: footer dropped → the two Text blocks become adjacent and stitch.
        let stitched = Structurer().structure(page(blocks))
        XCTAssertEqual(stitched.elements.count, 1)
        XCTAssertEqual(
            paragraphs(stitched).first?.text,
            "This sentence continues past the page footer and finishes here.")

        // With running headers kept, the footer sits between them and breaks the stitch.
        let kept = Structurer(options: .init(dropRunningHeaders: false)).structure(page(blocks))
        XCTAssertEqual(kept.elements.count, 3)
    }

    // MARK: - Routing

    func testTableIsPreservedAsHTML() {
        let html = "<table><tr><td>a</td><td>b</td></tr></table>"
        let doc = Structurer().structure(page([block("Table", html, order: 0)]))
        XCTAssertEqual(doc.elements.count, 1)
        guard case .table(let out) = doc.elements[0] else { return XCTFail("expected table") }
        XCTAssertTrue(out.contains("<table"))
    }

    func testListGroupBecomesList() {
        let doc = Structurer().structure(
            page([block("ListGroup", "<ul><li>first</li><li>second</li></ul>", order: 0)]))
        guard case .list(let ordered, let items) = doc.elements[0] else { return XCTFail("expected list") }
        XCTAssertFalse(ordered)
        XCTAssertEqual(items, ["first", "second"])
    }

    func testSkippedFigureBecomesFigurePlaceholder() {
        let doc = Structurer().structure(page([block("Picture", "", order: 0, skipped: true)]))
        XCTAssertEqual(doc.elements.count, 1)
        guard case .figure = doc.elements[0] else { return XCTFail("expected figure") }
    }

    func testDecodesHTMLEntities() {
        let doc = Structurer().structure(
            page([block("Text", "<p>Tom &amp; Jerry cost &#36;5 &mdash; cheap.</p>", order: 0)]))
        XCTAssertEqual(paragraphs(doc).first?.text, "Tom & Jerry cost $5 — cheap.")
    }

    // MARK: - Segmentation

    func testSegmentsMultipleSentences() {
        let doc = Structurer().structure(
            page([block("Text", "<p>First sentence. Second sentence! Third one?</p>", order: 0)]))
        XCTAssertEqual(
            paragraphs(doc).first?.sentences,
            ["First sentence.", "Second sentence!", "Third one?"])
    }

    func testDoesNotOverSplitOnDecimalsAndAbbreviations() {
        // A decimal point and a lowercase-continued "e.g." must not be read as sentence ends.
        let doc = Structurer().structure(
            page([block("Text", "<p>The ratio is about 3.14, e.g. as seen in the table.</p>", order: 0)]))
        XCTAssertEqual(paragraphs(doc).first?.sentences.count, 1)
    }

    func testSegmentationDisabledKeepsWholeText() {
        let doc = Structurer(options: .init(segmentSentences: false)).structure(
            page([block("Text", "<p>One. Two. Three.</p>", order: 0)]))
        XCTAssertEqual(paragraphs(doc).first?.sentences, ["One. Two. Three."])
    }

    // MARK: - Orthography normalization

    func testNormalizationFoldsLongSButKeepsFaithfulText() {
        let doc = Structurer(options: .init(normalizeOrthography: true)).structure(
            page([block("Text", "<p>Da Rappreſentarſi nel nuouo Teatro Zane.</p>", order: 0)]))
        let para = paragraphs(doc).first
        XCTAssertEqual(para?.text, "Da Rappreſentarſi nel nuouo Teatro Zane.")  // faithful: ſ kept
        XCTAssertEqual(para?.normalizedText, "Da Rappresentarsi nel nuouo Teatro Zane.")  // ſ → s
    }

    func testNormalizationOffLeavesNormalizedTextNil() {
        let doc = Structurer().structure(
            page([block("Text", "<p>Da Rappreſentarſi nel nuouo Teatro.</p>", order: 0)]))
        XCTAssertNil(paragraphs(doc).first?.normalizedText)
    }

    // MARK: - Rendering & Codable

    func testMarkdownRendering() {
        let doc = Structurer().structure(
            page([
                block("SectionHeader", "<h1>Title</h1>", order: 0),
                block("Text", "<p>Body text here.</p>", order: 1),
                block("ListGroup", "<ul><li>one</li><li>two</li></ul>", order: 2),
            ]))
        XCTAssertEqual(doc.markdown(), "# Title\n\nBody text here.\n\n- one\n- two")
    }

    func testJSONRoundTrip() throws {
        let doc = Structurer().structure(
            page([
                block("SectionHeader", "<h2>Heading</h2>", order: 0),
                block("Text", "<p>A paragraph with two sentences. Here is the second.</p>", order: 1),
                block("Table", "<table><tr><td>x</td></tr></table>", order: 2),
            ]))
        let data = try JSONEncoder().encode(doc)
        let decoded = try JSONDecoder().decode(StructuredDocument.self, from: data)
        XCTAssertEqual(decoded, doc)
    }
}
