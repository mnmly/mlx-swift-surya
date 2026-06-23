import XCTest

@testable import MLXSurya

/// Pure-Swift tests for the VLM output parsers (`SuryaParsers`). No model / Metal needed —
/// these pin the port of `surya/inference/parsers.py`.
final class ParserTests: XCTestCase {

    func testDenormBbox() {
        // 0..1000 normalized → pixels on a 2000×1000 image.
        let px = SuryaParsers.denormBbox([100, 200, 500, 800], imageWidth: 2000, imageHeight: 1000)
        XCTAssertEqual(px, [200, 200, 1000, 800])
    }

    func testParseLayoutJSON() {
        let raw = """
            ```json
            [
              {"label": "Section-Header", "bbox": "0 0 1000 100", "count": 12},
              {"label": "Image", "bbox": "0 100 500 600", "count": 0}
            ]
            ```
            """
        let blocks = SuryaParsers.parseLayout(raw, imageWidth: 1000, imageHeight: 1000)
        XCTAssertEqual(blocks.count, 2)
        XCTAssertEqual(blocks[0].label, "Section-Header")
        XCTAssertEqual(blocks[0].bbox, [0, 0, 1000, 100])
        XCTAssertEqual(blocks[0].count, 12)
        // Canonicalization happens in the pipeline, not the parser.
        XCTAssertEqual(SuryaPrompts.canonicalLabel(blocks[0].label), "SectionHeader")
        XCTAssertEqual(SuryaPrompts.canonicalLabel(blocks[1].label), "Picture")
    }

    func testParseLayoutBboxAsArray() {
        let raw = #"[{"label":"Text","bbox":[10,20,30,40],"count":5}]"#
        let blocks = SuryaParsers.parseLayout(raw, imageWidth: 1000, imageHeight: 1000)
        XCTAssertEqual(blocks.count, 1)
        XCTAssertEqual(blocks[0].bbox, [10, 20, 30, 40])
    }

    func testParseTableRecFiltersLabels() {
        let raw = """
            [{"label":"Row","bbox":"0 0 1000 50"},
             {"label":"Col","bbox":"0 0 200 1000"},
             {"label":"Junk","bbox":"0 0 1 1"}]
            """
        let els = SuryaParsers.parseTableRec(raw, imageWidth: 1000, imageHeight: 1000)
        XCTAssertEqual(els.count, 2)
        XCTAssertEqual(els.map(\.label), ["Row", "Col"])
    }

    func testParseFullPageHTMLTopLevelDivs() {
        let raw = """
            <div data-bbox="0 0 1000 100" data-label="Text">
              <p data-bbox="10 10 20 20">Hello <b>world</b></p>
            </div>
            <div data-bbox="0 100 1000 200" data-label="Image"></div>
            """
        let blocks = SuryaParsers.parseFullPageHTML(raw, imageWidth: 1000, imageHeight: 2000)
        XCTAssertEqual(blocks.count, 2)
        XCTAssertEqual(blocks[0].label, "Text")
        XCTAssertEqual(blocks[0].bbox, [0, 0, 1000, 200])  // y scaled by 2000/1000
        // Inner data-bbox stripped from children, content preserved.
        XCTAssertFalse(blocks[0].html.contains("data-bbox"))
        XCTAssertTrue(blocks[0].html.contains("Hello"))
        XCTAssertTrue(blocks[0].html.contains("<b>world</b>"))
        XCTAssertEqual(blocks[1].label, "Image")
    }

    func testParseFullPageHTMLNestedDivDepth() {
        // A nested <div> must not prematurely close the top-level block.
        let raw = """
            <div data-bbox="0 0 100 100" data-label="Table">
              <div data-bbox="0 0 50 50" data-label="Cell">inner</div>
            </div>
            """
        let blocks = SuryaParsers.parseFullPageHTML(raw, imageWidth: 100, imageHeight: 100)
        XCTAssertEqual(blocks.count, 1)
        XCTAssertEqual(blocks[0].label, "Table")
        XCTAssertTrue(blocks[0].html.contains("inner"))
    }

    func testCleanBlockHTMLStripsFences() {
        let raw = "```html\n<p>Line</p>\n```"
        XCTAssertEqual(SuryaParsers.cleanBlockHTML(raw), "<p>Line</p>")
    }
}
