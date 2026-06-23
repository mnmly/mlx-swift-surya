import XCTest

@testable import MLXSurya

/// Skeleton-level contract tests. These exercise the pure-Swift utilities that
/// are real today (no MLX / Metal), so they run under `swift test` as well as
/// Xcode. Model-backed tests arrive with each slice.
final class SkeletonTests: XCTestCase {

    func testParsePageRange() {
        XCTAssertEqual(parsePageRange("1-5,7,9-12"), [1, 2, 3, 4, 5, 7, 9, 10, 11, 12])
        XCTAssertEqual(parsePageRange("3"), [3])
        XCTAssertEqual(parsePageRange("2,2,1"), [1, 2])
        XCTAssertEqual(parsePageRange(""), [])
    }

    func testPolygonBoxFromBBox() {
        let box = PolygonBox(bbox: [10, 20, 110, 220])
        XCTAssertEqual(box.polygon, [[10, 20], [110, 20], [110, 220], [10, 220]])
        XCTAssertEqual(box.bbox, [10, 20, 110, 220])
        XCTAssertEqual(box.width, 100)
        XCTAssertEqual(box.height, 200)
        XCTAssertEqual(box.area, 20_000)
    }

    func testPolygonBoxRescaleAndClamp() {
        var box = PolygonBox(bbox: [10, 10, 20, 20])
        box.rescale(widthScale: 2, heightScale: 3)
        XCTAssertEqual(box.bbox, [20, 30, 40, 60])
        box.clamp(width: 35, height: 100)
        XCTAssertEqual(box.bbox, [20, 30, 35, 60])
    }

    func testConfigDefaultsMatchUpstream() {
        let cfg = SuryaConfiguration()
        XCTAssertEqual(cfg.modelCheckpoint, "datalab-to/surya-ocr-2")
        XCTAssertEqual(cfg.bboxScale, 1000)
        XCTAssertEqual(cfg.maxTokensFullPage, 12288)
    }

    func testSessionLoadsOffline() async throws {
        // load() is cheap + offline — models load lazily on first use.
        let session = try await SuryaSession.load(SuryaSessionConfig())
        XCTAssertEqual(session.config.configuration.modelCheckpoint, "datalab-to/surya-ocr-2")
    }
}
