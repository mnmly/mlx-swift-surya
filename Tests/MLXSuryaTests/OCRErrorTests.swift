import XCTest

@testable import MLXSurya

/// Pure-Swift tests for the DistilBERT OCR-error config decoding (no model / Metal).
final class OCRErrorTests: XCTestCase {

    func testDistilBertConfigDecodes() throws {
        let json = """
            {"activation":"gelu","dim":768,"hidden_dim":3072,"n_heads":12,"n_layers":6,
             "max_position_embeddings":512,"vocab_size":119547,"model_type":"distilbert"}
            """
        let cfg = try JSONDecoder().decode(DistilBertConfig.self, from: Data(json.utf8))
        XCTAssertEqual(cfg.dim, 768)
        XCTAssertEqual(cfg.hiddenDim, 3072)
        XCTAssertEqual(cfg.nLayers, 6)
        XCTAssertEqual(cfg.nHeads, 12)
        XCTAssertEqual(cfg.vocabSize, 119_547)
        XCTAssertEqual(cfg.numLabels, 2)
    }

    func testOCRErrorLabels() {
        XCTAssertEqual(OCRErrorEngine.id2label, ["good", "bad"])
    }
}
