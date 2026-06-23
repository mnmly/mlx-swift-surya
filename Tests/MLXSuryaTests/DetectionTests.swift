import XCTest

@testable import MLXSurya

/// Pure-Swift tests for the detection geometry + postprocessing (`DetectionPostProcess`) and
/// config decoding. No model / Metal needed.
final class DetectionTests: XCTestCase {

    func testConfigDecodesFromJSON() throws {
        let json = """
            {"num_classes":2,"widths":[32,64,128,256,512],"depths":[1,1,1,6,6],
             "strides":[2,2,2,2,2],"head_dim":32,"num_stages":4,
             "decoder_layer_hidden_size":128,"decoder_hidden_size":512,"layer_norm_eps":1e-6}
            """
        let cfg = try JSONDecoder().decode(EfficientViTConfig.self, from: Data(json.utf8))
        XCTAssertEqual(cfg.widths, [32, 64, 128, 256, 512])
        XCTAssertEqual(cfg.depths, [1, 1, 1, 6, 6])
        XCTAssertEqual(cfg.numLabels, 2)
        XCTAssertEqual(cfg.headDim, 32)
    }

    func testConnectedComponents4() {
        // 5×3 grid, two separate components (left 2×2 block, right single pixel).
        // .  rows top→bottom
        let w = 5, h = 3
        var m = [Bool](repeating: false, count: w * h)
        func set(_ x: Int, _ y: Int) { m[y * w + x] = true }
        set(0, 0); set(1, 0); set(0, 1); set(1, 1)  // 2×2 block
        set(4, 2)  // single
        let comps = DetectionPostProcess.connectedComponents4(m, w: w, h: h)
        XCTAssertEqual(comps.count, 2)
        let big = comps.max { $0.pixels.count < $1.pixels.count }!
        XCTAssertEqual(big.pixels.count, 4)
        XCTAssertEqual([big.minX, big.minY, big.maxX, big.maxY], [0, 0, 1, 1])
    }

    func testConvexHullAndMinAreaBox() {
        // Axis-aligned rectangle point cloud 0..10 × 0..4.
        var pts: [SIMD2<Float>] = []
        for x in 0...10 { for y in 0...4 { pts.append(SIMD2(Float(x), Float(y))) } }
        let hull = DetectionPostProcess.convexHull(&pts)
        XCTAssertGreaterThanOrEqual(hull.count, 4)
        let box = DetectionPostProcess.minAreaBox(hull)
        XCTAssertEqual(box.count, 4)
        // Min-area box of a 10×4 rectangle has area ≈ 40.
        let xs = box.map { $0.x }, ys = box.map { $0.y }
        let area = (xs.max()! - xs.min()!) * (ys.max()! - ys.min()!)
        XCTAssertEqual(area, 40, accuracy: 2)
    }

    func testClockwiseStartsAtMinSumCorner() {
        let box = [SIMD2<Float>(10, 0), SIMD2(10, 5), SIMD2(0, 5), SIMD2(0, 0)]
        let cw = DetectionPostProcess.clockwise(box)
        XCTAssertEqual(cw[0], SIMD2<Float>(0, 0))  // min x+y corner first
    }

    func testDilateGrowsRegion() {
        // single pixel at center of 5×5 → radius-1 dilation = 3×3 block (9 px).
        let w = 5, h = 5
        var m = [Bool](repeating: false, count: w * h)
        m[2 * w + 2] = true
        let d = DetectionPostProcess.dilate(m, w: w, h: h, radius: 1)
        XCTAssertEqual(d.filter { $0 }.count, 9)
    }

    func testDynamicThresholdsClamp() {
        // Uniform high heatmap → scaling≈1 → thresholds ≈ inputs (within clamp).
        let hm = [Float](repeating: 0.7, count: 1000)
        let (text, low) = DetectionPostProcess.dynamicThresholds(
            hm, textThreshold: 0.6, lowText: 0.35)
        XCTAssertEqual(text, 0.6, accuracy: 0.05)
        XCTAssertEqual(low, 0.35, accuracy: 0.05)
    }
}
