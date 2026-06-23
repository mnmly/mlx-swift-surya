import Foundation

/// A quadrilateral region with optional confidence — the geometric primitive
/// shared across detection, layout, and recognition results. Ports the core of
/// `surya.common.polygon.PolygonBox`.
///
/// Points are `[x, y]` pairs ordered clockwise from the top-left corner. The
/// axis-aligned ``bbox`` is derived on demand from the corner extents.
public struct PolygonBox: Sendable, Codable, Equatable {
    /// Corner points as `[x, y]` pairs, clockwise from top-left.
    public var polygon: [[Double]]
    /// Optional detection/recognition confidence in `0...1`.
    public var confidence: Double?

    /// Create a box from explicit corner points.
    public init(polygon: [[Double]], confidence: Double? = nil) {
        self.polygon = polygon
        self.confidence = confidence
    }

    /// Create an axis-aligned box from `[x0, y0, x1, y1]`, expanded to four
    /// clockwise corners (top-left, top-right, bottom-right, bottom-left).
    public init(bbox: [Double], confidence: Double? = nil) {
        precondition(bbox.count == 4, "bbox must be [x0, y0, x1, y1]")
        let (x0, y0, x1, y1) = (bbox[0], bbox[1], bbox[2], bbox[3])
        self.polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        self.confidence = confidence
    }

    /// Axis-aligned bounding box `[x0, y0, x1, y1]` over all corner points.
    public var bbox: [Double] {
        let xs = polygon.map { $0[0] }
        let ys = polygon.map { $0[1] }
        return [xs.min() ?? 0, ys.min() ?? 0, xs.max() ?? 0, ys.max() ?? 0]
    }

    /// Width of the axis-aligned bounding box.
    public var width: Double { let b = bbox; return b[2] - b[0] }
    /// Height of the axis-aligned bounding box.
    public var height: Double { let b = bbox; return b[3] - b[1] }
    /// Area of the axis-aligned bounding box.
    public var area: Double { width * height }

    /// Scale every corner by independent x/y factors (e.g. processor-space →
    /// image-space). Mirrors `PolygonBox.rescale`.
    public mutating func rescale(widthScale: Double, heightScale: Double) {
        polygon = polygon.map { [$0[0] * widthScale, $0[1] * heightScale] }
    }

    /// Clamp every corner into `[0, width] × [0, height]`. Mirrors `PolygonBox.fit_to_bounds`.
    public mutating func clamp(width: Double, height: Double) {
        polygon = polygon.map {
            [min(max($0[0], 0), width), min(max($0[1], 0), height)]
        }
    }
}
