import Foundation

/// CRAFT-style heatmap → text-line polygons. Pure-Swift port of
/// `surya.detection.heatmap.detect_boxes` (cv2 connectedComponents + dilate + minAreaRect),
/// plus the dynamic thresholding and box rescale/expand from `get_and_clean_boxes` /
/// `parallel_get_boxes`.
enum DetectionPostProcess {

    /// Run the full post-process: heatmap (row-major `[hmH*hmW]`, values 0..1) → boxes in
    /// original-image pixels.
    static func boxes(
        heatmap: [Float], hmW: Int, hmH: Int, imageW: Int, imageH: Int,
        textThreshold: Float, lowText: Float, yExpandMargin: Float
    ) -> [PolygonBox] {
        let (textT, lowT) = dynamicThresholds(heatmap, textThreshold: textThreshold, lowText: lowText)

        // Binary mask + 4-connected components.
        var mask = [Bool](repeating: false, count: hmW * hmH)
        for i in 0..<(hmW * hmH) { mask[i] = heatmap[i] > lowT }
        let comps = connectedComponents4(mask, w: hmW, h: hmH)

        var rawBoxes: [[SIMD2<Float>]] = []
        var confs: [Float] = []
        var maxConf: Float = 0

        for comp in comps {
            if comp.pixels.count < 10 { continue }
            let cw = comp.maxX - comp.minX + 1
            let ch = comp.maxY - comp.minY + 1
            let niter = Int(Double(min(cw, ch)).squareRoot())
            let buffer = 1

            var lineMax: Float = 0
            for p in comp.pixels { lineMax = max(lineMax, heatmap[p]) }
            if lineMax < textT { continue }

            let sx = max(0, comp.minX - niter - buffer)
            let sy = max(0, comp.minY - niter - buffer)
            let ex = min(hmW, comp.maxX + niter + buffer + 1)
            let ey = min(hmH, comp.maxY + niter + buffer + 1)
            let rw = ex - sx, rh = ey - sy
            if rw <= 0 || rh <= 0 { continue }

            // Local component mask, then dilate (square kernel, ksize = buffer + niter).
            var local = [Bool](repeating: false, count: rw * rh)
            for p in comp.pixels {
                let px = p % hmW, py = p / hmW
                local[(py - sy) * rw + (px - sx)] = true
            }
            let radius = max(0, (buffer + niter) / 2)
            let dilated = radius > 0 ? dilate(local, w: rw, h: rh, radius: radius) : local

            var pts: [SIMD2<Float>] = []
            var dlMinX = Float(hmW), dlMinY = Float(hmH), dlMaxX: Float = 0, dlMaxY: Float = 0
            for j in 0..<(rw * rh) where dilated[j] {
                let x = Float((j % rw) + sx), y = Float((j / rw) + sy)
                pts.append(SIMD2<Float>(x, y))
                dlMinX = min(dlMinX, x); dlMinY = min(dlMinY, y)
                dlMaxX = max(dlMaxX, x); dlMaxY = max(dlMaxY, y)
            }
            if pts.count < 4 { continue }

            var hull = convexHull(&pts)
            if hull.count < 3 { continue }
            var box = minAreaBox(hull)

            // Diamond-shape alignment: near-square boxes → axis-aligned from contour bounds.
            let wEdge = simd_distance(box[0], box[1])
            let hEdge = simd_distance(box[1], box[2])
            let ratio = max(wEdge, hEdge) / (min(wEdge, hEdge) + 1e-5)
            if abs(1 - ratio) <= 0.1 {
                box = [
                    SIMD2(dlMinX, dlMinY), SIMD2(dlMaxX, dlMinY),
                    SIMD2(dlMaxX, dlMaxY), SIMD2(dlMinX, dlMaxY),
                ]
            }

            // Clockwise order starting from the min-sum corner.
            box = clockwise(box)
            maxConf = max(maxConf, lineMax)
            confs.append(lineMax)
            rawBoxes.append(box)
            _ = hull
        }

        if maxConf > 0 { confs = confs.map { $0 / maxConf } }

        // Rescale heatmap-space boxes → image pixels, fit to bounds, y-expand non-vertical boxes.
        let sxScale = Float(imageW) / Float(hmW)
        let syScale = Float(imageH) / Float(hmH)
        var result: [PolygonBox] = []
        for (box, conf) in zip(rawBoxes, confs) {
            var poly = box.map { p -> [Double] in
                [
                    Double(min(max(p.x * sxScale, 0), Float(imageW))),
                    Double(min(max(p.y * syScale, 0), Float(imageH))),
                ]
            }
            var pb = PolygonBox(polygon: poly, confidence: Double(conf))
            let bb = pb.bbox
            let bw = bb[2] - bb[0], bh = bb[3] - bb[1]
            if bw <= 0 || bh <= 0 { continue }
            // Expand y for non-vertical boxes (height < 3*width).
            if bh < 3 * bw {
                let dy = Double(yExpandMargin) * bh
                poly = pb.polygon.map { [$0[0], $0[1]] }
                pb = PolygonBox(
                    polygon: [
                        [bb[0], bb[1] - dy], [bb[2], bb[1] - dy],
                        [bb[2], bb[3] + dy], [bb[0], bb[3] + dy],
                    ], confidence: Double(conf))
                pb.clamp(width: Double(imageW), height: Double(imageH))
            }
            result.append(pb)
        }
        return result
    }

    // MARK: - Dynamic thresholds

    static func dynamicThresholds(
        _ heatmap: [Float], textThreshold: Float, lowText: Float, typicalTop10Avg: Float = 0.7
    ) -> (text: Float, low: Float) {
        let n = heatmap.count
        if n == 0 { return (textThreshold, lowText) }
        let top10Count = Int(Double(n) * 0.9)
        var sorted = heatmap
        sorted.sort()
        let tail = sorted[top10Count...]
        let avg = tail.reduce(0, +) / Float(max(1, tail.count))
        let scaling = pow(min(max(avg / typicalTop10Avg, 0), 1), 0.5)
        let low = min(max(lowText * scaling, 0.1), 0.6)
        let text = min(max(textThreshold * scaling, 0.15), 0.8)
        return (text, low)
    }

    // MARK: - Connected components (4-connectivity)

    struct Component { var pixels: [Int]; var minX: Int; var minY: Int; var maxX: Int; var maxY: Int }

    static func connectedComponents4(_ mask: [Bool], w: Int, h: Int) -> [Component] {
        var visited = [Bool](repeating: false, count: w * h)
        var comps: [Component] = []
        var stack: [Int] = []
        for start in 0..<(w * h) {
            if !mask[start] || visited[start] { continue }
            stack.removeAll(keepingCapacity: true)
            stack.append(start)
            visited[start] = true
            var pixels: [Int] = []
            var minX = w, minY = h, maxX = 0, maxY = 0
            while let p = stack.popLast() {
                pixels.append(p)
                let px = p % w, py = p / w
                minX = min(minX, px); minY = min(minY, py)
                maxX = max(maxX, px); maxY = max(maxY, py)
                if px > 0 { let q = p - 1; if mask[q] && !visited[q] { visited[q] = true; stack.append(q) } }
                if px < w - 1 { let q = p + 1; if mask[q] && !visited[q] { visited[q] = true; stack.append(q) } }
                if py > 0 { let q = p - w; if mask[q] && !visited[q] { visited[q] = true; stack.append(q) } }
                if py < h - 1 { let q = p + w; if mask[q] && !visited[q] { visited[q] = true; stack.append(q) } }
            }
            comps.append(Component(pixels: pixels, minX: minX, minY: minY, maxX: maxX, maxY: maxY))
        }
        return comps
    }

    /// Separable square-kernel binary dilation (radius `r` each side).
    static func dilate(_ m: [Bool], w: Int, h: Int, radius r: Int) -> [Bool] {
        var horiz = [Bool](repeating: false, count: w * h)
        for y in 0..<h {
            let row = y * w
            for x in 0..<w where m[row + x] {
                let lo = max(0, x - r), hi = min(w - 1, x + r)
                for xx in lo...hi { horiz[row + xx] = true }
            }
        }
        var out = [Bool](repeating: false, count: w * h)
        for y in 0..<h {
            for x in 0..<w where horiz[y * w + x] {
                let lo = max(0, y - r), hi = min(h - 1, y + r)
                for yy in lo...hi { out[yy * w + x] = true }
            }
        }
        return out
    }

    // MARK: - Geometry

    /// Andrew's monotone-chain convex hull (returns CCW hull).
    static func convexHull(_ pts: inout [SIMD2<Float>]) -> [SIMD2<Float>] {
        pts.sort { $0.x < $1.x || ($0.x == $1.x && $0.y < $1.y) }
        if pts.count <= 2 { return pts }
        func cross(_ o: SIMD2<Float>, _ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }
        var lower: [SIMD2<Float>] = []
        for p in pts {
            while lower.count >= 2 && cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 {
                lower.removeLast()
            }
            lower.append(p)
        }
        var upper: [SIMD2<Float>] = []
        for p in pts.reversed() {
            while upper.count >= 2 && cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 {
                upper.removeLast()
            }
            upper.append(p)
        }
        lower.removeLast(); upper.removeLast()
        return lower + upper
    }

    /// Minimum-area enclosing rectangle of a convex polygon (rotating calipers). Returns 4 corners.
    static func minAreaBox(_ hull: [SIMD2<Float>]) -> [SIMD2<Float>] {
        let n = hull.count
        if n < 3 { return Array(repeating: hull.first ?? .zero, count: 4) }
        var bestArea = Float.greatestFiniteMagnitude
        var best = [SIMD2<Float>](repeating: .zero, count: 4)
        for i in 0..<n {
            let a = hull[i], b = hull[(i + 1) % n]
            var edge = SIMD2<Float>(b.x - a.x, b.y - a.y)
            let len = (edge.x * edge.x + edge.y * edge.y).squareRoot()
            if len < 1e-6 { continue }
            edge /= len
            let normal = SIMD2<Float>(-edge.y, edge.x)
            var minU = Float.greatestFiniteMagnitude, maxU = -Float.greatestFiniteMagnitude
            var minV = Float.greatestFiniteMagnitude, maxV = -Float.greatestFiniteMagnitude
            for p in hull {
                let u = p.x * edge.x + p.y * edge.y
                let v = p.x * normal.x + p.y * normal.y
                minU = min(minU, u); maxU = max(maxU, u)
                minV = min(minV, v); maxV = max(maxV, v)
            }
            let area = (maxU - minU) * (maxV - minV)
            if area < bestArea {
                bestArea = area
                func pt(_ u: Float, _ v: Float) -> SIMD2<Float> {
                    SIMD2<Float>(u * edge.x + v * normal.x, u * edge.y + v * normal.y)
                }
                best = [pt(minU, minV), pt(maxU, minV), pt(maxU, maxV), pt(minU, maxV)]
            }
        }
        return best
    }

    /// Reorder corners clockwise starting at the minimum-sum (top-left-ish) corner.
    static func clockwise(_ box: [SIMD2<Float>]) -> [SIMD2<Float>] {
        guard box.count == 4 else { return box }
        var startIdx = 0
        var best = Float.greatestFiniteMagnitude
        for (i, p) in box.enumerated() where p.x + p.y < best {
            best = p.x + p.y; startIdx = i
        }
        return (0..<4).map { box[($0 + startIdx) % 4] }
    }

    private static func simd_distance(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        let dx = a.x - b.x, dy = a.y - b.y
        return (dx * dx + dy * dy).squareRoot()
    }
}
