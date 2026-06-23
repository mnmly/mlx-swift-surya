import CoreGraphics
import Foundation
import MLX

/// Preprocessing for the EfficientViT detector. Ports `DetectionPredictor.prepare_image` and
/// `surya.detection.util.split_image`.
enum DetectionPreprocess {
    // ImageNet normalization (SegFormer processor defaults).
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    /// Split a tall image into `chunkHeight`-tall chunks (only when height exceeds `maxHeight`).
    /// The last chunk is padded to `chunkHeight` with white. Returns `(chunk, realHeight)` pairs.
    /// Ports `split_image`.
    static func splitImage(
        _ cg: CGImage, chunkHeight: Int = 1200, maxHeight: Int = 1400
    ) -> [(image: CGImage, realHeight: Int)] {
        let w = cg.width, h = cg.height
        if h <= maxHeight { return [(cg, h)] }
        let numSplits = Int(ceil(Double(h) / Double(chunkHeight)))
        var out: [(CGImage, Int)] = []
        for i in 0..<numSplits {
            let top = i * chunkHeight
            let bottom = min((i + 1) * chunkHeight, h)
            let realH = bottom - top
            guard let crop = cg.cropping(to: CGRect(x: 0, y: top, width: w, height: realH)) else {
                continue
            }
            if realH < chunkHeight {
                out.append((padBottom(crop, toHeight: chunkHeight), realH))
            } else {
                out.append((crop, realH))
            }
        }
        return out
    }

    /// Pad an image's bottom with white to reach `toHeight` (top-left anchored). Mirrors
    /// `PIL.ImageOps.pad(..., color=255, centering=(0,0))` for the vertical case.
    private static func padBottom(_ cg: CGImage, toHeight: Int) -> CGImage {
        let w = cg.width
        guard
            let ctx = CGContext(
                data: nil, width: w, height: toHeight, bitsPerComponent: 8, bytesPerRow: 0,
                space: ImageOps.sRGB, bitmapInfo: ImageOps.bitmapInfo)
        else { return cg }
        ctx.setFillColor(gray: 1, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: toHeight))
        // CoreGraphics origin is bottom-left; draw the crop at the top.
        ctx.draw(cg, in: CGRect(x: 0, y: toHeight - cg.height, width: w, height: cg.height))
        return ctx.makeImage() ?? cg
    }

    /// `prepare_image`: thumbnail (aspect-preserving downscale) to fit `size`, then stretch to
    /// exactly `size×size`, rescale to [0,1], ImageNet-normalize. Returns NHWC `(1, size, size, 3)`.
    static func prepare(_ cg: CGImage, size: Int = 1200) -> MLXArray {
        let w = cg.width, h = cg.height
        var img = cg
        // thumbnail: only downscale, preserve aspect ratio.
        let s = min(Double(size) / Double(w), Double(size) / Double(h))
        if s < 1 {
            let tw = max(1, Int((Double(w) * s).rounded()))
            let th = max(1, Int((Double(h) * s).rounded()))
            img = ImageOps.resizeExact(img, width: tw, height: th)
        }
        // stretch to exactly size×size.
        img = ImageOps.resizeExact(img, width: size, height: size)
        return normalizedNHWC(img)
    }

    /// Render a (size×size) CGImage to a normalized NHWC float32 `MLXArray` `(1, H, W, 3)`.
    static func normalizedNHWC(_ cg: CGImage) -> MLXArray {
        let w = cg.width, h = cg.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        rgba.withUnsafeMutableBytes { buf in
            if let ctx = CGContext(
                data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: w * 4, space: ImageOps.sRGB, bitmapInfo: ImageOps.bitmapInfo)
            {
                ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            }
        }
        var floats = [Float](repeating: 0, count: w * h * 3)
        for p in 0..<(w * h) {
            let r = Float(rgba[p * 4 + 0]) / 255
            let g = Float(rgba[p * 4 + 1]) / 255
            let b = Float(rgba[p * 4 + 2]) / 255
            floats[p * 3 + 0] = (r - mean[0]) / std[0]
            floats[p * 3 + 1] = (g - mean[1]) / std[1]
            floats[p * 3 + 2] = (b - mean[2]) / std[2]
        }
        return MLXArray(floats, [1, h, w, 3])
    }
}
