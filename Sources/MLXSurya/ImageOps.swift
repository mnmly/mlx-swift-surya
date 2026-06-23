import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

#if canImport(PDFKit)
    import PDFKit
#endif

/// Image utilities for the surya pipeline. `CGImage` (top-left origin) is the
/// canonical representation. Ports `surya.input.processing` / PDF rasterization
/// and the grid-aligned `surya.inference.util.scale_to_fit`.
public enum ImageOps {
    static let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
    static let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

    /// Load an image file into a `CGImage`. Mirrors `Image.open(...).convert("RGB")`.
    public static func load(_ url: URL) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw SuryaError.imageLoadFailed(url) }
        return cg
    }

    /// Number of pages in a PDF, or nil if not a readable PDF.
    public static func pdfPageCount(_ url: URL) -> Int? {
        #if canImport(PDFKit)
            return PDFDocument(url: url)?.pageCount
        #else
            return nil
        #endif
    }

    /// Render one PDF page at the given DPI (surya uses `IMAGE_DPI` / `IMAGE_DPI_HIGHRES`).
    public static func renderPDFPage(_ url: URL, page: Int, imageDPI: CGFloat) throws -> CGImage {
        #if canImport(PDFKit)
            guard let doc = PDFDocument(url: url), let pdfPage = doc.page(at: page) else {
                throw SuryaError.pdfPageUnavailable(url, page)
            }
            let bounds = pdfPage.bounds(for: .mediaBox)
            let scale = imageDPI / 72.0
            let w = max(1, Int((bounds.width * scale).rounded()))
            let h = max(1, Int((bounds.height * scale).rounded()))
            guard
                let ctx = CGContext(
                    data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                    space: sRGB, bitmapInfo: bitmapInfo)
            else { throw SuryaError.contextCreationFailed }
            ctx.setFillColor(gray: 1, alpha: 1)
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            ctx.interpolationQuality = .high
            ctx.scaleBy(x: scale, y: scale)
            ctx.translateBy(x: -bounds.origin.x, y: -bounds.origin.y)
            pdfPage.draw(with: .mediaBox, to: ctx)
            guard let cg = ctx.makeImage() else { throw SuryaError.contextCreationFailed }
            return cg
        #else
            throw SuryaError.pdfUnsupported
        #endif
    }

    /// Resize to an exact pixel size (aspect-distorting) with high-quality interpolation.
    /// Python uses PIL `LANCZOS`; CoreGraphics `.high` is a close (not bit-identical) substitute.
    public static func resizeExact(_ cg: CGImage, width: Int, height: Int) -> CGImage {
        guard
            let ctx = CGContext(
                data: nil, width: max(1, width), height: max(1, height), bitsPerComponent: 8,
                bytesPerRow: 0, space: sRGB, bitmapInfo: bitmapInfo)
        else { return cg }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage() ?? cg
    }

    /// Crop a pixel-coordinate rect (top-left origin), clamped to bounds. Mirrors `PIL.Image.crop`.
    public static func cropPixels(_ cg: CGImage, _ rect: CGRect) -> CGImage? {
        let bounds = CGRect(x: 0, y: 0, width: cg.width, height: cg.height)
        let r = rect.integral.intersection(bounds)
        if r.width < 1 || r.height < 1 { return nil }
        return cg.cropping(to: r)
    }

    /// Write a `CGImage` to disk, encoded by file extension (`.png`, `.jpg`, `.webp`).
    public static func write(_ cg: CGImage, to url: URL) throws {
        let ext = url.pathExtension.lowercased()
        let type: UTType =
            ext == "jpg" || ext == "jpeg" ? .jpeg : ext == "webp" ? .webP : .png
        guard
            let dest = CGImageDestinationCreateWithURL(
                url as CFURL, type.identifier as CFString, 1, nil)
        else { throw SuryaError.contextCreationFailed }
        CGImageDestinationAddImage(dest, cg, nil)
        if !CGImageDestinationFinalize(dest) { throw SuryaError.contextCreationFailed }
    }

    // MARK: - scale_to_fit

    /// surya's full-resolution VLM pixel cap (`scale_to_fit` max, 3072×2048 = ~6.3 MP).
    public static let defaultMaxImagePixels = 3072 * 2048
    /// surya's VLM minimum pixel floor (`scale_to_fit` min, 1792×28).
    public static let defaultMinImagePixels = 1792 * 28

    /// Grid-aligned resize ported from `surya.inference.util.scale_to_fit`.
    ///
    /// Guarantees output dimensions are multiples of `gridSize` (28) and that the total pixel
    /// count stays within `[minPixels, maxPixels]`, preferring moves that best preserve the
    /// original aspect ratio. Lowering `maxPixels` reduces the VLM's image-token count — the main
    /// lever for OCR/layout/table speed (at an accuracy cost on small text). Returns the
    /// (possibly unchanged) resized image.
    public static func scaleToFit(
        _ cg: CGImage,
        maxPixels: Int = defaultMaxImagePixels,
        minPixels: Int = defaultMinImagePixels,
        gridSize: Int = 28
    ) -> CGImage {
        let width = cg.width, height = cg.height
        if width <= 0 || height <= 0 { return cg }

        let originalAR = Double(width) / Double(height)
        let currentPixels = Double(width * height)
        let maxP = Double(maxPixels)
        let minP = Double(minPixels)

        var scale = 1.0
        if currentPixels > maxP {
            scale = (maxP / currentPixels).squareRoot()
        } else if currentPixels < minP {
            scale = (minP / currentPixels).squareRoot()
        }

        let g = Double(gridSize)
        var wBlocks = max(1, Int((Double(width) * scale / g).rounded(.toNearestOrEven)))
        var hBlocks = max(1, Int((Double(height) * scale / g).rounded(.toNearestOrEven)))

        while Double(wBlocks * hBlocks * gridSize * gridSize) > maxP {
            if wBlocks == 1 && hBlocks == 1 { break }
            if wBlocks == 1 { hBlocks -= 1; continue }
            if hBlocks == 1 { wBlocks -= 1; continue }
            let arWLoss = abs(Double(wBlocks - 1) / Double(hBlocks) - originalAR)
            let arHLoss = abs(Double(wBlocks) / Double(hBlocks - 1) - originalAR)
            if arWLoss < arHLoss { wBlocks -= 1 } else { hBlocks -= 1 }
        }

        let newWidth = wBlocks * gridSize
        let newHeight = hBlocks * gridSize
        if newWidth == width && newHeight == height { return cg }
        return resizeExact(cg, width: newWidth, height: newHeight)
    }
}
