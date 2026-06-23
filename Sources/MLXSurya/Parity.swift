import Foundation
import MLX

/// Numerical-parity helpers: load pinned Python reference tensors (safetensors) and compare the
/// Swift/MLX forward pass against them. Used by the `surya-cli parity` subcommand.
public enum SuryaParity {
    /// One parity check outcome.
    public struct Result: Sendable {
        public let name: String
        public let maxAbsDiff: Float
        public let tolerance: Float
        public var pass: Bool { maxAbsDiff <= tolerance }
        public let detail: String
    }

    /// OCR-error (DistilBERT) logit parity. `ref` holds `input_ids` + Python `logits`.
    public static func ocrError(refURL: URL, engine: OCRErrorEngine, tolerance: Float = 0.1) throws
        -> Result
    {
        let a = try MLX.loadArrays(url: refURL)
        guard let idsA = a["input_ids"], let refA = a["logits"] else {
            throw SuryaError.missingPinnedInputs("input_ids/logits in \(refURL.lastPathComponent)")
        }
        let ids = idsA.reshaped([-1]).asType(.int32).asArray(Int32.self).map(Int.init)
        let swift = engine.logits(inputIds: ids)
        let ref = refA.reshaped([-1]).asArray(Float.self)
        let maxDiff = zip(swift, ref).map { abs($0 - $1) }.max() ?? .greatestFiniteMagnitude
        return Result(
            name: "ocr-error logits", maxAbsDiff: maxDiff, tolerance: tolerance,
            detail: "swift=\(fmt(swift)) python=\(fmt(ref))")
    }

    /// Detection (EfficientViT) heatmap parity. `ref` holds NCHW `pixel_values` + sigmoid
    /// `heatmap` `(1, numLabels, H/4, W/4)`. The Swift forward runs NHWC; channels are compared.
    public static func detection(refURL: URL, engine: DetectionEngine, tolerance: Float = 0.02)
        throws -> Result
    {
        let a = try MLX.loadArrays(url: refURL)
        guard let pv = a["pixel_values"], let refHeat = a["heatmap"] else {
            throw SuryaError.missingPinnedInputs("pixel_values/heatmap in \(refURL.lastPathComponent)")
        }
        // NCHW (1,3,H,W) → NHWC (1,H,W,3).
        let nhwc = pv.transposed(0, 2, 3, 1)
        let out = engine.rawForward(nhwc.asType(.float32))  // (1, h, w, C) NHWC sigmoid
        // refHeat NCHW (1,C,h,w) → NHWC (1,h,w,C) for an aligned comparison.
        let refNHWC = refHeat.transposed(0, 2, 3, 1).asType(.float32)
        let diff = MLX.abs(out - refNHWC)
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = diff.mean().item(Float.self)
        // Per-channel breakdown (channel 0 = text heatmap used for boxes; channel 1 = affinity).
        let ch0 = diff[0..., 0..., 0..., 0]
        let ch1 = diff[0..., 0..., 0..., 1]
        let c0max = ch0.max().item(Float.self), c0mean = ch0.mean().item(Float.self)
        let c1max = ch1.max().item(Float.self), c1mean = ch1.mean().item(Float.self)
        let frac = (MLX.sum(diff .> 0.1).item(Int.self))
        return Result(
            name: "detection heatmap", maxAbsDiff: maxDiff, tolerance: tolerance,
            detail: String(
                format:
                    "mean=%.5f | ch0(text) max=%.4f mean=%.5f | ch1(affinity) max=%.4f mean=%.5f | px>0.1=%d/%d",
                meanDiff, c0max, c0mean, c1max, c1mean, frac, out.dim(1) * out.dim(2) * 2))
    }

    /// Stage-by-stage detection bisection: compare each encoder stage + the decode logits against
    /// pinned Python intermediates. The first stage with a large relative diff is the divergence.
    public static func detectionStages(pixelValuesURL: URL, stagesURL: URL, engine: DetectionEngine)
        throws -> [Result]
    {
        guard let pv = try MLX.loadArrays(url: pixelValuesURL)["pixel_values"] else {
            throw SuryaError.missingPinnedInputs("pixel_values")
        }
        let refs = try MLX.loadArrays(url: stagesURL)
        let nhwc = pv.transposed(0, 2, 3, 1).asType(.float32)
        let states = engine.encoderStates(nhwc)

        func compare(_ name: String, _ swift: MLXArray, _ refNCHW: MLXArray) -> Result {
            let refN = refNCHW.transposed(0, 2, 3, 1).asType(.float32)
            let d = MLX.abs(swift - refN)
            let maxDiff = d.max().item(Float.self)
            let refMax = MLX.abs(refN).max().item(Float.self)
            let rel = refMax > 0 ? maxDiff / refMax : maxDiff
            return Result(
                name: name, maxAbsDiff: maxDiff, tolerance: 1e-2,
                detail: String(
                    format: "rel=%.4f mean=%.5f refMax=%.3f shape=%@", rel,
                    d.mean().item(Float.self), refMax, "\(swift.shape)"))
        }

        var results: [Result] = []
        // Stem internals first (divergence is early), then stages.
        let stem = engine.stemTrace(nhwc)
        for k in ["in_conv_conv", "in_conv_bn", "in_conv", "stem"] {
            if let r = refs[k], let s = stem[k] { results.append(compare(k, s, r)) }
        }
        for i in 0..<states.count {
            if let r = refs["stage\(i)"] { results.append(compare("stage\(i)", states[i], r)) }
        }
        if let dl = refs["decode_logits"] {
            results.append(compare("decode_logits", engine.decodeLogits(nhwc), dl))
        }
        return results
    }

    /// VLM input parity: render the chat template + tokenize a fixed image+prompt with
    /// ``SuryaWordLevelTokenizer`` and assert the `input_ids` match the Python reference exactly
    /// (single `<|image_pad|>`, pre-grid-expansion). Validates the WordLevel tokenizer + Jinja
    /// chat template.
    public static func vlmInputs(refURL: URL, tokenizer: SuryaWordLevelTokenizer) throws -> Result {
        let obj =
            (try JSONSerialization.jsonObject(with: Data(contentsOf: refURL))) as? [String: Any]
        guard let raw = obj?["input_ids"] as? [Any] else {
            throw SuryaError.missingPinnedInputs("input_ids in \(refURL.lastPathComponent)")
        }
        let refIds = raw.compactMap { ($0 as? NSNumber)?.intValue }
        let prompt =
            "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000)."
        let content: [[String: String]] = [["type": "image"], ["type": "text", "text": prompt]]
        let messages: [[String: any Sendable]] = [["role": "user", "content": content]]
        let ids = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)

        let match = ids == refIds
        var firstDiff = -1
        for i in 0..<min(ids.count, refIds.count) where ids[i] != refIds[i] { firstDiff = i; break }
        let detail =
            match
            ? "input_ids identical (\(ids.count) tokens)"
            : "swift n=\(ids.count) python n=\(refIds.count) firstDiff@\(firstDiff)"
        return Result(
            name: "vlm input_ids", maxAbsDiff: match ? 0 : 1, tolerance: 0, detail: detail)
    }

    private static func fmt(_ v: [Float]) -> String {
        "[" + v.map { String(format: "%.4f", $0) }.joined(separator: ", ") + "]"
    }
}
