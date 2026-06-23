import Foundation

/// Errors surfaced by the MLXSurya pipeline.
public enum SuryaError: Error, CustomStringConvertible {
    /// A pipeline stage that is scaffolded but not yet ported. The associated
    /// string names the slice that will implement it (e.g. `"detection model"`).
    case notImplemented(String)
    /// The model snapshot directory is missing a required file.
    case missingModelFile(String)
    /// No model snapshot was provided and none is cached. The string suggests how
    /// to obtain one (e.g. `SuryaModel.download()`).
    case modelNotAvailable(String)
    /// An input file could not be loaded or rasterized.
    case invalidInput(String)
    /// A Hugging Face / S3 repo id could not be parsed.
    case invalidRepoID(String)
    /// An image file failed to decode.
    case imageLoadFailed(URL)
    /// A PDF page could not be rendered.
    case pdfPageUnavailable(URL, Int)
    /// PDF rendering is unavailable on this platform.
    case pdfUnsupported
    /// A CoreGraphics context could not be created.
    case contextCreationFailed
    /// An unsupported input file type was supplied.
    case unsupportedFileType(String)
    /// Pinned parity inputs were not found at the given path.
    case missingPinnedInputs(String)
    /// A prefill forward unexpectedly produced tokens instead of logits.
    case prefillProducedTokens

    public var description: String {
        switch self {
        case .notImplemented(let what): return "Not implemented yet: \(what)"
        case .missingModelFile(let f): return "Missing model file: \(f)"
        case .modelNotAvailable(let hint): return "Model not available. \(hint)"
        case .invalidInput(let s): return "Invalid input: \(s)"
        case .invalidRepoID(let s): return "Invalid repo id: \(s)"
        case .imageLoadFailed(let u): return "Failed to load image: \(u.path)"
        case .pdfPageUnavailable(let u, let p): return "PDF page \(p) unavailable in \(u.path)"
        case .pdfUnsupported: return "PDF rendering requires PDFKit (macOS/iOS)."
        case .contextCreationFailed: return "Failed to create CoreGraphics context."
        case .unsupportedFileType(let e): return "Unsupported file type: \(e)"
        case .missingPinnedInputs(let p): return "Pinned parity inputs not found at \(p)."
        case .prefillProducedTokens: return "Prefill returned tokens instead of logits."
        }
    }
}
