import Foundation
import MLX
import MLXNN
import Tokenizers

/// Acquires the DistilBERT OCR-error checkpoint from the datalab model store, mirroring
/// `surya.common.s3`.
public enum OCRErrorModel {
    /// The default checkpoint path (matches `settings.OCR_ERROR_MODEL_CHECKPOINT`).
    public static let checkpoint = "ocr_error_detection/2025_02_18"
    static let s3Base = "https://models.datalab.to"
    static let requiredFiles = [
        "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
        "vocab.txt", "special_tokens_map.json",
    ]

    /// Local cache directory: `~/Library/Caches/datalab/models/<checkpoint>`.
    public static func cacheDirectory(_ checkpoint: String = checkpoint) -> URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Caches/datalab/models")
            .appendingPathComponent(checkpoint)
    }

    /// Ensure all required files are present locally, downloading if needed.
    @discardableResult
    public static func download(_ checkpoint: String = checkpoint) async throws -> URL {
        let dir = cacheDirectory(checkpoint)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in requiredFiles {
            let dest = dir.appendingPathComponent(file)
            if FileManager.default.fileExists(atPath: dest.path) { continue }
            guard let url = URL(string: "\(s3Base)/\(checkpoint)/\(file)") else {
                throw SuryaError.invalidRepoID("\(s3Base)/\(checkpoint)/\(file)")
            }
            let (tmp, response) = try await URLSession.shared.download(from: url)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                throw SuryaError.modelNotAvailable("HTTP \(http.statusCode) for \(file)")
            }
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: tmp, to: dest)
        }
        return dir
    }
}

/// Loads the DistilBERT OCR-error classifier + its WordPiece tokenizer and classifies text spans
/// as good/bad. Ports `surya.ocr_error.OCRErrorPredictor`. Single-caller contract.
public final class OCRErrorEngine {
    let model: DistilBertForSequenceClassification
    let tokenizer: any Tokenizer
    let maxLength: Int

    static let id2label = ["good", "bad"]

    /// Build from a snapshot directory (config.json + model.safetensors + tokenizer files).
    public init(modelDirectory: URL, dtype: DType = .float32, maxLength: Int = 512) async throws {
        let config: DistilBertConfig
        if let data = try? Data(contentsOf: modelDirectory.appendingPathComponent("config.json")) {
            config = (try? JSONDecoder().decode(DistilBertConfig.self, from: data))
                ?? DistilBertConfig()
        } else {
            config = DistilBertConfig()
        }
        self.model = DistilBertForSequenceClassification(config)
        self.maxLength = maxLength

        // Weights are all Linear/Embedding/LayerNorm — load directly, no transpose.
        let weights = try MLX.loadArrays(
            url: modelDirectory.appendingPathComponent("model.safetensors"))
        let remapped = weights.map { ($0.key, $0.value.asType(dtype)) }
        try model.update(
            parameters: ModuleParameters.unflattened(remapped), verify: [.noUnusedKeys])
        MLX.eval(model.parameters())

        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)
    }

    /// Raw classifier logits for pinned token ids (no tokenizer) — for numerical parity tests.
    public func logits(inputIds: [Int]) -> [Float] {
        let input = MLXArray(inputIds.map { Int32($0) }, [1, inputIds.count])
        let out = model(input)
        MLX.eval(out)
        return out[0].asArray(Float.self)
    }

    /// Classify one text span. Returns the predicted label (`"good"`/`"bad"`) + its probability.
    public func detect(_ text: String) -> OCRErrorVerdict {
        var ids = tokenizer.encode(text: text)  // includes [CLS] … [SEP]
        if ids.count > maxLength { ids = Array(ids.prefix(maxLength)) }
        let input = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        let probs = softmax(model(input), axis: -1)
        MLX.eval(probs)
        let p = probs[0].asArray(Float.self)
        let argmax = p[1] > p[0] ? 1 : 0
        return OCRErrorVerdict(label: Self.id2label[argmax], confidence: Double(p[argmax]))
    }
}
