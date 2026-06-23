import CoreGraphics
import Foundation
import MLX

/// Acquires the EfficientViT detection checkpoint from the datalab model store
/// (`https://models.datalab.to/text_detection/...`), mirroring `surya.common.s3`.
public enum DetectionModel {
    /// The default checkpoint path (matches `settings.DETECTOR_MODEL_CHECKPOINT`).
    public static let checkpoint = "text_detection/2025_05_07"
    static let s3Base = "https://models.datalab.to"
    static let requiredFiles = ["config.json", "model.safetensors"]

    /// Local cache directory (matches Python's `platformdirs` datalab cache so a Python download
    /// is reused): `~/Library/Caches/datalab/models/<checkpoint>`.
    public static func cacheDirectory(_ checkpoint: String = checkpoint) -> URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Caches/datalab/models")
            .appendingPathComponent(checkpoint)
    }

    /// Ensure `config.json` + `model.safetensors` are present locally, downloading if needed.
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

/// Loads the EfficientViT detector and runs the full text-line detection pipeline. Ports
/// `surya.detection.DetectionPredictor`. Single-caller contract (driven by ``SuryaSession``).
public final class DetectionEngine {
    let model: EfficientViTForSemanticSegmentation
    let procSize: Int

    /// Build the engine from a snapshot directory (config.json + model.safetensors).
    public init(modelDirectory: URL, procSize: Int = 1200, dtype: DType = .float32) throws {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let config: EfficientViTConfig
        if let data = try? Data(contentsOf: configURL) {
            config = (try? JSONDecoder().decode(EfficientViTConfig.self, from: data))
                ?? EfficientViTConfig()
        } else {
            config = EfficientViTConfig()
        }
        self.model = EfficientViTForSemanticSegmentation(config)
        self.procSize = procSize
        try DetectionWeights.load(
            into: model, url: modelDirectory.appendingPathComponent("model.safetensors"),
            dtype: dtype)
        MLX.eval(model.parameters())
    }

    /// Detect text lines on a page. Returns boxes in the page's pixel coordinates.
    public func detect(
        _ page: CGImage, textThreshold: Float, lowText: Float, yExpandMargin: Float
    ) -> DetectionResult {
        let imageW = page.width, imageH = page.height
        let chunks = DetectionPreprocess.splitImage(page, chunkHeight: procSize, maxHeight: 1400)

        // Preprocess + batch all chunks, run once.
        let inputs = chunks.map { DetectionPreprocess.prepare($0.image, size: procSize) }
        let batch = concatenated(inputs, axis: 0)  // (B, procSize, procSize, 3)
        var logits = model(batch)  // (B, procSize/4, procSize/4, numLabels), sigmoided
        logits = detBilinearResize(logits, toH: procSize, toW: procSize)
        MLX.eval(logits)

        // Stack per-chunk text heatmaps (channel 0), cutting padding on non-first short chunks.
        var combined: [Float] = []
        var totalRows = 0
        for (i, chunk) in chunks.enumerated() {
            let hm = logits[i, 0..., 0..., 0]  // (procSize, procSize)
            var rows = procSize
            if i > 0 && chunk.realHeight < procSize { rows = chunk.realHeight }
            let slice = rows < procSize ? hm[0..<rows, 0...] : hm
            combined.append(contentsOf: slice.asArray(Float.self))
            totalRows += rows
        }

        let hmW = procSize
        let hmH = totalRows
        let boxes = DetectionPostProcess.boxes(
            heatmap: combined, hmW: hmW, hmH: hmH, imageW: imageW, imageH: imageH,
            textThreshold: textThreshold, lowText: lowText, yExpandMargin: yExpandMargin)
        return DetectionResult(lines: boxes.map { TextLine(box: $0) }, imageSize: [imageW, imageH])
    }
}
