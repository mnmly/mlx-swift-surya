import CoreGraphics
import CoreImage
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import MLXVLM
import Tokenizers

/// Loads the `surya-ocr-2` checkpoint (`qwen3_5` VLM) and runs single-image generations. Thin
/// bridge over `mlx-swift-lm`; surya orchestration (prompt selection + parsing) lives in
/// ``SuryaPipeline``. Mirrors `mlx-swift-chandra`'s engine — same architecture family.
public actor SuryaEngine {
    /// The loaded MLXVLM model container.
    public let container: ModelContainer

    /// Load the model once and reuse it across pages. `gpuCacheLimit` bounds MLX's reusable
    /// Metal buffer cache so a long-lived process doesn't keep the multi-GB peak footprint
    /// resident (the cache is not a leak, but it is large). Pass `nil` for MLX defaults.
    ///
    /// `precision: .int8` quantizes the model in-memory after load (group size 64, 8 bits) —
    /// ~30% faster decode + ~half the live memory, near-lossless. Linear/Embedding layers whose
    /// feature dimension isn't a multiple of the group size stay full-precision.
    public init(
        modelDirectory: URL, gpuCacheLimit: Int? = 512 * 1024 * 1024,
        precision: SuryaPrecision = .bf16
    ) async throws {
        if let gpuCacheLimit {
            MLX.Memory.cacheLimit = gpuCacheLimit
        }
        // surya-ocr-2 ships a char-level WordLevel tokenizer that the default loader can't
        // parse (it falls back to BPE and crashes on missing merges), so inject our own.
        self.container = try await VLMModelFactory.shared.loadContainer(
            from: modelDirectory, using: SuryaTokenizerLoader())

        if precision == .int8 {
            let groupSize = 64
            await container.perform { (context: ModelContext) in
                // Quantize the language model only — leave the vision tower full-precision.
                // Quantizing vision breaks image conditioning (the model hallucinates a
                // degenerate sequence and never stops). Mirrors mlx_lm's VLM int8 recipe.
                quantize(model: context.model, groupSize: groupSize, bits: 8) { path, module in
                    guard path.hasPrefix("language_model") else { return false }
                    if let lin = module as? Linear { return lin.weight.dim(-1) % groupSize == 0 }
                    if let emb = module as? Embedding { return emb.weight.dim(-1) % groupSize == 0 }
                    return false
                }
                MLX.eval(context.model)
            }
        }
    }

    /// A snapshot of MLX GPU memory (active = live arrays, cache = reusable buffers,
    /// peak = high-water mark). Flat `active` across repeated runs ⇒ no leak.
    public struct MemorySnapshot: Sendable {
        public let active: Int
        public let cache: Int
        public let peak: Int
    }

    /// Current MLX GPU memory snapshot.
    public static func memorySnapshot() -> MemorySnapshot {
        MemorySnapshot(
            active: MLX.Memory.activeMemory, cache: MLX.Memory.cacheMemory,
            peak: MLX.Memory.peakMemory)
    }

    /// One generation: the model's raw text output plus the number of tokens it emitted.
    public struct Generation: Sendable {
        public let raw: String
        public let tokenCount: Int
    }

    /// Run one image + prompt generation with greedy decoding (`ArgMaxSampler`) to match the
    /// reference (temperature 0). The image must already be prepared (`scaleToFit`).
    ///
    /// `<|im_end|>` is added as an extra stop token (the `qwen3_5` model emits it at turn
    /// boundaries, which the base `generation_config` omits).
    public func generate(
        image: CGImage, prompt: String, maxTokens: Int
    ) async throws -> Generation {
        let ci = CIImage(cgImage: image)
        return try await container.perform { (context: ModelContext) in
            let input = UserInput(chat: [.user(prompt, images: [.ciImage(ci)])])
            let lmInput = try await context.processor.prepare(input: input)

            var configuration = context.configuration
            configuration.extraEOSTokens.insert("<|im_end|>")

            let iterator = try TokenIterator(
                input: lmInput, model: context.model, cache: nil,
                processor: nil, sampler: ArgMaxSampler(), maxTokens: maxTokens)

            let (stream, _) = MLXLMCommon.generateTask(
                promptTokenCount: lmInput.text.tokens.size,
                modelConfiguration: configuration,
                tokenizer: context.tokenizer,
                iterator: iterator,
                wiredMemoryTicket: nil)

            var output = ""
            for await gen in stream {
                if case .chunk(let s) = gen { output += s }
            }
            let tokenCount = context.tokenizer.encode(text: output).count
            return Generation(raw: output, tokenCount: tokenCount)
        }
    }
}
