import Foundation
import MLX
import MLXFast
import MLXNN

// Port of surya/ocr_error/model/encoder.py (DistilBertForSequenceClassification).
// All weights are Linear/Embedding/LayerNorm (2-D/1-D) — no NCHW→NHWC transpose needed.
// Module attribute names / @ModuleInfo keys mirror the HF DistilBert tree.

/// Configuration for the DistilBERT OCR-error classifier. Ports `DistilBertConfig`.
public struct DistilBertConfig: Codable, Sendable {
    public var dim: Int
    public var hiddenDim: Int
    public var nLayers: Int
    public var nHeads: Int
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var numLabels: Int
    public var layerNormEps: Float

    public init(
        dim: Int = 768, hiddenDim: Int = 3072, nLayers: Int = 6, nHeads: Int = 12,
        vocabSize: Int = 119_547, maxPositionEmbeddings: Int = 512, numLabels: Int = 2,
        layerNormEps: Float = 1e-12
    ) {
        self.dim = dim
        self.hiddenDim = hiddenDim
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.vocabSize = vocabSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.numLabels = numLabels
        self.layerNormEps = layerNormEps
    }

    enum CodingKeys: String, CodingKey {
        case dim, hiddenDim = "hidden_dim", nLayers = "n_layers", nHeads = "n_heads"
        case vocabSize = "vocab_size", maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 768
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 3072
        nLayers = try c.decodeIfPresent(Int.self, forKey: .nLayers) ?? 6
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 12
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 119_547
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 512
        numLabels = 2
        layerNormEps = 1e-12
    }
}

/// Word + position embeddings → LayerNorm. Ports `Embeddings` (DistilBert has no token-type emb).
final class DistilBertEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(_ config: DistilBertConfig) {
        self._wordEmbeddings = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.dim),
            key: "word_embeddings")
        self._positionEmbeddings = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: config.maxPositionEmbeddings, dimensions: config.dim),
            key: "position_embeddings")
        self._layerNorm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.dim, eps: config.layerNormEps),
            key: "LayerNorm")
        super.init()
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let s = ids.dim(-1)
        let positions = MLXArray(0..<Int32(s))
        return layerNorm(wordEmbeddings(ids) + positionEmbeddings(positions))
    }
}

/// Multi-head self-attention (`MultiHeadSelfAttention`). q/k/v/out linear projections.
final class DistilBertAttention: Module {
    @ModuleInfo(key: "q_lin") var qLin: Linear
    @ModuleInfo(key: "k_lin") var kLin: Linear
    @ModuleInfo(key: "v_lin") var vLin: Linear
    @ModuleInfo(key: "out_lin") var outLin: Linear
    let nHeads: Int
    let headDim: Int
    let scale: Float

    init(_ config: DistilBertConfig) {
        self.nHeads = config.nHeads
        self.headDim = config.dim / config.nHeads
        self.scale = 1.0 / Float(config.dim / config.nHeads).squareRoot()
        self._qLin = ModuleInfo(wrappedValue: Linear(config.dim, config.dim), key: "q_lin")
        self._kLin = ModuleInfo(wrappedValue: Linear(config.dim, config.dim), key: "k_lin")
        self._vLin = ModuleInfo(wrappedValue: Linear(config.dim, config.dim), key: "v_lin")
        self._outLin = ModuleInfo(wrappedValue: Linear(config.dim, config.dim), key: "out_lin")
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let b = x.dim(0), s = x.dim(1)
        func split(_ t: MLXArray) -> MLXArray {
            t.reshaped([b, s, nHeads, headDim]).transposed(0, 2, 1, 3)
        }
        let q = split(qLin(x)), k = split(kLin(x)), v = split(vLin(x))
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return outLin(out.transposed(0, 2, 1, 3).reshaped([b, s, nHeads * headDim]))
    }
}

/// Position-wise feed-forward (`FFN`): lin1 → GELU → lin2.
final class DistilBertFFN: Module {
    @ModuleInfo(key: "lin1") var lin1: Linear
    @ModuleInfo(key: "lin2") var lin2: Linear

    init(_ config: DistilBertConfig) {
        self._lin1 = ModuleInfo(wrappedValue: Linear(config.dim, config.hiddenDim), key: "lin1")
        self._lin2 = ModuleInfo(wrappedValue: Linear(config.hiddenDim, config.dim), key: "lin2")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { lin2(gelu(lin1(x))) }
}

/// One transformer block (`TransformerBlock`): post-norm attention + FFN.
final class DistilBertLayer: Module {
    @ModuleInfo(key: "attention") var attention: DistilBertAttention
    @ModuleInfo(key: "sa_layer_norm") var saLayerNorm: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: DistilBertFFN
    @ModuleInfo(key: "output_layer_norm") var outputLayerNorm: LayerNorm

    init(_ config: DistilBertConfig) {
        self._attention = ModuleInfo(wrappedValue: DistilBertAttention(config), key: "attention")
        self._saLayerNorm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.dim, eps: config.layerNormEps),
            key: "sa_layer_norm")
        self._ffn = ModuleInfo(wrappedValue: DistilBertFFN(config), key: "ffn")
        self._outputLayerNorm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.dim, eps: config.layerNormEps),
            key: "output_layer_norm")
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let attn = saLayerNorm(x + attention(x, mask: mask))
        return outputLayerNorm(attn + ffn(attn))
    }
}

/// Stack of transformer blocks (`Transformer`).
final class DistilBertTransformer: Module {
    @ModuleInfo(key: "layer") var layer: [DistilBertLayer]

    init(_ config: DistilBertConfig) {
        self._layer = ModuleInfo(
            wrappedValue: (0..<config.nLayers).map { _ in DistilBertLayer(config) }, key: "layer")
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x
        for l in layer { h = l(h, mask: mask) }
        return h
    }
}

/// The DistilBert encoder (`distilbert`).
final class DistilBertModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: DistilBertEmbeddings
    @ModuleInfo(key: "transformer") var transformer: DistilBertTransformer

    init(_ config: DistilBertConfig) {
        self._embeddings = ModuleInfo(wrappedValue: DistilBertEmbeddings(config), key: "embeddings")
        self._transformer = ModuleInfo(
            wrappedValue: DistilBertTransformer(config), key: "transformer")
        super.init()
    }

    func callAsFunction(_ ids: MLXArray, mask: MLXArray?) -> MLXArray {
        transformer(embeddings(ids), mask: mask)
    }
}

/// DistilBert sequence classifier. Ports `DistilBertForSequenceClassification`.
/// Output logits `(B, numLabels)`.
public final class DistilBertForSequenceClassification: Module {
    @ModuleInfo(key: "distilbert") var distilbert: DistilBertModel
    @ModuleInfo(key: "pre_classifier") var preClassifier: Linear
    @ModuleInfo(key: "classifier") var classifier: Linear

    public init(_ config: DistilBertConfig) {
        self._distilbert = ModuleInfo(wrappedValue: DistilBertModel(config), key: "distilbert")
        self._preClassifier = ModuleInfo(
            wrappedValue: Linear(config.dim, config.dim), key: "pre_classifier")
        self._classifier = ModuleInfo(
            wrappedValue: Linear(config.dim, config.numLabels), key: "classifier")
        super.init()
    }

    /// `ids`: `(B, S)` token ids. Returns logits `(B, numLabels)`.
    public func callAsFunction(_ ids: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let hidden = distilbert(ids, mask: mask)
        let pooled = hidden[0..., 0]  // [CLS]
        return classifier(relu(preClassifier(pooled)))
    }
}
