import Foundation

/// Configuration for the EfficientViT text-detection model. Ports
/// `surya.detection.model.config.EfficientViTConfig`. Defaults match the
/// `datalab-to/line_det` checkpoint's `config.json`.
public struct EfficientViTConfig: Codable, Sendable {
    public var numClasses: Int
    public var numChannels: Int
    public var widths: [Int]
    public var headDim: Int
    public var numStages: Int
    public var depths: [Int]
    public var strides: [Int]
    public var decoderLayerHiddenSize: Int
    public var decoderHiddenSize: Int
    public var layerNormEps: Float

    /// `num_labels` (== `num_classes`) — the detection head outputs this many heatmaps
    /// (channel 0 = text, channel 1 = affinity).
    public var numLabels: Int { numClasses }

    public init(
        numClasses: Int = 2,
        numChannels: Int = 3,
        widths: [Int] = [32, 64, 128, 256, 512],
        headDim: Int = 32,
        numStages: Int = 4,
        depths: [Int] = [1, 1, 1, 6, 6],
        strides: [Int] = [2, 2, 2, 2, 2],
        decoderLayerHiddenSize: Int = 128,
        decoderHiddenSize: Int = 512,
        layerNormEps: Float = 1e-6
    ) {
        self.numClasses = numClasses
        self.numChannels = numChannels
        self.widths = widths
        self.headDim = headDim
        self.numStages = numStages
        self.depths = depths
        self.strides = strides
        self.decoderLayerHiddenSize = decoderLayerHiddenSize
        self.decoderHiddenSize = decoderHiddenSize
        self.layerNormEps = layerNormEps
    }

    enum CodingKeys: String, CodingKey {
        case numClasses = "num_classes"
        case numChannels = "num_channels"
        case widths
        case headDim = "head_dim"
        case numStages = "num_stages"
        case depths
        case strides
        case decoderLayerHiddenSize = "decoder_layer_hidden_size"
        case decoderHiddenSize = "decoder_hidden_size"
        case layerNormEps = "layer_norm_eps"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numClasses = try c.decodeIfPresent(Int.self, forKey: .numClasses) ?? 2
        numChannels = try c.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        widths = try c.decodeIfPresent([Int].self, forKey: .widths) ?? [32, 64, 128, 256, 512]
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 32
        numStages = try c.decodeIfPresent(Int.self, forKey: .numStages) ?? 4
        depths = try c.decodeIfPresent([Int].self, forKey: .depths) ?? [1, 1, 1, 6, 6]
        strides = try c.decodeIfPresent([Int].self, forKey: .strides) ?? [2, 2, 2, 2, 2]
        decoderLayerHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .decoderLayerHiddenSize) ?? 128
        decoderHiddenSize = try c.decodeIfPresent(Int.self, forKey: .decoderHiddenSize) ?? 512
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
    }
}
