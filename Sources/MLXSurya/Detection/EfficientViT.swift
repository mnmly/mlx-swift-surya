import Foundation
import MLX
import MLXNN

// Port of surya/detection/model/encoderdecoder.py (EfficientViT-Large segmentation).
// PyTorch is NCHW; MLX is NHWC. Conv weights are transposed NCHW→NHWC at load time
// (see DetectionWeights). Module attribute names / @ModuleInfo keys mirror the PyTorch tree
// exactly so weight keys line up.

/// Backbone activation (Hardswish) / decode-head activation (ReLU) / none.
enum DetActivation: Sendable {
    case none, relu, relu6, hardswish

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch self {
        case .none: return x
        case .relu: return maximum(x, 0)
        case .relu6: return clip(x, min: 0, max: 6)
        case .hardswish: return x * clip(x + 3, min: 0, max: 6) / 6
        }
    }
}

private func detPadding(_ kernel: Int, _ stride: Int) -> Int {
    ((stride - 1) + (kernel - 1)) / 2
}

/// Bilinear resize an NHWC tensor to an exact target size (PyTorch `interpolate(align_corners=false)`).
func detBilinearResize(_ x: MLXArray, toH: Int, toW: Int) -> MLXArray {
    let h = x.dim(1)
    let w = x.dim(2)
    if h == toH && w == toW { return x }
    let up = Upsample(
        scaleFactor: .array([Float(toH) / Float(h), Float(toW) / Float(w)]),
        mode: .linear(alignCorners: false))
    return up(x)
}

/// conv → (optional BatchNorm) → activation. Ports `ConvNormAct`.
final class ConvNormAct: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: BatchNorm?
    let act: DetActivation

    init(
        _ inCh: Int, _ outCh: Int, kernel: Int = 3, stride: Int = 1, groups: Int = 1,
        bias: Bool = false, hasNorm: Bool = true, act: DetActivation = .hardswish, eps: Float
    ) {
        let pad = detPadding(kernel, stride)
        self._conv = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: inCh, outputChannels: outCh, kernelSize: .init(kernel),
                stride: .init(stride), padding: .init(pad), groups: groups, bias: bias),
            key: "conv")
        self._norm = ModuleInfo(
            wrappedValue: hasNorm ? BatchNorm(featureCount: outCh, eps: eps) : nil, key: "norm")
        self.act = act
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        if let norm { h = norm(h) }
        return act(h)
    }
}

/// `ConvBlock` (conv1 → conv2). Used by the stem.
final class DetConvBlock: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: ConvNormAct
    @ModuleInfo(key: "conv2") var conv2: ConvNormAct

    init(_ inCh: Int, _ outCh: Int, kernel: Int, stride: Int, act: DetActivation, eps: Float) {
        let mid = inCh  // expand_ratio == 1
        self._conv1 = ModuleInfo(
            wrappedValue: ConvNormAct(inCh, mid, kernel: kernel, stride: stride, act: act, eps: eps),
            key: "conv1")
        self._conv2 = ModuleInfo(
            wrappedValue: ConvNormAct(mid, outCh, kernel: kernel, stride: 1, act: .none, eps: eps),
            key: "conv2")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv2(conv1(x)) }
}

/// `MBConv` (inverted_conv → depth_conv → point_conv). `fewerNorm` puts bias on the first two
/// convs and BatchNorm only on the point conv (matches the checkpoint).
final class DetMBConv: Module, UnaryLayer {
    @ModuleInfo(key: "inverted_conv") var invertedConv: ConvNormAct
    @ModuleInfo(key: "depth_conv") var depthConv: ConvNormAct
    @ModuleInfo(key: "point_conv") var pointConv: ConvNormAct

    init(
        _ inCh: Int, _ outCh: Int, kernel: Int, stride: Int, expandRatio: Int,
        fewerNorm: Bool, act: DetActivation, eps: Float
    ) {
        let mid = inCh * expandRatio
        self._invertedConv = ModuleInfo(
            wrappedValue: ConvNormAct(
                inCh, mid, kernel: 1, stride: 1, bias: fewerNorm, hasNorm: !fewerNorm, act: act,
                eps: eps), key: "inverted_conv")
        self._depthConv = ModuleInfo(
            wrappedValue: ConvNormAct(
                mid, mid, kernel: kernel, stride: stride, groups: mid, bias: fewerNorm,
                hasNorm: !fewerNorm, act: act, eps: eps), key: "depth_conv")
        self._pointConv = ModuleInfo(
            wrappedValue: ConvNormAct(
                mid, outCh, kernel: 1, stride: 1, bias: false, hasNorm: true, act: .none, eps: eps),
            key: "point_conv")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { pointConv(depthConv(invertedConv(x))) }
}

/// `FusedMBConv` (spatial_conv → point_conv).
final class DetFusedMBConv: Module, UnaryLayer {
    @ModuleInfo(key: "spatial_conv") var spatialConv: ConvNormAct
    @ModuleInfo(key: "point_conv") var pointConv: ConvNormAct

    init(
        _ inCh: Int, _ outCh: Int, kernel: Int, stride: Int, expandRatio: Int,
        fewerNorm: Bool, act: DetActivation, eps: Float
    ) {
        let mid = inCh * expandRatio
        self._spatialConv = ModuleInfo(
            wrappedValue: ConvNormAct(
                inCh, mid, kernel: kernel, stride: stride, bias: fewerNorm, hasNorm: !fewerNorm,
                act: act, eps: eps), key: "spatial_conv")
        self._pointConv = ModuleInfo(
            wrappedValue: ConvNormAct(
                mid, outCh, kernel: 1, stride: 1, bias: false, hasNorm: true, act: .none, eps: eps),
            key: "point_conv")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { pointConv(spatialConv(x)) }
}

/// Lightweight multi-scale linear attention. Ports `LiteMLA`. Operates in NHWC; the attention
/// math runs in float32 (matching the Python `_attn` cast).
///
/// `aggreg` is `nn.ModuleList([ nn.Sequential(conv0, conv1) ])`, i.e. an outer list (one per
/// scale) of an inner 2-conv sequence — modeled as `[[Conv2d]]` so the weight keys
/// `aggreg.0.0` / `aggreg.0.1` line up (numeric path segments are array indices in MLX).
final class DetLiteMLA: Module, UnaryLayer {
    @ModuleInfo(key: "qkv") var qkv: ConvNormAct
    @ModuleInfo(key: "aggreg") var aggreg: [[Conv2d]]
    @ModuleInfo(key: "proj") var proj: ConvNormAct

    let dim: Int
    let heads: Int
    let eps: Float = 1e-5

    init(_ inCh: Int, _ outCh: Int, headDim: Int, scales: [Int] = [5], eps: Float) {
        let heads = inCh / headDim
        let totalDim = heads * headDim
        self.dim = headDim
        self.heads = heads
        self._qkv = ModuleInfo(
            wrappedValue: ConvNormAct(
                inCh, 3 * totalDim, kernel: 1, bias: false, hasNorm: false, act: .none, eps: eps),
            key: "qkv")
        self._aggreg = ModuleInfo(
            wrappedValue: scales.map { scale in
                [
                    Conv2d(
                        inputChannels: 3 * totalDim, outputChannels: 3 * totalDim,
                        kernelSize: .init(scale), stride: .init(1), padding: .init(scale / 2),
                        groups: 3 * totalDim, bias: false),
                    Conv2d(
                        inputChannels: 3 * totalDim, outputChannels: 3 * totalDim,
                        kernelSize: .init(1), groups: 3 * heads, bias: false),
                ]
            }, key: "aggreg")
        self._proj = ModuleInfo(
            wrappedValue: ConvNormAct(
                totalDim * (1 + scales.count), outCh, kernel: 1, bias: false, hasNorm: true,
                act: .none, eps: eps), key: "proj")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2)
        let qkvOut = qkv(x)
        var multi = [qkvOut]
        for op in aggreg { multi.append(op[1](op[0](qkvOut))) }
        let cat = concatenated(multi, axis: -1)  // (B,H,W, 3*totalDim*(1+scales))
        let c = cat.dim(-1)
        let groups = c / (3 * dim)  // == heads * (1+scales)

        // (B,H,W,C) -> (B, HW, groups, 3*dim) -> (B, groups, HW, 3*dim)
        var qkvR = cat.reshaped([b, h * w, groups, 3 * dim]).transposed(0, 2, 1, 3)
        let parts = qkvR.split(parts: 3, axis: -1)  // each (B, groups, HW, dim)
        var q = maximum(parts[0], 0).asType(.float32)
        var k = maximum(parts[1], 0).asType(.float32)
        var v = parts[2].asType(.float32)
        v = padded(v, widths: [.init(0), .init(0), .init(0), .init((0, 1))], value: MLXArray(Float(1)))

        let kv = matmul(k.transposed(0, 1, 3, 2), v)  // (B,groups,dim,dim+1)
        let out = matmul(q, kv)  // (B,groups,HW,dim+1)
        let num = out[.ellipsis, ..<dim]
        let den = out[.ellipsis, dim...]
        var attn = (num / (den + eps)).asType(x.dtype)  // (B,groups,HW,dim)

        // (B,groups,HW,dim) -> (B,HW,groups,dim) -> (B,H,W,groups*dim)
        attn = attn.transposed(0, 2, 1, 3).reshaped([b, h, w, groups * dim])
        _ = qkvR
        return proj(attn)
    }
}

/// `ResidualBlock`: `main(x)` plus an optional identity shortcut. `pre_norm` is always Identity.
final class DetResidualBlock: Module, UnaryLayer {
    @ModuleInfo(key: "main") var main: UnaryLayer
    let hasShortcut: Bool

    init(main: UnaryLayer, hasShortcut: Bool) {
        self._main = ModuleInfo(wrappedValue: main, key: "main")
        self.hasShortcut = hasShortcut
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let res = main(x)
        return hasShortcut ? res + x : res
    }
}

/// `EfficientVitBlock`: residual LiteMLA (context) + residual MBConv (local).
final class DetEfficientVitBlock: Module, UnaryLayer {
    @ModuleInfo(key: "context_module") var contextModule: DetResidualBlock
    @ModuleInfo(key: "local_module") var localModule: DetResidualBlock

    init(_ inCh: Int, headDim: Int, expandRatio: Int, eps: Float) {
        self._contextModule = ModuleInfo(
            wrappedValue: DetResidualBlock(
                main: DetLiteMLA(inCh, inCh, headDim: headDim, eps: eps), hasShortcut: true),
            key: "context_module")
        self._localModule = ModuleInfo(
            wrappedValue: DetResidualBlock(
                main: DetMBConv(
                    inCh, inCh, kernel: 3, stride: 1, expandRatio: expandRatio, fewerNorm: true,
                    act: .hardswish, eps: eps), hasShortcut: true), key: "local_module")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { localModule(contextModule(x)) }
}

/// Input stem: `in_conv` + `res0` (depth 1). Ports `Stem` (block_type="large" → ConvBlock).
final class DetStem: Module, UnaryLayer {
    @ModuleInfo(key: "in_conv") var inConv: ConvNormAct
    @ModuleInfo(key: "res0") var res0: DetResidualBlock

    init(_ inCh: Int, _ outCh: Int, stride: Int, act: DetActivation, eps: Float) {
        self._inConv = ModuleInfo(
            wrappedValue: ConvNormAct(
                inCh, outCh, kernel: stride + 1, stride: stride, act: act, eps: eps),
            key: "in_conv")
        self._res0 = ModuleInfo(
            wrappedValue: DetResidualBlock(
                main: DetConvBlock(outCh, outCh, kernel: 3, stride: 1, act: act, eps: eps),
                hasShortcut: true), key: "res0")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { res0(inConv(x)) }
}

/// One encoder stage (`EfficientVitLargeStage`): a downsampling head block + `depth` body blocks.
final class DetStage: Module, UnaryLayer {
    @ModuleInfo(key: "blocks") var blocks: [UnaryLayer]

    init(
        _ inCh: Int, _ outCh: Int, depth: Int, stride: Int, headDim: Int, vitStage: Bool,
        fewerNorm: Bool, act: DetActivation, eps: Float
    ) {
        var built: [UnaryLayer] = []
        // Head block: FusedMBConv (non-fewer) or MBConv (fewer), no shortcut.
        let headExpand = vitStage ? 24 : 16
        let headFewer = vitStage || fewerNorm
        let headMain: UnaryLayer =
            headFewer
            ? DetMBConv(
                inCh, outCh, kernel: stride + 1, stride: stride, expandRatio: headExpand,
                fewerNorm: true, act: act, eps: eps)
            : DetFusedMBConv(
                inCh, outCh, kernel: stride + 1, stride: stride, expandRatio: headExpand,
                fewerNorm: false, act: act, eps: eps)
        built.append(DetResidualBlock(main: headMain, hasShortcut: false))

        if vitStage {
            for _ in 0..<depth {
                built.append(
                    DetEfficientVitBlock(outCh, headDim: headDim, expandRatio: 6, eps: eps))
            }
        } else {
            for _ in 0..<depth {
                let bodyMain: UnaryLayer =
                    fewerNorm
                    ? DetMBConv(
                        outCh, outCh, kernel: 3, stride: 1, expandRatio: 4, fewerNorm: true,
                        act: act, eps: eps)
                    : DetFusedMBConv(
                        outCh, outCh, kernel: 3, stride: 1, expandRatio: 4, fewerNorm: false,
                        act: act, eps: eps)
                built.append(DetResidualBlock(main: bodyMain, hasShortcut: true))
            }
        }
        self._blocks = ModuleInfo(wrappedValue: built, key: "blocks")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for b in blocks { h = b(h) }
        return h
    }
}

/// EfficientViT-Large encoder. Returns the 4 stage hidden states (NHWC).
final class DetEfficientVitLarge: Module {
    @ModuleInfo(key: "stem") var stem: DetStem
    @ModuleInfo(key: "stages") var stages: [DetStage]

    init(_ config: EfficientViTConfig) {
        let eps = config.layerNormEps
        let act = DetActivation.hardswish
        self._stem = ModuleInfo(
            wrappedValue: DetStem(
                config.numChannels, config.widths[0], stride: config.strides[0], act: act, eps: eps),
            key: "stem")
        var built: [DetStage] = []
        var inCh = config.widths[0]
        for i in 0..<(config.widths.count - 1) {
            let w = config.widths[i + 1]
            let d = config.depths[i + 1]
            let s = config.strides[i + 1]
            built.append(
                DetStage(
                    inCh, w, depth: d, stride: s, headDim: config.headDim, vitStage: i >= 3,
                    fewerNorm: i >= 2, act: act, eps: eps))
            inCh = w
        }
        self._stages = ModuleInfo(wrappedValue: built, key: "stages")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        var h = stem(x)
        var states: [MLXArray] = []
        for stage in stages {
            h = stage(h)
            states.append(h)
        }
        return states
    }
}

/// Per-stage MLP that unifies channels to `decoder_layer_hidden_size`. Ports `DecodeMLP`.
final class DetDecodeMLP: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ inDim: Int, _ outDim: Int) {
        self._proj = ModuleInfo(wrappedValue: Linear(inDim, outDim), key: "proj")
        super.init()
    }

    /// Input NHWC `(B,H,W,C)` → `(B, H*W, outDim)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        return proj(x.reshaped([b, h * w, c]))
    }
}

/// SegFormer-style decode head. Ports `DecodeHead`.
final class DetDecodeHead: Module {
    @ModuleInfo(key: "linear_c") var linearC: [DetDecodeMLP]
    @ModuleInfo(key: "linear_fuse") var linearFuse: Conv2d
    @ModuleInfo(key: "batch_norm") var batchNorm: BatchNorm
    @ModuleInfo(key: "classifier") var classifier: Conv2d

    init(_ config: EfficientViTConfig) {
        let lh = config.decoderLayerHiddenSize
        self._linearC = ModuleInfo(
            wrappedValue: config.widths[1...].map { DetDecodeMLP($0, lh) }, key: "linear_c")
        self._linearFuse = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: lh * config.numStages, outputChannels: config.decoderHiddenSize,
                kernelSize: .init(1), bias: false), key: "linear_fuse")
        self._batchNorm = ModuleInfo(
            wrappedValue: BatchNorm(featureCount: config.decoderHiddenSize), key: "batch_norm")
        self._classifier = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: config.decoderHiddenSize, outputChannels: config.numLabels,
                kernelSize: .init(1), bias: true), key: "classifier")
        super.init()
    }

    func callAsFunction(_ states: [MLXArray]) -> MLXArray {
        let h0 = states[0].dim(1), w0 = states[0].dim(2)
        var ups: [MLXArray] = []
        for (state, mlp) in zip(states, linearC) {
            let hs = state.dim(1), ws = state.dim(2)
            let m = mlp(state).reshaped([state.dim(0), hs, ws, mlp.proj.weight.dim(0)])
            ups.append(detBilinearResize(m, toH: h0, toW: w0))
        }
        let fused = linearFuse(concatenated(ups.reversed(), axis: -1))
        let bn = maximum(batchNorm(fused), 0)
        return classifier(bn)
    }
}

/// Top-level detection model: encoder + decode head + sigmoid. Ports
/// `EfficientViTForSemanticSegmentation`. Output is NHWC `(B, H/4, W/4, numLabels)` in `[0,1]`.
public final class EfficientViTForSemanticSegmentation: Module {
    @ModuleInfo(key: "vit") var vit: DetEfficientVitLarge
    @ModuleInfo(key: "decode_head") var decodeHead: DetDecodeHead

    public init(_ config: EfficientViTConfig) {
        self._vit = ModuleInfo(wrappedValue: DetEfficientVitLarge(config), key: "vit")
        self._decodeHead = ModuleInfo(wrappedValue: DetDecodeHead(config), key: "decode_head")
        super.init()
    }

    /// `pixelValues`: NHWC `(B, H, W, 3)`. Returns sigmoided logits NHWC `(B, H/4, W/4, numLabels)`.
    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        sigmoid(decodeHead(vit(pixelValues)))
    }
}
