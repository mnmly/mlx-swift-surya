import Foundation
import MLX
import MLXNN

/// Loads PyTorch-origin EfficientViT detection weights into the MLX model.
///
/// Conversions applied at load time:
/// - every 4-D tensor is a conv weight → transpose NCHW `(O, I, kH, kW)` → NHWC `(O, kH, kW, I)`;
/// - `num_batches_tracked` buffers are dropped (not MLX parameters);
/// - everything is cast to `dtype`.
/// 1-D (BatchNorm / bias) and 2-D (Linear) tensors pass through unchanged.
enum DetectionWeights {
    static func load(
        into model: EfficientViTForSemanticSegmentation, url: URL, dtype: DType = .float32
    ) throws {
        let weights = try MLX.loadArrays(url: url)
        var remapped: [(String, MLXArray)] = []
        remapped.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key.hasSuffix("num_batches_tracked") { continue }
            var v = value
            if v.ndim == 4 { v = v.transposed(0, 2, 3, 1) }
            remapped.append((key, v.asType(dtype)))
        }
        try model.update(
            parameters: ModuleParameters.unflattened(remapped), verify: [.noUnusedKeys])
    }
}
