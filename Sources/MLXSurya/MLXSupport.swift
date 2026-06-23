import MLX
import MLXNN

// Proves the mlx-swift dependency resolves, compiles, and links in the skeleton,
// and centralizes the dtype boundary the ported models will load weights at.
// (Real model code replaces / extends this as each slice lands.)

/// MLX-side conventions shared by the ported models.
enum MLXSupport {
    /// The default compute dtype weights are cast to at load time. The native
    /// EfficientViT / DistilBERT models load in float16; the VLM loads in the
    /// precision of its snapshot (bf16) via MLXVLM.
    static let defaultDType: DType = .float16
}
