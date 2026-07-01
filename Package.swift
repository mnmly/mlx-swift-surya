// swift-tools-version: 6.1
import PackageDescription

// mlx-swift-surya — Swift/MLX port of datalab-to/surya (OCR, layout, table & reading order).
//
// surya 0.20 is three independent pieces, ported with three strategies:
//   1. surya-ocr-2  — a `qwen3_5` VLM (layout + table + recognition + reading order).
//                     Loaded through MLXVLM (same arch family as mlx-swift-chandra), NOT
//                     hand-ported. Wiring + prompts + parsers only.
//   2. text detection — a native EfficientViT segmentation net. Ported natively to MLX.
//   3. ocr-error      — a native DistilBERT classifier. Ported natively to MLX.
//
// This is the SKELETON: package shape, the shared `SuryaSession` driver API, Swift 6
// strict concurrency, and DocC are in place; the model slices fill in behind the stable
// Session API. The heavy VLM dependency stack (mlx-swift-lm / swift-huggingface) is
// declared-but-commented below and gets enabled when the VLM slice begins.

let package = Package(
    name: "mlx-swift-surya",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "MLXSurya", targets: ["MLXSurya"]),
        .executable(name: "surya-cli", targets: ["surya-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3"),

        // VLM (surya-ocr-2, qwen3_5) — same stack as mlx-swift-chandra.
        // Upstream tag carrying qwen3_5 VLM support.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", .upToNextMinor(from: "3.31.4")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // surya-ocr-2's char-level WordLevel tokenizer is unsupported by swift-transformers,
        // so we implement our own Tokenizer and render its chat template via Jinja directly.
        .package(url: "https://github.com/huggingface/swift-jinja", .upToNextMinor(from: "2.3.6")),
    ],
    targets: [
        .target(
            name: "MLXSurya",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Jinja", package: "swift-jinja"),
            ],
            path: "Sources/MLXSurya"
        ),
        .executableTarget(
            name: "surya-cli",
            dependencies: [
                "MLXSurya",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Examples/surya-cli"
        ),
        .testTarget(
            name: "MLXSuryaTests",
            dependencies: ["MLXSurya"],
            path: "Tests/MLXSuryaTests"
        ),
    ],
    swiftLanguageModes: [.v6]
)
