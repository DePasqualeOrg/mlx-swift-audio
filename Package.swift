// swift-tools-version:6.2
import PackageDescription

let package = Package(
  name: "mlx-audio",
  platforms: [.macOS("15.4"), .iOS("18.4")],
  products: [
    // Core library without Kokoro TTS (no GPLv3 dependencies)
    .library(
      name: "MLXAudio",
      targets: ["MLXAudio"]
    ),
    // Full library including Kokoro TTS (includes GPLv3 espeak-ng dependency)
    .library(
      name: "MLXAudioWithKokoro",
      targets: ["MLXAudioWithKokoro"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", branch: "main"),
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.0"),
    .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.0")),
    // espeak-ng is GPLv3 licensed - only used by Kokoro TTS
    // TODO: Switch back to upstream after https://github.com/espeak-ng/espeak-ng/pull/2327 is merged
    .package(url: "https://github.com/DePasqualeOrg/espeak-ng-spm.git", branch: "fix-path-espeak-data-macro"),
  ],
  targets: [
    // Core target - excludes Kokoro TTS (no GPLv3 code)
    .target(
      name: "MLXAudio",
      dependencies: [
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXFFT", package: "mlx-swift"),
        .product(name: "Transformers", package: "swift-transformers"),
      ],
      path: "package",
      exclude: ["Tests", "TTS/Kokoro"],
      resources: [
        .process("TTS/OuteTTS/default_speaker.json"),
      ],
      swiftSettings: [
        .define("MLXAUDIO_EXCLUDE_KOKORO"),
      ]
    ),
    // Full target - includes Kokoro TTS with espeak-ng (GPLv3)
    .target(
      name: "MLXAudioWithKokoro",
      dependencies: [
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXFFT", package: "mlx-swift"),
        .product(name: "Transformers", package: "swift-transformers"),
        .product(name: "libespeak-ng", package: "espeak-ng-spm"),
        .product(name: "espeak-ng-data", package: "espeak-ng-spm"),
      ],
      path: "package",
      exclude: ["Tests"],
      resources: [
        .process("TTS/OuteTTS/default_speaker.json"),
      ]
    ),
    .testTarget(
      name: "MLXAudioTests",
      dependencies: ["MLXAudioWithKokoro"],
      path: "package/Tests"
    ),
  ]
)
