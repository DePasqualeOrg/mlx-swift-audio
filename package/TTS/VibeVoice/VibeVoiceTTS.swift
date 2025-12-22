// Copyright © Microsoft (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/microsoft/VibeVoice
// License: licenses/vibevoice.txt

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Actor wrapper for VibeVoiceModel that provides thread-safe generation
public actor VibeVoiceTTS {
  // MARK: - Properties

  /// Model is nonisolated(unsafe) because it contains non-Sendable types (MLXArray)
  /// but is only accessed within the actor's methods
  private nonisolated(unsafe) let model: VibeVoiceModel

  /// Text tokenizer (Qwen2)
  private let textTokenizer: Tokenizer

  /// Output sample rate (24kHz)
  public static let outputSampleRate: Int = VibeVoiceConstants.sampleRate

  // MARK: - Initialization

  private init(model: VibeVoiceModel, textTokenizer: Tokenizer) {
    self.model = model
    self.textTokenizer = textTokenizer
  }

  /// Default HuggingFace repository ID for VibeVoice
  public static let defaultRepoId = VibeVoiceConstants.defaultRepoId

  /// Load VibeVoiceTTS from HuggingFace repository
  /// - Parameters:
  ///   - repoId: HuggingFace repository ID
  ///   - progressHandler: Optional progress handler for download
  /// - Returns: Loaded VibeVoiceTTS instance
  public static func load(
    repoId: String = defaultRepoId,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> VibeVoiceTTS {
    // Download model snapshot from HuggingFace
    let modelDirectory = try await HubConfiguration.shared.snapshot(
      from: repoId,
      progressHandler: progressHandler
    )

    // Load configuration
    let config = try VibeVoiceConfig.fromPretrained(modelPath: modelDirectory.path)

    // Load model weights
    let modelURL = modelDirectory.appendingPathComponent("model.safetensors")
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
      throw VibeVoiceError.modelNotLoaded
    }

    let allWeights = try MLX.loadArrays(url: modelURL)

    // Create model
    let model = try createModel(config: config, weights: allWeights)

    // Load text tokenizer
    let textTokenizer = try await loadTokenizer(modelDirectory: modelDirectory)

    return VibeVoiceTTS(model: model, textTokenizer: textTokenizer)
  }

  /// Load tokenizer from model directory
  private static func loadTokenizer(modelDirectory: URL) async throws -> Tokenizer {
    try await AutoTokenizer.from(modelFolder: modelDirectory)
  }

  /// Create model from config and weights
  private static func createModel(
    config: VibeVoiceConfig,
    weights: [String: MLXArray]
  ) throws -> VibeVoiceModel {
    let model = VibeVoiceModel(config: config)

    // Sanitize and load weights
    let sanitizedWeights = sanitizeWeights(weights, model: model)

    // Quantize if needed
    quantize(model: model) { path, _ in
      if sanitizedWeights["\(path).scales"] != nil {
        return (groupSize: 64, bits: 4, mode: .affine)
      }
      return nil
    }

    // Load weights
    let weightsList = sanitizedWeights.map { (key: $0.key, value: $0.value) }
    try model.update(parameters: ModuleParameters.unflattened(weightsList), verify: [])

    return model
  }

  /// Sanitize weights for loading from HuggingFace format
  private static func sanitizeWeights(_ weights: [String: MLXArray], model: VibeVoiceModel) -> [String: MLXArray] {
    var newWeights: [String: MLXArray] = [:]

    // Get current model shapes
    var currShapes: [String: [Int]] = [:]
    for (key, value) in model.parameters().flattened() {
      currShapes[key] = value.shape
    }

    func transformKey(_ key: String) -> String {
      var k = key

      // Remove "model." prefix
      if k.hasPrefix("model.") {
        k = String(k.dropFirst("model.".count))
      }

      // Prediction head transformations
      // t_embedder.mlp.0 -> t_embedder.mlp.layers.0
      k = k.replacingOccurrences(
        of: #"\.t_embedder\.mlp\.(\d+)\."#,
        with: ".t_embedder.mlp.layers.$1.",
        options: .regularExpression
      )
      k = k.replacingOccurrences(
        of: #"\.t_embedder\.mlp\.(\d+)\.weight$"#,
        with: ".t_embedder.mlp.layers.$1.weight",
        options: .regularExpression
      )

      // adaLN_modulation.1 -> adaLN_modulation.layers.1
      k = k.replacingOccurrences(
        of: #"\.adaLN_modulation\.(\d+)\."#,
        with: ".adaLN_modulation.layers.$1.",
        options: .regularExpression
      )
      k = k.replacingOccurrences(
        of: #"\.adaLN_modulation\.(\d+)\.weight$"#,
        with: ".adaLN_modulation.layers.$1.weight",
        options: .regularExpression
      )

      return k
    }

    for (k, v) in weights {
      let newKey = transformKey(k)

      // Handle scaling factors specially
      if newKey.contains("speech_scaling_factor") {
        newWeights["speech_scaling_factor"] = v
        continue
      }
      if newKey.contains("speech_bias_factor") {
        newWeights["speech_bias_factor"] = v
        continue
      }

      // Skip rotary embedding inv_freq (computed at init)
      if newKey.contains("rotary_emb.inv_freq") {
        continue
      }

      // Check if key exists in model
      guard let targetShape = currShapes[newKey] else {
        continue
      }

      // Handle shape mismatches
      if v.shape == targetShape {
        newWeights[newKey] = v
      } else if v.ndim == 2, v.T.shape == targetShape {
        // Transpose Linear weights (PyTorch vs MLX layout)
        newWeights[newKey] = v.T
      } else if v.ndim == 3 {
        // Check if it's a transposed conv weight
        let isConvtr = newKey.contains("convtr")

        if isConvtr {
          // ConvTranspose1d: PyTorch (C_in, C_out, K) -> MLX (C_out, K, C_in)
          if v.shape[1] == targetShape[0], v.shape[2] == targetShape[1], v.shape[0] == targetShape[2] {
            newWeights[newKey] = v.transposed(1, 2, 0)
          } else {
            newWeights[newKey] = v
          }
        } else {
          // Conv1d weights: PyTorch (C_out, C_in, K) -> MLX (C_out, K, C_in)
          if v.shape[0] == targetShape[0], v.shape[1] == targetShape[2], v.shape[2] == targetShape[1] {
            newWeights[newKey] = v.transposed(0, 2, 1)
          } else {
            newWeights[newKey] = v
          }
        }
      } else {
        newWeights[newKey] = v
      }
    }

    return newWeights
  }

  // MARK: - Voice Loading

  /// Load a voice cache for conditioning
  /// - Parameter voiceName: Name of the voice (without .safetensors extension)
  public func loadVoice(_ voiceName: String) throws {
    guard let modelPath = model.config.modelPath else {
      throw VibeVoiceError.invalidInput("Model path not set")
    }

    let voicePath = URL(fileURLWithPath: modelPath)
      .appendingPathComponent("voices")
      .appendingPathComponent("\(voiceName).safetensors")

    guard FileManager.default.fileExists(atPath: voicePath.path) else {
      throw VibeVoiceError.invalidInput("Voice cache not found: \(voicePath.path)")
    }

    try model.loadVoice(voicePath: voicePath)
  }

  // MARK: - Generation

  /// Generate audio from text
  /// - Parameters:
  ///   - text: Input text to synthesize
  ///   - maxTokens: Maximum number of speech tokens to generate
  ///   - cfgScale: Classifier-free guidance scale
  ///   - ddpmSteps: Override diffusion inference steps
  /// - Returns: Generation result with audio and metadata
  public func generate(
    text: String,
    maxTokens: Int = 512,
    cfgScale: Float = 1.5,
    ddpmSteps: Int? = nil
  ) -> TTSGenerationResult {
    let startTime = CFAbsoluteTimeGetCurrent()

    // Tokenize input
    let textTokenIds = textTokenizer.encode(text: text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n", addSpecialTokens: false)
    let inputIds = MLXArray(textTokenIds.map { Int32($0) }).reshaped([1, -1])

    // Generate
    let audio = model.generate(
      inputIds: inputIds,
      maxTokens: maxTokens,
      cfgScale: cfgScale,
      ddpmSteps: ddpmSteps
    )

    audio.eval()

    let processingTime = CFAbsoluteTimeGetCurrent() - startTime

    return TTSGenerationResult(
      audio: audio.asArray(Float.self),
      sampleRate: Self.outputSampleRate,
      processingTime: processingTime
    )
  }

  // MARK: - Tokenization

  /// Encode text to token IDs
  public func encode(text: String, addSpecialTokens: Bool = false) -> [Int] {
    textTokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
  }

  /// Decode token IDs back to text
  public func decode(tokens: [Int], skipSpecialTokens: Bool = true) -> String {
    textTokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
  }

  /// Output sample rate
  public var sampleRate: Int {
    Self.outputSampleRate
  }
}
