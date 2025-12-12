import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN

/// Main Whisper model containing AudioEncoder and TextDecoder
class WhisperModel: Module {
  let encoder: AudioEncoder
  let decoder: TextDecoder
  let dims: ModelDimensions

  // Alignment heads for word-level timestamps
  // Shape: (num_alignment_heads, 2) where each row is [layer_idx, head_idx]
  @ParameterInfo(key: "alignment_heads") var alignmentHeads: MLXArray

  static let defaultRepoId = "mlx-community/whisper-base-mlx"

  init(dims: ModelDimensions) {
    self.dims = dims
    encoder = AudioEncoder(
      nMels: dims.n_mels,
      nCtx: dims.n_audio_ctx,
      nState: dims.n_audio_state,
      nHead: dims.n_audio_head,
      nLayer: dims.n_audio_layer
    )
    decoder = TextDecoder(
      nVocab: dims.n_vocab,
      nCtx: dims.n_text_ctx,
      nState: dims.n_text_state,
      nHead: dims.n_text_head,
      nLayer: dims.n_text_layer
    )

    // Initialize alignment_heads to empty array (will be loaded from checkpoint)
    _alignmentHeads.wrappedValue = MLXArray([])
  }

  /// Encode audio features (matches Python's embed_audio method)
  ///
  /// - Parameter mel: Mel spectrogram (batch, n_mels, n_frames)
  /// - Returns: Encoded audio features (batch, n_audio_ctx, n_audio_state)
  func encode(_ mel: MLXArray) -> MLXArray {
    encoder(mel)
  }

  /// Decode tokens with audio features
  ///
  /// - Parameters:
  ///   - tokens: Token indices (batch, n_tokens)
  ///   - audioFeatures: Encoded audio features (batch, n_audio_ctx, n_audio_state)
  ///   - kvCache: Optional cached key/value tensors
  /// - Returns: Tuple of (logits, new_kv_cache, cross_attention_weights)
  func decode(
    _ tokens: MLXArray,
    audioFeatures: MLXArray,
    kvCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)]? = nil
  ) -> (MLXArray, [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)], [MLXArray?]) {
    decoder(tokens, xa: audioFeatures, kvCache: kvCache)
  }

  /// Get logits for tokens given audio features (matches Python's logits method)
  ///
  /// - Parameters:
  ///   - tokens: Token indices (batch, n_tokens)
  ///   - audioFeatures: Encoded audio features (batch, n_audio_ctx, n_audio_state)
  /// - Returns: Logits (batch, n_tokens, n_vocab)
  func logits(_ tokens: MLXArray, audioFeatures: MLXArray) -> MLXArray {
    let (logits, _, _) = decode(tokens, audioFeatures: audioFeatures)
    return logits
  }

  /// Forward pass with cross-attention weights (for word-level timestamps)
  ///
  /// - Parameters:
  ///   - mel: Mel spectrogram (batch, n_mels, n_frames)
  ///   - tokens: Token indices (batch, n_tokens)
  /// - Returns: Tuple of (logits, cross_attention_weights)
  func forwardWithCrossQK(_ mel: MLXArray, tokens: MLXArray) -> (MLXArray, [MLXArray?]) {
    let audioFeatures = encode(mel)
    let (logits, _, crossQK) = decode(tokens, audioFeatures: audioFeatures)
    return (logits, crossQK)
  }

  /// Main forward pass (matches Python's __call__ method)
  ///
  /// - Parameters:
  ///   - mel: Mel spectrogram (batch, n_mels, n_frames)
  ///   - tokens: Token indices (batch, n_tokens)
  /// - Returns: Logits (batch, n_tokens, n_vocab)
  func callAsFunction(_ mel: MLXArray, tokens: MLXArray) -> MLXArray {
    logits(tokens, audioFeatures: encode(mel))
  }

  /// Whether this is a multilingual model
  ///
  /// Detection logic:
  /// - Multilingual models have n_vocab = 51866 (use multilingual.tiktoken)
  ///   - 50,257 base tokens + 1,609 special tokens = 51,866
  ///   - Models: tiny, base, small, medium, large, large-v2, large-v3, large-v3-turbo
  ///
  /// - English-only models have n_vocab = 51864 (use gpt2.tiktoken)
  ///   - 50,256 base tokens + 1,608 special tokens = 51,864
  ///   - Models: tiny.en, base.en, small.en, medium.en
  var isMultilingual: Bool {
    dims.n_vocab >= 51865
  }

  /// Number of supported languages
  var numLanguages: Int {
    dims.n_vocab - 51765 - (isMultilingual ? 1 : 0)
  }

  /// Set alignment heads for word-level timestamps
  ///
  /// - Parameter heads: Array of shape (num_heads, 2) where each row is [layer_idx, head_idx]
  func setAlignmentHeads(_ heads: MLXArray) {
    alignmentHeads = heads
  }

  /// Load Whisper model from Hugging Face Hub
  ///
  /// - Parameters:
  ///   - modelSize: Model size (tiny, base, small, medium, large, largeTurbo)
  ///   - progressHandler: Optional callback for download progress
  /// - Returns: Initialized WhisperModel with loaded weights
  static func load(
    modelSize: WhisperModelSize,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> WhisperModel {
    // Validate model is available (has safetensors weights)
    guard modelSize.isAvailable else {
      throw STTError.modelUnavailable(
        "Model '\(modelSize.rawValue)' requires .npz format which MLX Swift doesn't support. Available models: tiny, base, large-v3-turbo"
      )
    }

    let repoId = modelSize.repoId
    Log.model.info("Loading Whisper from \(repoId)...")

    // Download model files from HuggingFace
    // MLX Whisper models use different safetensors formats:
    // 1. weights.safetensors (newer models like large-v3-turbo)
    // 2. model.safetensors (alternative naming, quantized models)
    // 3. encoder.safetensors + decoder.safetensors (split weights, jkrukowski repos)
    //
    // Note: MLX Swift only supports .safetensors format (not .npz)
    let modelDirectory = try await Hub.snapshot(
      from: repoId,
      matching: [
        "weights.safetensors",
        "model.safetensors",
        "encoder.safetensors",
        "decoder.safetensors",
        "config.json",
      ],
      progressHandler: progressHandler
    )

    // Load config to get model dimensions
    let configURL = modelDirectory.appending(path: "config.json")
    let dims = try ModelDimensions.load(from: configURL)

    // Try loading weights from various safetensors formats
    let weightsSafetensors = modelDirectory.appending(path: "weights.safetensors")
    let modelSafetensors = modelDirectory.appending(path: "model.safetensors")
    let encoderSafetensors = modelDirectory.appending(path: "encoder.safetensors")
    let decoderSafetensors = modelDirectory.appending(path: "decoder.safetensors")

    let weights: [String: MLXArray]

    if FileManager.default.fileExists(atPath: weightsSafetensors.path) {
      // Format 1: weights.safetensors (single file, e.g., large-v3-turbo)
      weights = try MLX.loadArrays(url: weightsSafetensors)
    } else if FileManager.default.fileExists(atPath: modelSafetensors.path) {
      // Format 2: model.safetensors (quantized models)
      weights = try MLX.loadArrays(url: modelSafetensors)
    } else if FileManager.default.fileExists(atPath: encoderSafetensors.path),
              FileManager.default.fileExists(atPath: decoderSafetensors.path)
    {
      // Format 3: Split weights (encoder + decoder, e.g., jkrukowski repos)
      let encoderWeights = try MLX.loadArrays(url: encoderSafetensors)
      let decoderWeights = try MLX.loadArrays(url: decoderSafetensors)
      weights = encoderWeights.merging(decoderWeights) { _, new in new }
    } else {
      throw STTError.modelUnavailable(
        "No safetensors weights file found (tried weights.safetensors, model.safetensors, encoder/decoder.safetensors)"
      )
    }

    // Initialize model
    let model = WhisperModel(dims: dims)

    // Check if model is quantized (has .scales weights)
    let isQuantized = weights.keys.contains { $0.contains(".scales") }
    if isQuantized {
      Log.model.info("Detected quantized Whisper model weights")
      quantize(model: model) { path, _ in
        if weights["\(path).scales"] != nil {
          return (64, 4, .affine)
        }
        return nil
      }
    }

    // Load weights into model
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    // Set to eval mode for inference
    model.train(false)

    // Evaluate model to ensure weights are loaded
    eval(model)

    Log.model.info("Whisper \(modelSize.rawValue) model loaded successfully")

    return model
  }

  /// Detect the spoken language in the audio
  ///
  /// - Parameter mel: Mel spectrogram (batch, n_mels, n_frames)
  /// - Returns: Tuple of (language_code, probability)
  func detectLanguage(_ mel: MLXArray) -> (String, Float) {
    // Encode audio
    let audioFeatures = encode(mel)

    // Create SOT sequence for language detection
    // [SOT] token ID is 50258
    let sotToken = MLXArray([Int32(50258)]).expandedDimensions(axis: 0)

    // Get logits for the first token after SOT
    let (logits, _, _) = decode(sotToken, audioFeatures: audioFeatures)

    // Get language token logits (tokens 50259-50357 are language tokens)
    let languageTokenStart = 50259
    let languageTokenEnd = 50358 // Exclusive
    let languageLogits = logits[0, 0, languageTokenStart ..< languageTokenEnd]

    // Find the language with highest probability
    let probs = MLX.softmax(languageLogits, axis: -1)
    let maxIdx = MLX.argMax(probs).item(Int32.self)
    let maxProb = probs[Int(maxIdx)].item(Float.self)

    // Map index to language code
    let languageIdx = Int(maxIdx)
    let languageCode = LANGUAGES[languageIdx] ?? "en"

    return (languageCode, maxProb)
  }
}

/// Whisper language codes (ISO 639-1)
///
/// Index corresponds to language token offset from 50259
private let LANGUAGES: [Int: String] = [
  0: "en", 1: "zh", 2: "de", 3: "es", 4: "ru", 5: "ko",
  6: "fr", 7: "ja", 8: "pt", 9: "tr", 10: "pl", 11: "ca",
  12: "nl", 13: "ar", 14: "sv", 15: "it", 16: "id", 17: "hi",
  18: "fi", 19: "vi", 20: "he", 21: "uk", 22: "el", 23: "ms",
  24: "cs", 25: "ro", 26: "da", 27: "hu", 28: "ta", 29: "no",
  30: "th", 31: "ur", 32: "hr", 33: "bg", 34: "lt", 35: "la",
  36: "mi", 37: "ml", 38: "cy", 39: "sk", 40: "te", 41: "fa",
  42: "lv", 43: "bn", 44: "sr", 45: "az", 46: "sl", 47: "kn",
  48: "et", 49: "mk", 50: "br", 51: "eu", 52: "is", 53: "hy",
  54: "ne", 55: "mn", 56: "bs", 57: "kk", 58: "sq", 59: "sw",
  60: "gl", 61: "mr", 62: "pa", 63: "si", 64: "km", 65: "sn",
  66: "yo", 67: "so", 68: "af", 69: "oc", 70: "ka", 71: "be",
  72: "tg", 73: "sd", 74: "gu", 75: "am", 76: "yi", 77: "lo",
  78: "uz", 79: "fo", 80: "ht", 81: "ps", 82: "tk", 83: "nn",
  84: "mt", 85: "sa", 86: "lb", 87: "my", 88: "bo", 89: "tl",
  90: "mg", 91: "as", 92: "tt", 93: "haw", 94: "ln", 95: "ha",
  96: "ba", 97: "jw", 98: "su",
]
