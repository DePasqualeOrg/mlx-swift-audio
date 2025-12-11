//  Main Chatterbox model (Module-based neural network)

import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Constants

/// Sample rate for speech tokenizer (16kHz)
let ChatterboxS3Sr = 16000

/// Sample rate for vocoder output (24kHz)
let ChatterboxS3GenSr = 24000

/// Size of speech token vocabulary (3^8 = 6561)
let ChatterboxSpeechVocabSize = 6561

// MARK: - Special Tokens

let ChatterboxSOT = "[START]"
let ChatterboxEOT = "[STOP]"
let ChatterboxSPACE = "[SPACE]"

// MARK: - Text Normalization

/// Normalize punctuation for TTS input
func puncNorm(_ text: String) -> String {
  var result = text

  if result.isEmpty {
    return "You need to add some text for me to talk."
  }

  // Capitalize first letter
  if let first = result.first, first.isLowercase {
    result = first.uppercased() + String(result.dropFirst())
  }

  // Remove multiple space chars
  let components = result.components(separatedBy: .whitespaces)
  result = components.filter { !$0.isEmpty }.joined(separator: " ")

  // Replace uncommon/LLM punctuation
  let replacements: [(String, String)] = [
    ("...", ", "),
    ("…", ", "),
    (":", ","),
    (" - ", ", "),
    (";", ", "),
    ("—", "-"),
    ("–", "-"),
    (" ,", ","),
    ("\u{201C}", "\""),
    ("\u{201D}", "\""),
    ("'", "'"),
    ("'", "'"),
  ]
  for (old, new) in replacements {
    result = result.replacingOccurrences(of: old, with: new)
  }

  // Add full stop if no ending punctuation
  result = result.trimmingCharacters(in: .whitespaces)
  let sentenceEnders: Set<Character> = [".", "!", "?", "-", ","]
  if let last = result.last, !sentenceEnders.contains(last) {
    result += "."
  }

  return result
}

/// Drop SOS and EOS tokens, extracting only the speech content between them
func dropInvalidTokens(_ x: MLXArray) -> MLXArray {
  let SOS = ChatterboxSpeechVocabSize // 6561
  let EOS = ChatterboxSpeechVocabSize + 1 // 6562

  let xFlat = x.flattened()

  // Find SOS position
  let sosMask = xFlat .== SOS
  var s = 0
  if MLX.any(sosMask).item(Bool.self) {
    s = Int(MLX.argMax(sosMask).item(Int32.self)) + 1
  }

  // Find EOS position
  let eosMask = xFlat .== EOS
  var e = xFlat.shape[0]
  if MLX.any(eosMask).item(Bool.self) {
    e = Int(MLX.argMax(eosMask).item(Int32.self))
  }

  return xFlat[s ..< e]
}

// MARK: - Conditionals

/// Container for T3 and S3Gen conditioning.
///
/// Marked `@unchecked Sendable` because it contains non-Sendable MLXArray fields
/// (via `T3Cond` and `S3GenRefDict`), but all access is controlled within the
/// `ChatterboxTTS` actor's methods.
struct ChatterboxConditionals: @unchecked Sendable {
  /// T3 conditioning (speaker embedding, prompt tokens, emotion)
  var t3: T3Cond

  /// S3Gen reference dictionary
  var gen: S3GenRefDict

  init(t3: T3Cond, gen: S3GenRefDict) {
    self.t3 = t3
    self.gen = gen
  }
}

// MARK: - ChatterboxModel

/// Chatterbox neural network model (Module-based)
///
/// This integrates:
/// - T3: LLaMA-based text-to-speech-token generator
/// - S3Gen: Flow matching decoder with HiFi-GAN vocoder
/// - VoiceEncoder: Speaker embedding extractor
/// - S3Tokenizer: Speech tokenizer for reference audio (loaded from separate repo)
///
/// This class is a `Module` subclass because Chatterbox uses a single weight file that contains
/// parameters for multiple sub-models. The MLX `Module` system's `@ModuleInfo` property wrappers
/// enable hierarchical weight distribution - when `update(parameters:)` is called on this model,
/// weights are automatically routed to the appropriate sub-modules based on their key paths.
///
/// Other TTS engines (Marvis, Orpheus, Kokoro, OuteTTS) don't need this pattern because they load
/// separate weight files directly into each component model.
///
/// Note: Use `ChatterboxTTS` (actor) for thread-safe access.
class ChatterboxModel: Module {
  /// Base Hugging Face repository name for Chatterbox TTS
  private static let baseRepoName = "Chatterbox-TTS"

  /// Hugging Face repository for S3TokenizerV2 (shared across TTS models)
  static let s3TokenizerRepoId = "mlx-community/S3TokenizerV2"

  /// Get Hugging Face repository ID for specified quantization level
  ///
  /// - Parameter quantization: The quantization level (default: q4)
  /// - Returns: The HuggingFace repository ID
  static func repoId(quantization: ChatterboxQuantization = .q4) -> String {
    "mlx-community/\(baseRepoName)-\(quantization.rawValue)"
  }

  /// Default Hugging Face repository ID (4-bit quantized)
  ///
  /// Convenience property that returns the 4-bit variant.
  /// Use `repoId(quantization:)` for other quantization levels.
  static var defaultRepoId: String {
    repoId(quantization: .q4)
  }

  /// Encoder conditioning length (6 seconds at 16kHz)
  static let encCondLen = 6 * ChatterboxS3Sr

  /// Decoder conditioning length (10 seconds at 24kHz)
  static let decCondLen = 10 * ChatterboxS3GenSr

  /// Output sample rate
  let sr: Int = ChatterboxS3GenSr

  /// Configuration
  let config: ChatterboxModelConfig

  /// T3 model (text to speech tokens)
  @ModuleInfo(key: "t3") var t3: T3

  /// S3Gen model (speech tokens to waveform)
  @ModuleInfo(key: "s3gen") var s3gen: S3Token2Wav

  /// Voice encoder (speaker embedding)
  @ModuleInfo(key: "ve") var ve: VoiceEncoder

  /// S3 tokenizer (speech tokenization)
  @ModuleInfo(key: "s3_tokenizer") var s3Tokenizer: S3TokenizerV2

  /// Text tokenizer (text to token IDs)
  var textTokenizer: EnTokenizer?

  /// Pre-computed conditionals (optional)
  var conds: ChatterboxConditionals?

  init(config: ChatterboxModelConfig? = nil) {
    self.config = config ?? ChatterboxModelConfig()

    _t3.wrappedValue = T3(config: self.config.t3Config)
    _s3gen.wrappedValue = S3Token2Wav()
    _ve.wrappedValue = VoiceEncoder()
    _s3Tokenizer.wrappedValue = S3TokenizerV2()
  }

  /// Load text tokenizer from vocabulary file
  ///
  /// - Parameter vocabFilePath: Path to tokenizer.json vocabulary file
  func loadTextTokenizer(vocabFilePath: String) throws {
    textTokenizer = try EnTokenizer(vocabFilePath: vocabFilePath)
  }

  /// Output sample rate
  var sampleRate: Int {
    ChatterboxS3GenSr
  }

  // MARK: - Model Loading

  /// Sanitize weights by removing keys that shouldn't be loaded
  ///
  /// Some keys like `freqs_cis` and `trim_fade` are computed buffers that are not trainable
  /// and should not be loaded from weights.
  /// `embed_tokens` is skipped because T3 uses custom text_emb/speech_emb instead.
  /// `s3_tokenizer.*` keys are skipped because S3Tokenizer is loaded from a separate repo.
  /// Also renames BatchNorm weight/bias to gamma/beta for MLX compatibility.
  private static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]

    for (key, value) in weights {
      // Skip S3Tokenizer weights - loaded from separate repo (mlx-community/S3TokenizerV2)
      if key.hasPrefix("s3_tokenizer.") {
        continue
      }

      // Skip computed buffers that are generated during initialization
      if key.contains("freqsCis") || key.contains("freqs_cis") {
        continue
      }
      if key.contains("trimFade") || key.contains("trim_fade") {
        continue
      }
      if key.contains("randNoise") || key.contains("rand_noise") {
        continue
      }
      if key.contains("stftWindow") || key.contains("stft_window") {
        continue
      }
      if key.contains("pos_enc.pe") || key.contains("posEnc.pe") {
        continue
      }
      // T3 uses custom text_emb and speech_emb, not the LlamaModel's embed_tokens
      if key.contains("embed_tokens") {
        continue
      }
      // T3 uses custom text_head and speech_head, not the LlamaModel's lm_head
      if key.contains("lm_head") {
        continue
      }
      // Skip BatchNorm's num_batches_tracked (not used in MLX)
      if key.contains("num_batches_tracked") {
        continue
      }

      var finalKey = key

      // Convert Python block naming to Swift array indexing
      // down_blocks_0 -> down_blocks.0, mid_blocks_0 -> mid_blocks.0, up_blocks_0 -> up_blocks.0
      let blocksPattern = try! NSRegularExpression(pattern: #"(down_blocks|mid_blocks|up_blocks)_(\d+)"#)
      finalKey = blocksPattern.stringByReplacingMatches(
        in: finalKey, range: NSRange(finalKey.startIndex..., in: finalKey),
        withTemplate: "$1.$2",
      )

      // Convert Python transformer naming to Swift array indexing
      // transformer_0 -> transformers.0
      let transformerPattern = try! NSRegularExpression(pattern: #"\.transformer_(\d+)\."#)
      finalKey = transformerPattern.stringByReplacingMatches(
        in: finalKey, range: NSRange(finalKey.startIndex..., in: finalKey),
        withTemplate: ".transformers.$1.",
      )

      // === CAMPPlus (speaker_encoder) weight name mappings ===

      // Helper function for 1-indexed to 0-indexed regex replacement
      func replaceWithZeroIndexed(_ input: String, pattern: String, prefix: String, suffix: String) -> String {
        let regex = try! NSRegularExpression(pattern: pattern)
        guard let match = regex.firstMatch(in: input, range: NSRange(input.startIndex..., in: input)) else {
          return input
        }
        let fullRange = Range(match.range, in: input)!
        let numberRange = Range(match.range(at: 1), in: input)!
        let number = Int(input[numberRange])! - 1
        return input.replacingCharacters(in: fullRange, with: "\(prefix)\(number)\(suffix)")
      }

      // xvector.block1 -> blocks.0, xvector.block2 -> blocks.1, etc.
      finalKey = replaceWithZeroIndexed(finalKey, pattern: #"xvector\.block(\d+)\."#, prefix: "blocks.", suffix: ".")

      // xvector.transit1 -> transits.0, etc.
      finalKey = replaceWithZeroIndexed(finalKey, pattern: #"xvector\.transit(\d+)\."#, prefix: "transits.", suffix: ".")

      // xvector.tdnn. -> tdnn.
      finalKey = finalKey.replacingOccurrences(of: "xvector.tdnn.", with: "tdnn.")

      // xvector.dense. -> dense.
      finalKey = finalKey.replacingOccurrences(of: "xvector.dense.", with: "dense.")

      // xvector.out_nonlinear. -> out_nonlinear.
      finalKey = finalKey.replacingOccurrences(of: "xvector.out_nonlinear.", with: "out_nonlinear.")

      // .tdnnd1. -> .layers.0., .tdnnd2. -> .layers.1., etc.
      finalKey = replaceWithZeroIndexed(finalKey, pattern: #"\.tdnnd(\d+)\."#, prefix: ".layers.", suffix: ".")

      // .nonlinear1.batchnorm. -> .nonlinear1.0.
      let nonlinear1BnPattern = try! NSRegularExpression(pattern: #"\.nonlinear(\d+)\.batchnorm\."#)
      finalKey = nonlinear1BnPattern.stringByReplacingMatches(
        in: finalKey, range: NSRange(finalKey.startIndex..., in: finalKey),
        withTemplate: ".nonlinear$1.0.",
      )

      // .nonlinear.batchnorm. -> .nonlinear.0.
      finalKey = finalKey.replacingOccurrences(of: ".nonlinear.batchnorm.", with: ".nonlinear.0.")

      // out_nonlinear.batchnorm. -> out_nonlinear.0.
      finalKey = finalKey.replacingOccurrences(of: ".out_nonlinear.batchnorm.", with: ".out_nonlinear.0.")
      if finalKey.hasPrefix("out_nonlinear.batchnorm.") {
        finalKey = finalKey.replacingOccurrences(of: "out_nonlinear.batchnorm.", with: "out_nonlinear.0.")
      }

      // Note: MLX Swift BatchNorm uses weight/bias, not gamma/beta, so no renaming needed

      // === Conv1d weight transposition for CAMPPlus ===
      // PyTorch Conv1d: (O, I, K) -> MLX Conv1d: (O, K, I)
      // Note: Conv2d weights are already in MLX format from HuggingFace, only Conv1d needs fix
      // Heuristic: if dim[1] > dim[2], shape looks like PyTorch format (in_channels > kernel_size)
      var finalValue = value
      if finalKey.contains("speaker_encoder"), finalKey.hasSuffix(".weight"), value.ndim == 3 {
        if value.shape[1] > value.shape[2] {
          finalValue = value.swappedAxes(1, 2)
        }
      }

      sanitized[finalKey] = finalValue
    }
    return sanitized
  }

  /// Load Chatterbox TTS model from Hugging Face Hub
  ///
  /// Loads the main model weights (T3, S3Gen, VoiceEncoder) from the Chatterbox repo
  /// and the S3Tokenizer weights from a separate shared repo.
  ///
  /// - Parameters:
  ///   - quantization: Quantization level (fp16, 8bit, 4bit). Default is 4bit.
  ///   - s3TokenizerRepoId: Hugging Face repository ID for S3Tokenizer (default: mlx-community/S3TokenizerV2)
  ///   - progressHandler: Optional callback for download progress
  /// - Returns: Initialized ChatterboxModel with loaded weights
  static func load(
    quantization: ChatterboxQuantization = .q4,
    s3TokenizerRepoId: String = s3TokenizerRepoId,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in },
  ) async throws -> ChatterboxModel {
    let repoId = repoId(quantization: quantization)

    // Download both repos in parallel
    Log.model.info("Loading Chatterbox (\(quantization.rawValue)) from \(repoId) and S3Tokenizer from \(s3TokenizerRepoId)...")

    async let modelDirectoryTask = HubConfiguration.shared.snapshot(
      from: repoId,
      matching: ["model.safetensors", "tokenizer.json", "config.json"],
      progressHandler: progressHandler
    )

    async let s3TokenizerDirectoryTask = HubConfiguration.shared.snapshot(
      from: s3TokenizerRepoId,
      matching: ["model.safetensors", "config.json"],
      progressHandler: progressHandler
    )

    let (modelDirectory, s3TokenizerDirectory) = try await (modelDirectoryTask, s3TokenizerDirectoryTask)

    // Load Chatterbox weights and sanitize (remove computed buffers like freqs_cis)
    let weightFileURL = modelDirectory.appending(path: "model.safetensors")
    let rawWeights = try MLX.loadArrays(url: weightFileURL)
    let weights = sanitizeWeights(rawWeights)

    // Initialize model
    let model = ChatterboxModel()

    // Check if model is quantized (has .scales weights)
    let isQuantized = weights.keys.contains { $0.contains(".scales") }
    if isQuantized {
      Log.model.info("Detected quantized Chatterbox model weights")
      quantize(model: model) { path, _ in
        if weights["\(path).scales"] != nil {
          return (64, 4, .affine)
        }
        return nil
      }
    }

    // Load Chatterbox weights into model (T3, S3Gen, VE - excludes s3_tokenizer)
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    // Load S3Tokenizer weights separately
    let s3TokenizerWeightURL = s3TokenizerDirectory.appending(path: "model.safetensors")
    let s3TokenizerRawWeights = try MLX.loadArrays(url: s3TokenizerWeightURL)

    // Add s3_tokenizer prefix for the Module system
    var s3TokenizerWeights: [String: MLXArray] = [:]
    for (key, value) in s3TokenizerRawWeights {
      s3TokenizerWeights["s3_tokenizer.\(key)"] = value
    }

    let s3TokenizerParameters = ModuleParameters.unflattened(s3TokenizerWeights)
    try model.update(parameters: s3TokenizerParameters, verify: [.noUnusedKeys])

    // Set to eval mode for inference (important for BatchNorm to use running stats)
    model.train(false)

    // Evaluate model to ensure weights are loaded
    eval(model)

    // Load text tokenizer
    let tokenizerPath = modelDirectory.appending(path: "tokenizer.json")
    try model.loadTextTokenizer(vocabFilePath: tokenizerPath.path)

    Log.model.info("Chatterbox TTS model (\(quantization.rawValue)) loaded successfully")

    return model
  }

  /// Prepare conditioning from a reference audio clip
  ///
  /// - Parameters:
  ///   - refWav: Reference waveform (samples,) or (1, samples)
  ///   - refSr: Reference sample rate
  ///   - exaggeration: Emotion exaggeration factor (0-1)
  /// - Returns: Conditionals object with T3 and S3Gen conditioning
  func prepareConditionals(
    refWav: MLXArray,
    refSr: Int,
    exaggeration: Float = 0.5,
  ) -> ChatterboxConditionals {
    var wav = refWav
    if wav.ndim == 2 {
      wav = wav.squeezed(axis: 0)
    }

    // Resample to 24kHz for S3Gen
    var refWav24k = wav
    if refSr != ChatterboxS3GenSr {
      refWav24k = AudioResampler.resample(wav, from: refSr, to: ChatterboxS3GenSr)
    }
    // Truncate to decoder conditioning length (10s)
    if refWav24k.shape[0] > Self.decCondLen {
      refWav24k = refWav24k[0 ..< Self.decCondLen]
    }

    // Resample 24kHz to 16kHz for S3Gen tokenization
    let refWav16kFrom24k = AudioResampler.resample(refWav24k, from: ChatterboxS3GenSr, to: ChatterboxS3Sr)

    // Resample original to 16kHz for T3 encoder conditioning
    var refWav16kFull = wav
    if refSr != ChatterboxS3Sr {
      refWav16kFull = AudioResampler.resample(wav, from: refSr, to: ChatterboxS3Sr)
    }

    // Truncate to encoder conditioning length (6s) for T3 tokens
    var refWav16k = refWav16kFull
    if refWav16k.shape[0] > Self.encCondLen {
      refWav16k = refWav16k[0 ..< Self.encCondLen]
    }

    // Get S3Gen reference embeddings
    var s3genRefDict = S3GenRefDict(
      promptToken: MLXArray.zeros([1, 0]),
      promptTokenLen: MLXArray([Int32(0)]),
      promptFeat: MLXArray.zeros([1, 0, 80]),
      promptFeatLen: MLXArray([Int32(0)]),
      embedding: MLXArray.zeros([1, 192]),
    )
    var t3CondPromptTokens: MLXArray?

    // S3Gen tokens (from 10s audio, resampled 24k->16k)
    let s3genMel = logMelSpectrogramChatterbox(audio: refWav16kFrom24k)
    let s3genMelBatch = s3genMel.expandedDimensions(axis: 0)
    let s3genMelLen = MLXArray([Int32(s3genMel.shape[1])])

    let (s3genTokens, _) = s3Tokenizer.quantize(mel: s3genMelBatch, melLen: s3genMelLen)

    // Get S3Gen embeddings
    s3genRefDict = s3gen.embedRef(
      refWav: refWav24k.expandedDimensions(axis: 0),
      refSr: ChatterboxS3GenSr,
      refSpeechTokens: s3genTokens,
      refSpeechTokenLens: MLXArray([Int32(s3genTokens.shape[1])]),
    )

    // T3 conditioning tokens (from 6s audio)
    let t3Mel = logMelSpectrogramChatterbox(audio: refWav16k)
    let t3MelBatch = t3Mel.expandedDimensions(axis: 0)
    let t3MelLen = MLXArray([Int32(t3Mel.shape[1])])

    let (t3Tokens, _) = s3Tokenizer.quantize(mel: t3MelBatch, melLen: t3MelLen)

    // Limit T3 tokens to prompt length
    let plen = config.t3Config.speechCondPromptLen
    t3CondPromptTokens = t3Tokens[0..., 0 ..< min(plen, t3Tokens.shape[1])]

    // Voice encoder speaker embedding
    let veEmbed = ve.embedsFromWavs(wavs: [refWav16kFull])
    let veEmbedMean = veEmbed.mean(axis: 0, keepDims: true)

    let t3Cond = T3Cond(
      speakerEmb: veEmbedMean,
      condPromptSpeechTokens: t3CondPromptTokens,
      emotionAdv: MLXArray([exaggeration]).reshaped([1, 1, 1]),
    )

    return ChatterboxConditionals(t3: t3Cond, gen: s3genRefDict)
  }

  /// Generate speech from text
  ///
  /// - Parameters:
  ///   - text: Input text to synthesize
  ///   - audioPrompt: Reference audio for voice matching
  ///   - audioPromptSr: Sample rate of audio prompt
  ///   - conds: Pre-computed conditionals (optional)
  ///   - exaggeration: Emotion exaggeration factor (0-1)
  ///   - cfgWeight: Classifier-free guidance weight
  ///   - temperature: Sampling temperature
  ///   - repetitionPenalty: Penalty for repeated tokens
  ///   - minP: Minimum probability threshold
  ///   - topP: Top-p (nucleus) sampling threshold
  ///   - maxNewTokens: Maximum number of tokens to generate
  /// - Returns: Generated audio waveform
  func generate(
    text: String,
    audioPrompt: MLXArray? = nil,
    audioPromptSr: Int? = nil,
    conds: ChatterboxConditionals? = nil,
    exaggeration: Float = 0.1,
    cfgWeight: Float = 0.5,
    temperature: Float = 0.8,
    repetitionPenalty: Float = 1.2,
    minP: Float = 0.05,
    topP: Float = 1.0,
    maxNewTokens: Int = 1000,
  ) -> MLXArray {
    // Prepare conditionals if needed
    var conditionals = conds
    if conditionals == nil {
      if let prompt = audioPrompt, let sr = audioPromptSr {
        conditionals = prepareConditionals(refWav: prompt, refSr: sr, exaggeration: exaggeration)
      } else if let cached = self.conds {
        conditionals = cached
      } else {
        fatalError("Reference audio is required for Chatterbox TTS")
      }
    }

    guard var cond = conditionals else {
      fatalError("Failed to prepare conditionals")
    }

    // Update exaggeration if needed
    cond.t3.emotionAdv = MLXArray([exaggeration]).reshaped([1, 1, 1])

    // Normalize text
    let normalizedText = puncNorm(text)

    // Tokenize text using EnTokenizer
    guard let tokenizer = textTokenizer else {
      fatalError("Text tokenizer not loaded. Call loadTextTokenizer(vocabFilePath:) first.")
    }
    let textTokens = tokenizer.textToTokens(normalizedText)

    var tokens = textTokens
    if cfgWeight > 0.0 {
      tokens = MLX.concatenated([tokens, tokens], axis: 0)
    }

    // Add start/end tokens
    let sot = config.t3Config.startTextToken
    let eot = config.t3Config.stopTextToken

    let sotTokens = MLXArray.full([tokens.shape[0], 1], values: MLXArray(Int32(sot)))
    let eotTokens = MLXArray.full([tokens.shape[0], 1], values: MLXArray(Int32(eot)))
    tokens = MLX.concatenated([sotTokens, tokens, eotTokens], axis: 1)

    // Generate speech tokens with T3
    let speechTokens = t3.inference(
      t3Cond: &cond.t3,
      textTokens: tokens,
      maxNewTokens: maxNewTokens,
      temperature: temperature,
      topP: topP,
      minP: minP,
      repetitionPenalty: repetitionPenalty,
      cfgWeight: cfgWeight,
    )

    // Check for truncation: if we generated maxNewTokens and last token isn't EOS
    let generatedCount = speechTokens.shape[1]
    if generatedCount >= maxNewTokens {
      let lastToken: Int32 = speechTokens[0, generatedCount - 1].item()
      if lastToken != Int32(config.t3Config.stopSpeechToken) {
        Log.tts.warning(
          "ChatterboxTTS: Generation hit token limit (\(maxNewTokens)), audio may be truncated.",
        )
      }
    }

    // Extract conditional batch (first in CFG pair)
    var filteredTokens = speechTokens[0 ..< 1]

    // Drop invalid tokens
    filteredTokens = dropInvalidTokens(filteredTokens)

    // Filter out tokens >= SPEECH_VOCAB_SIZE
    // Pull mask to CPU once (single bulk transfer) rather than per-token .item() calls
    let mask = filteredTokens .< ChatterboxSpeechVocabSize
    let maskValues = mask.asArray(Bool.self)
    let validIndices = maskValues.enumerated().compactMap { $0.element ? Int32($0.offset) : nil }
    if !validIndices.isEmpty {
      filteredTokens = filteredTokens[MLXArray(validIndices)]
    } else {
      filteredTokens = MLXArray([Int32]())
    }

    // Reshape for S3Gen
    filteredTokens = filteredTokens.expandedDimensions(axis: 0)

    // Generate waveform with S3Gen
    var wav = s3gen(speechTokens: filteredTokens, refDict: cond.gen, finalize: true)

    // Flatten to 1D if needed
    if wav.ndim == 2 {
      wav = wav.squeezed(axis: 0)
    }

    return wav
  }
}
