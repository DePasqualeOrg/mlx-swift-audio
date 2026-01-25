// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

// Chatterbox Turbo TTS - Fast distilled text-to-speech model

import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Constants

/// Sample rate for speech tokenizer (16kHz)
let ChatterboxTurboS3Sr = 16000

/// Sample rate for vocoder output (24kHz)
let ChatterboxTurboS3GenSr = 24000

/// Size of speech token vocabulary (3^8 = 6561)
let ChatterboxTurboSpeechVocabSize = 6561

/// Silence token for S3Gen
let ChatterboxTurboS3GenSil = 4299

// MARK: - Quantization Options

/// Quantization options for ChatterboxTurbo
public enum ChatterboxTurboQuantization: String, Sendable, CaseIterable {
  /// 16-bit floating point (best quality, larger size)
  case fp16

  /// 8-bit quantization (good balance of quality and size)
  case q8 = "8bit"

  /// 4-bit quantization (smallest size, some quality tradeoff)
  case q4 = "4bit"

  /// Display name
  public var displayName: String {
    switch self {
      case .fp16: "FP16 (Best Quality)"
      case .q8: "8-bit (Balanced)"
      case .q4: "4-bit (Smallest)"
    }
  }

  /// Approximate size multiplier relative to fp16
  public var sizeMultiplier: Float {
    switch self {
      case .fp16: 1.0
      case .q8: 0.5
      case .q4: 0.25
    }
  }
}

// MARK: - Conditionals for Turbo

/// Container for T3Turbo and S3Gen conditioning
struct ChatterboxTurboConditionals: @unchecked Sendable {
  var t3: T3TurboCond
  var gen: CBTRefDict

  init(t3: T3TurboCond, gen: CBTRefDict) {
    self.t3 = t3
    self.gen = gen
  }
}

// MARK: - ChatterboxTurboModel

/// Chatterbox Turbo TTS model
///
/// A fast distilled version of Chatterbox using:
/// - GPT2-Medium backbone (instead of LLaMA)
/// - 2-step meanflow CFM (instead of 10-step)
/// - No CFG, exaggeration, or min_p support
class ChatterboxTurboModel: Module {
  /// Base Hugging Face repository name
  private static let baseRepoName = "Chatterbox-Turbo-TTS"

  /// Hugging Face repository for S3TokenizerV2
  static let s3TokenizerRepoId = "mlx-community/S3TokenizerV2"

  /// Get Hugging Face repository ID for specified quantization
  static func repoId(quantization: ChatterboxTurboQuantization = .q4) -> String {
    "mlx-community/\(baseRepoName)-\(quantization.rawValue)"
  }

  /// Default repository ID (4-bit quantized)
  static var defaultRepoId: String {
    repoId(quantization: .q4)
  }

  /// Encoder conditioning length (15 seconds at 16kHz)
  static let encCondLen = 15 * ChatterboxTurboS3Sr

  /// Decoder conditioning length (10 seconds at 24kHz)
  static let decCondLen = 10 * ChatterboxTurboS3GenSr

  /// Output sample rate
  let sr: Int = ChatterboxTurboS3GenSr

  /// T3 Turbo model (text to speech tokens)
  @ModuleInfo(key: "t3") var t3: T3Turbo

  /// S3Gen model (speech tokens to waveform)
  @ModuleInfo(key: "s3gen") var s3gen: S3Token2WavTurbo

  /// Voice encoder (speaker embedding)
  @ModuleInfo(key: "ve") var ve: VoiceEncoderTurbo

  /// S3 tokenizer (speech tokenization) - loaded separately
  var s3Tokenizer: S3TokenizerV2?

  /// Text tokenizer (GPT-2 BPE from Hugging Face)
  var textTokenizer: Tokenizer?

  /// Pre-computed conditionals
  var conds: ChatterboxTurboConditionals?

  override init() {
    _t3.wrappedValue = T3Turbo()
    _s3gen.wrappedValue = S3Token2WavTurbo(meanflow: true)
    _ve.wrappedValue = VoiceEncoderTurbo()
  }

  /// Output sample rate
  var sampleRate: Int {
    ChatterboxTurboS3GenSr
  }

  // MARK: - Text Tokenizer

  /// Load text tokenizer from model directory
  func loadTextTokenizer(modelDirectory: URL) async throws {
    textTokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)
  }

  // MARK: - Weight Loading

  /// Sanitize weights by removing computed buffers and renaming keys
  private static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]

    for (key, value) in weights {
      // Skip S3Tokenizer weights - loaded separately
      if key.hasPrefix("s3_tokenizer.") {
        continue
      }

      // Skip computed buffers
      if key.contains("freqs_cis") || key.contains("trim_fade") ||
        key.contains("stft_window") || key.contains("pos_enc.pe")
      {
        continue
      }

      // Skip num_batches_tracked
      if key.contains("num_batches_tracked") {
        continue
      }

      var finalKey = key
      var finalValue = value

      // Convert block naming conventions
      let blocksPattern = try! NSRegularExpression(pattern: #"(down_blocks|mid_blocks|up_blocks)_(\d+)"#)
      finalKey = blocksPattern.stringByReplacingMatches(
        in: finalKey, range: NSRange(finalKey.startIndex..., in: finalKey),
        withTemplate: "$1.$2"
      )

      // Remap ff.net array indices to named keys (prevents array unflattening during quantization)
      // ff.net.0.* -> ff.net.gelu.* (CBTGELUProj)
      // ff.net.1.* -> ff.net.linear.* (output Linear)
      finalKey = finalKey.replacingOccurrences(of: ".net.0.", with: ".net.gelu.")
      finalKey = finalKey.replacingOccurrences(of: ".net.1.", with: ".net.linear.")

      // Remove Conv1dPT wrapper key in mel2wav (Python uses wrapper class with inner .conv)
      // e.g., mel2wav.conv_pre.conv.weight -> mel2wav.conv_pre.weight
      if finalKey.contains("mel2wav") {
        finalKey = finalKey.replacingOccurrences(of: ".conv.weight", with: ".weight")
        finalKey = finalKey.replacingOccurrences(of: ".conv.bias", with: ".bias")
      }

      // CAMPPlus weight fixes
      if finalKey.contains("speaker_encoder"), finalKey.hasSuffix(".weight"), value.ndim == 3 {
        if value.shape[1] > value.shape[2] {
          finalValue = value.swappedAxes(1, 2)
        }
      }

      sanitized[finalKey] = finalValue
    }

    return sanitized
  }

  /// Load ChatterboxTurbo model from Hugging Face Hub
  static func load(
    quantization: ChatterboxTurboQuantization = .q4,
    s3TokenizerRepoId: String = s3TokenizerRepoId,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> ChatterboxTurboModel {
    let repoId = repoId(quantization: quantization)

    Log.model.info("Loading ChatterboxTurbo (\(quantization.rawValue)) from \(repoId)...")

    // Download model and S3Tokenizer
    async let modelDirectoryTask = HubConfiguration.shared.snapshot(
      from: repoId,
      matching: ["model.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json", "conds.safetensors"],
      progressHandler: progressHandler
    )

    async let s3TokenizerDirectoryTask = HubConfiguration.shared.snapshot(
      from: s3TokenizerRepoId,
      matching: ["model.safetensors", "config.json"],
      progressHandler: progressHandler
    )

    let (modelDirectory, s3TokenizerDirectory) = try await (modelDirectoryTask, s3TokenizerDirectoryTask)

    // Load and sanitize weights
    let weightFileURL = modelDirectory.appending(path: "model.safetensors")
    let rawWeights = try MLX.loadArrays(url: weightFileURL)
    let weights = sanitizeWeights(rawWeights)

    // Initialize model
    let model = ChatterboxTurboModel()

    // Check for quantized weights and apply quantization to matching layers
    let isQuantized = weights.keys.contains { $0.contains(".scales") }
    let quantBits: Int? = {
      switch quantization {
        case .q8: return 8
        case .q4: return 4
        case .fp16: return nil
      }
    }()
    if isQuantized, let quantBits {
      Log.model.info("Detected quantized ChatterboxTurbo weights")
      quantize(model: model) { path, _ in
        if weights["\(path).scales"] != nil {
          return (64, quantBits, .affine)
        }
        return nil
      }
    }

    // Load weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    // Load S3Tokenizer
    let s3TokenizerWeightURL = s3TokenizerDirectory.appending(path: "model.safetensors")
    let s3TokenizerWeights = try MLX.loadArrays(url: s3TokenizerWeightURL)
    let s3Tokenizer = S3TokenizerV2()
    let s3TokenizerParams = ModuleParameters.unflattened(s3TokenizerWeights)
    try s3Tokenizer.update(parameters: s3TokenizerParams, verify: [.noUnusedKeys])
    model.s3Tokenizer = s3Tokenizer

    // Set to eval mode
    model.train(false)
    eval(model)

    // Load text tokenizer
    try await model.loadTextTokenizer(modelDirectory: modelDirectory)

    // Load pre-computed conditionals if available
    let condsPath = modelDirectory.appending(path: "conds.safetensors")
    if FileManager.default.fileExists(atPath: condsPath.path) {
      let condsData = try MLX.loadArrays(url: condsPath)

      let speakerEmb = condsData["t3.speaker_emb"] ?? MLXArray.zeros([1, 256])
      let condTokens = condsData["t3.cond_prompt_speech_tokens"]

      let t3Cond = T3TurboCond(
        speakerEmb: speakerEmb,
        condPromptSpeechTokens: condTokens
      )

      var genDict: [String: MLXArray] = [:]
      for (k, v) in condsData {
        if k.hasPrefix("gen.") {
          genDict[String(k.dropFirst(4))] = v
        }
      }

      // Create CBTRefDict from loaded data
      let s3GenRef = CBTRefDict(
        promptToken: genDict["prompt_token"] ?? MLXArray.zeros([1, 0]),
        promptTokenLen: genDict["prompt_token_len"] ?? MLXArray([Int32(0)]),
        promptFeat: genDict["prompt_feat"] ?? MLXArray.zeros([1, 0, 80]),
        promptFeatLen: genDict["prompt_feat_len"] ?? MLXArray([Int32(0)]),
        embedding: genDict["embedding"] ?? MLXArray.zeros([1, 192])
      )

      model.conds = ChatterboxTurboConditionals(t3: t3Cond, gen: s3GenRef)
      Log.model.info("Loaded pre-computed conditionals")
    }

    Log.model.info("ChatterboxTurbo model loaded successfully")
    return model
  }

  // MARK: - Conditioning

  /// Prepare conditioning from reference audio
  func prepareConditionals(
    refWav: MLXArray,
    refSr: Int,
    exaggeration _: Float = 0.0 // Ignored in Turbo
  ) -> ChatterboxTurboConditionals {
    var wav = refWav
    if wav.ndim == 2 {
      wav = wav.squeezed(axis: 0)
    }

    // Resample to 24kHz for S3Gen
    var refWav24k = wav
    if refSr != ChatterboxTurboS3GenSr {
      refWav24k = AudioResampler.resample(wav, from: refSr, to: ChatterboxTurboS3GenSr)
    }
    if refWav24k.shape[0] > Self.decCondLen {
      refWav24k = refWav24k[0 ..< Self.decCondLen]
    }

    // Resample to 16kHz
    let refWav16kFrom24k = AudioResampler.resample(refWav24k, from: ChatterboxTurboS3GenSr, to: ChatterboxTurboS3Sr)

    var refWav16kFull = wav
    if refSr != ChatterboxTurboS3Sr {
      refWav16kFull = AudioResampler.resample(wav, from: refSr, to: ChatterboxTurboS3Sr)
    }

    var refWav16k = refWav16kFull
    if refWav16k.shape[0] > Self.encCondLen {
      refWav16k = refWav16k[0 ..< Self.encCondLen]
    }

    // Get S3Gen tokens
    guard let s3Tokenizer else {
      fatalError("S3Tokenizer not loaded")
    }

    let s3genMel = logMelSpectrogramChatterbox(audio: refWav16kFrom24k)
    let s3genMelBatch = s3genMel.expandedDimensions(axis: 0)
    let s3genMelLen = MLXArray([Int32(s3genMel.shape[1])])
    let (s3genTokens, _) = s3Tokenizer.quantize(mel: s3genMelBatch, melLen: s3genMelLen)

    // Get S3Gen reference dict
    let s3genRefDict = s3gen.embedRef(
      refWav: refWav24k.expandedDimensions(axis: 0),
      refSr: ChatterboxTurboS3GenSr,
      refSpeechTokens: s3genTokens,
      refSpeechTokenLens: MLXArray([Int32(s3genTokens.shape[1])])
    )

    // Get T3 tokens
    let t3Mel = logMelSpectrogramChatterbox(audio: refWav16k)
    let t3MelBatch = t3Mel.expandedDimensions(axis: 0)
    let t3MelLen = MLXArray([Int32(t3Mel.shape[1])])
    let (t3Tokens, _) = s3Tokenizer.quantize(mel: t3MelBatch, melLen: t3MelLen)

    let plen = t3.config.speechCondPromptLen
    let t3CondPromptTokens = t3Tokens[0..., 0 ..< min(plen, t3Tokens.shape[1])]

    // Voice encoder embedding
    let veEmbed = ve.embedsFromWavs(wavs: [refWav16kFull])
    let veEmbedMean = veEmbed.mean(axis: 0, keepDims: true)

    let t3Cond = T3TurboCond(
      speakerEmb: veEmbedMean,
      condPromptSpeechTokens: t3CondPromptTokens
    )

    let conditionals = ChatterboxTurboConditionals(t3: t3Cond, gen: s3genRefDict)

    // Force evaluation and clear memory cache to release intermediate allocations
    eval(conditionals.t3.speakerEmb, conditionals.gen.embedding)
    MLXMemory.clearCache()

    return conditionals
  }

  // MARK: - Generation

  /// Generate speech from text
  func generate(
    text: String,
    audioPrompt: MLXArray? = nil,
    audioPromptSr: Int? = nil,
    conds: ChatterboxTurboConditionals? = nil,
    temperature: Float = 0.8,
    repetitionPenalty: Float = 1.2,
    topP: Float = 0.95,
    topK: Int = 1000,
    maxNewTokens: Int = 1000
  ) -> MLXArray {
    // Prepare conditionals
    var conditionals = conds
    if conditionals == nil {
      if let prompt = audioPrompt, let sr = audioPromptSr {
        conditionals = prepareConditionals(refWav: prompt, refSr: sr)
      } else if let cached = self.conds {
        conditionals = cached
      } else {
        fatalError("Reference audio is required")
      }
    }

    guard var cond = conditionals else {
      fatalError("Failed to prepare conditionals")
    }

    // Normalize text
    let normalizedText = puncNorm(text)

    // Tokenize text
    guard let tokenizer = textTokenizer else {
      fatalError("Text tokenizer not loaded")
    }
    let tokenIds = tokenizer.encode(text: normalizedText, addSpecialTokens: false)
    let textTokens = MLXArray(tokenIds.map { Int32($0) }).reshaped([1, -1])

    // Generate speech tokens with T3 Turbo
    let speechTokens = t3.inferenceTurbo(
      t3Cond: &cond.t3,
      textTokens: textTokens,
      temperature: temperature,
      topK: topK,
      topP: topP,
      repetitionPenalty: repetitionPenalty,
      maxGenLen: maxNewTokens
    )

    // Filter valid tokens
    var filteredTokens = speechTokens.flattened()
    let mask = filteredTokens .< ChatterboxTurboSpeechVocabSize
    let maskValues = mask.asArray(Bool.self)
    let validIndices = maskValues.enumerated().compactMap { $0.element ? Int32($0.offset) : nil }
    if !validIndices.isEmpty {
      filteredTokens = filteredTokens[MLXArray(validIndices)]
    }

    // Add silence tokens
    let silence = MLXArray([Int32(ChatterboxTurboS3GenSil), Int32(ChatterboxTurboS3GenSil), Int32(ChatterboxTurboS3GenSil)])
    filteredTokens = MLX.concatenated([filteredTokens, silence], axis: 0)
    filteredTokens = filteredTokens.expandedDimensions(axis: 0)

    // Generate waveform
    let (wav, _) = s3gen.inference(
      speechTokens: filteredTokens,
      refDict: cond.gen,
      nCfmTimesteps: 2
    )

    // Flatten to 1D
    if wav.ndim == 2 {
      return wav.squeezed(axis: 0)
    }
    return wav
  }
}

// MARK: - CBTRefDict Extension

extension S3Token2WavTurbo {
  /// Get reference embeddings for S3Gen
  func embedRef(
    refWav: MLXArray,
    refSr: Int,
    refSpeechTokens: MLXArray? = nil,
    refSpeechTokenLens: MLXArray? = nil
  ) -> CBTRefDict {
    var wav = refWav
    if wav.ndim == 1 {
      wav = wav.expandedDimensions(axis: 0)
    }

    // Resample to 24kHz for mel extraction
    var refWav24k = wav[0]
    if refSr != S3GenTurboConstants.s3GenSR {
      refWav24k = AudioResampler.resample(refWav24k, from: refSr, to: S3GenTurboConstants.s3GenSR)
    }

    // Extract mel features - returns (80, T), need (B, T, 80)
    var refMels = melSpectrogramS3Gen(audio: refWav24k)
    refMels = refMels.expandedDimensions(axis: 0) // (1, 80, T)
    refMels = refMels.transposed(0, 2, 1) // (B, T, 80)

    // Process tokens
    var promptTokens: MLXArray
    var promptTokenLen: MLXArray

    if let tokens = refSpeechTokens, let lens = refSpeechTokenLens {
      promptTokens = tokens
      promptTokenLen = lens

      let actualLen = promptTokens.shape[1]
      let expectedLen = refMels.shape[1] / 2

      if actualLen != expectedLen {
        if actualLen < expectedLen {
          refMels = refMels[0..., 0 ..< (2 * actualLen), 0...]
        } else {
          promptTokens = promptTokens[0..., 0 ..< expectedLen]
        }
        promptTokenLen = MLXArray([Int32(min(actualLen, expectedLen))])
      }
    } else {
      promptTokens = MLXArray.zeros([1, refMels.shape[1] / 2], dtype: .int32)
      promptTokenLen = MLXArray([Int32(refMels.shape[1] / 2)])
    }

    // Resample to 16kHz for speaker encoder
    var refWav16k = wav[0]
    if refSr != S3GenTurboConstants.s3SR {
      refWav16k = AudioResampler.resample(refWav16k, from: refSr, to: S3GenTurboConstants.s3SR)
    }

    // Get speaker embedding
    let xVector = speakerEncoder.inference(refWav16k)
    eval(xVector)

    return CBTRefDict(
      promptToken: promptTokens,
      promptTokenLen: promptTokenLen,
      promptFeat: refMels,
      promptFeatLen: MLXArray([Int32(refMels.shape[1])]),
      embedding: xVector
    )
  }
}

// MARK: - Mel Spectrogram for S3Gen

/// Compute mel spectrogram for S3Gen (24kHz)
/// Uses s3genMelSpectrogram with correct parameters: nFft=1920, numMels=80, hopSize=480
func melSpectrogramS3Gen(audio: MLXArray) -> MLXArray {
  // Ensure batch dimension for s3genMelSpectrogram
  var audioArray = audio
  let was1D = audio.ndim == 1
  if was1D {
    audioArray = audio.expandedDimensions(axis: 0)
  }

  // Use S3Gen mel spectrogram (80 mels, 24kHz parameters)
  var mel = s3genMelSpectrogram(
    y: audioArray,
    nFft: 1920,
    numMels: 80,
    samplingRate: 24000,
    hopSize: 480,
    winSize: 1920,
    fmin: 0,
    fmax: 8000,
    center: false
  )

  // Return without batch dim if input was 1D
  if was1D {
    mel = mel.squeezed(axis: 0)
  }

  return mel
}
