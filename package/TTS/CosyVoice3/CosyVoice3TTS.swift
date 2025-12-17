// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Actor wrapper for CosyVoice3Model that provides thread-safe generation
actor CosyVoice3TTS {
  // MARK: - Properties

  /// Model is nonisolated(unsafe) because it contains non-Sendable types (MLXArray)
  /// but is only accessed within the actor's methods
  private nonisolated(unsafe) let model: CosyVoice3Model

  /// Speaker encoder for extracting embeddings from reference audio (shared with CosyVoice2)
  private nonisolated(unsafe) let speakerEncoder: CAMPlusSpeakerEncoder

  /// Text tokenizer (Qwen2 with CosyVoice3 special tokens)
  private let textTokenizer: Tokenizer

  /// Output sample rate (24kHz for CosyVoice3)
  static let outputSampleRate: Int = CosyVoice3Constants.sampleRate

  /// S3 tokenizer sample rate (16kHz)
  static let tokenizerSampleRate: Int = CosyVoice3Constants.s3TokenizerRate

  // MARK: - Constants

  /// Maximum character count for a single chunk
  private static let maxChunkCharacters = 300

  /// Token ID for <|endofprompt|> special token (used in instruct mode)
  private static let endOfPromptTokenId: Int = 151_646

  /// CosyVoice3-specific special tokens for speech control
  static let specialTokens: [String] = [
    "<|endofprompt|>",
    "[breath]",
    "<strong>",
    "</strong>",
    "[noise]",
    "[laughter]",
    "[cough]",
    "[clucking]",
    "[accent]",
    "[quick_breath]",
    "<laughter>",
    "</laughter>",
    "[hissing]",
    "[sigh]",
    "[vocalized-noise]",
    "[lipsmack]",
    "[mn]",
  ]

  // MARK: - Initialization

  private init(
    model: CosyVoice3Model, speakerEncoder: CAMPlusSpeakerEncoder, textTokenizer: Tokenizer
  ) {
    self.model = model
    self.speakerEncoder = speakerEncoder
    self.textTokenizer = textTokenizer
  }

  /// Default HuggingFace repository ID for CosyVoice3
  static let defaultRepoId = CosyVoice3Constants.defaultRepoId

  /// Load CosyVoice3TTS from HuggingFace repository
  /// - Parameters:
  ///   - repoId: HuggingFace repository ID (default: mlx-community/Fun-CosyVoice3-0.5B-2512-4bit)
  ///   - progressHandler: Optional progress handler for download
  /// - Returns: Loaded CosyVoice3TTS instance
  static func load(
    repoId: String = defaultRepoId,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> CosyVoice3TTS {
    // Download model snapshot from HuggingFace
    let modelDirectory = try await HubConfiguration.shared.snapshot(
      from: repoId,
      progressHandler: progressHandler
    )

    // Load configuration
    let config = try CosyVoice3Config.fromPretrained(modelPath: modelDirectory.path)

    // Load model weights
    let modelURL = modelDirectory.appendingPathComponent("model.safetensors")
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
      throw CosyVoice3Error.modelNotLoaded
    }

    let allWeights = try MLX.loadArrays(url: modelURL)

    // Create model components
    let model = try createModel(config: config, weights: allWeights)

    // Create speaker encoder and load weights from main model.safetensors
    let speakerEncoder = CAMPlusSpeakerEncoder()
    let campplusWeights = allWeights.filter { $0.key.hasPrefix("campplus.") }
      .reduce(into: [:]) { result, pair in
        result[String(pair.key.dropFirst("campplus.".count))] = pair.value
      }
    if !campplusWeights.isEmpty {
      speakerEncoder.loadWeights(from: campplusWeights)
    } else {
      print("Warning: No campplus weights found in model. Speaker embeddings will be zeros.")
    }

    // Load text tokenizer (Qwen2 with CosyVoice3 special tokens)
    let textTokenizer = try await loadTokenizer(modelDirectory: modelDirectory)

    return CosyVoice3TTS(model: model, speakerEncoder: speakerEncoder, textTokenizer: textTokenizer)
  }

  /// Load Qwen2 tokenizer from model directory
  /// - Parameter modelDirectory: URL to the model directory containing tokenizer files
  /// - Returns: Tokenizer configured for CosyVoice3
  ///
  /// Supports both tokenizer.json (fast format) and vocab.json + merges.txt (BPE format)
  private static func loadTokenizer(modelDirectory: URL) async throws -> Tokenizer {
    // Use AutoTokenizer which handles both tokenizer.json and vocab.json formats
    try await AutoTokenizer.from(modelFolder: modelDirectory)
  }

  /// Create model from config and weights
  private static func createModel(
    config: CosyVoice3Config,
    weights: [String: MLXArray]
  ) throws -> CosyVoice3Model {
    // Create Qwen2 config for LLM
    var qwen2Config = CosyVoice3Qwen2Config()
    qwen2Config.hiddenSize = config.llm.hiddenSize
    qwen2Config.numHiddenLayers = config.llm.numHiddenLayers
    qwen2Config.intermediateSize = config.llm.intermediateSize
    qwen2Config.numAttentionHeads = config.llm.numAttentionHeads
    qwen2Config.numKeyValueHeads = config.llm.numKeyValueHeads
    qwen2Config.rmsNormEps = config.llm.rmsNormEps
    qwen2Config.vocabSize = config.llm.vocabSize

    // Create LLM with extended vocabulary
    let llm = CosyVoice3LM(
      llmInputSize: config.llm.llmInputSize,
      llmOutputSize: config.llm.llmOutputSize,
      speechTokenSize: config.llm.speechTokenSize,
      extendedVocabSize: config.llm.extendedVocabSize,
      qwen2Config: qwen2Config,
      mixRatio: config.llm.mixRatio
    )

    // Set sampling function
    llm.sampling = { logits, decodedTokens, topK in
      cosyVoice3RasSampling(logits: logits, decodedTokens: decodedTokens, sampling: topK)
    }

    // Load LLM weights
    let llmWeights: [String: MLXArray] = weights.filter { $0.key.hasPrefix("llm.") }
      .reduce(into: [:]) { result, pair in
        result[String(pair.key.dropFirst("llm.".count))] = pair.value
      }

    // Quantize LLM if weights are quantized
    quantize(model: llm) { path, _ in
      if llmWeights["\(path).scales"] != nil {
        return (groupSize: 64, bits: 4, mode: .affine)
      }
      return nil
    }

    if !llmWeights.isEmpty {
      let weightsList = llmWeights.map { (key: $0.key, value: $0.value) }
      try llm.update(parameters: ModuleParameters.unflattened(weightsList), verify: [])
    }

    // Load Qwen2 backbone weights
    let qwen2Weights: [String: MLXArray] = weights.filter { $0.key.hasPrefix("qwen2.") && $0.key != "qwen2.lm_head.weight" }
      .reduce(into: [:]) { result, pair in
        result[String(pair.key.dropFirst("qwen2.".count))] = pair.value
      }

    // Quantize Qwen2 model if weights are quantized
    quantize(model: llm.llm) { path, _ in
      if qwen2Weights["\(path).scales"] != nil {
        return (groupSize: 64, bits: 4, mode: .affine)
      }
      return nil
    }

    if !qwen2Weights.isEmpty {
      let weightsList = qwen2Weights.map { (key: $0.key, value: $0.value) }
      try llm.llm.update(parameters: ModuleParameters.unflattened(weightsList), verify: [])
    }

    // Create Causal HiFi-GAN vocoder
    let hifigan = CausalHiFTGenerator(
      inChannels: config.hifigan.inChannels,
      baseChannels: config.hifigan.baseChannels,
      nbHarmonics: config.hifigan.nbHarmonics,
      samplingRate: config.hifigan.samplingRate,
      nsfAlpha: config.hifigan.nsfAlpha,
      nsfSigma: config.hifigan.nsfSigma,
      nsfVoicedThreshold: config.hifigan.nsfVoicedThreshold,
      upsampleRates: config.hifigan.upsampleRates,
      upsampleKernelSizes: config.hifigan.upsampleKernelSizes,
      istftParams: ["n_fft": config.hifigan.istftNFft, "hop_len": config.hifigan.istftHopLen],
      resblockKernelSizes: config.hifigan.resblockKernelSizes,
      resblockDilationSizes: config.hifigan.resblockDilationSizes,
      sourceResblockKernelSizes: config.hifigan.sourceResblockKernelSizes,
      sourceResblockDilationSizes: config.hifigan.sourceResblockDilationSizes,
      convPreLookRight: config.hifigan.convPreLookRight
    )

    // Load HiFi-GAN weights
    let hifiganWeights: [String: MLXArray] = weights.filter { $0.key.hasPrefix("hifigan.") || $0.key.hasPrefix("hift.") }
      .reduce(into: [:]) { result, pair in
        var key = pair.key
        if key.hasPrefix("hift.") {
          key = String(key.dropFirst("hift.".count))
        } else {
          key = String(key.dropFirst("hifigan.".count))
        }
        // Remap indexed-style keys to list-style keys
        key = key.replacingOccurrences(of: #"(ups|resblocks|source_downs|source_resblocks)_(\d+)"#, with: "$1.$2", options: .regularExpression)
        key = key.replacingOccurrences(of: #"(convs1|convs2|activations1|activations2)_(\d+)"#, with: "$1.$2", options: .regularExpression)
        result[key] = pair.value
      }

    // Quantize HiFi-GAN if weights are quantized
    quantize(model: hifigan) { path, _ in
      if hifiganWeights["\(path).scales"] != nil {
        return (groupSize: 64, bits: 4, mode: .affine)
      }
      return nil
    }

    if !hifiganWeights.isEmpty {
      let weightsList = hifiganWeights.map { ($0.key, $0.value) }
      try hifigan.update(parameters: ModuleParameters.unflattened(weightsList), verify: [])
    }

    // Create DiT for flow model
    let dit = DiT(
      dim: config.flow.dit.dim,
      depth: config.flow.dit.depth,
      heads: config.flow.dit.heads,
      dimHead: config.flow.dit.dimHead,
      dropout: config.flow.dit.dropout,
      ffMult: config.flow.dit.ffMult,
      melDim: config.flow.dit.melDim,
      muDim: config.flow.dit.muDim,
      longSkipConnection: config.flow.dit.longSkipConnection,
      spkDim: config.flow.dit.spkDim,
      outChannels: config.flow.dit.outChannels,
      staticChunkSize: config.flow.dit.staticChunkSize,
      numDecodingLeftChunks: config.flow.dit.numDecodingLeftChunks
    )

    // Create CFM with DiT estimator
    let cfm = CosyVoice3ConditionalCFM(
      estimator: dit,
      sigmaMin: config.flow.cfmSigmaMin,
      tScheduler: config.flow.cfmTScheduler,
      inferenceCfgRate: config.flow.cfmInferenceCfgRate
    )

    // Create PreLookaheadLayer
    let preLookahead = CosyVoice3PreLookaheadLayer(
      inChannels: config.flow.inputSize,
      channels: config.flow.preLookaheadChannels,
      preLookaheadLen: config.flow.preLookaheadLen
    )

    // Create flow module
    let flow = CosyVoice3FlowModule(
      inputSize: config.flow.inputSize,
      outputSize: config.flow.outputSize,
      spkEmbedDim: config.flow.spkEmbedDim,
      vocabSize: config.flow.vocabSize,
      tokenMelRatio: config.flow.tokenMelRatio,
      preLookaheadLen: config.flow.preLookaheadLen,
      nTimesteps: config.flow.nTimesteps,
      preLookahead: preLookahead,
      decoder: cfm
    )

    // Load flow weights
    let flowWeights: [String: MLXArray] = weights.filter { $0.key.hasPrefix("flow.") }
      .reduce(into: [:]) { result, pair in
        var key = String(pair.key.dropFirst("flow.".count))
        // Remap transformer_blocks_N to transformerBlocks.N
        key = key.replacingOccurrences(of: #"transformer_blocks_(\d+)"#, with: "transformer_blocks.$1", options: .regularExpression)
        // Filter out rotary_embed.inv_freq which is computed at runtime
        if !key.contains("rotary_embed.inv_freq") {
          result[key] = pair.value
        }
      }

    // Quantize flow model if weights are quantized
    quantize(model: flow) { path, _ in
      if flowWeights["\(path).scales"] != nil {
        return (groupSize: 64, bits: 4, mode: .affine)
      }
      return nil
    }

    if !flowWeights.isEmpty {
      let weightsList = flowWeights.map { (key: $0.key, value: $0.value) }
      try flow.update(parameters: ModuleParameters.unflattened(weightsList), verify: [])
    }

    // Create and return model
    return CosyVoice3Model(
      config: config,
      llm: llm,
      flow: flow,
      hifigan: hifigan
    )
  }

  // MARK: - Conditionals

  /// Prepare conditioning from reference audio
  ///
  /// Returns pre-computed conditionals that can be reused across multiple generation calls.
  ///
  /// - Parameters:
  ///   - refWav: Reference audio waveform at 24kHz
  ///   - refText: Optional transcription of reference audio (enables zero-shot mode)
  ///   - s3Tokenizer: S3 speech tokenizer
  /// - Returns: Pre-computed conditionals for generation
  func prepareConditionals(
    refWav: MLXArray,
    refText: String? = nil,
    s3Tokenizer: @Sendable @escaping (MLXArray, MLXArray) -> (MLXArray, MLXArray)
  ) -> CosyVoice3Conditionals {
    // Truncate to max 30 seconds
    let maxSamples = 30 * Self.outputSampleRate
    var audio = refWav
    if audio.shape[0] > maxSamples {
      audio = audio[0 ..< maxSamples]
    }

    // Resample to 16kHz for tokenizer
    let audio16k = resampleAudio(audio, fromRate: Self.outputSampleRate, toRate: Self.tokenizerSampleRate)

    // Extract 128-mel for S3 tokenizer
    let mel128 = logMelSpectrogramCAMPPlus(audio: audio16k, sampleRate: Self.tokenizerSampleRate, numMelBins: 128)
    let mel128Batched = mel128.expandedDimensions(axis: 0)
    let mel128Len = MLXArray([Int32(mel128.shape[1])])

    // Get speech tokens
    let (speechTokens, speechTokenLens) = s3Tokenizer(mel128Batched, mel128Len)

    // Extract 80-mel for flow model (at 24kHz)
    let mel80 = computeMelSpectrogram80(audio: refWav)
    let mel80Batched = mel80.expandedDimensions(axis: 0) // (1, T, D)

    // Align mel and token lengths
    let tokenLen = Int(speechTokenLens[0].item(Int32.self))
    let melLen = min(tokenLen * 2, mel80.shape[0])

    let alignedMel = mel80Batched[0..., 0 ..< melLen, 0...]
    let alignedMelLen = MLXArray([Int32(melLen)])
    let alignedTokens = speechTokens[0..., 0 ..< (melLen / 2)]
    let alignedTokenLen = MLXArray([Int32(melLen / 2)])

    // Extract speaker embedding
    let speakerEmb = speakerEncoder(audio16k)

    // Tokenize reference text if provided
    var promptText: MLXArray?
    var promptTextLen: MLXArray?
    if let text = refText?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
      let tokens = encode(text: text, addSpecialTokens: false)
      promptText = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
      promptTextLen = MLXArray([Int32(tokens.count)])
    }

    return CosyVoice3Conditionals(
      promptSpeechToken: alignedTokens,
      promptSpeechTokenLen: alignedTokenLen,
      promptMel: alignedMel,
      promptMelLen: alignedMelLen,
      speakerEmbedding: speakerEmb,
      promptText: promptText,
      promptTextLen: promptTextLen
    )
  }

  // MARK: - Generation

  /// Generate audio from text using pre-computed conditionals (zero-shot mode)
  func generateZeroShot(
    text _: String,
    textTokens: [Int],
    conditionals: CosyVoice3Conditionals,
    sampling: Int = 25,
    nTimesteps: Int = 10
  ) throws -> TTSGenerationResult {
    let startTime = CFAbsoluteTimeGetCurrent()

    let textArray = MLXArray(textTokens.map { Int32($0) }).reshaped(1, -1)
    let textLen = MLXArray([Int32(textTokens.count)])

    guard let promptText = conditionals.promptText,
          let promptTextLen = conditionals.promptTextLen
    else {
      throw CosyVoice3Error.invalidInput("Zero-shot mode requires reference text in conditionals")
    }

    let audio = try model.synthesizeZeroShot(
      text: textArray,
      textLen: textLen,
      promptText: promptText,
      promptTextLen: promptTextLen,
      promptSpeechToken: conditionals.promptSpeechToken,
      promptSpeechTokenLen: conditionals.promptSpeechTokenLen,
      promptMel: conditionals.promptMel,
      promptMelLen: conditionals.promptMelLen,
      speakerEmbedding: conditionals.speakerEmbedding,
      sampling: sampling,
      nTimesteps: nTimesteps
    )

    audio.eval()

    let processingTime = CFAbsoluteTimeGetCurrent() - startTime

    return TTSGenerationResult(
      audio: audio.squeezed().asArray(Float.self),
      sampleRate: Self.outputSampleRate,
      processingTime: processingTime
    )
  }

  /// Generate audio from text using cross-lingual mode (no reference transcription)
  func generateCrossLingual(
    text _: String,
    textTokens: [Int],
    conditionals: CosyVoice3Conditionals,
    sampling: Int = 25,
    nTimesteps: Int = 10
  ) throws -> TTSGenerationResult {
    let startTime = CFAbsoluteTimeGetCurrent()

    let textArray = MLXArray(textTokens.map { Int32($0) }).reshaped(1, -1)
    let textLen = MLXArray([Int32(textTokens.count)])

    let audio = try model.synthesizeCrossLingual(
      text: textArray,
      textLen: textLen,
      promptSpeechToken: conditionals.promptSpeechToken,
      promptSpeechTokenLen: conditionals.promptSpeechTokenLen,
      promptMel: conditionals.promptMel,
      promptMelLen: conditionals.promptMelLen,
      speakerEmbedding: conditionals.speakerEmbedding,
      sampling: sampling,
      nTimesteps: nTimesteps
    )

    audio.eval()

    let processingTime = CFAbsoluteTimeGetCurrent() - startTime

    return TTSGenerationResult(
      audio: audio.squeezed().asArray(Float.self),
      sampleRate: Self.outputSampleRate,
      processingTime: processingTime
    )
  }

  /// Generate audio using voice conversion mode
  func generateVoiceConversion(
    sourceTokens: MLXArray,
    sourceTokenLen: MLXArray,
    conditionals: CosyVoice3Conditionals,
    nTimesteps: Int = 10
  ) throws -> TTSGenerationResult {
    let startTime = CFAbsoluteTimeGetCurrent()

    let audio = try model.synthesizeVC(
      sourceSpeechToken: sourceTokens,
      sourceSpeechTokenLen: sourceTokenLen,
      promptSpeechToken: conditionals.promptSpeechToken,
      promptSpeechTokenLen: conditionals.promptSpeechTokenLen,
      promptMel: conditionals.promptMel,
      promptMelLen: conditionals.promptMelLen,
      speakerEmbedding: conditionals.speakerEmbedding,
      nTimesteps: nTimesteps
    )

    audio.eval()

    let processingTime = CFAbsoluteTimeGetCurrent() - startTime

    return TTSGenerationResult(
      audio: audio.squeezed().asArray(Float.self),
      sampleRate: Self.outputSampleRate,
      processingTime: processingTime
    )
  }

  /// Generate audio using instruct mode with style control
  func generateInstruct(
    text _: String,
    textTokens: [Int],
    instructText: String,
    conditionals: CosyVoice3Conditionals,
    sampling: Int = 25,
    nTimesteps: Int = 10
  ) throws -> TTSGenerationResult {
    let startTime = CFAbsoluteTimeGetCurrent()

    let textArray = MLXArray(textTokens.map { Int32($0) }).reshaped(1, -1)
    let textLen = MLXArray([Int32(textTokens.count)])

    // Tokenize instruct text and manually append <|endofprompt|> token
    var instructTokens = encode(text: instructText, addSpecialTokens: false)
    instructTokens.append(Self.endOfPromptTokenId)
    let instructArray = MLXArray(instructTokens.map { Int32($0) }).reshaped(1, -1)
    let instructLen = MLXArray([Int32(instructTokens.count)])

    let audio = try model.synthesizeInstruct(
      text: textArray,
      textLen: textLen,
      instructText: instructArray,
      instructTextLen: instructLen,
      promptSpeechToken: conditionals.promptSpeechToken,
      promptSpeechTokenLen: conditionals.promptSpeechTokenLen,
      promptMel: conditionals.promptMel,
      promptMelLen: conditionals.promptMelLen,
      speakerEmbedding: conditionals.speakerEmbedding,
      sampling: sampling,
      nTimesteps: nTimesteps
    )

    audio.eval()

    let processingTime = CFAbsoluteTimeGetCurrent() - startTime

    return TTSGenerationResult(
      audio: audio.squeezed().asArray(Float.self),
      sampleRate: Self.outputSampleRate,
      processingTime: processingTime
    )
  }

  // MARK: - Voice Conversion Source Audio

  /// Stored source audio tokens for voice conversion
  private var vcSourceTokens: MLXArray?
  private var vcSourceTokenLen: MLXArray?

  /// Prepare source audio for voice conversion
  func prepareSourceAudioForVC(
    audio: MLXArray,
    s3Tokenizer: @Sendable @escaping (MLXArray, MLXArray) -> (MLXArray, MLXArray)
  ) {
    // Truncate to max 30 seconds
    let maxSamples = 30 * Self.outputSampleRate
    var audioTruncated = audio
    if audio.shape[0] > maxSamples {
      audioTruncated = audio[0 ..< maxSamples]
    }

    // Resample to 16kHz for tokenizer
    let audio16k = resampleAudio(audioTruncated, fromRate: Self.outputSampleRate, toRate: Self.tokenizerSampleRate)

    // Extract 128-mel for S3 tokenizer
    let mel128 = logMelSpectrogramCAMPPlus(audio: audio16k, sampleRate: Self.tokenizerSampleRate, numMelBins: 128)
    let mel128Batched = mel128.expandedDimensions(axis: 0)
    let mel128Len = MLXArray([Int32(mel128.shape[1])])

    // Get and store speech tokens
    let (tokens, length) = s3Tokenizer(mel128Batched, mel128Len)
    vcSourceTokens = tokens
    vcSourceTokenLen = length
  }

  /// Check if source audio is prepared for voice conversion
  var isSourceAudioPrepared: Bool {
    vcSourceTokens != nil
  }

  /// Clear stored source audio tokens
  func clearSourceAudio() {
    vcSourceTokens = nil
    vcSourceTokenLen = nil
  }

  /// Generate voice conversion using stored source audio
  func generateVoiceConversionFromPrepared(
    conditionals: CosyVoice3Conditionals,
    nTimesteps: Int = 10
  ) throws -> TTSGenerationResult {
    guard let sourceTokens = vcSourceTokens, let sourceTokenLen = vcSourceTokenLen else {
      throw CosyVoice3Error.invalidInput("No source audio prepared for voice conversion")
    }

    let startTime = CFAbsoluteTimeGetCurrent()

    let audio = try model.synthesizeVC(
      sourceSpeechToken: sourceTokens,
      sourceSpeechTokenLen: sourceTokenLen,
      promptSpeechToken: conditionals.promptSpeechToken,
      promptSpeechTokenLen: conditionals.promptSpeechTokenLen,
      promptMel: conditionals.promptMel,
      promptMelLen: conditionals.promptMelLen,
      speakerEmbedding: conditionals.speakerEmbedding,
      nTimesteps: nTimesteps
    )

    audio.eval()

    let processingTime = CFAbsoluteTimeGetCurrent() - startTime

    return TTSGenerationResult(
      audio: audio.squeezed().asArray(Float.self),
      sampleRate: Self.outputSampleRate,
      processingTime: processingTime
    )
  }

  /// Output sample rate
  var sampleRate: Int {
    Self.outputSampleRate
  }

  /// Check if speaker encoder is loaded
  var isSpeakerEncoderLoaded: Bool {
    speakerEncoder.isLoaded
  }

  // MARK: - Tokenization

  /// Encode text to token IDs using the Qwen2 tokenizer
  func encode(text: String, addSpecialTokens: Bool = false) -> [Int] {
    textTokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
  }

  /// Decode token IDs back to text
  func decode(tokens: [Int], skipSpecialTokens: Bool = true) -> String {
    textTokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
  }

  /// Get the token ID for a specific token string
  func tokenToId(_ token: String) -> Int? {
    textTokenizer.convertTokenToId(token)
  }

  // MARK: - Helper Functions

  /// Simple audio resampling using linear interpolation
  private func resampleAudio(_ audio: MLXArray, fromRate: Int, toRate: Int) -> MLXArray {
    if fromRate == toRate {
      return audio
    }

    let ratio = Float(toRate) / Float(fromRate)
    let audioBTC = audio.reshaped(1, -1, 1) // (1, T, 1)
    let interpolated = linearInterpolate1d(audioBTC, scaleFactor: ratio)
    return interpolated.squeezed() // Back to (T',)
  }

  /// Compute 80-mel spectrogram for flow model (at 24kHz)
  private func computeMelSpectrogram80(audio: MLXArray) -> MLXArray {
    // CosyVoice3 flow model mel spectrogram parameters
    let melSpec = s3genMelSpectrogram(
      y: audio,
      nFft: 1920,
      numMels: 80,
      samplingRate: 24000,
      hopSize: 480,
      winSize: 1920,
      fmin: 0,
      fmax: 8000,
      center: false
    )
    // s3genMelSpectrogram returns (num_mels, T') for 1D input
    // We need (T', num_mels) for the flow model
    return melSpec.transposed(1, 0)
  }
}
