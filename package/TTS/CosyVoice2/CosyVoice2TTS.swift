// CosyVoice2 TTS wrapper with thread-safe generation
// Similar to ChatterboxTTS actor pattern

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Actor wrapper for CosyVoice2Model that provides thread-safe generation
actor CosyVoice2TTS {
  // MARK: - Properties

  /// Model is nonisolated(unsafe) because it contains non-Sendable types (MLXArray)
  /// but is only accessed within the actor's methods
  private nonisolated(unsafe) let model: CosyVoice2Model

  /// Speaker encoder for extracting embeddings from reference audio
  private nonisolated(unsafe) let speakerEncoder: CAMPlusSpeakerEncoder

  /// Text tokenizer (Qwen2 with CosyVoice2 special tokens)
  private let textTokenizer: PreTrainedTokenizer

  /// Output sample rate (24kHz for CosyVoice2)
  static let outputSampleRate: Int = 24000

  /// S3 tokenizer sample rate (16kHz)
  static let tokenizerSampleRate: Int = 16000

  // MARK: - Constants

  /// Maximum character count for a single chunk
  private static let maxChunkCharacters = 300

  /// Token ID for <|endofprompt|> special token (used in instruct mode)
  /// This is the first token added after the base Qwen2 vocabulary (151643-151645)
  /// The base tokenizer may not recognize this token, so we handle it manually
  private static let endOfPromptTokenId: Int = 151_646

  /// CosyVoice2-specific special tokens for speech control
  /// These tokens are added dynamically at runtime in the original implementation
  /// and may not be in the base tokenizer vocabulary
  static let specialTokens: [String] = [
    "<|endofprompt|>", // For instruct mode
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
    model: CosyVoice2Model, speakerEncoder: CAMPlusSpeakerEncoder, textTokenizer: PreTrainedTokenizer
  ) {
    self.model = model
    self.speakerEncoder = speakerEncoder
    self.textTokenizer = textTokenizer
  }

  /// Default HuggingFace repository ID for CosyVoice2
  static let defaultRepoId = "mlx-community/CosyVoice2-0.5B-4bit"

  /// Load CosyVoice2TTS from HuggingFace repository
  /// - Parameters:
  ///   - repoId: HuggingFace repository ID (default: mlx-community/CosyVoice2-0.5B-4bit)
  ///   - progressHandler: Optional progress handler for download
  /// - Returns: Loaded CosyVoice2TTS instance
  static func load(
    repoId: String = defaultRepoId,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> CosyVoice2TTS {
    // Download model snapshot from HuggingFace
    let modelDirectory = try await HubConfiguration.shared.snapshot(
      from: repoId,
      progressHandler: progressHandler
    )

    // Load configuration
    let config = try CosyVoice2Config.fromPretrained(modelPath: modelDirectory.path)

    // Load model weights
    let modelURL = modelDirectory.appendingPathComponent("model.safetensors")
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
      throw CosyVoice2Error.modelNotLoaded
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

    // Load text tokenizer (Qwen2 with CosyVoice2 special tokens)
    let textTokenizer = try loadTokenizer(modelDirectory: modelDirectory)

    return CosyVoice2TTS(model: model, speakerEncoder: speakerEncoder, textTokenizer: textTokenizer)
  }

  /// Load Qwen2 tokenizer from model directory
  /// - Parameter modelDirectory: URL to the model directory containing tokenizer.json and tokenizer_config.json
  /// - Returns: PreTrainedTokenizer configured for CosyVoice2
  private static func loadTokenizer(modelDirectory: URL) throws -> PreTrainedTokenizer {
    let tokenizerConfigURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
    let tokenizerDataURL = modelDirectory.appendingPathComponent("tokenizer.json")

    guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
      throw CosyVoice2Error.tokenizerNotFound("tokenizer_config.json not found at \(modelDirectory.path)")
    }
    guard FileManager.default.fileExists(atPath: tokenizerDataURL.path) else {
      throw CosyVoice2Error.tokenizerNotFound("tokenizer.json not found at \(modelDirectory.path)")
    }

    let tokenizerConfigData = try Data(contentsOf: tokenizerConfigURL)
    let tokenizerDataData = try Data(contentsOf: tokenizerDataURL)

    let decoder = JSONDecoder()
    let tokenizerConfig = try decoder.decode(Config.self, from: tokenizerConfigData)
    let tokenizerData = try decoder.decode(Config.self, from: tokenizerDataData)

    return try PreTrainedTokenizer(
      tokenizerConfig: tokenizerConfig,
      tokenizerData: tokenizerData
    )
  }

  /// Create model from config and weights
  private static func createModel(
    config: CosyVoice2Config,
    weights: [String: MLXArray]
  ) throws -> CosyVoice2Model {
    // Create Qwen2 config
    var qwen2Config = Qwen2Config()
    qwen2Config.hiddenSize = config.llm.hiddenSize
    qwen2Config.numHiddenLayers = config.llm.numHiddenLayers
    qwen2Config.intermediateSize = config.llm.intermediateSize
    qwen2Config.numAttentionHeads = config.llm.numAttentionHeads
    qwen2Config.numKeyValueHeads = config.llm.numKeyValueHeads
    qwen2Config.rmsNormEps = config.llm.rmsNormEps
    qwen2Config.vocabSize = config.llm.vocabSize

    // Create LLM
    let llm = Qwen2LM(
      llmInputSize: config.llm.llmInputSize,
      llmOutputSize: config.llm.llmOutputSize,
      speechTokenSize: config.llm.speechTokenSize,
      qwen2Config: qwen2Config,
      mixRatio: config.llm.mixRatio
    )

    // Set sampling function
    llm.sampling = { logits, decodedTokens, topK in
      rasSampling(logits: logits, decodedTokens: decodedTokens, sampling: topK)
    }

    // Load LLM weights (non-Qwen2 parts: llm_embedding, llm_decoder, speech_embedding)
    let llmWeights: [String: MLXArray] = weights.filter { $0.key.hasPrefix("llm.") }
      .reduce(into: [:]) { result, pair in
        result[String(pair.key.dropFirst("llm.".count))] = pair.value
      }

    // Quantize LLM if weights are quantized (check for .scales keys)
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

    // Create HiFi-GAN vocoder
    let hifigan = CosyHiFTGenerator(
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
      sourceResblockDilationSizes: config.hifigan.sourceResblockDilationSizes
    )

    // Load HiFi-GAN weights
    let hiftWeights: [String: MLXArray] = weights.filter { $0.key.hasPrefix("hift.") }
      .reduce(into: [:]) { result, pair in
        var key = String(pair.key.dropFirst("hift.".count))
        // Remap indexed-style keys to list-style keys
        key = key.replacingOccurrences(of: #"(ups|resblocks|source_downs|source_resblocks)_(\d+)"#, with: "$1.$2", options: .regularExpression)
        key = key.replacingOccurrences(of: #"(convs1|convs2|activations1|activations2)_(\d+)"#, with: "$1.$2", options: .regularExpression)
        result[key] = pair.value
      }

    // Quantize HiFi-GAN if weights are quantized
    quantize(model: hifigan) { path, _ in
      if hiftWeights["\(path).scales"] != nil {
        return (groupSize: 64, bits: 4, mode: .affine)
      }
      return nil
    }

    if !hiftWeights.isEmpty {
      let weightsList = hiftWeights.map { ($0.key, $0.value) }
      try hifigan.update(parameters: ModuleParameters.unflattened(weightsList), verify: [])
    }

    // Create flow decoder (ConditionalDecoder) - the estimator for CFM
    let flowDecoder = ConditionalDecoder(
      inChannels: config.flow.decoderInChannels,
      outChannels: config.flow.decoderOutChannel,
      causal: true,
      channels: config.flow.decoderChannels,
      dropout: config.flow.decoderDropout,
      attentionHeadDim: config.flow.decoderAttentionHeadDim,
      nBlocks: config.flow.decoderNBlocks,
      numMidBlocks: config.flow.decoderNumMidBlocks,
      numHeads: config.flow.decoderNumHeads,
      actFn: config.flow.decoderActFn
    )

    // Create CFM config
    var cfmConfig = CosyVoice2CFMConfig()
    cfmConfig.sigmaMin = config.flow.cfmSigmaMin
    cfmConfig.tScheduler = config.flow.cfmTScheduler
    cfmConfig.inferenceCfgRate = config.flow.cfmInferenceCfgRate

    // Create CFM with decoder
    let cfm = CosyVoice2ConditionalCFM(
      inChannels: config.flow.cfmInChannels,
      cfmConfig: cfmConfig,
      nSpks: 1,
      spkEmbDim: config.flow.outputSize,
      estimator: flowDecoder
    )

    // Create flow encoder
    let flowEncoder = UpsampleConformerEncoder(
      inputSize: config.flow.encoderInputSize,
      outputSize: config.flow.encoderOutputSize,
      attentionHeads: config.flow.encoderAttentionHeads,
      linearUnits: config.flow.encoderLinearUnits,
      numBlocks: config.flow.encoderNumBlocks,
      numUpBlocks: config.flow.encoderNumUpBlocks,
      dropoutRate: config.flow.encoderDropoutRate,
      positionalDropoutRate: config.flow.encoderPositionalDropoutRate,
      attentionDropoutRate: config.flow.encoderAttentionDropoutRate,
      normalizeBefore: config.flow.encoderNormalizeBefore,
      staticChunkSize: config.flow.encoderStaticChunkSize,
      macaronStyle: config.flow.encoderMacaronStyle,
      useCnnModule: config.flow.encoderUseCnnModule,
      cnnModuleKernel: config.flow.encoderCnnModuleKernel,
      causal: config.flow.encoderCausal,
      preLookaheadLen: config.flow.preLookaheadLen,
      upsampleStride: config.flow.encoderUpsampleStride
    )

    // Create flow module with pre-built components
    let flow = CosyVoice2FlowModule(
      inputSize: config.flow.inputSize,
      outputSize: config.flow.outputSize,
      spkEmbedDim: config.flow.spkEmbedDim,
      vocabSize: config.flow.vocabSize,
      tokenMelRatio: config.flow.tokenMelRatio,
      nTimesteps: config.flow.nTimesteps,
      encoder: flowEncoder,
      decoder: cfm
    )

    // Load flow weights
    let flowWeights: [String: MLXArray] = weights.filter { $0.key.hasPrefix("flow.") }
      .reduce(into: [:]) { result, pair in
        var key = String(pair.key.dropFirst("flow.".count))
        // Remap indexed-style keys (underscore) to list-style keys (dot) for encoder layers
        // Python saves as encoders_0, encoders_1, etc. but Swift expects encoders.0, encoders.1
        key = key.replacingOccurrences(of: #"encoder\.encoders_(\d+)"#, with: "encoder.encoders.$1", options: .regularExpression)
        key = key.replacingOccurrences(of: #"encoder\.up_encoders_(\d+)"#, with: "encoder.up_encoders.$1", options: .regularExpression)
        // Remap decoder (CFM estimator) block keys
        // Python saves as down_blocks_0, mid_blocks_0, up_blocks_0 but Swift expects down_blocks.0, etc.
        key = key.replacingOccurrences(of: #"decoder\.estimator\.down_blocks_(\d+)"#, with: "decoder.estimator.down_blocks.$1", options: .regularExpression)
        key = key.replacingOccurrences(of: #"decoder\.estimator\.mid_blocks_(\d+)"#, with: "decoder.estimator.mid_blocks.$1", options: .regularExpression)
        key = key.replacingOccurrences(of: #"decoder\.estimator\.up_blocks_(\d+)"#, with: "decoder.estimator.up_blocks.$1", options: .regularExpression)
        // Remap transformer blocks within each down/mid/up block
        // Python saves as transformer_0, transformer_1, etc. but Swift expects transformers.0, transformers.1
        key = key.replacingOccurrences(of: #"\.transformer_(\d+)\."#, with: ".transformers.$1.", options: .regularExpression)
        result[key] = pair.value
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
    return CosyVoice2Model(
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
  ) -> CosyVoice2Conditionals {
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
    // computeMelSpectrogram80 returns (T, D) format where T=time, D=80 mel bins
    let mel80 = computeMelSpectrogram80(audio: refWav)
    let mel80Batched = mel80.expandedDimensions(axis: 0) // (1, T, D)

    // Align mel and token lengths
    // mel80.shape[0] is the time dimension (T), mel80.shape[1] is the mel dimension (D)
    let tokenLen = Int(speechTokenLens[0].item(Int32.self))
    let melLen = min(tokenLen * 2, mel80.shape[0]) // Use shape[0] for time dimension!

    let alignedMel = mel80Batched[0..., 0 ..< melLen, 0...]
    let alignedMelLen = MLXArray([Int32(melLen)])
    let alignedTokens = speechTokens[0..., 0 ..< (melLen / 2)]
    let alignedTokenLen = MLXArray([Int32(melLen / 2)])

    // Extract speaker embedding
    let speakerEmb = speakerEncoder(audio16k)

    // Tokenize reference text if provided (using built-in tokenizer)
    var promptText: MLXArray?
    var promptTextLen: MLXArray?
    if let text = refText {
      let tokens = encode(text: text, addSpecialTokens: false)
      promptText = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
      promptTextLen = MLXArray([Int32(tokens.count)])
    }

    return CosyVoice2Conditionals(
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
  ///
  /// - Parameters:
  ///   - text: Text to synthesize
  ///   - textTokens: Tokenized text
  ///   - conditionals: Pre-computed reference audio conditionals
  ///   - sampling: Top-k sampling parameter
  ///   - nTimesteps: Number of flow matching steps
  /// - Returns: Generated audio result
  func generateZeroShot(
    text _: String,
    textTokens: [Int],
    conditionals: CosyVoice2Conditionals,
    sampling: Int = 25,
    nTimesteps: Int = 10
  ) throws -> TTSGenerationResult {
    let startTime = CFAbsoluteTimeGetCurrent()

    let textArray = MLXArray(textTokens.map { Int32($0) }).reshaped(1, -1)
    let textLen = MLXArray([Int32(textTokens.count)])

    guard let promptText = conditionals.promptText,
          let promptTextLen = conditionals.promptTextLen
    else {
      throw CosyVoice2Error.invalidInput("Zero-shot mode requires reference text in conditionals")
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
    conditionals: CosyVoice2Conditionals,
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
    conditionals: CosyVoice2Conditionals,
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
  ///
  /// Instruct mode allows controlling the style of speech generation with instructions
  /// like "Speak slowly and calmly" or "Read with excitement".
  ///
  /// - Parameters:
  ///   - text: Text to synthesize
  ///   - textTokens: Tokenized text
  ///   - instructText: Style instruction (should end with <|endofprompt|>)
  ///   - conditionals: Pre-computed reference audio conditionals
  ///   - sampling: Top-k sampling parameter
  ///   - nTimesteps: Number of flow matching steps
  /// - Returns: Generated audio result
  func generateInstruct(
    text _: String,
    textTokens: [Int],
    instructText: String,
    conditionals: CosyVoice2Conditionals,
    sampling: Int = 25,
    nTimesteps: Int = 10
  ) throws -> TTSGenerationResult {
    let startTime = CFAbsoluteTimeGetCurrent()

    let textArray = MLXArray(textTokens.map { Int32($0) }).reshaped(1, -1)
    let textLen = MLXArray([Int32(textTokens.count)])

    // Tokenize instruct text and manually append <|endofprompt|> token
    // The base tokenizer doesn't have this special token, so we add it manually
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

  /// Stored source audio tokens for voice conversion (stays within actor)
  private var vcSourceTokens: MLXArray?
  private var vcSourceTokenLen: MLXArray?

  /// Prepare source audio for voice conversion
  ///
  /// Stores source audio tokens internally for use with voice conversion generation.
  ///
  /// - Parameters:
  ///   - audio: Source audio waveform at 24kHz
  ///   - s3Tokenizer: S3 speech tokenizer function
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
    conditionals: CosyVoice2Conditionals,
    nTimesteps: Int = 10
  ) throws -> TTSGenerationResult {
    guard let sourceTokens = vcSourceTokens, let sourceTokenLen = vcSourceTokenLen else {
      throw CosyVoice2Error.invalidInput("No source audio prepared for voice conversion")
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
  /// - Parameters:
  ///   - text: Text to encode
  ///   - addSpecialTokens: Whether to add special tokens (default: false for TTS)
  /// - Returns: Array of token IDs
  func encode(text: String, addSpecialTokens: Bool = false) -> [Int] {
    textTokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
  }

  /// Decode token IDs back to text
  /// - Parameters:
  ///   - tokens: Token IDs to decode
  ///   - skipSpecialTokens: Whether to skip special tokens (default: true)
  /// - Returns: Decoded text
  func decode(tokens: [Int], skipSpecialTokens: Bool = true) -> String {
    textTokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
  }

  /// Get the token ID for a specific token string
  /// - Parameter token: Token string to look up
  /// - Returns: Token ID or nil if not found
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
    // linearInterpolate1d expects (B, T, C) format
    // audio is (T,), so we reshape to (1, T, 1), interpolate, then squeeze back to (T,)
    let audioBTC = audio.reshaped(1, -1, 1) // (1, T, 1)
    let interpolated = linearInterpolate1d(audioBTC, scaleFactor: ratio)
    return interpolated.squeezed() // Back to (T',)
  }

  /// Compute 80-mel spectrogram for flow model (at 24kHz)
  ///
  /// Uses the same mel spectrogram computation as S3Gen with CosyVoice2 parameters:
  /// - n_fft=1920, hop_size=480, win_size=1920, fmin=0, fmax=8000
  /// - 80 mel bins at 24kHz sample rate
  ///
  /// - Parameter audio: Audio waveform (T,) at 24kHz
  /// - Returns: Mel spectrogram (T', 80) where T' = num_frames
  private func computeMelSpectrogram80(audio: MLXArray) -> MLXArray {
    // CosyVoice2 flow model mel spectrogram parameters
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

// MARK: - Log Mel Spectrogram for CAMPlus

/// Compute log mel spectrogram for CAMPlus/S3 tokenizer (128 mels at 16kHz)
///
/// This uses the same mel spectrogram computation as the S3 tokenizer:
/// - n_fft=400, hop_length=160, win_length=400
/// - 128 mel bins at 16kHz sample rate
/// - S3Tokenizer-style log normalization
///
/// - Parameters:
///   - audio: Audio waveform (T,) at 16kHz
///   - sampleRate: Sample rate (should be 16000)
///   - numMelBins: Number of mel bins (default 128)
/// - Returns: Log-Mel spectrogram (num_mels, T')
func logMelSpectrogramCAMPPlus(
  audio: MLXArray,
  sampleRate _: Int = 16000,
  numMelBins: Int = 128
) -> MLXArray {
  // Use the Chatterbox log mel spectrogram which matches S3 tokenizer behavior
  // It uses: n_fft=400, hop_length=160, slaney mel scale, and S3-style normalization
  logMelSpectrogramChatterbox(audio: audio, nMels: numMelBins, padding: 0)
}
