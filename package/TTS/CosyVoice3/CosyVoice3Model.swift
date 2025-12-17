// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

import Foundation
import MLX
import MLXNN

// MARK: - CosyVoice3 Main Model

/// CosyVoice3 Text-to-Speech model
///
/// This model generates high-quality speech from text using:
/// 1. Qwen2-based LLM for speech token generation
/// 2. DiT-based Flow Matching for mel spectrogram synthesis
/// 3. Causal HiFi-GAN for waveform generation
///
/// Supports:
/// - Zero-shot voice matching with prompt audio
/// - Cross-lingual mode (no reference transcription needed)
/// - Instruct mode for style control
/// - Voice conversion
/// - Full streaming support
class CosyVoice3Model: Module {
  let config: CosyVoice3Config

  @ModuleInfo(key: "llm") var llm: CosyVoice3LM?
  @ModuleInfo(key: "flow") var flow: CosyVoice3FlowModule?
  @ModuleInfo(key: "hifigan") var hifigan: CausalHiFTGenerator?

  var sampleRate: Int {
    config.hifigan.samplingRate
  }

  init(
    config: CosyVoice3Config = CosyVoice3Config(),
    llm: CosyVoice3LM? = nil,
    flow: CosyVoice3FlowModule? = nil,
    hifigan: CausalHiFTGenerator? = nil
  ) {
    self.config = config
    _llm.wrappedValue = llm
    _flow.wrappedValue = flow
    _hifigan.wrappedValue = hifigan
  }

  /// Generate speech tokens from text
  /// - Parameters:
  ///   - text: Input text token IDs (1, T)
  ///   - textLen: Text length (1,)
  ///   - promptText: Prompt text token IDs for voice matching (1, T_p)
  ///   - promptTextLen: Prompt text length (1,)
  ///   - promptSpeechToken: Prompt speech tokens for voice matching (1, T_s)
  ///   - promptSpeechTokenLen: Prompt speech token length (1,)
  ///   - embedding: Speaker embedding (unused, for API compat)
  ///   - sampling: Top-k sampling parameter
  ///   - maxTokenTextRatio: Maximum speech/text token ratio
  ///   - minTokenTextRatio: Minimum speech/text token ratio
  /// - Returns: Array of generated speech token IDs
  func generateTokens(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    embedding: MLXArray? = nil,
    sampling: Int = 25,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0
  ) throws -> [Int] {
    guard let llm else {
      throw CosyVoice3Error.modelNotLoaded
    }

    return try llm.inference(
      text: text,
      textLen: textLen,
      promptText: promptText,
      promptTextLen: promptTextLen,
      promptSpeechToken: promptSpeechToken,
      promptSpeechTokenLen: promptSpeechTokenLen,
      embedding: embedding,
      sampling: sampling,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio
    )
  }

  /// Convert speech tokens to mel spectrogram
  /// - Parameters:
  ///   - tokens: Speech tokens (1, T)
  ///   - tokenLen: Token length (1,)
  ///   - promptToken: Prompt speech tokens (1, T_p)
  ///   - promptTokenLen: Prompt token length (1,)
  ///   - promptFeat: Prompt mel features (1, T_mel, D)
  ///   - promptFeatLen: Prompt feature length (1,)
  ///   - embedding: Speaker embedding (1, D_spk)
  ///   - finalize: Whether this is the final chunk
  ///   - nTimesteps: Number of diffusion steps
  ///   - streaming: Whether in streaming mode
  /// - Returns: Tuple of (mel_spectrogram, flow_cache)
  func tokensToMel(
    tokens: MLXArray,
    tokenLen: MLXArray,
    promptToken: MLXArray,
    promptTokenLen: MLXArray,
    promptFeat: MLXArray,
    promptFeatLen: MLXArray,
    embedding: MLXArray,
    finalize: Bool = true,
    nTimesteps: Int? = nil,
    streaming: Bool = false
  ) throws -> (MLXArray, MLXArray?) {
    guard let flow else {
      throw CosyVoice3Error.modelNotLoaded
    }

    return flow.inference(
      token: tokens,
      tokenLen: tokenLen,
      promptToken: promptToken,
      promptTokenLen: promptTokenLen,
      promptFeat: promptFeat,
      promptFeatLen: promptFeatLen,
      embedding: embedding,
      streaming: streaming,
      finalize: finalize,
      nTimesteps: nTimesteps ?? config.flow.nTimesteps
    )
  }

  /// Convert mel spectrogram to audio waveform
  /// - Parameters:
  ///   - mel: Mel spectrogram (1, D, T)
  ///   - finalize: Whether this is the final chunk
  /// - Returns: Audio waveform (1, T_audio)
  func melToAudio(mel: MLXArray, finalize: Bool = true) throws -> MLXArray {
    guard let hifigan else {
      throw CosyVoice3Error.modelNotLoaded
    }

    let (audio, _) = hifigan(mel, finalize: finalize)
    return audio
  }

  /// Full TTS pipeline: text -> audio
  func synthesize(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    sampling: Int = 25,
    nTimesteps: Int = 10,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0
  ) throws -> MLXArray {
    // Step 1: Generate speech tokens
    let tokens = try generateTokens(
      text: text,
      textLen: textLen,
      promptText: promptText,
      promptTextLen: promptTextLen,
      promptSpeechToken: promptSpeechToken,
      promptSpeechTokenLen: promptSpeechTokenLen,
      sampling: sampling,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio
    )

    if tokens.isEmpty {
      throw CosyVoice3Error.invalidInput("No tokens generated")
    }

    // Convert to array
    let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
    let tokenLen = MLXArray([Int32(tokens.count)])

    // Step 2: Convert tokens to mel spectrogram
    let (mel, _) = try tokensToMel(
      tokens: tokenArray,
      tokenLen: tokenLen,
      promptToken: promptSpeechToken,
      promptTokenLen: promptSpeechTokenLen,
      promptFeat: promptMel,
      promptFeatLen: promptMelLen,
      embedding: speakerEmbedding,
      finalize: true,
      nTimesteps: nTimesteps
    )

    // Step 3: Convert mel to audio
    let audio = try melToAudio(mel: mel)

    return audio
  }

  /// Zero-shot voice matching TTS pipeline
  ///
  /// This mode requires a transcription of the reference audio (promptText).
  func synthesizeZeroShot(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    sampling: Int = 25,
    nTimesteps: Int = 10,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0
  ) throws -> MLXArray {
    try synthesize(
      text: text,
      textLen: textLen,
      promptText: promptText,
      promptTextLen: promptTextLen,
      promptSpeechToken: promptSpeechToken,
      promptSpeechTokenLen: promptSpeechTokenLen,
      promptMel: promptMel,
      promptMelLen: promptMelLen,
      speakerEmbedding: speakerEmbedding,
      sampling: sampling,
      nTimesteps: nTimesteps,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio
    )
  }

  /// Cross-lingual TTS pipeline (no reference transcription needed)
  ///
  /// In this mode:
  /// - LLM receives NO promptText and NO promptSpeechToken
  /// - LLM generates speech tokens based purely on the input text
  /// - Flow model still uses promptSpeechToken and promptMel for speaker identity
  func synthesizeCrossLingual(
    text: MLXArray,
    textLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    sampling: Int = 25,
    nTimesteps: Int = 10,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0
  ) throws -> MLXArray {
    // Generate tokens WITHOUT prompt context
    let emptyPromptText = MLXArray.zeros([1, 0], dtype: .int32)
    let emptyPromptTextLen = MLXArray([Int32(0)])
    let emptySpeechToken = MLXArray.zeros([1, 0], dtype: .int32)
    let emptySpeechTokenLen = MLXArray([Int32(0)])

    let tokens = try generateTokens(
      text: text,
      textLen: textLen,
      promptText: emptyPromptText,
      promptTextLen: emptyPromptTextLen,
      promptSpeechToken: emptySpeechToken,
      promptSpeechTokenLen: emptySpeechTokenLen,
      sampling: sampling,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio
    )

    if tokens.isEmpty {
      throw CosyVoice3Error.invalidInput("No tokens generated")
    }

    print("[Flow] Starting flow model with \(tokens.count) tokens")
    let flowStart = CFAbsoluteTimeGetCurrent()

    let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
    let tokenLen = MLXArray([Int32(tokens.count)])

    // Flow model uses prompt for speaker identity
    print("[Flow] Calling tokensToMel...")
    let (mel, _) = try tokensToMel(
      tokens: tokenArray,
      tokenLen: tokenLen,
      promptToken: promptSpeechToken,
      promptTokenLen: promptSpeechTokenLen,
      promptFeat: promptMel,
      promptFeatLen: promptMelLen,
      embedding: speakerEmbedding,
      finalize: true,
      nTimesteps: nTimesteps
    )

    let flowTime = CFAbsoluteTimeGetCurrent() - flowStart
    print("[Flow] tokensToMel completed in \(String(format: "%.2f", flowTime))s, mel shape: \(mel.shape)")

    print("[HiFi-GAN] Converting mel to audio...")
    let hiStart = CFAbsoluteTimeGetCurrent()
    let audio = try melToAudio(mel: mel)
    let hiTime = CFAbsoluteTimeGetCurrent() - hiStart
    print("[HiFi-GAN] Completed in \(String(format: "%.2f", hiTime))s, audio shape: \(audio.shape)")

    return audio
  }

  /// Instruct-mode TTS pipeline with style control
  ///
  /// This allows controlling the style of speech generation with instructions
  /// like "Speak slowly and calmly" or "Read with excitement".
  func synthesizeInstruct(
    text: MLXArray,
    textLen: MLXArray,
    instructText: MLXArray,
    instructTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    sampling: Int = 25,
    nTimesteps: Int = 10,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0
  ) throws -> MLXArray {
    // Generate with instruct context
    let emptySpeechToken = MLXArray.zeros([1, 0], dtype: .int32)
    let emptySpeechTokenLen = MLXArray([Int32(0)])

    let tokens = try generateTokens(
      text: text,
      textLen: textLen,
      promptText: instructText,
      promptTextLen: instructTextLen,
      promptSpeechToken: emptySpeechToken,
      promptSpeechTokenLen: emptySpeechTokenLen,
      sampling: sampling,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio
    )

    if tokens.isEmpty {
      throw CosyVoice3Error.invalidInput("No tokens generated")
    }

    let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
    let tokenLen = MLXArray([Int32(tokens.count)])

    // Flow model uses prompt for speaker identity
    let (mel, _) = try tokensToMel(
      tokens: tokenArray,
      tokenLen: tokenLen,
      promptToken: promptSpeechToken,
      promptTokenLen: promptSpeechTokenLen,
      promptFeat: promptMel,
      promptFeatLen: promptMelLen,
      embedding: speakerEmbedding,
      finalize: true,
      nTimesteps: nTimesteps
    )

    return try melToAudio(mel: mel)
  }

  /// Voice Conversion (VC) pipeline: convert source speech to target speaker voice
  ///
  /// In this mode:
  /// - No LLM inference - source speech tokens are used directly
  /// - Flow model converts source tokens to target speaker voice
  func synthesizeVC(
    sourceSpeechToken: MLXArray,
    sourceSpeechTokenLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    nTimesteps: Int = 10
  ) throws -> MLXArray {
    // VC mode: use source speech tokens directly
    let (mel, _) = try tokensToMel(
      tokens: sourceSpeechToken,
      tokenLen: sourceSpeechTokenLen,
      promptToken: promptSpeechToken,
      promptTokenLen: promptSpeechTokenLen,
      promptFeat: promptMel,
      promptFeatLen: promptMelLen,
      embedding: speakerEmbedding,
      finalize: true,
      nTimesteps: nTimesteps
    )

    return try melToAudio(mel: mel)
  }

  /// Streaming speech synthesis with chunked output
  ///
  /// Generates audio in chunks as tokens are produced, reducing latency.
  func synthesizeStreaming(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    sampling: Int = 25,
    nTimesteps: Int = 10,
    chunkSize: Int = 50,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0
  ) throws -> [MLXArray] {
    guard let flow else {
      throw CosyVoice3Error.modelNotLoaded
    }

    var speechTokens: [Int] = []
    var audioChunks: [MLXArray] = []
    var chunkIdx = 0
    let preLookaheadLen = flow.preLookaheadLen
    let tokenMelRatio = flow.tokenMelRatio

    // Generate all tokens first (could be made streaming later)
    speechTokens = try generateTokens(
      text: text,
      textLen: textLen,
      promptText: promptText,
      promptTextLen: promptTextLen,
      promptSpeechToken: promptSpeechToken,
      promptSpeechTokenLen: promptSpeechTokenLen,
      sampling: sampling,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio
    )

    // Process in chunks
    var processedTokens = 0
    while processedTokens < speechTokens.count {
      let targetEnd = (chunkIdx + 1) * chunkSize + preLookaheadLen
      let isLastChunk = targetEnd >= speechTokens.count

      let endIdx: Int
      let finalize: Bool
      if isLastChunk {
        endIdx = speechTokens.count
        finalize = true
      } else {
        endIdx = min(targetEnd, speechTokens.count)
        finalize = false
      }

      let chunkTokens = Array(speechTokens[0 ..< endIdx])
      let tokenArray = MLXArray(chunkTokens.map { Int32($0) }).reshaped(1, -1)
      let tokenLen = MLXArray([Int32(chunkTokens.count)])

      let (mel, _) = try tokensToMel(
        tokens: tokenArray,
        tokenLen: tokenLen,
        promptToken: promptSpeechToken,
        promptTokenLen: promptSpeechTokenLen,
        promptFeat: promptMel,
        promptFeatLen: promptMelLen,
        embedding: speakerEmbedding,
        finalize: finalize,
        nTimesteps: nTimesteps,
        streaming: true
      )

      let audio = try melToAudio(mel: mel, finalize: finalize)

      // Extract just the new chunk
      let chunkStart = chunkIdx * chunkSize * tokenMelRatio
      if audio.shape[audio.ndim - 1] > chunkStart {
        let chunkAudio = audio[0..., chunkStart...]
        audioChunks.append(chunkAudio.squeezed(axis: 0))
      }

      processedTokens = endIdx
      chunkIdx += 1

      if isLastChunk {
        break
      }
    }

    return audioChunks
  }
}

// MARK: - Conditionals Container

/// Container for precomputed reference audio conditionals
struct CosyVoice3Conditionals: @unchecked Sendable {
  let promptSpeechToken: MLXArray
  let promptSpeechTokenLen: MLXArray
  let promptMel: MLXArray
  let promptMelLen: MLXArray
  let speakerEmbedding: MLXArray
  let promptText: MLXArray?
  let promptTextLen: MLXArray?
}

// MARK: - CosyVoice3FlowModule

/// Flow module that wraps PreLookaheadLayer + DiT CFM for CosyVoice3
class CosyVoice3FlowModule: Module {
  let inputSize: Int
  let outputSize: Int
  let spkEmbedDim: Int
  let vocabSize: Int
  let inputFrameRate: Int
  let tokenMelRatio: Int
  let preLookaheadLen: Int
  let nTimesteps: Int

  @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
  @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffineLayer: Linear
  @ModuleInfo(key: "pre_lookahead_layer") var preLookaheadLayer: CosyVoice3PreLookaheadLayer
  @ModuleInfo(key: "decoder") var decoder: CosyVoice3ConditionalCFM

  init(
    inputSize: Int = 512,
    outputSize: Int = 80,
    spkEmbedDim: Int = 192,
    vocabSize: Int = 6561,
    inputFrameRate: Int = 25,
    tokenMelRatio: Int = 2,
    preLookaheadLen: Int = 3,
    nTimesteps: Int = 10,
    preLookahead: CosyVoice3PreLookaheadLayer? = nil,
    decoder: CosyVoice3ConditionalCFM? = nil
  ) {
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.spkEmbedDim = spkEmbedDim
    self.vocabSize = vocabSize
    self.inputFrameRate = inputFrameRate
    self.tokenMelRatio = tokenMelRatio
    self.preLookaheadLen = preLookaheadLen
    self.nTimesteps = nTimesteps

    _inputEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: inputSize)
    _spkEmbedAffineLayer.wrappedValue = Linear(spkEmbedDim, outputSize)

    if let layer = preLookahead {
      _preLookaheadLayer.wrappedValue = layer
    } else {
      _preLookaheadLayer.wrappedValue = CosyVoice3PreLookaheadLayer(
        inChannels: inputSize,
        channels: inputSize,
        preLookaheadLen: preLookaheadLen
      )
    }

    if let dec = decoder {
      _decoder.wrappedValue = dec
    } else {
      // Create a default CFM with nil estimator - will need to be set before use
      _decoder.wrappedValue = CosyVoice3ConditionalCFM(estimator: nil)
    }
  }

  /// Inference: convert tokens to mel spectrogram
  func inference(
    token: MLXArray,
    tokenLen: MLXArray,
    promptToken: MLXArray,
    promptTokenLen: MLXArray,
    promptFeat: MLXArray,
    promptFeatLen _: MLXArray,
    embedding: MLXArray,
    streaming: Bool = false,
    finalize: Bool = true,
    nTimesteps: Int? = nil
  ) -> (MLXArray, MLXArray?) {
    // Normalize speaker embedding
    let norm = MLX.sqrt(MLX.sum(embedding * embedding, axis: 1, keepDims: true))
    let embeddingNorm = embedding / (norm + 1e-8)
    let spks = spkEmbedAffineLayer(embeddingNorm)

    // Concatenate prompt and target tokens
    let fullToken = MLX.concatenated([promptToken, token], axis: 1)
    let fullTokenLen = tokenLen + promptTokenLen

    // Create mask for token embedding
    let maxTokenLen = fullToken.shape[1]
    let seqRange = MLXArray(0 ..< Int32(maxTokenLen))
    let tokenMask = (seqRange .< fullTokenLen.expandedDimensions(axis: 1)).asType(.float32)
    let tokenMaskExpanded = tokenMask.expandedDimensions(axis: -1) // (B, T, 1)

    // Clip tokens to valid embedding range
    let numEmbeddings = inputEmbedding.weight.shape[0]
    let clippedToken = MLX.clip(fullToken, min: 0, max: numEmbeddings - 1)

    // Embed tokens and apply mask
    var tokenEmb = inputEmbedding(clippedToken)
    tokenEmb = tokenEmb * tokenMaskExpanded

    // Process through PreLookaheadLayer
    var encoderOut = preLookaheadLayer(tokenEmb, context: nil)

    // Upsample by tokenMelRatio
    encoderOut = MLX.repeated(encoderOut, count: tokenMelRatio, axis: 1)

    // Trim lookahead for streaming (unless finalizing)
    if !finalize {
      let trimLen = preLookaheadLen * tokenMelRatio
      if encoderOut.shape[1] > trimLen {
        encoderOut = encoderOut[0..., 0 ..< (encoderOut.shape[1] - trimLen), 0...]
      }
    }

    // Calculate mel lengths
    let melLen1 = promptFeat.shape[1]
    let melLen2 = encoderOut.shape[1] - melLen1

    // Prepare conditioning
    let totalLen = melLen1 + melLen2
    var conds: MLXArray
    if melLen1 > 0 {
      let promptSlice = promptFeat[0..., 0 ..< melLen1, 0...]
      let zerosPad = MLXArray.zeros([1, melLen2, outputSize])
      conds = MLX.concatenated([promptSlice, zerosPad], axis: 1)
    } else {
      conds = MLXArray.zeros([1, totalLen, outputSize])
    }
    conds = conds.swappedAxes(1, 2) // (1, D, T)

    // Create mask for decoder
    let mask = MLXArray.ones([1, 1, totalLen])

    // Transpose encoder output to (B, C, T) for decoder - mu stays at input_size dims
    let mu = encoderOut.swappedAxes(1, 2)

    // Run CFM decoder
    let (mel, cache) = decoder(
      mu: mu,
      mask: mask,
      spks: spks,
      cond: conds,
      nTimesteps: nTimesteps ?? self.nTimesteps,
      streaming: streaming
    )

    // Extract only the new portion (after prompt)
    let outputMel = mel[0..., 0..., melLen1...]

    return (outputMel, cache)
  }
}
