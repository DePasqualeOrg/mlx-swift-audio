// CosyVoice2 TTS model for Swift MLX
// Ported from mlx-audio-plus cosyvoice2/cosyvoice2.py

import Foundation
import MLX
import MLXNN

// MARK: - CosyVoice2 Main Model

/// CosyVoice2 Text-to-Speech model
///
/// This model generates high-quality speech from text using:
/// 1. Qwen2-based LLM for speech token generation
/// 2. Flow Matching for mel spectrogram synthesis
/// 3. HiFi-GAN for waveform generation
///
/// Supports:
/// - Zero-shot voice cloning with prompt audio
/// - Streaming generation
/// - Multiple languages (Chinese, English, Japanese, Korean)
class CosyVoice2Model: Module {
  let config: CosyVoice2Config

  @ModuleInfo(key: "llm") var llm: Qwen2LM?
  @ModuleInfo(key: "flow") var flow: CosyVoice2FlowModule?
  @ModuleInfo(key: "hifigan") var hifigan: CosyHiFTGenerator?

  init(
    config: CosyVoice2Config = CosyVoice2Config(),
    llm: Qwen2LM? = nil,
    flow: CosyVoice2FlowModule? = nil,
    hifigan: CosyHiFTGenerator? = nil
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
  ///   - promptText: Prompt text token IDs for voice cloning (1, T_p)
  ///   - promptTextLen: Prompt text length (1,)
  ///   - promptSpeechToken: Prompt speech tokens for voice cloning (1, T_s)
  ///   - promptSpeechTokenLen: Prompt speech token length (1,)
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
      throw CosyVoice2Error.modelNotLoaded
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
    nTimesteps: Int? = nil
  ) throws -> (MLXArray, MLXArray?) {
    guard let flow else {
      throw CosyVoice2Error.modelNotLoaded
    }

    return flow.inference(
      token: tokens,
      tokenLen: tokenLen,
      promptToken: promptToken,
      promptTokenLen: promptTokenLen,
      promptFeat: promptFeat,
      promptFeatLen: promptFeatLen,
      embedding: embedding,
      finalize: finalize,
      nTimesteps: nTimesteps ?? config.flow.nTimesteps
    )
  }

  /// Convert mel spectrogram to audio waveform
  /// - Parameters:
  ///   - mel: Mel spectrogram (1, D, T)
  ///   - f0: Optional F0 contour (not used - CosyVoice2 has built-in F0 predictor)
  /// - Returns: Audio waveform (1, T_audio)
  func melToAudio(mel: MLXArray, f0 _: MLXArray? = nil) throws -> MLXArray {
    guard let hifigan else {
      throw CosyVoice2Error.modelNotLoaded
    }

    // CosyHiFTGenerator has built-in F0 predictor
    let (audio, _) = hifigan(mel)
    return audio
  }

  /// Full TTS pipeline: text -> audio
  /// - Parameters:
  ///   - text: Input text token IDs (1, T)
  ///   - textLen: Text length (1,)
  ///   - promptText: Prompt text token IDs (1, T_p)
  ///   - promptTextLen: Prompt text length (1,)
  ///   - promptSpeechToken: Prompt speech tokens (1, T_s)
  ///   - promptSpeechTokenLen: Prompt speech token length (1,)
  ///   - promptMel: Prompt mel spectrogram (1, T_mel, D)
  ///   - promptMelLen: Prompt mel length (1,)
  ///   - speakerEmbedding: Speaker embedding (1, D_spk)
  ///   - sampling: Top-k sampling parameter
  ///   - nTimesteps: Number of flow matching steps
  ///   - maxTokenTextRatio: Maximum speech/text ratio
  ///   - minTokenTextRatio: Minimum speech/text ratio
  /// - Returns: Audio waveform (1, T_audio)
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

    print("[Synthesize] Generated \(tokens.count) speech tokens")
    if tokens.count >= 10 {
      print("[Synthesize] First 10 tokens: \(Array(tokens.prefix(10)))")
      print("[Synthesize] Last 10 tokens: \(Array(tokens.suffix(10)))")
    } else {
      print("[Synthesize] Tokens: \(tokens)")
    }

    if tokens.isEmpty {
      throw CosyVoice2Error.invalidInput("No tokens generated")
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

  /// Zero-shot voice cloning TTS pipeline
  ///
  /// This mode requires a transcription of the reference audio (promptText).
  /// The LLM receives both the prompt text and prompt speech tokens, allowing
  /// it to learn the semantic alignment between text and speech.
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
    // Step 1: Generate speech tokens WITHOUT prompt context
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
      throw CosyVoice2Error.invalidInput("No tokens generated")
    }

    print("[DEBUG] Generated \(tokens.count) speech tokens")

    let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
    let tokenLen = MLXArray([Int32(tokens.count)])

    // Step 2: Flow model uses prompt for speaker identity
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

    // Debug: check mel for NaN
    mel.eval()
    let melFlat = mel.reshaped(-1)
    let melMin = MLX.min(melFlat).item(Float.self)
    let melMax = MLX.max(melFlat).item(Float.self)
    print("[DEBUG] Mel shape: \(mel.shape), min: \(melMin), max: \(melMax)")

    // Step 3: Convert mel to audio
    let audio = try melToAudio(mel: mel)

    // Debug: check audio for NaN
    audio.eval()
    let audioFlat = audio.reshaped(-1)
    let audioMin = MLX.min(audioFlat).item(Float.self)
    let audioMax = MLX.max(audioFlat).item(Float.self)
    print("[DEBUG] Audio shape: \(audio.shape), min: \(audioMin), max: \(audioMax)")

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
    // Step 1: Generate with instruct context
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
      throw CosyVoice2Error.invalidInput("No tokens generated")
    }

    let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
    let tokenLen = MLXArray([Int32(tokens.count)])

    // Step 2: Flow model uses prompt for speaker identity
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
}

// MARK: - CosyVoice2FlowModule (Flow Module Wrapper)

/// Flow module that wraps encoder + CFM for CosyVoice2
/// This is a placeholder that should be integrated with the actual implementation
class CosyVoice2FlowModule: Module {
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
  @ModuleInfo(key: "encoder") var encoder: UpsampleConformerEncoder
  @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
  @ModuleInfo(key: "decoder") var decoder: CosyVoice2ConditionalCFM

  init(
    inputSize: Int = 512,
    outputSize: Int = 80,
    spkEmbedDim: Int = 192,
    vocabSize: Int = 6561,
    inputFrameRate: Int = 25,
    tokenMelRatio: Int = 2,
    preLookaheadLen: Int = 3,
    nTimesteps: Int = 10,
    encoder: UpsampleConformerEncoder? = nil,
    decoder: CosyVoice2ConditionalCFM? = nil
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

    if let enc = encoder {
      _encoder.wrappedValue = enc
    } else {
      _encoder.wrappedValue = UpsampleConformerEncoder(
        inputSize: inputSize,
        outputSize: inputSize
      )
    }

    // Encoder proj: project encoder output (512) to outputSize (80)
    let encoderOutputSize = encoder?.outputSize() ?? inputSize
    _encoderProj.wrappedValue = Linear(encoderOutputSize, outputSize)

    if let dec = decoder {
      _decoder.wrappedValue = dec
    } else {
      _decoder.wrappedValue = CosyVoice2ConditionalCFM(
        inChannels: outputSize * 3,
        nSpks: 1,
        spkEmbDim: outputSize
      )
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
    finalize: Bool = true,
    nTimesteps: Int? = nil
  ) -> (MLXArray, MLXArray?) {
    print("[Flow] Input shapes:")
    print("  token: \(token.shape), tokenLen: \(tokenLen)")
    print("  promptToken: \(promptToken.shape), promptTokenLen: \(promptTokenLen)")
    print("  promptFeat: \(promptFeat.shape)")
    print("  embedding: \(embedding.shape)")

    // Normalize speaker embedding first (like Python: embedding / (norm + 1e-8))
    let norm = MLX.sqrt(MLX.sum(embedding * embedding, axis: 1, keepDims: true))
    let embeddingNorm = embedding / (norm + 1e-8)
    let spks = spkEmbedAffineLayer(embeddingNorm)

    spks.eval()
    let spksMin = MLX.min(spks).item(Float.self)
    let spksMax = MLX.max(spks).item(Float.self)
    print("[Flow] spks shape: \(spks.shape), min: \(spksMin), max: \(spksMax)")

    // Concatenate prompt and target tokens
    let fullToken = MLX.concatenated([promptToken, token], axis: 1)
    let fullTokenLen = tokenLen + promptTokenLen

    // Create mask for token embedding (matching Python)
    let maxTokenLen = fullToken.shape[1]
    let seqRange = MLXArray(0 ..< Int32(maxTokenLen))
    let tokenMask = (seqRange .< fullTokenLen.expandedDimensions(axis: 1)).asType(.float32)
    let tokenMaskExpanded = tokenMask.expandedDimensions(axis: -1) // (B, T, 1)

    // Embed tokens and apply mask (matching Python: token = embedding(token) * mask)
    var tokenEmb = inputEmbedding(fullToken)
    tokenEmb = tokenEmb * tokenMaskExpanded

    // Encode - use streaming=false for full context attention (non-streaming mode)
    print("[Flow] Calling encoder with tokenEmb shape: \(tokenEmb.shape), fullTokenLen: \(fullTokenLen)")
    let (encoderOutRaw, _) = encoder(tokenEmb, xsLens: fullTokenLen.asType(.int32), streaming: false)
    print("[Flow] Encoder output shape: \(encoderOutRaw.shape)")

    // Trim lookahead for streaming (unless finalizing) - matching Python
    var encoderOut = encoderOutRaw
    if !finalize {
      let trimLen = preLookaheadLen * tokenMelRatio
      if encoderOut.shape[1] > trimLen {
        encoderOut = encoderOut[0..., 0 ..< (encoderOut.shape[1] - trimLen), 0...]
      }
    }

    // Calculate mel lengths (matching Python: mel_len1 = prompt_feat.shape[1], mel_len2 = h.shape[1] - mel_len1)
    let melLen1 = promptFeat.shape[1]
    let melLen2 = encoderOut.shape[1] - melLen1
    print("[Flow] melLen1 (prompt): \(melLen1), melLen2 (generated): \(melLen2)")

    // Project encoder output (512 -> 80)
    let h = encoderProj(encoderOut) // (B, T, 80)

    h.eval()
    let hMin = MLX.min(h).item(Float.self)
    let hMax = MLX.max(h).item(Float.self)
    print("[Flow] h (encoder_proj output) shape: \(h.shape), min: \(hMin), max: \(hMax)")

    // Prepare conditioning (matching Python exactly)
    // conds = mx.zeros([1, mel_len1 + mel_len2, output_size])
    // conds[:, :mel_len1] = prompt_feat
    // conds = mx.transpose(conds, [0, 2, 1])
    let totalLen = melLen1 + melLen2
    var conds = MLXArray.zeros([1, totalLen, outputSize])
    if melLen1 > 0 {
      // Copy prompt_feat into the first melLen1 positions
      let promptSlice = promptFeat[0..., 0 ..< melLen1, 0...]
      // Use concatenation to build conds with prompt at the start
      let zerosPad = MLXArray.zeros([1, melLen2, outputSize])
      conds = MLX.concatenated([promptSlice, zerosPad], axis: 1)
    }

    conds.eval()
    let condsMin = MLX.min(conds).item(Float.self)
    let condsMax = MLX.max(conds).item(Float.self)
    print("[Flow] conds (before transpose) shape: \(conds.shape), min: \(condsMin), max: \(condsMax)")

    conds = conds.swappedAxes(1, 2) // (1, D, T)

    // Create mask for decoder (all ones, float, matching Python)
    let mask = MLXArray.ones([1, 1, totalLen])

    // Transpose h to (B, C, T) for decoder
    let mu = h.swappedAxes(1, 2) // (B, 80, T)

    mu.eval()
    let muMin = MLX.min(mu).item(Float.self)
    let muMax = MLX.max(mu).item(Float.self)
    print("[Flow] mu shape: \(mu.shape), min: \(muMin), max: \(muMax)")
    print("[Flow] conds shape: \(conds.shape), mask shape: \(mask.shape)")
    print("[Flow] Running CFM decoder with nTimesteps: \(nTimesteps ?? self.nTimesteps)")

    // Run CFM decoder
    let (mel, _) = decoder(
      mu: mu,
      mask: mask,
      nTimesteps: nTimesteps ?? self.nTimesteps,
      temperature: 1.0,
      spks: spks,
      cond: conds,
      streaming: false
    )

    mel.eval()
    let melMin = MLX.min(mel).item(Float.self)
    let melMax = MLX.max(mel).item(Float.self)
    print("[Flow] CFM output mel shape: \(mel.shape), min: \(melMin), max: \(melMax)")

    // Extract only the new portion (after prompt) - matching Python: feat = feat[:, :, mel_len1:]
    let outputMel = mel[0..., 0..., melLen1...]
    print("[Flow] Output mel (after prompt trim) shape: \(outputMel.shape)")

    return (outputMel, nil)
  }
}

// MARK: - Conditionals Container

/// Container for precomputed reference audio conditionals
struct CosyVoice2Conditionals: @unchecked Sendable {
  let promptSpeechToken: MLXArray
  let promptSpeechTokenLen: MLXArray
  let promptMel: MLXArray
  let promptMelLen: MLXArray
  let speakerEmbedding: MLXArray
  let promptText: MLXArray?
  let promptTextLen: MLXArray?
}
