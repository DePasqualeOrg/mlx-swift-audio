//
//  T3.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/t3/t3.py
//  Token-To-Token (T3) TTS model using LLaMA as backbone.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXRandom

/// Token-To-Token (T3) TTS model using LLaMA as backbone.
///
/// Generates speech tokens from text tokens, conditioned on speaker embeddings
/// and optional emotion/prompt controls.
class T3: Module {
  let config: T3Config
  let llamaConfig: T3LlamaConfig
  let dim: Int

  @ModuleInfo(key: "tfmr") var tfmr: T3LlamaBackbone
  @ModuleInfo(key: "cond_enc") var condEnc: T3CondEnc
  @ModuleInfo(key: "text_emb") var textEmb: Embedding
  @ModuleInfo(key: "speech_emb") var speechEmb: Embedding
  @ModuleInfo(key: "text_head") var textHead: Linear
  @ModuleInfo(key: "speech_head") var speechHead: Linear
  @ModuleInfo(key: "text_pos_emb") var textPosEmb: LearnedPositionEmbeddings
  @ModuleInfo(key: "speech_pos_emb") var speechPosEmb: LearnedPositionEmbeddings

  init(config: T3Config? = nil) {
    let hp = config ?? T3Config.englishOnly()
    self.config = hp

    // Create LLaMA config from T3 config
    llamaConfig = T3LlamaConfig.llama520M
    dim = llamaConfig.hiddenSize

    // LLaMA transformer backbone
    _tfmr.wrappedValue = T3LlamaBackbone(llamaConfig)

    // Conditioning encoder
    _condEnc.wrappedValue = T3CondEnc(config: hp)

    // Text and speech token embeddings
    _textEmb.wrappedValue = Embedding(embeddingCount: hp.textTokensDictSize, dimensions: dim)
    _speechEmb.wrappedValue = Embedding(embeddingCount: hp.speechTokensDictSize, dimensions: dim)

    // Output projection heads
    _textHead.wrappedValue = Linear(llamaConfig.hiddenSize, hp.textTokensDictSize, bias: false)
    _speechHead.wrappedValue = Linear(llamaConfig.hiddenSize, hp.speechTokensDictSize, bias: false)

    // Learned position embeddings - always create since model weights include them
    let maxTextSeqLen = hp.maxTextTokens + 2
    _textPosEmb.wrappedValue = LearnedPositionEmbeddings(seqLen: maxTextSeqLen, modelDim: dim)

    let maxMelSeqLen = hp.maxSpeechTokens + 2 + 2
    _speechPosEmb.wrappedValue = LearnedPositionEmbeddings(seqLen: maxMelSeqLen, modelDim: dim)

    super.init()
  }

  /// Prepare conditioning embeddings from T3Cond
  func prepareConditioning(_ t3Cond: inout T3Cond) -> MLXArray {
    // Embed speech prompt tokens if provided
    if t3Cond.condPromptSpeechTokens != nil, t3Cond.condPromptSpeechEmb == nil {
      t3Cond.condPromptSpeechEmb =
        speechEmb(t3Cond.condPromptSpeechTokens!) +
        speechPosEmb(t3Cond.condPromptSpeechTokens!)
    }

    return condEnc(t3Cond)
  }

  /// Prepare input embeddings for the transformer
  func prepareInputEmbeds(
    t3Cond: inout T3Cond,
    textTokens: MLXArray,
    speechTokens: MLXArray,
    cfgWeight: Float = 0.0,
  ) -> (MLXArray, Int) {
    // Prepare conditioning embeddings
    var condEmb = prepareConditioning(&t3Cond)

    // Text embeddings
    var textEmbeddings = textEmb(textTokens)

    // CFG: zero out second batch item for unconditional
    if cfgWeight > 0.0, textEmbeddings.shape[0] > 1 {
      textEmbeddings = MLX.concatenated([
        textEmbeddings[0 ..< 1],
        MLXArray.zeros(like: textEmbeddings[1 ..< 2]),
      ], axis: 0)
    }

    // Speech embeddings
    var speechEmbeddings = speechEmb(speechTokens)

    // Add position embeddings if using learned positions
    if config.inputPosEmb == "learned" {
      textEmbeddings = textEmbeddings + textPosEmb(textTokens)
      speechEmbeddings = speechEmbeddings + speechPosEmb(speechTokens)
    }

    let lenCond = condEmb.shape[1]

    // Broadcast conditioning if batch sizes don't match
    if condEmb.shape[0] != textEmbeddings.shape[0] {
      condEmb = MLX.broadcast(
        condEmb,
        to: [textEmbeddings.shape[0]] + Array(condEmb.shape.dropFirst()),
      )
    }

    // Broadcast speech embeddings if batch sizes don't match (e.g., CFG)
    if speechEmbeddings.shape[0] != textEmbeddings.shape[0] {
      speechEmbeddings = MLX.broadcast(
        speechEmbeddings,
        to: [textEmbeddings.shape[0]] + Array(speechEmbeddings.shape.dropFirst()),
      )
    }

    // Concatenate: [conditioning | text | speech]
    let embeds = MLX.concatenated([condEmb, textEmbeddings, speechEmbeddings], axis: 1)

    return (embeds, lenCond)
  }

  /// Forward pass through T3 model
  func callAsFunction(
    t3Cond: inout T3Cond,
    textTokens: MLXArray,
    textTokenLens _: MLXArray,
    speechTokens: MLXArray,
    speechTokenLens _: MLXArray,
    cache: [KVCache]? = nil,
  ) -> [String: MLXArray] {
    // Prepare input embeddings
    let (embeds, lenCond) = prepareInputEmbeds(
      t3Cond: &t3Cond,
      textTokens: textTokens,
      speechTokens: speechTokens,
    )

    // Forward through LLaMA backbone
    let hiddenStates = tfmr(embeds, cache: cache)

    // Extract text and speech portions of hidden states
    let B = textTokens.shape[0]
    let lenText = textTokens.shape[1]
    let lenSpeech = speechTokens.shape[1]
    let hiddenDim = hiddenStates.shape[hiddenStates.ndim - 1]

    // Split hidden states by sequence position
    let textStart = lenCond
    let textEnd = lenCond + lenText
    let speechStart = lenCond + lenText
    let speechEnd = speechStart + lenSpeech

    let textLatents = hiddenStates[0..., textStart ..< textEnd, 0...]
    let speechLatents = hiddenStates[0..., speechStart ..< speechEnd, 0...]

    // Project to vocabulary
    let textLogits = textHead(textLatents)
    let speechLogits = speechHead(speechLatents)

    return [
      "text_logits": textLogits,
      "text_latents": textLatents,
      "speech_logits": speechLogits,
      "speech_latents": speechLatents,
      "hidden_states": hiddenStates,
    ]
  }

  /// Generate speech tokens from text tokens
  func inference(
    t3Cond: inout T3Cond,
    textTokens: MLXArray,
    initialSpeechTokens _: MLXArray? = nil,
    maxNewTokens: Int = 1024,
    temperature: Float = 0.8,
    topP: Float = 0.95,
    minP: Float = 0.05,
    repetitionPenalty: Float = 1.2,
    cfgWeight: Float = 0.5,
  ) -> MLXArray {
    // Ensure text_tokens is 2D
    var tokens = textTokens
    if tokens.ndim == 1 {
      tokens = tokens.expandedDimensions(axis: 0)
    }

    // Default initial speech token (BOS)
    let bosToken = MLXArray([Int32(config.startSpeechToken)]).reshaped([1, 1])

    // Prepare conditioning and text embeddings
    var condEmb = prepareConditioning(&t3Cond)

    var textEmbeddings = textEmb(tokens)

    if cfgWeight > 0.0 {
      // Zero out second batch for unconditional
      textEmbeddings = MLX.concatenated([
        textEmbeddings[0 ..< 1],
        MLXArray.zeros(like: textEmbeddings[0 ..< 1]),
      ], axis: 0)
    }

    if config.inputPosEmb == "learned" {
      textEmbeddings = textEmbeddings + textPosEmb(tokens)
    }

    // Broadcast conditioning if needed
    if condEmb.shape[0] != textEmbeddings.shape[0] {
      condEmb = MLX.broadcast(
        condEmb,
        to: [textEmbeddings.shape[0]] + Array(condEmb.shape.dropFirst()),
      )
    }

    // Create BOS embedding with position embedding at position 0
    var bosEmbed = speechEmb(bosToken)
    bosEmbed = bosEmbed + speechPosEmb.getFixedEmbedding(0)

    // For CFG, duplicate BOS embed
    if cfgWeight > 0.0 {
      bosEmbed = MLX.concatenated([bosEmbed, bosEmbed], axis: 0)
    }

    // Build initial input: [cond | text | bos]
    let inputEmbeddings = MLX.concatenated([condEmb, textEmbeddings, bosEmbed], axis: 1)

    // Create KV cache
    let cache = tfmr.createCache(batchSize: cfgWeight > 0.0 ? 2 : 1)

    // Track generated tokens - use Int32 directly to avoid conversion overhead
    var generatedIds: [Int32] = [Int32(config.startSpeechToken)]

    // Pre-allocate penalty scalar outside loop
    let penaltyScalar = MLXArray(repetitionPenalty)
    let applyRepetitionPenalty = repetitionPenalty != 1.0

    // Initial forward pass to fill cache
    var hidden = tfmr(inputEmbeddings, cache: cache)

    // Generation loop - restructured for better pipelining
    // Key insight: start next forward pass BEFORE extracting token ID
    // This allows GPU to compute while we wait for .item()
    for step in 0 ..< maxNewTokens {
      // Get logits for last position
      var logits = speechHead(hidden[0..., -1 ..< hidden.shape[1], 0...])
      logits = logits.squeezed(axis: 1)

      // Apply CFG
      if cfgWeight > 0.0, logits.shape[0] > 1 {
        let condLogits = logits[0 ..< 1, 0...]
        let uncondLogits = logits[1 ..< 2, 0...]
        logits = condLogits + cfgWeight * (condLogits - uncondLogits)
      } else {
        logits = logits[0 ..< 1, 0...]
      }

      // Apply repetition penalty (vectorized)
      if applyRepetitionPenalty, !generatedIds.isEmpty {
        let tokenIndices = MLXArray(generatedIds)
        let gatheredScores = logits[0, tokenIndices]
        let penalizedScores = MLX.where(
          gatheredScores .> 0,
          gatheredScores / penaltyScalar,
          gatheredScores * penaltyScalar,
        )
        logits[0, tokenIndices] = penalizedScores
      }

      // Sample next token (returns MLXArray - computation graph, not yet evaluated)
      let nextToken = sampleToken(
        logits: logits,
        temperature: temperature,
        topP: topP,
        minP: minP,
      )

      // Create embedding for next token BEFORE extracting ID
      // This uses the MLXArray directly, keeping computation on GPU
      var nextTokenEmbed = speechEmb(nextToken.reshaped([1, 1]))
      nextTokenEmbed = nextTokenEmbed + speechPosEmb.getFixedEmbedding(step + 1)

      // For CFG, duplicate
      if cfgWeight > 0.0 {
        nextTokenEmbed = MLX.concatenated([nextTokenEmbed, nextTokenEmbed], axis: 0)
      }

      // Start next forward pass (async) - GPU computes while we extract token ID
      hidden = tfmr(nextTokenEmbed, cache: cache)
      asyncEval(hidden)

      // NOW extract token ID - GPU is already working on next step
      let nextTokenId = nextToken.item(Int32.self)

      // Check for EOS
      if nextTokenId == Int32(config.stopSpeechToken) {
        generatedIds.append(nextTokenId)
        break
      }

      generatedIds.append(nextTokenId)
    }

    return MLXArray(generatedIds).reshaped([1, -1])
  }
}

// MARK: - Sampling Helpers

/// Sample next token with temperature, top-p, and min-p filtering
/// Matches Python mlx_lm implementation exactly for correctness
private func sampleToken(
  logits: MLXArray,
  temperature: Float,
  topP: Float,
  minP: Float,
) -> MLXArray {
  // Handle greedy decoding
  if temperature == 0 {
    return MLX.argMax(logits, axis: -1)
  }

  // Work with log-probs like Python implementation
  var logprobs = logSoftmax(logits / temperature, axis: -1)

  // Fast path: if both topP and minP are disabled, just sample directly
  if topP >= 1.0, minP <= 0.0 {
    return MLXRandom.categorical(logprobs)
  }

  // Apply top-p (nucleus sampling) - matches Python apply_top_p exactly
  if topP > 0, topP < 1.0 {
    let probs = MLX.exp(logprobs)
    // Sort in ascending order
    let sortedIndices = MLX.argSort(logprobs, axis: -1)
    let sortedProbs = MLX.takeAlong(probs, sortedIndices, axis: -1)
    let cumulativeProbs = MLX.cumsum(sortedProbs, axis: -1)

    // Create inverse indices to map back to original order
    let vocabSize = logprobs.shape[logprobs.ndim - 1]
    let arange = MLXArray(0 ..< vocabSize).asType(.int32)
    let inverseIndices = MLX.putAlong(
      MLXArray.zeros(sortedIndices.shape, dtype: .int32),
      sortedIndices,
      values: arange,
      axis: -1,
    )

    // Rearrange cumulative probs back to original order
    let originalCumulativeProbs = MLX.takeAlong(cumulativeProbs, inverseIndices, axis: -1)

    // Keep tokens where cumulative prob > (1 - topP)
    logprobs = MLX.where(
      originalCumulativeProbs .> (1 - topP),
      logprobs,
      MLXArray(-Float.infinity),
    )
  }

  // Apply min-p filtering - matches Python apply_min_p exactly
  if minP > 0 {
    let topLogprob = logprobs.max(axis: -1, keepDims: true)
    let threshold = topLogprob + log(minP)
    logprobs = MLX.where(
      logprobs .>= threshold,
      logprobs,
      MLXArray(-Float.infinity),
    )
  }

  // Sample from filtered distribution
  return MLXRandom.categorical(logprobs)
}
