// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

import Foundation
import MLX
import MLXNN

// MARK: - KV Cache

/// Key-Value cache for autoregressive generation in CosyVoice3
class CosyVoice3KVCache {
  var keys: MLXArray?
  var values: MLXArray?
  var offset: Int = 0

  func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
    if let existingKeys = self.keys, let existingValues = self.values {
      self.keys = MLX.concatenated([existingKeys, keys], axis: 2)
      self.values = MLX.concatenated([existingValues, values], axis: 2)
    } else {
      self.keys = keys
      self.values = values
    }
    offset += keys.shape[2]
    return (self.keys!, self.values!)
  }

  func reset() {
    keys = nil
    values = nil
    offset = 0
  }
}

// MARK: - Qwen2 Configuration

/// Configuration for Qwen2 model used in CosyVoice3
struct CosyVoice3Qwen2Config: Codable, Sendable {
  var hiddenSize: Int = 896
  var numHiddenLayers: Int = 24
  var intermediateSize: Int = 4864
  var numAttentionHeads: Int = 14
  var numKeyValueHeads: Int = 2
  var rmsNormEps: Float = 1e-6
  var vocabSize: Int = 151_936
  var maxPositionEmbeddings: Int = 32768
  var ropeTheta: Float = 1_000_000.0
  var ropeTraditional: Bool = false
  var tieWordEmbeddings: Bool = true

  enum CodingKeys: String, CodingKey {
    case hiddenSize = "hidden_size"
    case numHiddenLayers = "num_hidden_layers"
    case intermediateSize = "intermediate_size"
    case numAttentionHeads = "num_attention_heads"
    case numKeyValueHeads = "num_key_value_heads"
    case rmsNormEps = "rms_norm_eps"
    case vocabSize = "vocab_size"
    case maxPositionEmbeddings = "max_position_embeddings"
    case ropeTheta = "rope_theta"
    case ropeTraditional = "rope_traditional"
    case tieWordEmbeddings = "tie_word_embeddings"
  }

  init() {}
}

// MARK: - Qwen2 Attention

/// Attention module for Qwen2 in CosyVoice3
class CosyVoice3Attention: Module {
  let config: CosyVoice3Qwen2Config
  let scale: Float
  let headDim: Int

  @ModuleInfo(key: "q_proj") var wq: Linear
  @ModuleInfo(key: "k_proj") var wk: Linear
  @ModuleInfo(key: "v_proj") var wv: Linear
  @ModuleInfo(key: "o_proj") var wo: Linear

  let rope: RoPE

  init(_ config: CosyVoice3Qwen2Config) {
    self.config = config
    headDim = config.hiddenSize / config.numAttentionHeads
    scale = pow(Float(headDim), -0.5)

    _wq.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * headDim, bias: true)
    _wk.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * headDim, bias: true)
    _wv.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * headDim, bias: true)
    _wo.wrappedValue = Linear(config.numAttentionHeads * headDim, config.hiddenSize, bias: false)

    rope = RoPE(dimensions: headDim, traditional: config.ropeTraditional, base: config.ropeTheta)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: CosyVoice3KVCache?) -> MLXArray {
    let (B, L, _) = (x.shape[0], x.shape[1], x.shape[2])

    var queries = wq(x)
    var keys = wk(x)
    var values = wv(x)

    // Reshape for attention
    queries = queries.reshaped(B, L, config.numAttentionHeads, -1).transposed(0, 2, 1, 3)
    keys = keys.reshaped(B, L, config.numKeyValueHeads, -1).transposed(0, 2, 1, 3)
    values = values.reshaped(B, L, config.numKeyValueHeads, -1).transposed(0, 2, 1, 3)

    // Apply RoPE
    let offset = cache?.offset ?? 0
    queries = rope(queries, offset: offset)
    keys = rope(keys, offset: offset)

    // Update cache
    if let cache {
      (keys, values) = cache.update(keys: keys, values: values)
    }

    // Expand KV heads if needed (GQA)
    if config.numKeyValueHeads < config.numAttentionHeads {
      let nRep = config.numAttentionHeads / config.numKeyValueHeads
      keys = MLX.repeated(keys, count: nRep, axis: 1)
      values = MLX.repeated(values, count: nRep, axis: 1)
    }

    // Attention
    var scores = MLX.matmul(queries, keys.transposed(0, 1, 3, 2)) * scale

    if let mask {
      scores = scores + mask
    }

    let weights = MLX.softmax(scores, axis: -1)
    var output = MLX.matmul(weights, values)

    output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
    return wo(output)
  }
}

// MARK: - Qwen2 MLP

/// MLP module for Qwen2
class CosyVoice3MLP: Module, UnaryLayer {
  @ModuleInfo(key: "gate_proj") var gate: Linear
  @ModuleInfo(key: "down_proj") var down: Linear
  @ModuleInfo(key: "up_proj") var up: Linear

  init(dimensions: Int, hiddenDimensions: Int) {
    _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    down(silu(gate(x)) * up(x))
  }
}

// MARK: - Qwen2 Transformer Block

/// Transformer block for Qwen2
class CosyVoice3TransformerBlock: Module {
  @ModuleInfo(key: "self_attn") var attention: CosyVoice3Attention
  @ModuleInfo(key: "mlp") var mlp: CosyVoice3MLP
  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  init(_ config: CosyVoice3Qwen2Config) {
    _attention.wrappedValue = CosyVoice3Attention(config)
    _mlp.wrappedValue = CosyVoice3MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
    _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: CosyVoice3KVCache?) -> MLXArray {
    var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
    let h = x + r
    r = mlp(postAttentionLayerNorm(h))
    return h + r
  }
}

// MARK: - Qwen2 Model

/// Inner Qwen2 model (transformer layers + norm)
class CosyVoice3Qwen2ModelInner: Module {
  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

  let layers: [CosyVoice3TransformerBlock]
  @ModuleInfo(key: "norm") var norm: RMSNorm

  init(_ config: CosyVoice3Qwen2Config) {
    _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

    var layersArray: [CosyVoice3TransformerBlock] = []
    for _ in 0 ..< config.numHiddenLayers {
      layersArray.append(CosyVoice3TransformerBlock(config))
    }
    layers = layersArray

    _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }

  /// Forward with token input
  func callAsFunction(_ inputs: MLXArray, cache: [CosyVoice3KVCache]?) -> MLXArray {
    let h = embedTokens(inputs)
    return forward(embeddings: h, cache: cache)
  }

  /// Forward with embedding input (for CosyVoice3)
  func forward(embeddings: MLXArray, cache: [CosyVoice3KVCache]?) -> MLXArray {
    var h = embeddings

    // Create causal mask
    let mask = createCausalMask(h: h, cache: cache?.first)

    for (i, layer) in layers.enumerated() {
      h = layer(h, mask: mask, cache: cache?[i])
    }

    return norm(h)
  }

  /// Create causal attention mask using vectorized MLX operations
  private func createCausalMask(h: MLXArray, cache: CosyVoice3KVCache?) -> MLXArray? {
    let T = h.shape[1]
    if T == 1 {
      return nil // No mask needed for single token
    }

    let offset = cache?.offset ?? 0
    let totalLen = T + offset

    // Vectorized causal mask creation using broadcasting
    let rows = MLXArray(0 ..< Int32(T)).expandedDimensions(axis: 1)
    let cols = MLXArray(0 ..< Int32(totalLen)).expandedDimensions(axis: 0)

    // Causal condition: column index <= row index + offset
    let causalMask = cols .<= (rows + Int32(offset))

    // Convert boolean mask to attention bias: true -> 0, false -> -1e9
    let mask = MLX.where(causalMask, MLXArray(Float(0)), MLXArray(Float(-1e9)))

    return mask.reshaped(1, 1, T, totalLen)
  }
}

// MARK: - Qwen2 Encoder (for CosyVoice3)

/// Wrapper around Qwen2 model for CosyVoice3
class CosyVoice3Qwen2Encoder: Module {
  let model: CosyVoice3Qwen2ModelInner
  let config: CosyVoice3Qwen2Config

  init(config: CosyVoice3Qwen2Config) {
    self.config = config
    model = CosyVoice3Qwen2ModelInner(config)
  }

  /// Access the token embedding layer
  var embedTokens: Embedding {
    model.embedTokens
  }

  /// Full forward pass with attention mask
  func callAsFunction(_ xs: MLXArray, xsLens: MLXArray) -> (MLXArray, MLXArray) {
    let (B, T, _) = (xs.shape[0], xs.shape[1], xs.shape[2])

    // Create attention mask from lengths
    let positions = MLXArray(0 ..< Int32(T))
    let mask = positions .< xsLens.expandedDimensions(axis: 1) // (B, T)

    // Forward through Qwen2 with embeddings
    let hiddenStates = model.forward(embeddings: xs, cache: nil)

    return (hiddenStates, mask.reshaped(B, 1, T))
  }

  /// Single step forward with KV cache
  func forwardOneStep(_ xs: MLXArray, cache: [CosyVoice3KVCache]?) -> (MLXArray, [CosyVoice3KVCache]) {
    let cacheList = cache ?? (0 ..< config.numHiddenLayers).map { _ in CosyVoice3KVCache() }

    let hiddenStates = model.forward(embeddings: xs, cache: cacheList)

    return (hiddenStates, cacheList)
  }
}

// MARK: - CosyVoice3LM (Main LLM)

/// Qwen2-based Language Model for CosyVoice3 speech token generation
///
/// Key differences from Qwen2LM (CosyVoice2):
/// - Uses unified speech_embedding for all tokens including special tokens
/// - Special token indices: sos = speech_token_size + 0, eos = speech_token_size + 1, etc.
/// - llm_decoder has no bias and extended vocabulary (+200)
class CosyVoice3LM: Module {
  let llmInputSize: Int
  let llmOutputSize: Int
  let speechTokenSize: Int
  let extendedVocabSize: Int

  // CosyVoice3 special token indices (unified in speech_embedding)
  var sosToken: Int { speechTokenSize + 0 } // 6561
  var eosToken: Int { speechTokenSize + 1 } // 6562
  var taskIdToken: Int { speechTokenSize + 2 } // 6563
  var fillToken: Int { speechTokenSize + 3 } // 6564

  // Mix ratio for bidirectional streaming [text_tokens, speech_tokens]
  let mixRatio: [Int]

  // LLM backbone
  @ModuleInfo(key: "llm") var llm: CosyVoice3Qwen2Encoder

  // Output projection: no bias, extended vocabulary
  @ModuleInfo(key: "llm_decoder") var llmDecoder: Linear

  // Unified speech token embedding (includes special tokens)
  @ModuleInfo(key: "speech_embedding") var speechEmbedding: Embedding

  // Sampling function
  var sampling: ((MLXArray, [Int], Int) -> Int)?

  // Stop token IDs for generation (all extended vocab tokens >= speechTokenSize)
  var stopTokenIds: [Int] {
    (0 ..< extendedVocabSize).map { speechTokenSize + $0 }
  }

  init(
    llmInputSize: Int = 896,
    llmOutputSize: Int = 896,
    speechTokenSize: Int = 6561,
    extendedVocabSize: Int = 200,
    qwen2Config: CosyVoice3Qwen2Config = CosyVoice3Qwen2Config(),
    mixRatio: [Int] = [5, 15]
  ) {
    self.llmInputSize = llmInputSize
    self.llmOutputSize = llmOutputSize
    self.speechTokenSize = speechTokenSize
    self.extendedVocabSize = extendedVocabSize
    self.mixRatio = mixRatio

    // Unified speech embedding (no separate llm_embedding)
    _speechEmbedding.wrappedValue = Embedding(
      embeddingCount: speechTokenSize + extendedVocabSize,
      dimensions: llmInputSize
    )

    _llm.wrappedValue = CosyVoice3Qwen2Encoder(config: qwen2Config)

    // Output projection: no bias, extended vocabulary
    _llmDecoder.wrappedValue = Linear(llmOutputSize, speechTokenSize + extendedVocabSize, bias: false)
  }

  /// Sample token IDs with optional EOS rejection
  private func samplingIds(
    weightedScores: MLXArray,
    decodedTokens: [Int],
    sampling: Int,
    ignoreEos: Bool = true
  ) throws -> Int {
    guard let samplingFn = self.sampling else {
      fatalError("Sampling function not set")
    }

    var numTrials = 0
    let maxTrials = 100

    while true {
      let topIds = samplingFn(weightedScores, decodedTokens, sampling)

      // If not ignoring EOS, or sampled token is valid speech token, accept it
      if !ignoreEos || topIds < speechTokenSize {
        return topIds
      }

      numTrials += 1
      if numTrials > maxTrials {
        throw CosyVoice3Error.maxSamplingTrialsExceeded
      }
    }
  }

  /// Generate speech tokens autoregressively
  /// - Parameters:
  ///   - text: Text token IDs (1, T_text)
  ///   - textLen: Text length (1,)
  ///   - promptText: Prompt text token IDs (1, T_prompt_text)
  ///   - promptTextLen: Prompt text length (1,)
  ///   - promptSpeechToken: Prompt speech tokens (1, T_prompt_speech)
  ///   - promptSpeechTokenLen: Prompt speech token length (1,)
  ///   - sampling: Top-k sampling parameter
  ///   - maxTokenTextRatio: Maximum speech/text token ratio
  ///   - minTokenTextRatio: Minimum speech/text token ratio
  /// - Returns: Array of generated speech token IDs
  func inference(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    embedding _: MLXArray? = nil,
    sampling: Int = 25,
    maxTokenTextRatio: Float = 20,
    minTokenTextRatio: Float = 2
  ) throws -> [Int] {
    // Concatenate prompt and input text
    let fullText = MLX.concatenated([promptText, text], axis: 1)
    let fullTextLen = textLen + promptTextLen

    // Embed text tokens using Qwen2's embedding
    let textEmb = llm.embedTokens(fullText)

    // Get special token embeddings from unified speech_embedding
    let sosEmb = speechEmbedding.weight[sosToken].reshaped(1, 1, -1)
    let taskIdEmb = speechEmbedding.weight[taskIdToken].reshaped(1, 1, -1)

    // Embed prompt speech tokens if provided
    let promptSpeechTokenEmb: MLXArray = if promptSpeechTokenLen[0].item(Int.self) != 0 {
      speechEmbedding(promptSpeechToken)
    } else {
      MLXArray.zeros([1, 0, llmInputSize])
    }

    // Construct initial LM input: [sos, text, task_id, prompt_speech]
    let lmInput = MLX.concatenated([sosEmb, textEmb, taskIdEmb, promptSpeechTokenEmb], axis: 1)

    // Calculate min/max generation length
    let textLenInt = Int(fullTextLen[0].item(Int32.self))
    let promptTextLenInt = Int(promptTextLen[0].item(Int32.self))
    let minLen = Int(Float(textLenInt - promptTextLenInt) * minTokenTextRatio)
    let maxLen = Int(Float(textLenInt - promptTextLenInt) * maxTokenTextRatio)

    // Generate tokens
    return try inferenceLoop(lmInput: lmInput, sampling: sampling, minLen: minLen, maxLen: maxLen)
  }

  /// Core inference loop with KV caching
  private func inferenceLoop(
    lmInput: MLXArray,
    sampling: Int,
    minLen: Int,
    maxLen: Int
  ) throws -> [Int] {
    var outTokens: [Int] = []
    var cache: [CosyVoice3KVCache]? = nil
    var currentInput = lmInput

    print("[LLM] Starting inference: minLen=\(minLen), maxLen=\(maxLen), inputShape=\(lmInput.shape)")
    let startTime = CFAbsoluteTimeGetCurrent()

    for i in 0 ..< maxLen {
      // Forward one step
      let (yPred, newCache) = llm.forwardOneStep(currentInput, cache: cache)
      cache = newCache

      // Debug: print progress every 10 tokens or at the start
      if i == 0 || (i + 1) % 10 == 0 {
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("[LLM] Token \(i + 1)/\(maxLen), elapsed: \(String(format: "%.2f", elapsed))s")
      }

      // Get logits for last position
      let lastIdx = yPred.shape[1] - 1
      let logits = llmDecoder(yPred[0..., lastIdx, 0...])
      let logp = MLX.log(MLX.softmax(logits, axis: -1))

      // Sample next token
      let topIds = try samplingIds(
        weightedScores: logp.reshaped(-1),
        decodedTokens: outTokens,
        sampling: sampling,
        ignoreEos: i < minLen
      )

      // Check for any stop token (EOS or any extended vocab token)
      if topIds >= speechTokenSize {
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("[LLM] EOS detected at token \(i + 1), id=\(topIds), speechTokenSize=\(speechTokenSize)")
        print("[LLM] Generated \(outTokens.count) tokens in \(String(format: "%.2f", elapsed))s")
        break
      }

      // Add the token
      outTokens.append(topIds)

      // Prepare input for next step
      currentInput = speechEmbedding.weight[topIds].reshaped(1, 1, -1)
    }

    let totalElapsed = CFAbsoluteTimeGetCurrent() - startTime
    print("[LLM] Inference complete: \(outTokens.count) tokens in \(String(format: "%.2f", totalElapsed))s")
    return outTokens
  }

  /// Streaming inference - yields tokens one by one
  func inferenceStream(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    embedding _: MLXArray? = nil,
    sampling: Int = 25,
    maxTokenTextRatio: Float = 20,
    minTokenTextRatio: Float = 2
  ) throws -> AnySequence<Int> {
    // Concatenate prompt and input text
    let fullText = MLX.concatenated([promptText, text], axis: 1)
    let fullTextLen = textLen + promptTextLen

    // Embed text tokens using Qwen2's embedding
    let textEmb = llm.embedTokens(fullText)

    // Get special token embeddings from unified speech_embedding
    let sosEmb = speechEmbedding.weight[sosToken].reshaped(1, 1, -1)
    let taskIdEmb = speechEmbedding.weight[taskIdToken].reshaped(1, 1, -1)

    // Embed prompt speech tokens if provided
    let promptSpeechTokenEmb: MLXArray = if promptSpeechTokenLen[0].item(Int.self) != 0 {
      speechEmbedding(promptSpeechToken)
    } else {
      MLXArray.zeros([1, 0, llmInputSize])
    }

    // Construct initial LM input: [sos, text, task_id, prompt_speech]
    let lmInput = MLX.concatenated([sosEmb, textEmb, taskIdEmb, promptSpeechTokenEmb], axis: 1)

    // Calculate min/max generation length
    let textLenInt = Int(fullTextLen[0].item(Int32.self))
    let promptTextLenInt = Int(promptTextLen[0].item(Int32.self))
    let minLen = Int(Float(textLenInt - promptTextLenInt) * minTokenTextRatio)
    let maxLen = Int(Float(textLenInt - promptTextLenInt) * maxTokenTextRatio)

    // Return streaming sequence
    return try AnySequence(inferenceLoopStream(lmInput: lmInput, sampling: sampling, minLen: minLen, maxLen: maxLen))
  }

  /// Core streaming inference loop with KV caching
  private func inferenceLoopStream(
    lmInput: MLXArray,
    sampling: Int,
    minLen: Int,
    maxLen: Int
  ) throws -> [Int] {
    var outTokens: [Int] = []
    var cache: [CosyVoice3KVCache]? = nil
    var currentInput = lmInput

    for i in 0 ..< maxLen {
      // Forward one step
      let (yPred, newCache) = llm.forwardOneStep(currentInput, cache: cache)
      cache = newCache

      // Get logits for last position
      let lastIdx = yPred.shape[1] - 1
      let logits = llmDecoder(yPred[0..., lastIdx, 0...])
      let logp = MLX.log(MLX.softmax(logits, axis: -1))

      // Sample next token
      let topIds = try samplingIds(
        weightedScores: logp.reshaped(-1),
        decodedTokens: outTokens,
        sampling: sampling,
        ignoreEos: i < minLen
      )

      // Check for any stop token (EOS or any extended vocab token)
      if topIds >= speechTokenSize {
        break
      }

      // Add the token
      outTokens.append(topIds)

      // Prepare input for next step
      currentInput = speechEmbedding.weight[topIds].reshaped(1, 1, -1)
    }

    return outTokens
  }
}

// MARK: - Sampling Functions for CosyVoice3

/// Nucleus (top-p) sampling with top-k cutoff
func cosyVoice3NucleusSampling(logits: MLXArray, topP: Float = 0.8, topK: Int = 25) -> Int {
  // Convert logits to probabilities
  let probs = MLX.softmax(logits)

  // Sort by probability (descending)
  let sortedIndices = MLX.argSort(-probs)
  let sortedProbs = probs[sortedIndices]

  // Compute cumulative probabilities
  let cumsumProbs = MLX.cumsum(sortedProbs)

  // Find cutoff: where cumsum first exceeds top_p, limited by top_k
  let belowThreshold = cumsumProbs .< topP

  // Count how many tokens are below threshold, but cap at topK
  let nTokens = min(Int(MLX.sum(belowThreshold).item(Int32.self)) + 1, topK)

  // Get top-n token indices and their probabilities
  let topIndices = sortedIndices[0 ..< nTokens]
  var topProbs = sortedProbs[0 ..< nTokens]

  // Renormalize and sample
  topProbs = topProbs / MLX.sum(topProbs)
  let idx = MLXRandom.categorical(MLX.log(topProbs))

  return Int(topIndices[idx].item(Int32.self))
}

/// Repetition-Aware Sampling (RAS) for CosyVoice3
func cosyVoice3RasSampling(
  logits: MLXArray,
  decodedTokens: [Int],
  sampling _: Int,
  topP: Float = 0.8,
  topK: Int = 25,
  winSize: Int = 10,
  tauR: Float = 0.1
) -> Int {
  // First, try nucleus sampling
  var topIds = cosyVoice3NucleusSampling(logits: logits, topP: topP, topK: topK)

  // Check for repetition in recent window
  if !decodedTokens.isEmpty {
    let recentTokens = Array(decodedTokens.suffix(winSize))
    let repNum = recentTokens.filter { $0 == topIds }.count

    // If repetition exceeds threshold, fall back to random sampling
    if Float(repNum) >= Float(winSize) * tauR {
      let probs = MLX.softmax(logits)
      topIds = Int(MLXRandom.categorical(MLX.log(probs)).item(Int32.self))
    }
  }

  return topIds
}

/// Simple top-k sampling from logits for CosyVoice3
func cosyVoice3TopKSampling(logits: MLXArray, decodedTokens _: [Int], topK: Int = 25) -> Int {
  // Get top-k indices using argpartition
  let topKIndices = MLX.argPartition(-logits, kth: topK - 1)[0 ..< topK]

  // Get the values at those indices
  let topKValues = logits[topKIndices]

  // Sample from top-k using softmax probabilities
  let probs = MLX.softmax(topKValues)
  let idx = MLXRandom.categorical(MLX.log(probs))

  return Int(topKIndices[idx].item(Int32.self))
}

// MARK: - Error Types

enum CosyVoice3Error: LocalizedError {
  case maxSamplingTrialsExceeded
  case modelNotLoaded
  case invalidInput(String)
  case tokenizerNotFound(String)
  case audioProcessingFailed(String)

  var errorDescription: String? {
    switch self {
      case .maxSamplingTrialsExceeded:
        "Maximum sampling trials exceeded during token generation"
      case .modelNotLoaded:
        "CosyVoice3 model is not loaded"
      case let .invalidInput(message):
        "Invalid input: \(message)"
      case let .tokenizerNotFound(path):
        "Tokenizer not found at path: \(path)"
      case let .audioProcessingFailed(message):
        "Audio processing failed: \(message)"
    }
  }
}
