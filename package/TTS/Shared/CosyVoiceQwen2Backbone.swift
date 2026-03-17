// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration for Qwen2 model used in CosyVoice2 and CosyVoice3
struct CosyVoiceQwen2Config: Codable, Sendable {
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

// MARK: - Attention

class CosyVoiceQwen2Attention: Module {
  let config: CosyVoiceQwen2Config
  let scale: Float
  let headDim: Int

  @ModuleInfo(key: "q_proj") var wq: Linear
  @ModuleInfo(key: "k_proj") var wk: Linear
  @ModuleInfo(key: "v_proj") var wv: Linear
  @ModuleInfo(key: "o_proj") var wo: Linear

  let rope: RoPE

  init(_ config: CosyVoiceQwen2Config) {
    self.config = config
    headDim = config.hiddenSize / config.numAttentionHeads
    scale = pow(Float(headDim), -0.5)

    _wq.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * headDim, bias: true)
    _wk.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * headDim, bias: true)
    _wv.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * headDim, bias: true)
    _wo.wrappedValue = Linear(config.numAttentionHeads * headDim, config.hiddenSize, bias: false)

    rope = RoPE(dimensions: headDim, traditional: config.ropeTraditional, base: config.ropeTheta)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCacheSimple?) -> MLXArray {
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

    // Use optimized scaled dot product attention with automatic GQA handling
    // mask: .causal for prefill (L > 1), .none for single-token generation
    let output = MLXFast.scaledDotProductAttention(
      queries: queries,
      keys: keys,
      values: values,
      scale: scale,
      mask: L > 1 ? .causal : .none
    )
    .transposed(0, 2, 1, 3)
    .reshaped(B, L, -1)

    return wo(output)
  }
}

// MARK: - MLP

class CosyVoiceQwen2MLP: Module, UnaryLayer {
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

// MARK: - Transformer Block

class CosyVoiceQwen2TransformerBlock: Module {
  @ModuleInfo(key: "self_attn") var attention: CosyVoiceQwen2Attention
  @ModuleInfo var mlp: CosyVoiceQwen2MLP
  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  init(_ config: CosyVoiceQwen2Config) {
    _attention.wrappedValue = CosyVoiceQwen2Attention(config)
    _mlp.wrappedValue = CosyVoiceQwen2MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
    _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCacheSimple?) -> MLXArray {
    var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
    let h = x + r
    r = mlp(postAttentionLayerNorm(h))
    return h + r
  }
}

// MARK: - Model Inner

class CosyVoiceQwen2ModelInner: Module {
  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

  @ModuleInfo var layers: [CosyVoiceQwen2TransformerBlock]
  @ModuleInfo var norm: RMSNorm

  init(_ config: CosyVoiceQwen2Config) {
    _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

    var layersArray: [CosyVoiceQwen2TransformerBlock] = []
    for _ in 0 ..< config.numHiddenLayers {
      layersArray.append(CosyVoiceQwen2TransformerBlock(config))
    }
    _layers.wrappedValue = layersArray

    _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }

  /// Forward with token input
  func callAsFunction(_ inputs: MLXArray, cache: [KVCacheSimple]?) -> MLXArray {
    let h = embedTokens(inputs)
    return forward(embeddings: h, cache: cache)
  }

  /// Forward with embedding input
  func forward(embeddings: MLXArray, cache: [KVCacheSimple]?) -> MLXArray {
    var h = embeddings

    // scaledDotProductAttention handles causal masking internally
    for (i, layer) in layers.enumerated() {
      h = layer(h, mask: nil, cache: cache?[i])
    }

    return norm(h)
  }
}

// MARK: - Encoder

/// Wrapper around Qwen2 model for CosyVoice
/// Provides access to embeddings and single-step forward for autoregressive generation
class CosyVoiceQwen2Encoder: Module {
  @ModuleInfo var model: CosyVoiceQwen2ModelInner
  let config: CosyVoiceQwen2Config

  init(config: CosyVoiceQwen2Config) {
    self.config = config
    _model.wrappedValue = CosyVoiceQwen2ModelInner(config)
  }

  /// Access the token embedding layer
  var embedTokens: Embedding {
    model.embedTokens
  }

  /// Full forward pass with attention mask
  /// - Parameters:
  ///   - xs: Input embeddings (B, T, D)
  ///   - xsLens: Sequence lengths (B,)
  /// - Returns: Tuple of (hidden_states, mask)
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
  /// - Parameters:
  ///   - xs: Input embeddings (B, T, D) - typically T=1 for generation
  ///   - cache: List of KVCacheSimple objects, one per layer
  /// - Returns: Tuple of (hidden_states, cache)
  func forwardOneStep(_ xs: MLXArray, cache: [KVCacheSimple]?) -> (MLXArray, [KVCacheSimple]) {
    let cacheList = cache ?? (0 ..< config.numHiddenLayers).map { _ in KVCacheSimple() }

    let hiddenStates = model.forward(embeddings: xs, cache: cacheList)

    return (hiddenStates, cacheList)
  }
}
