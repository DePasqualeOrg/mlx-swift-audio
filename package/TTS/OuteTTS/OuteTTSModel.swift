//
//  OuteTTSModel.swift
//  MLXAudio
//
//  Custom Llama model for OuteTTS

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration for OuteTTS model (Llama architecture)
/// Loaded from config.json in the model repository
struct OuteTTSLlamaConfiguration: Codable, Sendable {
  var hiddenSize: Int
  var intermediateSize: Int
  var attentionHeads: Int
  var kvHeads: Int
  var hiddenLayers: Int
  var vocabularySize: Int
  var rmsNormEps: Float
  var ropeTheta: Float
  var ropeTraditional: Bool
  var maxPositionEmbeddings: Int?
  var tieWordEmbeddings: Bool
  var ropeScaling: [String: StringOrNumber]?

  var headDim: Int { hiddenSize / attentionHeads }

  enum CodingKeys: String, CodingKey {
    case hiddenSize = "hidden_size"
    case intermediateSize = "intermediate_size"
    case attentionHeads = "num_attention_heads"
    case kvHeads = "num_key_value_heads"
    case hiddenLayers = "num_hidden_layers"
    case vocabularySize = "vocab_size"
    case rmsNormEps = "rms_norm_eps"
    case ropeTheta = "rope_theta"
    case ropeTraditional = "rope_traditional"
    case maxPositionEmbeddings = "max_position_embeddings"
    case tieWordEmbeddings = "tie_word_embeddings"
    case ropeScaling = "rope_scaling"
  }

  init(from decoder: Swift.Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
    intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
    attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
    kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
    hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
    vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
    rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
    ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
    ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
    maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
    tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
  }
}

// MARK: - RoPE (Rotary Position Embedding)

/// Rotary Position Embedding with Llama3-style scaling support
class OuteTTSRoPE: Module {
  let dims: Int
  let traditional: Bool
  var base: Float?
  let scale: Float
  var freqs: MLXArray?

  init(config: OuteTTSLlamaConfiguration) {
    dims = config.headDim
    traditional = config.ropeTraditional
    base = config.ropeTheta
    scale = 1.0

    super.init()

    // Check for Llama3-style rope scaling
    if let ropeScaling = config.ropeScaling,
       case let .string(ropeType) = ropeScaling["type"] ?? ropeScaling["rope_type"],
       ropeType == "llama3",
       case let .float(factor) = ropeScaling["factor"],
       let base
    {
      let lowFreqFactor: Float = if case let .float(v) = ropeScaling["low_freq_factor"] {
        v
      } else {
        1.0
      }

      let highFreqFactor: Float = if case let .float(v) = ropeScaling["high_freq_factor"] {
        v
      } else {
        4.0
      }

      let oldContextLen: Float = if case let .float(v) = ropeScaling["original_max_position_embeddings"] {
        v
      } else {
        8192
      }

      let lowFreqWavelen = oldContextLen / lowFreqFactor
      let highFreqWavelen = oldContextLen / highFreqFactor

      let indices = MLXArray(stride(from: 0, to: dims, by: 2))
      var frequencies = MLX.pow(base, indices / Float(dims))
      let wavelens = 2 * Float.pi * frequencies

      frequencies = MLX.where(
        wavelens .> MLXArray(lowFreqWavelen),
        frequencies * factor,
        frequencies,
      )
      let isMediumFreq = MLX.logicalAnd(
        wavelens .> MLXArray(highFreqWavelen),
        wavelens .< MLXArray(lowFreqWavelen),
      )
      let smoothFactors = (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
      let smoothFreqs = frequencies / ((1 - smoothFactors) / factor + smoothFactors)

      freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
      self.base = nil
    }
  }

  func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
    MLXFast.RoPE(
      x,
      dimensions: dims,
      traditional: traditional,
      base: base,
      scale: scale,
      offset: offset,
      freqs: freqs,
    )
  }
}

// MARK: - Attention Module

/// Multi-head attention with Grouped Query Attention (GQA) support
class OuteTTSAttention: Module {
  let config: OuteTTSLlamaConfiguration
  let scale: Float

  @ModuleInfo(key: "q_proj") var wq: Linear
  @ModuleInfo(key: "k_proj") var wk: Linear
  @ModuleInfo(key: "v_proj") var wv: Linear
  @ModuleInfo(key: "o_proj") var wo: Linear

  let rope: OuteTTSRoPE

  init(_ config: OuteTTSLlamaConfiguration) {
    self.config = config
    scale = 1.0 / sqrt(Float(config.headDim))

    _wq.wrappedValue = Linear(config.hiddenSize, config.attentionHeads * config.headDim, bias: false)
    _wk.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
    _wv.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
    _wo.wrappedValue = Linear(config.attentionHeads * config.headDim, config.hiddenSize, bias: false)

    rope = OuteTTSRoPE(config: config)
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    cache: KVCache?,
  ) -> MLXArray {
    let (B, L) = (x.dim(0), x.dim(1))

    var queries = wq(x)
    var keys = wk(x)
    var values = wv(x)

    // Reshape for multi-head attention: [B, L, H, D] -> [B, H, L, D]
    queries = queries.reshaped(B, L, config.attentionHeads, -1).transposed(0, 2, 1, 3)
    keys = keys.reshaped(B, L, config.kvHeads, -1).transposed(0, 2, 1, 3)
    values = values.reshaped(B, L, config.kvHeads, -1).transposed(0, 2, 1, 3)

    // Apply RoPE with cache offset
    let offset = cache?.offset ?? 0
    queries = rope(queries, offset: offset)
    keys = rope(keys, offset: offset)

    // Update cache and compute attention
    let output: MLXArray
    if let cache {
      (keys, values) = cache.update(keys: keys, values: values)
      output = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: scale,
        mask: mask,
      )
    } else {
      output = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: scale,
        mask: mask,
      )
    }

    // Reshape back: [B, H, L, D] -> [B, L, H*D]
    let outputReshaped = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)

    return wo(outputReshaped)
  }
}

// MARK: - MLP Module

/// Feed-forward network with SiLU activation (SwiGLU variant)
class OuteTTSMLP: Module, UnaryLayer {
  @ModuleInfo(key: "gate_proj") var gate: Linear
  @ModuleInfo(key: "down_proj") var down: Linear
  @ModuleInfo(key: "up_proj") var up: Linear

  init(_ config: OuteTTSLlamaConfiguration) {
    _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
    _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    down(silu(gate(x)) * up(x))
  }
}

// MARK: - Transformer Block

/// Single transformer layer with attention and MLP
class OuteTTSTransformerBlock: Module {
  @ModuleInfo(key: "self_attn") var attention: OuteTTSAttention
  @ModuleInfo(key: "mlp") var mlp: OuteTTSMLP

  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  init(_ config: OuteTTSLlamaConfiguration) {
    _attention.wrappedValue = OuteTTSAttention(config)
    _mlp.wrappedValue = OuteTTSMLP(config)
    _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    cache: KVCache?,
  ) -> MLXArray {
    // Self-attention with residual
    let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
    // MLP with residual
    let out = h + mlp(postAttentionLayerNorm(h))
    return out
  }
}

// MARK: - Model Inner

/// Inner model (without LM head)
class OuteTTSModelInner: Module {
  let config: OuteTTSLlamaConfiguration

  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  @ModuleInfo(key: "norm") var norm: RMSNorm

  let layers: [OuteTTSTransformerBlock]

  init(_ config: OuteTTSLlamaConfiguration) {
    self.config = config

    _embedTokens.wrappedValue = Embedding(
      embeddingCount: config.vocabularySize,
      dimensions: config.hiddenSize,
    )
    _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

    layers = (0 ..< config.hiddenLayers).map { _ in OuteTTSTransformerBlock(config) }
  }

  func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
    var h = embedTokens(inputs)

    // Determine mask based on sequence length
    // Use simple causal mask for multi-token, none for single token
    let mask: MLXFast.ScaledDotProductAttentionMaskMode = if h.dim(1) > 1 {
      .causal
    } else {
      .none
    }

    for (i, layer) in layers.enumerated() {
      h = layer(h, mask: mask, cache: cache?[i])
    }

    return norm(h)
  }
}

// MARK: - LM Head Model

/// OuteTTS model with language modeling head
class OuteTTSLlamaModel: Module {
  @ModuleInfo(key: "model") var model: OuteTTSModelInner
  @ModuleInfo(key: "lm_head") var lmHead: Linear?

  let config: OuteTTSLlamaConfiguration

  init(_ config: OuteTTSLlamaConfiguration) {
    self.config = config
    _model.wrappedValue = OuteTTSModelInner(config)

    // Only create separate lm_head if not tying word embeddings
    if !config.tieWordEmbeddings {
      _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }
  }

  func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
    let out = model(inputs, cache: cache)

    if let lmHead {
      return lmHead(out)
    } else {
      // Tied embeddings: use embedding's asLinear method for output projection
      return model.embedTokens.asLinear(out)
    }
  }

  /// Create KV caches for all layers
  func newCache() -> [KVCache] {
    (0 ..< config.hiddenLayers).map { _ in KVCacheSimple() }
  }

  /// Sanitize weights - remove unused rotary embeddings
  func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    weights.filter {
      !$0.key.contains("self_attn.rotary_emb.inv_freq")
    }
  }
}
