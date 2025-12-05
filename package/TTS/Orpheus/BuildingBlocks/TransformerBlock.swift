//
//  TransformerBlock.swift
//  MLXAudio
//
//  Created by Ben Harraway on 21/05/2025.
//
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration for Orpheus model (Llama 3B architecture)
struct OrpheusConfig {
  let hiddenSize: Int = 3072
  let intermediateSize: Int = 8192
  let attentionHeads: Int = 24
  let kvHeads: Int = 8
  let hiddenLayers: Int = 28
  let vocabularySize: Int = 156_940 // Updated to match mlx-community/orpheus-3b-0.1-ft-4bit
  let rmsNormEps: Float = 1e-5
  let ropeTheta: Float = 500_000.0
  let ropeTraditional: Bool = false
  let ropeScaleFactor: Float = 32.0
  let ropeLowFreqFactor: Float = 1.0
  let ropeHighFreqFactor: Float = 4.0
  let ropeOldContextLen: Int = 8192
  let maxSeqLen: Int = 2048
  let tieWordEmbeddings: Bool = true // lm_head shares weights with embed_tokens

  var headDim: Int { hiddenSize / attentionHeads }
}

// MARK: - Attention Module

/// Multi-head attention with support for Grouped Query Attention (GQA)
private class OrpheusAttention: Module {
  let config: OrpheusConfig
  let scale: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear

  let rope: Llama3RoPE

  init(_ config: OrpheusConfig) {
    self.config = config
    scale = 1.0 / sqrt(Float(config.headDim))

    _qProj.wrappedValue = Linear(config.hiddenSize, config.attentionHeads * config.headDim, bias: false)
    _kProj.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
    _vProj.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
    _oProj.wrappedValue = Linear(config.attentionHeads * config.headDim, config.hiddenSize, bias: false)

    rope = Llama3RoPE(
      dims: config.headDim,
      traditional: config.ropeTraditional,
      base: config.ropeTheta,
      scaleFactor: config.ropeScaleFactor,
      lowFreqFactor: config.ropeLowFreqFactor,
      highFreqFactor: config.ropeHighFreqFactor,
      oldContextLen: Float(config.ropeOldContextLen),
    )
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    cache: KVCache?,
  ) -> MLXArray {
    let (B, L) = (x.dim(0), x.dim(1))

    var queries = qProj(x)
    var keys = kProj(x)
    var values = vProj(x)

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

    return oProj(outputReshaped)
  }
}

// MARK: - Transformer Block

/// Single transformer layer with attention and MLP
private class OrpheusTransformerBlock: Module {
  @ModuleInfo(key: "self_attn") var attention: OrpheusAttention
  @ModuleInfo(key: "mlp") var mlp: SwiGLUMLP

  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  init(_ config: OrpheusConfig) {
    _attention.wrappedValue = OrpheusAttention(config)
    _mlp.wrappedValue = SwiGLUMLP(
      hiddenSize: config.hiddenSize,
      intermediateSize: config.intermediateSize,
      bias: false,
    )
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

// MARK: - Orpheus Model

/// Main Orpheus model (Llama architecture for TTS)
class OrpheusModel: Module {
  let config: OrpheusConfig

  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  @ModuleInfo(key: "norm") var norm: RMSNorm

  fileprivate let layers: [OrpheusTransformerBlock]

  init(_ config: OrpheusConfig = OrpheusConfig()) {
    self.config = config

    _embedTokens.wrappedValue = Embedding(
      embeddingCount: config.vocabularySize,
      dimensions: config.hiddenSize,
    )
    _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

    layers = (0 ..< config.hiddenLayers).map { _ in OrpheusTransformerBlock(config) }
  }

  func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
    var h = embedTokens(inputs)

    // Determine mask based on sequence length and cache state
    let mask: MLXFast.ScaledDotProductAttentionMaskMode = if h.dim(1) > 1 {
      .causal // Multi-token (prompt processing)
    } else {
      .none // Single token (incremental generation)
    }

    for (i, layer) in layers.enumerated() {
      h = layer(h, mask: mask, cache: cache?[i])
    }

    return norm(h)
  }

  /// Create KV caches for all layers
  func newCache() -> [KVCache] {
    (0 ..< config.hiddenLayers).map { _ in KVCacheSimple() }
  }
}

// MARK: - LM Head Model

/// Orpheus model with language modeling head
class OrpheusLMHeadModel: Module {
  @ModuleInfo(key: "model") var model: OrpheusModel
  @ModuleInfo(key: "lm_head") var lmHead: Linear?

  let config: OrpheusConfig

  init(_ config: OrpheusConfig = OrpheusConfig()) {
    self.config = config
    _model.wrappedValue = OrpheusModel(config)

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
      // This works correctly for both regular and quantized embeddings
      return model.embedTokens.asLinear(out)
    }
  }

  /// Create KV caches for all layers
  func newCache() -> [KVCache] {
    model.newCache()
  }
}
