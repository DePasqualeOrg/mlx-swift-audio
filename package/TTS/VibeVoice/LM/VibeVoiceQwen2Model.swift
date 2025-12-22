// Copyright © Microsoft (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/microsoft/VibeVoice
// License: licenses/vibevoice.txt

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Rotary Position Embedding

/// Rotary Position Embedding (RoPE) for VibeVoice
class VibeVoiceRotaryEmbedding: Module {
  let dim: Int
  let maxPositionEmbeddings: Int
  let base: Float

  init(dim: Int, maxPositionEmbeddings: Int = 8192, base: Float = 1_000_000.0) {
    self.dim = dim
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.base = base
  }

  /// Compute inverse frequencies on the fly
  private func computeInvFreq() -> MLXArray {
    let arange = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32)
    return 1.0 / MLX.pow(MLXArray(base), arange / Float(dim))
  }

  /// Compute cos and sin for rotary embeddings
  /// - Parameter positionIds: Position indices, shape (L,) or (B, L)
  /// - Returns: Tuple of (cos, sin) each of shape matching positions x dim
  func callAsFunction(_ positionIds: MLXArray) -> (MLXArray, MLXArray) {
    var positions = positionIds
    if positions.ndim == 0 {
      positions = positions.expandedDimensions(axis: 0)
    }

    // Use first batch row if batched
    let t: MLXArray
    if positions.ndim > 1 {
      t = positions[0].asType(.float32)
    } else {
      t = positions.asType(.float32)
    }

    let invFreq = computeInvFreq()
    let freqs = MLX.outer(t, invFreq) // (L, dim/2)

    // Concatenate to get full dimension
    let emb = MLX.concatenated([freqs, freqs], axis: -1) // (L, dim)

    return (MLX.cos(emb), MLX.sin(emb))
  }
}

/// Rotate half the hidden dims of the input
func vibeVoiceRotateHalf(_ x: MLXArray) -> MLXArray {
  let halfDim = x.shape[x.ndim - 1] / 2
  let x1 = x[.ellipsis, 0 ..< halfDim]
  let x2 = x[.ellipsis, halfDim...]
  return MLX.concatenated([-x2, x1], axis: -1)
}

/// Apply rotary position embeddings to query and key tensors
/// - Parameters:
///   - q: Query tensor, shape (B, L, H, D)
///   - k: Key tensor, shape (B, L, H_kv, D)
///   - cos: Cosine embeddings, shape (1, L, D)
///   - sin: Sine embeddings, shape (1, L, D)
/// - Returns: Tuple of rotated (q, k)
func vibeVoiceApplyRotaryPosEmb(
  q: MLXArray,
  k: MLXArray,
  cos: MLXArray,
  sin: MLXArray
) -> (MLXArray, MLXArray) {
  // Expand dims for head dimension: (1, L, D) -> (1, L, 1, D)
  let cosExpanded = cos.expandedDimensions(axis: 2)
  let sinExpanded = sin.expandedDimensions(axis: 2)

  let qEmbed = (q * cosExpanded) + (vibeVoiceRotateHalf(q) * sinExpanded)
  let kEmbed = (k * cosExpanded) + (vibeVoiceRotateHalf(k) * sinExpanded)

  return (qEmbed, kEmbed)
}

// MARK: - Attention

/// Multi-head attention with grouped query attention support
class VibeVoiceAttention: Module {
  let numHeads: Int
  let numKVHeads: Int
  let headDim: Int
  let hiddenSize: Int
  let scale: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear

  init(config: Qwen2DecoderConfig) {
    numHeads = config.numAttentionHeads
    numKVHeads = config.numKeyValueHeads
    headDim = config.headDim ?? (config.hiddenSize / numHeads)
    hiddenSize = config.hiddenSize
    scale = 1.0 / sqrt(Float(headDim))

    _qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: true)
    _kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: true)
    _vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: true)
    _oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)
  }

  func callAsFunction(
    _ x: MLXArray,
    cos: MLXArray,
    sin: MLXArray,
    mask: MLXArray? = nil,
    cache: (MLXArray, MLXArray)? = nil
  ) -> (MLXArray, (MLXArray, MLXArray)) {
    let B = x.shape[0]
    let L = x.shape[1]

    var q = qProj(x)
    var k = kProj(x)
    var v = vProj(x)

    // Reshape: (B, L, num_heads * head_dim) -> (B, L, num_heads, head_dim)
    q = q.reshaped([B, L, numHeads, headDim])
    k = k.reshaped([B, L, numKVHeads, headDim])
    v = v.reshaped([B, L, numKVHeads, headDim])

    // Apply rotary embeddings
    (q, k) = vibeVoiceApplyRotaryPosEmb(q: q, k: k, cos: cos, sin: sin)

    // KV cache handling
    if let (kCache, vCache) = cache {
      k = MLX.concatenated([kCache, k], axis: 1)
      v = MLX.concatenated([vCache, v], axis: 1)
    }

    let newCache = (k, v)

    // Transpose for attention: (B, L, H, D) -> (B, H, L, D)
    q = q.transposed(0, 2, 1, 3)
    k = k.transposed(0, 2, 1, 3)
    v = v.transposed(0, 2, 1, 3)

    // Use optimized scaled dot-product attention
    let out = MLXFast.scaledDotProductAttention(
      queries: q,
      keys: k,
      values: v,
      scale: scale,
      mask: mask
    )

    // Reshape output: (B, H, L, D) -> (B, L, H * D)
    let outReshaped = out.transposed(0, 2, 1, 3).reshaped([B, L, -1])

    return (oProj(outReshaped), newCache)
  }
}

// MARK: - MLP

/// Feed-forward network with SwiGLU activation
class VibeVoiceMLP: Module {
  @ModuleInfo(key: "gate_proj") var gateProj: Linear
  @ModuleInfo(key: "up_proj") var upProj: Linear
  @ModuleInfo(key: "down_proj") var downProj: Linear

  init(config: Qwen2DecoderConfig) {
    _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
    _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
    _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    downProj(silu(gateProj(x)) * upProj(x))
  }
}

// MARK: - Decoder Layer

/// A single transformer decoder layer
class VibeVoiceDecoderLayer: Module {
  @ModuleInfo(key: "self_attn") var selfAttn: VibeVoiceAttention
  @ModuleInfo(key: "mlp") var mlp: VibeVoiceMLP
  @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

  init(config: Qwen2DecoderConfig) {
    _selfAttn.wrappedValue = VibeVoiceAttention(config: config)
    _mlp.wrappedValue = VibeVoiceMLP(config: config)
    _inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    _postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }

  func callAsFunction(
    _ x: MLXArray,
    cos: MLXArray,
    sin: MLXArray,
    mask: MLXArray? = nil,
    cache: (MLXArray, MLXArray)? = nil
  ) -> (MLXArray, (MLXArray, MLXArray)) {
    // Self attention with pre-norm
    var residual = x
    var h = inputLayernorm(x)
    let (attnOut, newCache) = selfAttn(h, cos: cos, sin: sin, mask: mask, cache: cache)
    h = residual + attnOut

    // MLP with pre-norm
    residual = h
    h = postAttentionLayernorm(h)
    h = residual + mlp(h)

    return (h, newCache)
  }
}

// MARK: - Speech Connector

/// Connector to project speech latents to LM hidden size
class SpeechConnector: Module {
  @ModuleInfo(key: "fc1") var fc1: Linear
  @ModuleInfo(key: "norm") var norm: RMSNorm
  @ModuleInfo(key: "fc2") var fc2: Linear

  init(inputDim: Int, outputDim: Int, eps: Float = 1e-6) {
    _fc1.wrappedValue = Linear(inputDim, outputDim)
    _norm.wrappedValue = RMSNorm(dimensions: outputDim, eps: eps)
    _fc2.wrappedValue = Linear(outputDim, outputDim)
  }

  func callAsFunction(_ features: MLXArray) -> MLXArray {
    var x = fc1(features)
    x = norm(x)
    x = fc2(x)
    return x
  }
}

// MARK: - Binary Classifier

/// Binary classifier for TTS end-of-speech detection
class BinaryClassifier: Module {
  @ModuleInfo(key: "fc1") var fc1: Linear
  @ModuleInfo(key: "fc2") var fc2: Linear

  init(hiddenSize: Int) {
    _fc1.wrappedValue = Linear(hiddenSize, hiddenSize)
    _fc2.wrappedValue = Linear(hiddenSize, 1)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = fc1(x)
    h = relu(h)
    h = fc2(h)
    return h
  }
}

// MARK: - Qwen2 Model

/// Qwen2 transformer model for text and speech processing
class VibeVoiceQwen2Model: Module {
  let config: Qwen2DecoderConfig
  let useNorm: Bool

  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
  @ModuleInfo(key: "layers") var layers: [VibeVoiceDecoderLayer]
  @ModuleInfo(key: "norm") var norm: RMSNorm?

  private let rotaryEmb: VibeVoiceRotaryEmbedding

  init(config: Qwen2DecoderConfig, useNorm: Bool = true) {
    self.config = config
    self.useNorm = useNorm

    // Token embeddings (only if vocab_size > 0)
    if config.vocabSize > 0 {
      _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
    } else {
      _embedTokens.wrappedValue = nil
    }

    // Transformer layers
    _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
      VibeVoiceDecoderLayer(config: config)
    }

    // Final norm (only for TTS LM, not base LM)
    if useNorm {
      _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    } else {
      _norm.wrappedValue = nil
    }

    // Rotary embeddings
    let headDim = config.headDim ?? (config.hiddenSize / config.numAttentionHeads)
    rotaryEmb = VibeVoiceRotaryEmbedding(
      dim: headDim,
      maxPositionEmbeddings: config.maxPositionEmbeddings,
      base: config.ropeTheta
    )
  }

  /// Forward pass
  /// - Parameters:
  ///   - inputsEmbeds: Embedded inputs, shape (B, L, D)
  ///   - inputIds: Token IDs, shape (B, L) - used if inputsEmbeds is nil
  ///   - mask: Attention mask
  ///   - cache: KV cache from previous steps
  ///   - isCausal: Whether to apply causal masking
  /// - Returns: Tuple of (hidden_states, new_cache)
  func callAsFunction(
    inputsEmbeds: MLXArray? = nil,
    inputIds: MLXArray? = nil,
    mask: MLXArray? = nil,
    cache: [(MLXArray, MLXArray)]? = nil,
    isCausal: Bool = true
  ) -> (MLXArray, [(MLXArray, MLXArray)]) {
    var h: MLXArray
    if let embeds = inputsEmbeds {
      h = embeds
    } else if let ids = inputIds, let embed = embedTokens {
      h = embed(ids)
    } else {
      fatalError("Either inputsEmbeds or inputIds must be provided")
    }

    let L = h.shape[1]

    // Compute position offset from cache
    var offset = 0
    if let c = cache, !c.isEmpty, let firstCache = c.first {
      offset = firstCache.0.shape[1]
    }

    // Position IDs
    let positionIds = MLXArray(offset ..< (offset + L))

    // Get rotary embeddings and add batch dimension
    var (cos, sin) = rotaryEmb(positionIds)
    cos = cos.expandedDimensions(axis: 0) // (L, D) -> (1, L, D)
    sin = sin.expandedDimensions(axis: 0)

    // Create causal mask if needed
    var attnMask = mask
    if attnMask == nil, isCausal, L > 1 {
      let kLen = offset + L
      let qPos = MLXArray(offset ..< (offset + L)).expandedDimensions(axis: 1) // (L, 1)
      let kPos = MLXArray(0 ..< kLen).expandedDimensions(axis: 0) // (1, K)
      let allow = qPos .>= kPos // (L, K)
      let negInf = MLXArray(-Float.infinity)
      let causalMask = MLX.where(allow, MLXArray(Float(0)), negInf)
      attnMask = causalMask.expandedDimensions(axes: [0, 1]) // (1, 1, L, K)
    }

    var newCaches: [(MLXArray, MLXArray)] = []

    for (i, layer) in layers.enumerated() {
      let layerCache = cache?[i]
      let (newH, c) = layer(h, cos: cos, sin: sin, mask: attnMask, cache: layerCache)
      h = newH
      newCaches.append(c)
    }

    if let normLayer = norm {
      h = normLayer(h)
    }

    return (h, newCaches)
  }
}
