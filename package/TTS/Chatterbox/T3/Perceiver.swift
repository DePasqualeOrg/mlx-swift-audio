//
//  Perceiver.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/t3/perceiver.py
//

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// MARK: - AttentionQKV

/// Multi-head attention with separate Q, K, V projections
class AttentionQKV: Module {
  let nHeads: Int
  let headDim: Int
  let scale: Float
  let dropoutRate: Float

  init(nHeads: Int, headDim: Int, dropoutRate: Float = 0.1, scale: Float? = nil) {
    self.nHeads = nHeads
    self.headDim = headDim
    self.scale = scale ?? pow(Float(headDim), -0.5)
    self.dropoutRate = dropoutRate
  }

  /// Forward pass through attention
  ///
  /// - Parameters:
  ///   - q: Query tensor (B, T_q, n_heads * head_dim)
  ///   - k: Key tensor (B, T_k, n_heads * head_dim)
  ///   - v: Value tensor (B, T_v, n_heads * head_dim)
  ///   - mask: Optional attention mask
  /// - Returns: Output tensor (B, T_q, n_heads * head_dim)
  func callAsFunction(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    mask: MLXArray? = nil,
  ) -> MLXArray {
    // Split heads: (B, T, D) -> (B, n_heads, T, head_dim)
    let qSplit = splitHeads(q)
    let kSplit = splitHeads(k)
    let vSplit = splitHeads(v)

    // Use MLX fast attention (fused kernel)
    let out = MLXFast.scaledDotProductAttention(
      queries: qSplit,
      keys: kSplit,
      values: vSplit,
      scale: scale,
      mask: mask,
    )

    return combineHeads(out)
  }

  /// Split heads: (B, T, D) -> (B, n_heads, T, head_dim)
  private func splitHeads(_ x: MLXArray) -> MLXArray {
    let B = x.shape[0]
    let T = x.shape[1]
    let reshaped = x.reshaped([B, T, nHeads, headDim])
    return reshaped.transposed(0, 2, 1, 3)
  }

  /// Combine heads: (B, n_heads, T, head_dim) -> (B, T, D)
  private func combineHeads(_ x: MLXArray) -> MLXArray {
    let B = x.shape[0]
    let T = x.shape[2]
    let transposed = x.transposed(0, 2, 1, 3)
    return transposed.reshaped([B, T, -1])
  }
}

// MARK: - AttentionBlock

/// Cross-attention block with separate Q, K, V linear transformations
class AttentionBlock: Module {
  let channels: Int
  let numHeads: Int

  @ModuleInfo(key: "norm") var norm: LayerNorm
  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "proj_out") var projOut: Linear

  // AttentionQKV has no trainable parameters, so no @ModuleInfo
  var attention: AttentionQKV

  init(channels: Int, numHeads: Int = 1, dropoutRate: Float = 0.2, scale: Float? = nil) {
    self.channels = channels
    self.numHeads = numHeads

    _norm.wrappedValue = LayerNorm(dimensions: channels)
    _toQ.wrappedValue = Linear(channels, channels)
    _toK.wrappedValue = Linear(channels, channels)
    _toV.wrappedValue = Linear(channels, channels)
    _projOut.wrappedValue = Linear(channels, channels)

    // No @ModuleInfo - AttentionQKV has no learnable parameters
    attention = AttentionQKV(
      nHeads: numHeads,
      headDim: channels / numHeads,
      dropoutRate: dropoutRate,
      scale: scale,
    )

    super.init()
  }

  /// Cross-attention from x1 to x2.
  ///
  /// - Parameters:
  ///   - x1: Query source (B, T1, C)
  ///   - x2: Key/Value source (B, T2, C)
  ///   - mask: Optional attention mask
  /// - Returns: Output (B, T1, C)
  func callAsFunction(_ x1: MLXArray, _ x2: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let x1Norm = norm(x1)
    let x2Norm = norm(x2)

    let q = toQ(x1Norm)
    let k = toK(x2Norm)
    let v = toV(x2Norm)

    let h = attention(q: q, k: k, v: v, mask: mask)
    let hProj = projOut(h)

    return x1 + hProj
  }
}

// MARK: - Perceiver

/// Perceiver-style resampler for conditioning embeddings.
/// Reduces variable-length input to fixed-length latent representation.
///
/// Note: Uses a single shared attention block for both cross-attention
/// and self-attention, matching the original PyTorch implementation.
class Perceiver: Module {
  let queryShape: [Int]

  // Learnable query tokens - stored as a parameter for weight loading
  @ParameterInfo(key: "pre_attention_query") var preAttentionQuery: MLXArray

  @ModuleInfo(key: "attn") var attn: AttentionBlock

  init(
    preAttentionQueryToken: Int = 32,
    preAttentionQuerySize: Int = 1024,
    embeddingDim: Int = 1024,
    numAttnHeads: Int = 4,
  ) {
    queryShape = [1, preAttentionQueryToken, preAttentionQuerySize]

    // Learnable query tokens - initialize with uniform distribution
    let queryVariance = sqrt(3.0) * sqrt(2.0 / Float(preAttentionQueryToken + preAttentionQueryToken))
    _preAttentionQuery.wrappedValue = MLXRandom.uniform(
      low: -queryVariance,
      high: queryVariance,
      queryShape,
    )

    // Single shared attention block (used for both cross and self attention)
    _attn.wrappedValue = AttentionBlock(channels: embeddingDim, numHeads: numAttnHeads)

    super.init()
  }

  /// Forward pass through Perceiver
  ///
  /// - Parameter h: Input embeddings (B, T, D) - variable length T
  /// - Returns: Fixed-length output (B, query_tokens, D)
  func callAsFunction(_ h: MLXArray) -> MLXArray {
    let B = h.shape[0]

    // Expand query to batch size
    let query = MLX.broadcast(preAttentionQuery, to: [B] + Array(preAttentionQuery.shape.dropFirst()))

    // Cross-attention: query attends to input
    let preAtt = attn(query, h)

    // Self-attention: query attends to itself
    let attnOut = attn(preAtt, preAtt)

    return attnOut
  }
}
