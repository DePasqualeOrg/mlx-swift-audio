//  Transformer components for flow matching decoder

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - DiffusersAttention

/// Attention module matching diffusers.models.attention_processor.Attention
class DiffusersAttention: Module {
  let heads: Int
  let dimHead: Int
  let innerDim: Int
  let scale: Float

  @ModuleInfo(key: "query_proj") var queryProj: Linear
  @ModuleInfo(key: "key_proj") var keyProj: Linear
  @ModuleInfo(key: "value_proj") var valueProj: Linear
  @ModuleInfo(key: "out_proj") var outProj: Linear

  init(queryDim: Int, heads: Int = 8, dimHead: Int = 64, qkvBias: Bool = false, outBias: Bool = true) {
    self.heads = heads
    self.dimHead = dimHead
    innerDim = heads * dimHead
    scale = pow(Float(dimHead), -0.5)

    // CosyVoice2 has no bias on q/k/v but has bias on out_proj
    _queryProj.wrappedValue = Linear(queryDim, innerDim, bias: qkvBias)
    _keyProj.wrappedValue = Linear(queryDim, innerDim, bias: qkvBias)
    _valueProj.wrappedValue = Linear(queryDim, innerDim, bias: qkvBias)
    _outProj.wrappedValue = Linear(innerDim, queryDim, bias: outBias)
  }

  func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
    let shape = hiddenStates.shape
    let B = shape[0]
    let T = shape[1]

    // Project to q, k, v
    let q = queryProj(hiddenStates) // (B, T, inner_dim)
    let k = keyProj(hiddenStates)
    let v = valueProj(hiddenStates)

    // Reshape to (B, heads, T, dim_head)
    let qReshaped = q.reshaped([B, T, heads, dimHead]).transposed(0, 2, 1, 3)
    let kReshaped = k.reshaped([B, T, heads, dimHead]).transposed(0, 2, 1, 3)
    let vReshaped = v.reshaped([B, T, heads, dimHead]).transposed(0, 2, 1, 3)

    // Prepare mask for scaled_dot_product_attention
    var mask: MLXArray? = nil
    if let attnMask = attentionMask {
      if attnMask.ndim == 2 {
        // (B, T) -> (B, 1, 1, T)
        mask = attnMask[0..., .newAxis, .newAxis, 0...]
      } else {
        mask = attnMask
      }
    }

    // Use MLX fast attention
    let out = MLXFast.scaledDotProductAttention(
      queries: qReshaped,
      keys: kReshaped,
      values: vReshaped,
      scale: scale,
      mask: mask,
    )

    // Reshape back: (B, heads, T, dim_head) -> (B, T, inner_dim)
    let outReshaped = out.transposed(0, 2, 1, 3).reshaped([B, T, innerDim])

    // Output projection
    return outProj(outReshaped)
  }
}

// MARK: - FeedForward

/// Feed-forward network with GELU activation
class FeedForward: Module {
  @ModuleInfo(key: "layers") var layers: [Linear]

  init(dim: Int, innerDim: Int) {
    _layers.wrappedValue = [
      Linear(dim, innerDim),
      Linear(innerDim, dim),
    ]
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = layers[0](x)
    h = gelu(h)
    h = layers[1](h)
    return h
  }
}

// MARK: - BasicTransformerBlock

/// Basic transformer block for decoder
class BasicTransformerBlock: Module {
  @ModuleInfo(key: "norm1") var norm1: LayerNorm
  @ModuleInfo(key: "norm3") var norm3: LayerNorm
  @ModuleInfo(key: "attn") var attn: DiffusersAttention
  @ModuleInfo(key: "ff") var ff: FeedForward

  init(
    dim: Int,
    numAttentionHeads: Int,
    attentionHeadDim: Int,
    dropout _: Float = 0.0,
    activationFn _: String = "gelu",
  ) {
    _norm1.wrappedValue = LayerNorm(dimensions: dim)
    _norm3.wrappedValue = LayerNorm(dimensions: dim)
    _attn.wrappedValue = DiffusersAttention(
      queryDim: dim,
      heads: numAttentionHeads,
      dimHead: attentionHeadDim,
      qkvBias: false,
      outBias: true
    )
    _ff.wrappedValue = FeedForward(dim: dim, innerDim: dim * 4)
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil,
    timestep _: MLXArray? = nil,
  ) -> MLXArray {
    // Self-attention
    var h = hiddenStates
    let normed1 = norm1(h)
    let attnOut = attn(normed1, attentionMask: attentionMask)
    h = h + attnOut

    // Feed-forward
    let normed3 = norm3(h)
    let ffOut = ff(normed3)
    h = h + ffOut

    return h
  }
}
