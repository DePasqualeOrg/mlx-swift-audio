//
//  Attention.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/s3gen/transformer/attention.py
//  Multi-head attention modules for Conformer encoder
//

import Foundation
import MLX
import MLXNN

// MARK: - MultiHeadedAttention

/// Multi-Head Attention layer
class MultiHeadedAttention: Module {
  let dK: Int
  let h: Int
  let dropoutRate: Float

  @ModuleInfo(key: "linear_q") var linearQ: Linear
  @ModuleInfo(key: "linear_k") var linearK: Linear
  @ModuleInfo(key: "linear_v") var linearV: Linear
  @ModuleInfo(key: "linear_out") var linearOut: Linear

  init(nHead: Int, nFeat: Int, dropoutRate: Float, keyBias: Bool = true) {
    precondition(nFeat % nHead == 0, "nFeat must be divisible by nHead")

    dK = nFeat / nHead
    h = nHead
    self.dropoutRate = dropoutRate

    _linearQ.wrappedValue = Linear(nFeat, nFeat)
    _linearK.wrappedValue = Linear(nFeat, nFeat, bias: keyBias)
    _linearV.wrappedValue = Linear(nFeat, nFeat)
    _linearOut.wrappedValue = Linear(nFeat, nFeat)
  }

  /// Transform query, key and value
  func forwardQKV(query: MLXArray, key: MLXArray, value: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let nBatch = query.shape[0]

    var q = linearQ(query).reshaped([nBatch, -1, h, dK])
    var k = linearK(key).reshaped([nBatch, -1, h, dK])
    var v = linearV(value).reshaped([nBatch, -1, h, dK])

    q = q.transposed(0, 2, 1, 3) // (B, h, T1, d_k)
    k = k.transposed(0, 2, 1, 3) // (B, h, T2, d_k)
    v = v.transposed(0, 2, 1, 3) // (B, h, T2, d_k)

    return (q, k, v)
  }

  /// Compute attention context vector
  func forwardAttention(value: MLXArray, scores: MLXArray, mask: MLXArray?) -> MLXArray {
    let nBatch = value.shape[0]
    var attnScores = scores
    var attn: MLXArray

    // Compute expanded mask ONCE and reuse for both pre- and post-softmax
    if let m = mask, m.shape[2] > 0 {
      // Expand mask: (B, 1, T2) -> (B, 1, 1, T2)
      var maskExpanded = m.expandedDimensions(axis: 1)
      // Truncate mask to match scores length
      if maskExpanded.shape[3] > attnScores.shape[3] {
        maskExpanded = maskExpanded[0..., 0..., 0..., 0 ..< attnScores.shape[3]]
      }
      // Pre-compute mask condition once
      let maskCondition = maskExpanded .== 0

      // Apply mask before softmax
      attnScores = MLX.where(maskCondition, MLXArray(-Float.infinity), attnScores)
      attn = softmax(attnScores, axis: -1)
      // Apply mask after softmax (reuse maskCondition)
      attn = MLX.where(maskCondition, MLXArray(0.0), attn)
    } else {
      attn = softmax(attnScores, axis: -1)
    }

    // Compute attention output
    var x = MLX.matmul(attn, value) // (B, h, T1, d_k)
    x = x.transposed(0, 2, 1, 3) // (B, T1, h, d_k)
    x = x.reshaped([nBatch, -1, h * dK]) // (B, T1, d_model)

    return linearOut(x)
  }

  /// Scaled dot product attention
  func callAsFunction(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    mask: MLXArray? = nil,
    posEmb _: MLXArray? = nil,
    cache: MLXArray? = nil,
  ) -> (MLXArray, MLXArray) {
    var (q, k, v) = forwardQKV(query: query, key: key, value: value)

    // Handle KV caching
    if let c = cache, c.shape[0] > 0 {
      let split = MLX.split(c, indices: [c.shape[3] / 2], axis: 3)
      let keyCache = split[0]
      let valueCache = split[1]
      k = MLX.concatenated([keyCache, k], axis: 2)
      v = MLX.concatenated([valueCache, v], axis: 2)
    }

    let newCache = MLX.concatenated([k, v], axis: -1)

    let scores = MLX.matmul(q, k.swappedAxes(-2, -1)) / sqrt(Float(dK))
    let output = forwardAttention(value: v, scores: scores, mask: mask)

    return (output, newCache)
  }
}

// MARK: - RelPositionMultiHeadedAttention

/// Multi-Head Attention with relative positional encoding
class RelPositionMultiHeadedAttention: MultiHeadedAttention {
  @ModuleInfo(key: "linear_pos") var linearPos: Linear
  @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
  @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

  override init(nHead: Int, nFeat: Int, dropoutRate: Float, keyBias: Bool = true) {
    // Initialize learnable biases for relative position
    let dK = nFeat / nHead
    let scale = Float(sqrt(6.0 / Float(nHead + dK)))
    _posBiasU.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [nHead, dK])
    _posBiasV.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [nHead, dK])

    // Linear transformation for positional encoding
    _linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)

    super.init(nHead: nHead, nFeat: nFeat, dropoutRate: dropoutRate, keyBias: keyBias)
  }

  /// Compute relative positional encoding
  func relShift(_ x: MLXArray) -> MLXArray {
    let zeroPad = MLXArray.zeros([x.shape[0], x.shape[1], x.shape[2], 1])
    var xPadded = MLX.concatenated([zeroPad, x], axis: -1)

    xPadded = xPadded.reshaped([x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2]])
    let xSliced = xPadded[0..., 0..., 1..., 0...].reshaped(x.shape)
    return xSliced[0..., 0..., 0..., 0 ..< x.shape[3] / 2 + 1]
  }

  override func callAsFunction(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    mask: MLXArray? = nil,
    posEmb: MLXArray? = nil,
    cache: MLXArray? = nil,
  ) -> (MLXArray, MLXArray) {
    var (q, k, v) = forwardQKV(query: query, key: key, value: value)
    q = q.transposed(0, 2, 1, 3) // (B, T1, h, d_k)

    // Handle KV caching
    if let c = cache, c.shape[0] > 0 {
      let split = MLX.split(c, indices: [c.shape[3] / 2], axis: 3)
      let keyCache = split[0]
      let valueCache = split[1]
      k = MLX.concatenated([keyCache, k], axis: 2)
      v = MLX.concatenated([valueCache, v], axis: 2)
    }

    let newCache = MLX.concatenated([k, v], axis: -1)

    // Process positional embeddings
    guard let posEmbInput = posEmb else {
      fatalError("posEmb required for RelPositionMultiHeadedAttention")
    }

    let nBatchPos = posEmbInput.shape[0]
    var p = linearPos(posEmbInput).reshaped([nBatchPos, -1, h, dK])
    p = p.transposed(0, 2, 1, 3) // (B, h, T1, d_k)

    // Add biases to query
    let qWithBiasU = (q + posBiasU).transposed(0, 2, 1, 3) // (B, h, T1, d_k)
    let qWithBiasV = (q + posBiasV).transposed(0, 2, 1, 3) // (B, h, T1, d_k)

    // Compute attention scores with relative position
    let matrixAC = MLX.matmul(qWithBiasU, k.swappedAxes(-2, -1))
    var matrixBD = MLX.matmul(qWithBiasV, p.swappedAxes(-2, -1))

    // Apply relative shift if needed
    if matrixAC.shape != matrixBD.shape {
      matrixBD = relShift(matrixBD)
    }

    let scores = (matrixAC + matrixBD) / sqrt(Float(dK))
    let output = forwardAttention(value: v, scores: scores, mask: mask)

    return (output, newCache)
  }
}
