// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

import Foundation
import MLX
import MLXNN

// MARK: - Sinusoidal Position Embedding

/// Sinusoidal position embedding for timesteps
class SinusPositionEmbedding: Module {
  let dim: Int

  init(dim: Int) {
    self.dim = dim
  }

  /// Generate sinusoidal embeddings
  /// - Parameters:
  ///   - x: Input tensor (B,) of timesteps
  ///   - scale: Scale factor for embeddings
  /// - Returns: Embeddings (B, dim)
  func callAsFunction(_ x: MLXArray, scale: Float = 1000) -> MLXArray {
    let halfDim = dim / 2
    let emb = log(Float(10000)) / Float(halfDim - 1)
    let expRange = MLX.exp(MLXArray(0 ..< halfDim).asType(.float32) * -emb)
    let scaled = MLXArray(scale) * x.expandedDimensions(axis: 1) * expRange.expandedDimensions(axis: 0)
    return MLX.concatenated([MLX.sin(scaled), MLX.cos(scaled)], axis: -1)
  }
}

// MARK: - Timestep Embedding

/// Timestep embedding using sinusoidal position embedding + MLP
class CosyVoice3TimestepEmbedding: Module {
  @ModuleInfo(key: "time_embed") var timeEmbed: SinusPositionEmbedding
  @ModuleInfo(key: "time_mlp_0") var timeMlp0: Linear
  @ModuleInfo(key: "time_mlp_2") var timeMlp2: Linear

  init(dim: Int, freqEmbedDim: Int = 256) {
    _timeEmbed.wrappedValue = SinusPositionEmbedding(dim: freqEmbedDim)
    _timeMlp0.wrappedValue = Linear(freqEmbedDim, dim)
    _timeMlp2.wrappedValue = Linear(dim, dim)
  }

  /// Generate timestep embeddings
  /// - Parameter timestep: Timestep values (B,)
  /// - Returns: Embeddings (B, dim)
  func callAsFunction(_ timestep: MLXArray) -> MLXArray {
    var timeHidden = timeEmbed(timestep)
    timeHidden = timeHidden.asType(timestep.dtype)
    var time = timeMlp0(timeHidden)
    time = silu(time)
    time = timeMlp2(time)
    return time
  }
}

// MARK: - Causal Conv Position Embedding

/// Causal convolutional position embedding for streaming
class CausalConvPositionEmbedding: Module {
  let kernelSize: Int

  @ModuleInfo(key: "conv1") var conv1: Conv1d
  @ModuleInfo(key: "conv2") var conv2: Conv1d

  init(dim: Int, kernelSize: Int = 31, groups: Int = 16) {
    precondition(kernelSize % 2 != 0, "kernel_size must be odd")
    self.kernelSize = kernelSize
    _conv1.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: kernelSize, padding: 0, groups: groups)
    _conv2.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: kernelSize, padding: 0, groups: groups)
  }

  /// Apply causal convolutional position embedding
  /// - Parameters:
  ///   - x: Input (B, N, D)
  ///   - mask: Optional mask (B, N)
  /// - Returns: Position-embedded output (B, N, D)
  func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    var out = x

    if let m = mask {
      let maskExpanded = m.expandedDimensions(axis: -1)
      out = MLX.where(maskExpanded, out, MLXArray.zeros(like: out))
    }

    // Causal padding on left
    out = MLX.padded(out, widths: [IntOrPair(0), IntOrPair((kernelSize - 1, 0)), IntOrPair(0)])
    out = conv1(out)
    out = mish(out)

    out = MLX.padded(out, widths: [IntOrPair(0), IntOrPair((kernelSize - 1, 0)), IntOrPair(0)])
    out = conv2(out)
    out = mish(out)

    if let m = mask {
      let maskExpanded = m.expandedDimensions(axis: -1)
      out = MLX.where(maskExpanded, out, MLXArray.zeros(like: out))
    }

    return out
  }
}

// MARK: - Input Embedding

/// Input embedding for combining noised audio, condition, mu, and speaker
class InputEmbedding: Module {
  let spkDim: Int

  @ModuleInfo(key: "proj") var proj: Linear
  @ModuleInfo(key: "conv_pos_embed") var convPosEmbed: CausalConvPositionEmbedding

  init(melDim: Int, textDim: Int, outDim: Int, spkDim: Int? = nil) {
    let actualSpkDim = spkDim ?? 0
    self.spkDim = actualSpkDim
    _proj.wrappedValue = Linear(melDim * 2 + textDim + actualSpkDim, outDim)
    _convPosEmbed.wrappedValue = CausalConvPositionEmbedding(dim: outDim)
  }

  /// Combine inputs and apply position embedding
  /// - Parameters:
  ///   - x: Noised input audio (B, N, mel_dim)
  ///   - cond: Condition audio (B, N, mel_dim)
  ///   - textEmbed: Text/mu embeddings (B, N, text_dim)
  ///   - spks: Speaker embeddings (B, spk_dim)
  /// - Returns: Combined embeddings (B, N, out_dim)
  func callAsFunction(
    x: MLXArray,
    cond: MLXArray,
    textEmbed: MLXArray,
    spks: MLXArray
  ) -> MLXArray {
    var toCat = [x, cond, textEmbed]

    if spkDim > 0 {
      // Repeat speaker embedding for each time step
      let spksExpanded = MLX.broadcast(
        spks.expandedDimensions(axis: 1),
        to: [spks.shape[0], x.shape[1], spks.shape[spks.shape.count - 1]]
      )
      toCat.append(spksExpanded)
    }

    var out = proj(MLX.concatenated(toCat, axis: -1))
    out = convPosEmbed(out) + out
    return out
  }
}

// MARK: - Global Response Normalization

/// Global Response Normalization layer
class GRN: Module {
  @ParameterInfo(key: "gamma") var gamma: MLXArray
  @ParameterInfo(key: "beta") var beta: MLXArray

  init(dim: Int) {
    _gamma.wrappedValue = MLXArray.zeros([1, 1, dim])
    _beta.wrappedValue = MLXArray.zeros([1, 1, dim])
  }

  /// Apply Global Response Normalization
  /// - Parameter x: Input (B, N, D)
  /// - Returns: Normalized output (B, N, D)
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let gx = MLX.sqrt(MLX.sum(x * x, axis: 1, keepDims: true))
    let nx = gx / (MLX.mean(gx, axis: -1, keepDims: true) + 1e-6)
    return gamma * (x * nx) + beta + x
  }
}

// MARK: - Feed Forward

/// Feed-forward network with GELU activation
class DiTFeedForward: Module {
  let approximate: Bool

  @ModuleInfo(key: "ff_0_0") var ff00: Linear
  @ModuleInfo(key: "ff_1") var ff1: Dropout
  @ModuleInfo(key: "ff_2") var ff2: Linear

  init(dim: Int, dimOut: Int? = nil, mult: Int = 4, dropout: Float = 0.0, approximate: Bool = true) {
    let innerDim = dim * mult
    let actualDimOut = dimOut ?? dim
    self.approximate = approximate

    _ff00.wrappedValue = Linear(dim, innerDim)
    _ff1.wrappedValue = Dropout(p: dropout)
    _ff2.wrappedValue = Linear(innerDim, actualDimOut)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = ff00(x)
    out = approximate ? geluApproximate(out) : gelu(out)
    out = ff1(out)
    out = ff2(out)
    return out
  }
}

// MARK: - Adaptive Layer Norm

/// Adaptive Layer Normalization with zero initialization for DiT blocks
class AdaLayerNormZero: Module {
  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "norm") var norm: LayerNorm

  init(dim: Int) {
    _linear.wrappedValue = Linear(dim, dim * 6)
    _norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
  }

  /// Apply adaptive layer normalization
  /// - Parameters:
  ///   - x: Input (B, N, D)
  ///   - emb: Conditioning embedding (B, D)
  /// - Returns: Tuple of (normalized_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
  func callAsFunction(_ x: MLXArray, emb: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
    let embOut = linear(silu(emb))
    // Split into 6 parts
    let parts = MLX.split(embOut, parts: 6, axis: 1)
    let shiftMsa = parts[0]
    let scaleMsa = parts[1]
    let gateMsa = parts[2]
    let shiftMlp = parts[3]
    let scaleMlp = parts[4]
    let gateMlp = parts[5]

    let normalized = norm(x) * (1 + scaleMsa.expandedDimensions(axis: 1)) + shiftMsa.expandedDimensions(axis: 1)
    return (normalized, gateMsa, shiftMlp, scaleMlp, gateMlp)
  }
}

/// Final adaptive layer normalization (only shift and scale, no gate)
class AdaLayerNormZeroFinal: Module {
  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "norm") var norm: LayerNorm

  init(dim: Int) {
    _linear.wrappedValue = Linear(dim, dim * 2)
    _norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
  }

  /// Apply final adaptive layer normalization
  /// - Parameters:
  ///   - x: Input (B, N, D)
  ///   - emb: Conditioning embedding (B, D)
  /// - Returns: Normalized output (B, N, D)
  func callAsFunction(_ x: MLXArray, emb: MLXArray) -> MLXArray {
    let embOut = linear(silu(emb))
    let parts = MLX.split(embOut, parts: 2, axis: 1)
    let scale = parts[0]
    let shift = parts[1]
    return norm(x) * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
  }
}

// MARK: - Rotary Position Embedding

/// Rotate half the hidden dims of the input (for RoPE)
/// Matches x_transformers implementation
func rotateHalf(_ x: MLXArray) -> MLXArray {
  let shape = x.shape
  // Reshape to pairs: (..., d) -> (..., d//2, 2)
  var newShape = Array(shape.dropLast())
  newShape.append(shape.last! / 2)
  newShape.append(2)
  let reshaped = x.reshaped(newShape)

  // Split pairs - use .ellipsis to handle any number of leading dimensions
  let x1 = reshaped[.ellipsis, 0] // First element of each pair
  let x2 = reshaped[.ellipsis, 1] // Second element of each pair

  // Stack as (-x2, x1) interleaved
  let rotated = MLX.stacked([-x2, x1], axis: -1)

  // Flatten back
  return rotated.reshaped(shape)
}

/// Apply rotary position embedding to input tensor
/// Matches x_transformers implementation
/// - Parameters:
///   - t: Input tensor to rotate
///   - freqs: Frequency tensor from RotaryEmbedding
///   - scale: Scale factor (can be scalar or tensor for xpos)
func applyRotaryPosEmb(_ t: MLXArray, freqs: MLXArray, scale: MLXArray? = nil) -> MLXArray {
  let rotDim = freqs.shape[freqs.shape.count - 1]
  let seqLen = t.shape[t.shape.count - 2]
  let origDtype = t.dtype

  // Slice freqs to match sequence length: freqs[:, -seq_len:, :]
  var freqsSliced = freqs[0..., (freqs.shape[1] - seqLen)..., 0...]

  // Handle 4D tensor (B, H, N, D)
  if t.ndim == 4, freqsSliced.ndim == 3 {
    freqsSliced = freqsSliced.expandedDimensions(axis: 1)
  }

  // Partial rotary embeddings (GPT-J style)
  // Use .ellipsis to handle any number of leading dimensions: t[..., :rotDim]
  let tRot = t[.ellipsis, 0 ..< rotDim]
  let tUnrotated = t[.ellipsis, rotDim...]

  // Apply rotation: t * cos + rotate_half(t) * sin
  let rotated: MLXArray = if let s = scale {
    (tRot * MLX.cos(freqsSliced) * s) + (rotateHalf(tRot) * MLX.sin(freqsSliced) * s)
  } else {
    (tRot * MLX.cos(freqsSliced)) + (rotateHalf(tRot) * MLX.sin(freqsSliced))
  }

  let out = MLX.concatenated([rotated, tUnrotated], axis: -1)
  return out.asType(origDtype)
}

/// Rotary Position Embedding module
class RotaryEmbedding: Module {
  let dim: Int
  let interpolationFactor: Float
  let useXpos: Bool
  let scaleBase: Int

  private var invFreq: MLXArray
  private var scaleArray: MLXArray?
  private var cache: [Int: (MLXArray, MLXArray)] = [:]

  init(
    dim: Int,
    useXpos: Bool = false,
    scaleBase: Int = 512,
    interpolationFactor: Float = 1.0,
    base: Float = 10000.0,
    baseRescaleFactor: Float = 1.0
  ) {
    self.dim = dim
    self.interpolationFactor = interpolationFactor
    self.useXpos = useXpos
    self.scaleBase = scaleBase

    // Rescale base for longer sequences (NTK-aware scaling)
    let rescaledBase = base * pow(baseRescaleFactor, Float(dim) / Float(dim - 2))

    // Compute inv_freq
    let arange = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32)
    invFreq = 1.0 / MLX.pow(MLXArray(rescaledBase), arange / Float(dim))

    if useXpos {
      let scaleArange = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32)
      scaleArray = (scaleArange + 0.4 * Float(dim)) / (1.4 * Float(dim))
    }
  }

  /// Get rotary embeddings for a given sequence length
  func forwardFromSeqLen(_ seqLen: Int) -> (MLXArray, MLXArray) {
    if let cached = cache[seqLen] {
      return cached
    }
    let t = MLXArray(0 ..< seqLen).asType(.float32)
    let result = forward(t)
    cache[seqLen] = result
    return result
  }

  private func forward(_ t: MLXArray) -> (MLXArray, MLXArray) {
    var positions = t
    if positions.ndim == 1 {
      positions = positions.expandedDimensions(axis: 0) // (1, N)
    }

    // Compute frequencies: (B, N, dim/2)
    let freqs = MLX.einsum("bi,j->bij", positions, invFreq) / interpolationFactor

    // Stack and interleave: each angle appears twice for the pair
    let stacked = MLX.stacked([freqs, freqs], axis: -1)
    let freqsInterleaved = stacked.reshaped([stacked.shape[0], stacked.shape[1], -1])

    if scaleArray == nil {
      return (freqsInterleaved, MLXArray(1.0))
    }

    // Compute xpos scale
    let maxPos = Int(MLX.max(positions).item(Int32.self)) + 1
    let power = (positions - Float(maxPos / 2)) / Float(scaleBase)
    var scale = MLX.pow(scaleArray!, power.expandedDimensions(axis: -1))
    let scaleStacked = MLX.stacked([scale, scale], axis: -1)
    scale = scaleStacked.reshaped([scaleStacked.shape[0], scaleStacked.shape[1], -1])

    return (freqsInterleaved, scale)
  }
}

// MARK: - Attention

/// Multi-head attention with rotary position embedding support
class DiTAttention: Module {
  let dim: Int
  let heads: Int
  let dimHead: Int
  let innerDim: Int
  let dropout: Float

  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out_0") var toOut0: Linear
  @ModuleInfo(key: "to_out_1") var toOut1: Dropout

  init(dim: Int, heads: Int = 8, dimHead: Int = 64, dropout: Float = 0.0) {
    self.dim = dim
    self.heads = heads
    self.dimHead = dimHead
    innerDim = dimHead * heads
    self.dropout = dropout

    _toQ.wrappedValue = Linear(dim, innerDim)
    _toK.wrappedValue = Linear(dim, innerDim)
    _toV.wrappedValue = Linear(dim, innerDim)
    _toOut0.wrappedValue = Linear(innerDim, dim)
    _toOut1.wrappedValue = Dropout(p: dropout)
  }

  /// Apply multi-head attention
  /// - Parameters:
  ///   - x: Input (B, N, D)
  ///   - mask: Attention mask (B, N) or (B, 1, N, N)
  ///   - rope: Rotary position embedding tuple (freqs, scale)
  /// - Returns: Attention output (B, N, D)
  func callAsFunction(
    _ x: MLXArray,
    mask: MLXArray? = nil,
    rope: (MLXArray, MLXArray)? = nil
  ) -> MLXArray {
    let B = x.shape[0]
    let N = x.shape[1]

    // Project to Q, K, V
    var query = toQ(x)
    var key = toK(x)
    var value = toV(x)

    // Apply rotary position embedding BEFORE reshaping
    if let (freqs, xposScale) = rope {
      // When xposScale.ndim == 0, it's a scalar 1.0 (no xpos) - pass nil for no scaling
      // When xposScale.ndim > 0, it's a tensor (xpos enabled) - pass the tensor directly
      let qScale: MLXArray? = xposScale.ndim == 0 ? nil : xposScale
      let kScale: MLXArray? = xposScale.ndim == 0 ? nil : (1.0 / xposScale)
      query = applyRotaryPosEmb(query, freqs: freqs, scale: qScale)
      key = applyRotaryPosEmb(key, freqs: freqs, scale: kScale)
    }

    // Reshape for multi-head attention: (B, N, heads, head_dim)
    query = query.reshaped([B, N, heads, dimHead])
    key = key.reshaped([B, N, heads, dimHead])
    value = value.reshaped([B, N, heads, dimHead])

    // Transpose: (B, heads, N, head_dim)
    query = query.transposed(0, 2, 1, 3)
    key = key.transposed(0, 2, 1, 3)
    value = value.transposed(0, 2, 1, 3)

    // Prepare attention mask for scaledDotProductAttention
    // Convert from boolean mask to additive mask (False -> -inf, True -> 0)
    var attnMask: MLXArray? = nil
    if var inputMask = mask {
      if inputMask.ndim == 2 {
        // (B, N) -> (B, 1, 1, N)
        inputMask = inputMask.expandedDimensions(axes: [1, 2])
      } else if inputMask.ndim == 3 {
        inputMask = inputMask.expandedDimensions(axis: 1)
      }
      // Convert boolean mask to additive mask: False -> -inf, True -> 0
      let negInf = MLXArray(-Float.infinity)
      attnMask = MLX.where(inputMask, MLXArray(Float(0)), negInf)
    }

    // Use optimized fused attention kernel
    let scale = 1.0 / sqrt(Float(dimHead))
    var out = scaledDotProductAttention(
      queries: query,
      keys: key,
      values: value,
      scale: scale,
      mask: attnMask
    )

    // Reshape back
    out = out.transposed(0, 2, 1, 3).reshaped([B, N, innerDim])
    out = out.asType(query.dtype)

    // Output projection
    out = toOut0(out)
    out = toOut1(out)

    // Apply mask to output
    if var outMask = mask {
      if outMask.ndim == 2 {
        // (B, N) -> (B, N, 1)
        outMask = outMask.expandedDimensions(axis: -1)
      } else if outMask.ndim == 4 {
        // (B, 1, N, N) -> mask[:, 0, -1, :] -> (B, N) -> (B, N, 1)
        // Use single index -1 (last row) not a range
        let lastIdx = outMask.shape[2] - 1
        outMask = outMask[0..., 0, lastIdx, 0...].expandedDimensions(axis: -1)
      } else {
        outMask = outMask.expandedDimensions(axis: -1)
      }
      out = MLX.where(outMask, out, MLXArray.zeros(like: out))
    }

    return out
  }
}

// MARK: - DiT Block

/// Diffusion Transformer Block with adaptive layer norm
class DiTBlock: Module {
  @ModuleInfo(key: "attn_norm") var attnNorm: AdaLayerNormZero
  @ModuleInfo(key: "attn") var attn: DiTAttention
  @ModuleInfo(key: "ff_norm") var ffNorm: LayerNorm
  @ModuleInfo(key: "ff") var ff: DiTFeedForward

  init(dim: Int, heads: Int, dimHead: Int, ffMult: Int = 4, dropout: Float = 0.1) {
    _attnNorm.wrappedValue = AdaLayerNormZero(dim: dim)
    _attn.wrappedValue = DiTAttention(dim: dim, heads: heads, dimHead: dimHead, dropout: dropout)
    _ffNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    _ff.wrappedValue = DiTFeedForward(dim: dim, mult: ffMult, dropout: dropout, approximate: true)
  }

  /// Apply DiT block
  /// - Parameters:
  ///   - x: Input (B, N, D)
  ///   - t: Time embedding (B, D)
  ///   - mask: Attention mask
  ///   - rope: Rotary position embedding
  /// - Returns: Output (B, N, D)
  func callAsFunction(
    _ x: MLXArray,
    t: MLXArray,
    mask: MLXArray? = nil,
    rope: (MLXArray, MLXArray)? = nil
  ) -> MLXArray {
    var out = x

    // Pre-norm and modulation for attention
    let (norm, gateMsa, shiftMlp, scaleMlp, gateMlp) = attnNorm(out, emb: t)

    // Attention
    let attnOutput = attn(norm, mask: mask, rope: rope)

    // Apply attention with gate
    out = out + gateMsa.expandedDimensions(axis: 1) * attnOutput

    // FFN with modulation
    let ffNormOut = ffNorm(out) * (1 + scaleMlp.expandedDimensions(axis: 1)) + shiftMlp.expandedDimensions(axis: 1)
    let ffOutput = ff(ffNormOut)
    out = out + gateMlp.expandedDimensions(axis: 1) * ffOutput

    return out
  }
}

// MARK: - Chunk Mask Functions

/// Create mask for subsequent steps with chunk size (for streaming encoder)
func subsequentChunkMask(size: Int, chunkSize: Int) -> MLXArray {
  let posIdx = MLXArray(0 ..< size)
  // block_value[i] = (i // chunk_size + 1) * chunk_size
  let blockValue = ((posIdx / chunkSize) + 1) * chunkSize
  // Position j can be attended from position i if j < block_value[i]
  return posIdx.expandedDimensions(axis: 0) .< blockValue.expandedDimensions(axis: 1)
}

/// Create chunk-based attention mask for streaming inference
func addOptionalChunkMask(
  x: MLXArray,
  mask: MLXArray?,
  useDynamicChunk _: Bool,
  useDynamicLeftChunk _: Bool,
  decodingChunkSize _: Int,
  staticChunkSize: Int,
  numDecodingLeftChunks _: Int
) -> MLXArray {
  let B = x.shape[0]
  let N = x.shape[1]

  // Convert mask to boolean
  var boolMask: MLXArray? = nil
  if let m = mask {
    boolMask = m.asType(.bool)
  }

  var chunkMasks: MLXArray

  if staticChunkSize > 0 {
    // Streaming mode: create chunk-based mask
    var chunkMask = subsequentChunkMask(size: N, chunkSize: staticChunkSize)
    chunkMask = chunkMask.expandedDimensions(axis: 0)

    if let m = boolMask {
      chunkMasks = m.expandedDimensions(axis: 1) & chunkMask
    } else {
      chunkMasks = MLX.broadcast(chunkMask, to: [B, N, N])
    }
  } else {
    // Non-streaming: use base mask or full attention
    if let m = boolMask {
      chunkMasks = m
    } else {
      chunkMasks = MLXArray.ones([B, N]).asType(.bool)
    }
  }

  // Ensure correct output shape
  if chunkMasks.ndim == 2 {
    chunkMasks = chunkMasks.expandedDimensions(axis: 1)
    chunkMasks = MLX.broadcast(chunkMasks, to: [B, N, N])
  }

  // Safety fix: if any row is all-False, force to True (prevents NaN from softmax)
  // Use pure GPU operation to avoid CPU-GPU sync from .item() call
  let rowSums = MLX.sum(chunkMasks.asType(.float32), axis: -1, keepDims: true)
  let allFalseRows = rowSums .== 0
  chunkMasks = MLX.where(
    MLX.broadcast(allFalseRows, to: chunkMasks.shape),
    MLXArray.ones(like: chunkMasks),
    chunkMasks
  )

  // Add head dimension: (B, N, N) -> (B, 1, N, N)
  chunkMasks = chunkMasks.expandedDimensions(axis: 1)

  return chunkMasks
}

// MARK: - DiT (Main Transformer)

/// Diffusion Transformer for CosyVoice3
/// This is the main estimator that replaces the U-Net style decoder used in CosyVoice2
class DiT: Module {
  let dim: Int
  let depth: Int
  let staticChunkSize: Int
  let numDecodingLeftChunks: Int

  @ModuleInfo(key: "time_embed") var timeEmbed: CosyVoice3TimestepEmbedding
  @ModuleInfo(key: "input_embed") var inputEmbed: InputEmbedding
  @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [DiTBlock]
  @ModuleInfo(key: "long_skip_connection") var longSkipConnection: Linear?
  @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormZeroFinal
  @ModuleInfo(key: "proj_out") var projOut: Linear

  private var rotaryEmbed: RotaryEmbedding

  init(
    dim: Int = 1024,
    depth: Int = 22,
    heads: Int = 16,
    dimHead: Int = 64,
    dropout: Float = 0.1,
    ffMult: Int = 2,
    melDim: Int = 80,
    muDim: Int? = nil,
    longSkipConnection: Bool = false,
    spkDim: Int? = nil,
    outChannels: Int? = nil,
    staticChunkSize: Int = 50,
    numDecodingLeftChunks: Int = -1
  ) {
    self.dim = dim
    self.depth = depth
    self.staticChunkSize = staticChunkSize
    self.numDecodingLeftChunks = numDecodingLeftChunks

    let actualMuDim = muDim ?? melDim
    let actualOutChannels = outChannels ?? melDim

    _timeEmbed.wrappedValue = CosyVoice3TimestepEmbedding(dim: dim)
    _inputEmbed.wrappedValue = InputEmbedding(melDim: melDim, textDim: actualMuDim, outDim: dim, spkDim: spkDim)

    rotaryEmbed = RotaryEmbedding(dim: dimHead)

    var blocks: [DiTBlock] = []
    for _ in 0 ..< depth {
      blocks.append(DiTBlock(dim: dim, heads: heads, dimHead: dimHead, ffMult: ffMult, dropout: dropout))
    }
    _transformerBlocks.wrappedValue = blocks

    if longSkipConnection {
      _longSkipConnection.wrappedValue = Linear(dim * 2, dim, bias: false)
    } else {
      _longSkipConnection.wrappedValue = nil
    }

    _normOut.wrappedValue = AdaLayerNormZeroFinal(dim: dim)
    _projOut.wrappedValue = Linear(dim, actualOutChannels)
  }

  /// Forward pass of DiT
  /// - Parameters:
  ///   - x: Noised input (B, mel_dim, N) - channel-first
  ///   - mask: Mask (B, N)
  ///   - mu: Mu/condition (B, mel_dim, N)
  ///   - t: Timestep (B,) or scalar
  ///   - spks: Speaker embedding (B, D)
  ///   - cond: Condition audio (B, mel_dim, N)
  ///   - streaming: Whether in streaming mode
  /// - Returns: Output (B, mel_dim, N) - channel-first
  func callAsFunction(
    x: MLXArray,
    mask: MLXArray,
    mu: MLXArray,
    t: MLXArray,
    spks: MLXArray? = nil,
    cond: MLXArray? = nil,
    streaming: Bool = false
  ) -> MLXArray {
    // Transpose from channel-first to sequence-first
    var xSeq = x.swappedAxes(1, 2) // (B, N, mel_dim)
    let muSeq = mu.swappedAxes(1, 2) // (B, N, mu_dim)
    let condSeq = cond?.swappedAxes(1, 2) ?? MLXArray.zeros(like: xSeq) // (B, N, mel_dim)

    let B = xSeq.shape[0]
    let N = xSeq.shape[1]

    // Expand t for batch if needed
    var tBatch = t
    if tBatch.ndim == 0 {
      tBatch = MLX.broadcast(tBatch, to: [B])
    }

    // Time embedding
    tBatch = timeEmbed(tBatch)

    // Input embedding
    xSeq = inputEmbed(x: xSeq, cond: condSeq, textEmbed: muSeq, spks: spks ?? MLXArray.zeros([B, 0]))

    // Rotary embeddings
    let rope = rotaryEmbed.forwardFromSeqLen(N)

    var residual: MLXArray? = nil
    if longSkipConnection != nil {
      residual = xSeq
    }

    // Create attention mask
    let attnMask: MLXArray = if streaming {
      addOptionalChunkMask(
        x: xSeq, mask: mask, useDynamicChunk: false, useDynamicLeftChunk: false,
        decodingChunkSize: 0, staticChunkSize: staticChunkSize, numDecodingLeftChunks: -1
      )
    } else {
      addOptionalChunkMask(
        x: xSeq, mask: mask, useDynamicChunk: false, useDynamicLeftChunk: false,
        decodingChunkSize: 0, staticChunkSize: 0, numDecodingLeftChunks: -1
      )
    }

    // Transformer blocks
    for block in transformerBlocks {
      xSeq = block(xSeq, t: tBatch, mask: attnMask, rope: rope)
    }

    // Long skip connection
    if let skip = longSkipConnection, let res = residual {
      xSeq = skip(MLX.concatenated([xSeq, res], axis: -1))
    }

    // Final normalization and projection
    xSeq = normOut(xSeq, emb: tBatch)
    var output = projOut(xSeq)

    // Transpose back to channel-first
    output = output.swappedAxes(1, 2) // (B, mel_dim, N)

    return output
  }
}
