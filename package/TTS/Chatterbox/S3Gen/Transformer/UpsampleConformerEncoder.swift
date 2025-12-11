//  Conformer encoder with upsampling for speech synthesis

import Foundation
import MLX
import MLXNN

// MARK: - Upsample1D

/// A 1D upsampling layer with an optional convolution
class Upsample1D: Module {
  let channels: Int
  let outChannels: Int
  let stride: Int

  @ModuleInfo(key: "conv") var conv: Conv1d

  init(channels: Int, outChannels: Int, stride: Int = 2) {
    self.channels = channels
    self.outChannels = outChannels
    self.stride = stride

    // First repeat interpolate, then conv with stride=1
    _conv.wrappedValue = Conv1d(
      inputChannels: channels,
      outputChannels: outChannels,
      kernelSize: stride * 2 + 1,
      stride: 1,
      padding: 0,
    )
  }

  func callAsFunction(_ inputs: MLXArray, inputLengths: MLXArray) -> (MLXArray, MLXArray) {
    // inputs: (B, C, T) - PyTorch format

    // Upsample using nearest neighbor interpolation (repeat each timestep)
    var outputs = MLX.repeated(inputs, count: stride, axis: 2)

    // Pad on the left
    outputs = MLX.padded(outputs, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((stride * 2, 0))])

    // Transpose to MLX format (B, T, C) for conv
    outputs = outputs.transposed(0, 2, 1)

    // Apply convolution
    outputs = conv(outputs)

    // Transpose back to PyTorch format (B, C, T)
    outputs = outputs.transposed(0, 2, 1)

    return (outputs, inputLengths * stride)
  }
}

// MARK: - PreLookaheadLayer

/// Pre-lookahead layer for causal processing
class PreLookaheadLayer: Module {
  let channels: Int
  let preLookaheadLen: Int

  @ModuleInfo(key: "conv1") var conv1: Conv1d
  @ModuleInfo(key: "conv2") var conv2: Conv1d

  init(channels: Int, preLookaheadLen: Int = 1) {
    self.channels = channels
    self.preLookaheadLen = preLookaheadLen

    _conv1.wrappedValue = Conv1d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: preLookaheadLen + 1,
      stride: 1,
      padding: 0,
    )
    _conv2.wrappedValue = Conv1d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: 3,
      stride: 1,
      padding: 0,
    )
  }

  func callAsFunction(_ inputs: MLXArray) -> MLXArray {
    // inputs: (B, T, C) - MLX format
    var outputs = inputs

    // Look ahead padding on time dimension
    outputs = MLX.padded(outputs, widths: [IntOrPair(0), IntOrPair((0, preLookaheadLen)), IntOrPair(0)])
    outputs = leakyRelu(conv1(outputs))

    // Output padding on time dimension
    outputs = MLX.padded(outputs, widths: [IntOrPair(0), IntOrPair((2, 0)), IntOrPair(0)])
    outputs = conv2(outputs)

    // Residual connection
    return outputs + inputs
  }
}

// MARK: - Mask Helpers

/// Make mask tensor containing indices of padded part
func makePadMask(lengths: MLXArray, maxLen: Int = 0) -> MLXArray {
  let batchSize = lengths.shape[0]
  let actualMaxLen = maxLen > 0 ? maxLen : Int(lengths.max().item(Int32.self))

  let seqRange = MLXArray(0 ..< actualMaxLen)
  let seqRangeExpand = MLX.broadcast(
    seqRange.expandedDimensions(axis: 0),
    to: [batchSize, actualMaxLen],
  )

  let seqLengthExpand = lengths.expandedDimensions(axis: -1)
  return seqRangeExpand .>= seqLengthExpand
}

/// Create mask for subsequent steps with chunk size (for streaming encoder)
func subsequentChunkMask(size: Int, chunkSize: Int, numLeftChunks _: Int = -1) -> MLXArray {
  let posIdx = MLXArray(0 ..< size)
  let blockValue = ((posIdx / chunkSize) + 1) * chunkSize
  let ret = posIdx.expandedDimensions(axis: 0) .< blockValue.expandedDimensions(axis: 1)
  return ret
}

/// Apply optional mask for encoder
func addOptionalChunkMask(
  xs: MLXArray,
  masks: MLXArray,
  useDynamicChunk: Bool,
  useDynamicLeftChunk: Bool,
  decodingChunkSize: Int,
  staticChunkSize: Int,
  numDecodingLeftChunks: Int,
  enableFullContext: Bool = true,
) -> MLXArray {
  var chunkMasks = masks

  if useDynamicChunk {
    let maxLen = xs.shape[1]
    var chunkSize: Int
    var numLeftChunks: Int

    if decodingChunkSize < 0 {
      chunkSize = maxLen
      numLeftChunks = -1
    } else if decodingChunkSize > 0 {
      chunkSize = decodingChunkSize
      numLeftChunks = numDecodingLeftChunks
    } else {
      // For training, use random chunk size
      chunkSize = Int.random(in: 1 ..< maxLen)
      numLeftChunks = -1
      if chunkSize > maxLen / 2, enableFullContext {
        chunkSize = maxLen
      } else {
        chunkSize = chunkSize % 25 + 1
        if useDynamicLeftChunk {
          let maxLeftChunks = (maxLen - 1) / chunkSize
          numLeftChunks = Int.random(in: 0 ... maxLeftChunks)
        }
      }
    }

    var chunkMaskLocal = subsequentChunkMask(size: xs.shape[1], chunkSize: chunkSize, numLeftChunks: numLeftChunks)
    chunkMaskLocal = chunkMaskLocal.expandedDimensions(axis: 0)
    chunkMasks = masks & chunkMaskLocal
  } else if staticChunkSize > 0 {
    var chunkMaskLocal = subsequentChunkMask(
      size: xs.shape[1],
      chunkSize: staticChunkSize,
      numLeftChunks: numDecodingLeftChunks,
    )
    chunkMaskLocal = chunkMaskLocal.expandedDimensions(axis: 0)
    chunkMasks = masks & chunkMaskLocal
  }

  // Check for all-false masks and fix them
  let maskSums = chunkMasks.sum(axis: -1)
  let anyZero = MLX.any(maskSums .== 0).item(Bool.self)
  if anyZero {
    let zeroMask = (maskSums .== 0).expandedDimensions(axis: -1)
    chunkMasks = MLX.where(zeroMask, MLXArray(true), chunkMasks)
  }

  return chunkMasks
}

// MARK: - UpsampleConformerEncoder

/// Conformer encoder with upsampling for speech synthesis
class UpsampleConformerEncoder: Module {
  let _outputSize: Int
  let normalizeBefore: Bool
  let staticChunkSize: Int
  let useDynamicChunk: Bool
  let useDynamicLeftChunk: Bool
  let numUpBlocks: Int
  let upsampleStride: Int

  @ModuleInfo(key: "embed") var embed: LinearNoSubsampling
  @ModuleInfo(key: "after_norm") var afterNorm: LayerNorm
  @ModuleInfo(key: "pre_lookahead_layer") var preLookaheadLayer: PreLookaheadLayer
  @ModuleInfo(key: "encoders") var encoders: [ConformerEncoderLayer]
  @ModuleInfo(key: "up_layer") var upLayer: Upsample1D
  @ModuleInfo(key: "up_embed") var upEmbed: LinearNoSubsampling
  @ModuleInfo(key: "up_encoders") var upEncoders: [ConformerEncoderLayer]

  init(
    inputSize: Int = 512,
    outputSize: Int = 512,
    attentionHeads: Int = 8,
    linearUnits: Int = 2048,
    numBlocks: Int = 6,
    numUpBlocks: Int = 4,
    dropoutRate: Float = 0.1,
    positionalDropoutRate: Float = 0.1,
    attentionDropoutRate: Float = 0.1,
    inputLayer: String = "linear",
    posEncLayerType _: String = "rel_pos_espnet",
    normalizeBefore: Bool = true,
    staticChunkSize: Int = 0,
    useDynamicChunk: Bool = false,
    useDynamicLeftChunk: Bool = false,
    positionwiseConvKernelSize _: Int = 1,
    macaronStyle: Bool = false,
    selfattentionLayerType: String = "rel_selfattn",
    activationType _: String = "swish",
    useCnnModule: Bool = false,
    cnnModuleKernel: Int = 15,
    causal: Bool = false,
    cnnModuleNorm: String = "batch_norm",
    keyBias: Bool = true,
    preLookaheadLen: Int = 3,
    upsampleStride: Int = 2
  ) {
    precondition(inputLayer == "linear", "Only linear input layer supported")

    _outputSize = outputSize
    self.normalizeBefore = normalizeBefore
    self.staticChunkSize = staticChunkSize
    self.useDynamicChunk = useDynamicChunk
    self.useDynamicLeftChunk = useDynamicLeftChunk
    self.numUpBlocks = numUpBlocks
    self.upsampleStride = upsampleStride

    // Input embedding layer
    _embed.wrappedValue = LinearNoSubsampling(
      idim: inputSize,
      odim: outputSize,
      dropoutRate: dropoutRate,
      posEncClass: RelPositionalEncoding(dModel: outputSize, dropoutRate: positionalDropoutRate),
    )

    _afterNorm.wrappedValue = LayerNorm(dimensions: outputSize, eps: 1e-5)

    // Activation function
    let activation: UnaryLayer = SiLU()

    // Pre-lookahead layer (use outputSize instead of hardcoded 512)
    _preLookaheadLayer.wrappedValue = PreLookaheadLayer(channels: outputSize, preLookaheadLen: preLookaheadLen)

    // Main encoder layers
    var encoderLayers: [ConformerEncoderLayer] = []
    for _ in 0 ..< numBlocks {
      let selfAttn: Module = if selfattentionLayerType == "rel_selfattn" {
        RelPositionMultiHeadedAttention(
          nHead: attentionHeads,
          nFeat: outputSize,
          dropoutRate: attentionDropoutRate,
          keyBias: keyBias,
        )
      } else {
        MultiHeadedAttention(
          nHead: attentionHeads,
          nFeat: outputSize,
          dropoutRate: attentionDropoutRate,
          keyBias: keyBias,
        )
      }

      let feedForward = PositionwiseFeedForward(
        idim: outputSize,
        hiddenUnits: linearUnits,
        dropoutRate: dropoutRate,
        activation: activation,
      )

      let feedForwardMacaron: PositionwiseFeedForward? = macaronStyle
        ? PositionwiseFeedForward(
          idim: outputSize,
          hiddenUnits: linearUnits,
          dropoutRate: dropoutRate,
          activation: activation,
        )
        : nil

      let convMod: ConvolutionModule? = useCnnModule
        ? ConvolutionModule(
          channels: outputSize,
          kernelSize: cnnModuleKernel,
          activation: activation,
          norm: cnnModuleNorm,
          causal: causal,
        )
        : nil

      encoderLayers.append(ConformerEncoderLayer(
        size: outputSize,
        selfAttn: selfAttn,
        feedForward: feedForward,
        feedForwardMacaron: feedForwardMacaron,
        convModule: convMod,
        dropoutRate: dropoutRate,
        normalizeBefore: normalizeBefore,
      ))
    }
    _encoders.wrappedValue = encoderLayers

    // Upsampling layer (use outputSize and upsampleStride instead of hardcoded values)
    _upLayer.wrappedValue = Upsample1D(channels: outputSize, outChannels: outputSize, stride: upsampleStride)

    // Upsampling embedding layer
    _upEmbed.wrappedValue = LinearNoSubsampling(
      idim: inputSize,
      odim: outputSize,
      dropoutRate: dropoutRate,
      posEncClass: RelPositionalEncoding(dModel: outputSize, dropoutRate: positionalDropoutRate),
    )

    // Upsampling encoder layers (use numUpBlocks instead of hardcoded 4)
    var upEncoderLayers: [ConformerEncoderLayer] = []
    for _ in 0 ..< numUpBlocks {
      let selfAttn: Module = if selfattentionLayerType == "rel_selfattn" {
        RelPositionMultiHeadedAttention(
          nHead: attentionHeads,
          nFeat: outputSize,
          dropoutRate: attentionDropoutRate,
          keyBias: keyBias,
        )
      } else {
        MultiHeadedAttention(
          nHead: attentionHeads,
          nFeat: outputSize,
          dropoutRate: attentionDropoutRate,
          keyBias: keyBias,
        )
      }

      let feedForward = PositionwiseFeedForward(
        idim: outputSize,
        hiddenUnits: linearUnits,
        dropoutRate: dropoutRate,
        activation: activation,
      )

      let feedForwardMacaron: PositionwiseFeedForward? = macaronStyle
        ? PositionwiseFeedForward(
          idim: outputSize,
          hiddenUnits: linearUnits,
          dropoutRate: dropoutRate,
          activation: activation,
        )
        : nil

      let convMod: ConvolutionModule? = useCnnModule
        ? ConvolutionModule(
          channels: outputSize,
          kernelSize: cnnModuleKernel,
          activation: activation,
          norm: cnnModuleNorm,
          causal: causal,
        )
        : nil

      upEncoderLayers.append(ConformerEncoderLayer(
        size: outputSize,
        selfAttn: selfAttn,
        feedForward: feedForward,
        feedForwardMacaron: feedForwardMacaron,
        convModule: convMod,
        dropoutRate: dropoutRate,
        normalizeBefore: normalizeBefore,
      ))
    }
    _upEncoders.wrappedValue = upEncoderLayers
  }

  func outputSize() -> Int {
    _outputSize
  }

  /// Embed positions in tensor
  /// - Parameters:
  ///   - xs: Input tensor (B, T, D)
  ///   - xsLens: Input lengths (B,)
  ///   - decodingChunkSize: Chunk size for decoding (0=random, <0=full, >0=fixed)
  ///   - numDecodingLeftChunks: Number of left chunks (<0=all)
  ///   - streaming: Whether to use streaming (chunk-based) attention.
  ///     When False (default), uses full context attention (effective_chunk_size=0).
  ///     When True, uses static_chunk_size for causal masking.
  func callAsFunction(
    _ xs: MLXArray,
    xsLens: MLXArray,
    decodingChunkSize: Int = 0,
    numDecodingLeftChunks: Int = -1,
    streaming: Bool = false
  ) -> (MLXArray, MLXArray) {
    var xsOut = xs
    var xsLensOut = xsLens
    let T = xsOut.shape[1]

    var masks = MLX.logicalNot(makePadMask(lengths: xsLensOut, maxLen: T))
    masks = masks.expandedDimensions(axis: 1) // (B, 1, T)

    var posEmb: MLXArray
    (xsOut, posEmb, masks) = embed(xsOut, xMask: masks, offset: 0)
    let maskPad = masks

    // Use static_chunk_size when streaming, otherwise full context (0)
    let effectiveChunkSize = streaming ? staticChunkSize : 0

    var chunkMasks = addOptionalChunkMask(
      xs: xsOut,
      masks: masks,
      useDynamicChunk: useDynamicChunk,
      useDynamicLeftChunk: useDynamicLeftChunk,
      decodingChunkSize: decodingChunkSize,
      staticChunkSize: effectiveChunkSize,
      numDecodingLeftChunks: numDecodingLeftChunks,
    )

    // Lookahead + conformer encoder
    xsOut = preLookaheadLayer(xsOut)
    xsOut = forwardLayers(xs: xsOut, chunkMasks: chunkMasks, posEmb: posEmb, maskPad: maskPad)

    // Upsample + conformer encoder
    xsOut = xsOut.transposed(0, 2, 1) // (B, D, T)
    (xsOut, xsLensOut) = upLayer(xsOut, inputLengths: xsLensOut)
    xsOut = xsOut.transposed(0, 2, 1) // (B, T', D)

    let TUp = xsOut.shape[1]
    masks = MLX.logicalNot(makePadMask(lengths: xsLensOut, maxLen: TUp))
    masks = masks.expandedDimensions(axis: 1)

    (xsOut, posEmb, masks) = upEmbed(xsOut, xMask: masks, offset: 0)
    let maskPadUp = masks

    // Scale effective chunk size by upsample stride for upsampled encoder
    let effectiveUpChunkSize = effectiveChunkSize * upLayer.stride

    chunkMasks = addOptionalChunkMask(
      xs: xsOut,
      masks: masks,
      useDynamicChunk: useDynamicChunk,
      useDynamicLeftChunk: useDynamicLeftChunk,
      decodingChunkSize: decodingChunkSize,
      staticChunkSize: effectiveUpChunkSize,
      numDecodingLeftChunks: numDecodingLeftChunks,
    )

    xsOut = forwardUpLayers(xs: xsOut, chunkMasks: chunkMasks, posEmb: posEmb, maskPad: maskPadUp)

    if normalizeBefore {
      xsOut = afterNorm(xsOut)
    }

    return (xsOut, masks)
  }

  /// Forward through main encoder layers
  private func forwardLayers(xs: MLXArray, chunkMasks: MLXArray, posEmb: MLXArray, maskPad: MLXArray) -> MLXArray {
    var xsOut = xs
    var masks = chunkMasks
    for layer in encoders {
      (xsOut, masks, _, _) = layer(xsOut, mask: masks, posEmb: posEmb, maskPad: maskPad)
    }
    return xsOut
  }

  /// Forward through upsampling encoder layers
  private func forwardUpLayers(xs: MLXArray, chunkMasks: MLXArray, posEmb: MLXArray, maskPad: MLXArray) -> MLXArray {
    var xsOut = xs
    var masks = chunkMasks
    for layer in upEncoders {
      (xsOut, masks, _, _) = layer(xsOut, mask: masks, posEmb: posEmb, maskPad: maskPad)
    }
    return xsOut
  }
}
