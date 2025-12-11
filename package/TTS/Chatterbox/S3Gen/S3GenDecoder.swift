//  Conditional U-Net decoder for flow matching

import Foundation
import MLX
import MLXNN

// MARK: - CausalConv1d

/// Causal 1D convolution with left padding
class CausalConv1d: Module {
  @ModuleInfo(key: "conv") var conv: Conv1d
  let causalPadding: Int

  init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    dilation: Int = 1,
    bias: Bool = true,
  ) {
    precondition(stride == 1, "CausalConv1d only supports stride=1")
    _conv.wrappedValue = Conv1d(
      inputChannels: inChannels,
      outputChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: 0,
      dilation: dilation,
      bias: bias,
    )
    causalPadding = kernelSize - 1
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x: (B, C, T) - PyTorch format
    var out = x.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)
    // Pad on the left for causal convolution
    out = MLX.padded(out, widths: [IntOrPair(0), IntOrPair((causalPadding, 0)), IntOrPair(0)])
    out = conv(out)
    out = out.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)
    return out
  }
}

// MARK: - CausalBlock1D

/// Causal 1D block with LayerNorm
class CausalBlock1D: Module {
  @ModuleInfo(key: "conv") var conv: CausalConv1d
  @ModuleInfo(key: "norm") var norm: LayerNorm

  init(dim: Int, dimOut: Int) {
    _conv.wrappedValue = CausalConv1d(inChannels: dim, outChannels: dimOut, kernelSize: 3)
    _norm.wrappedValue = LayerNorm(dimensions: dimOut)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
    var output = conv(x * mask)
    // Transpose to (B, T, C), apply LayerNorm, transpose back
    output = output.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)
    output = norm(output)
    output = output.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)
    output = mish(output)
    return output * mask
  }
}

// MARK: - CausalResnetBlock1D

/// Causal ResNet block
class CausalResnetBlock1D: Module {
  @ModuleInfo(key: "mlp_linear") var mlpLinear: Linear
  @ModuleInfo(key: "block1") var block1: CausalBlock1D
  @ModuleInfo(key: "block2") var block2: CausalBlock1D
  @ModuleInfo(key: "res_conv") var resConv: Conv1d

  init(dim: Int, dimOut: Int, timeEmbDim: Int, groups _: Int = 8) {
    _mlpLinear.wrappedValue = Linear(timeEmbDim, dimOut)
    _block1.wrappedValue = CausalBlock1D(dim: dim, dimOut: dimOut)
    _block2.wrappedValue = CausalBlock1D(dim: dimOut, dimOut: dimOut)
    _resConv.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dimOut, kernelSize: 1)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> MLXArray {
    var h = block1(x, mask: mask)
    h = h + mlpLinear(mish(timeEmb)).expandedDimensions(axis: -1)
    h = block2(h, mask: mask)

    // res_conv
    var xRes = (x * mask).swappedAxes(1, 2)
    xRes = resConv(xRes)
    xRes = xRes.swappedAxes(1, 2)

    return h + xRes
  }
}

// MARK: - Block Containers

/// Container for down block components
class DownBlock: Module {
  @ModuleInfo(key: "resnet") var resnet: CausalResnetBlock1D
  @ModuleInfo(key: "transformers") var transformers: [BasicTransformerBlock]
  @ModuleInfo(key: "downsample") var downsample: Module

  init(resnet: CausalResnetBlock1D, transformerBlocks: [BasicTransformerBlock], downsample: Module) {
    _resnet.wrappedValue = resnet
    _transformers.wrappedValue = transformerBlocks
    _downsample.wrappedValue = downsample
  }
}

/// Container for mid block components
class MidBlock: Module {
  @ModuleInfo(key: "resnet") var resnet: CausalResnetBlock1D
  @ModuleInfo(key: "transformers") var transformers: [BasicTransformerBlock]

  init(resnet: CausalResnetBlock1D, transformerBlocks: [BasicTransformerBlock]) {
    _resnet.wrappedValue = resnet
    _transformers.wrappedValue = transformerBlocks
  }
}

/// Container for up block components
class UpBlock: Module {
  @ModuleInfo(key: "resnet") var resnet: CausalResnetBlock1D
  @ModuleInfo(key: "transformers") var transformers: [BasicTransformerBlock]
  @ModuleInfo(key: "upsample") var upsample: Module

  init(resnet: CausalResnetBlock1D, transformerBlocks: [BasicTransformerBlock], upsample: Module) {
    _resnet.wrappedValue = resnet
    _transformers.wrappedValue = transformerBlocks
    _upsample.wrappedValue = upsample
  }
}

// MARK: - ConditionalDecoder

/// Conditional U-Net decoder for flow matching
class ConditionalDecoder: Module {
  let inChannels: Int
  let outChannels: Int
  let causal: Bool
  let nDownBlocks: Int
  let nMidBlocks: Int
  let nUpBlocks: Int
  let staticChunkSize: Int
  let numDecodingLeftChunks: Int

  @ModuleInfo(key: "time_embeddings") var timeEmbeddings: SinusoidalPosEmb
  @ModuleInfo(key: "time_mlp") var timeMlp: TimestepEmbedding
  @ModuleInfo(key: "down_blocks") var downBlocks: [DownBlock]
  @ModuleInfo(key: "mid_blocks") var midBlocks: [MidBlock]
  @ModuleInfo(key: "up_blocks") var upBlocks: [UpBlock]
  @ModuleInfo(key: "final_block") var finalBlock: CausalBlock1D
  @ModuleInfo(key: "final_proj") var finalProj: Conv1d

  init(
    inChannels: Int = 320,
    outChannels: Int = 80,
    causal: Bool = true,
    channels: [Int] = [256],
    dropout: Float = 0.0,
    attentionHeadDim: Int = 64,
    nBlocks: Int = 4,
    numMidBlocks: Int = 12,
    numHeads: Int = 8,
    actFn: String = "gelu",
    staticChunkSize: Int = 50,
    numDecodingLeftChunks: Int = 2
  ) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.causal = causal
    self.staticChunkSize = staticChunkSize
    self.numDecodingLeftChunks = numDecodingLeftChunks

    // Time embeddings
    _timeEmbeddings.wrappedValue = SinusoidalPosEmb(dim: inChannels)
    let timeEmbedDim = channels[0] * 4
    _timeMlp.wrappedValue = TimestepEmbedding(inChannels: inChannels, timeEmbedDim: timeEmbedDim, actFn: "silu")

    // Down blocks
    var downBlocksArray: [DownBlock] = []
    var outputChannel = inChannels
    for (i, ch) in channels.enumerated() {
      let inputChannel = outputChannel
      outputChannel = ch
      let isLast = i == channels.count - 1

      let resnet = CausalResnetBlock1D(dim: inputChannel, dimOut: outputChannel, timeEmbDim: timeEmbedDim)

      var transformerBlocks: [BasicTransformerBlock] = []
      for _ in 0 ..< nBlocks {
        transformerBlocks.append(BasicTransformerBlock(
          dim: outputChannel,
          numAttentionHeads: numHeads,
          attentionHeadDim: attentionHeadDim,
          dropout: dropout,
          activationFn: actFn,
        ))
      }

      let downsample: Module = if !isLast {
        Downsample1D(dim: outputChannel)
      } else {
        CausalConv1d(inChannels: outputChannel, outChannels: outputChannel, kernelSize: 3)
      }

      downBlocksArray.append(DownBlock(resnet: resnet, transformerBlocks: transformerBlocks, downsample: downsample))
    }
    _downBlocks.wrappedValue = downBlocksArray
    nDownBlocks = channels.count

    // Mid blocks
    var midBlocksArray: [MidBlock] = []
    for _ in 0 ..< numMidBlocks {
      let resnet = CausalResnetBlock1D(dim: channels.last!, dimOut: channels.last!, timeEmbDim: timeEmbedDim)
      var transformerBlocks: [BasicTransformerBlock] = []
      for _ in 0 ..< nBlocks {
        transformerBlocks.append(BasicTransformerBlock(
          dim: channels.last!,
          numAttentionHeads: numHeads,
          attentionHeadDim: attentionHeadDim,
          dropout: dropout,
          activationFn: actFn,
        ))
      }
      midBlocksArray.append(MidBlock(resnet: resnet, transformerBlocks: transformerBlocks))
    }
    _midBlocks.wrappedValue = midBlocksArray
    nMidBlocks = numMidBlocks

    // Up blocks
    var upBlocksArray: [UpBlock] = []
    let channelsReversed = Array(channels.reversed()) + [channels[0]]
    for i in 0 ..< (channelsReversed.count - 1) {
      let inputChannel = channelsReversed[i] * 2
      let outCh = channelsReversed[i + 1]
      let isLast = i == channelsReversed.count - 2

      let resnet = CausalResnetBlock1D(dim: inputChannel, dimOut: outCh, timeEmbDim: timeEmbedDim)

      var transformerBlocks: [BasicTransformerBlock] = []
      for _ in 0 ..< nBlocks {
        transformerBlocks.append(BasicTransformerBlock(
          dim: outCh,
          numAttentionHeads: numHeads,
          attentionHeadDim: attentionHeadDim,
          dropout: dropout,
          activationFn: actFn,
        ))
      }

      let upsample: Module = if !isLast {
        MatchaUpsample1D(channels: outCh, useConvTranspose: true)
      } else {
        CausalConv1d(inChannels: outCh, outChannels: outCh, kernelSize: 3)
      }

      upBlocksArray.append(UpBlock(resnet: resnet, transformerBlocks: transformerBlocks, upsample: upsample))
    }
    _upBlocks.wrappedValue = upBlocksArray
    nUpBlocks = channelsReversed.count - 1

    // Final layers
    _finalBlock.wrappedValue = CausalBlock1D(dim: channelsReversed.last!, dimOut: channelsReversed.last!)
    _finalProj.wrappedValue = Conv1d(inputChannels: channelsReversed.last!, outputChannels: outChannels, kernelSize: 1)
  }

  func callAsFunction(
    x: MLXArray,
    mask: MLXArray,
    mu: MLXArray,
    t: MLXArray,
    spks: MLXArray? = nil,
    cond: MLXArray? = nil,
    streaming: Bool = false
  ) -> MLXArray {
    // Time embedding
    var tEmb = timeEmbeddings(t)
    tEmb = timeMlp(tEmb)

    // Concatenate conditioning
    var h = MLX.concatenated([x, mu], axis: 1)
    if let s = spks {
      let spksExpanded = MLX.broadcast(
        s.expandedDimensions(axis: -1),
        to: [s.shape[0], s.shape[1], h.shape[2]],
      )
      h = MLX.concatenated([h, spksExpanded], axis: 1)
    }
    if let c = cond {
      h = MLX.concatenated([h, c], axis: 1)
    }

    // Helper to compute attention bias from mask
    // streaming=false: full context attention (staticChunkSize=0)
    // streaming=true: chunk-based causal attention (staticChunkSize=self.staticChunkSize)
    func computeAttentionBias(_ hT: MLXArray, _ maskDown: MLXArray) -> MLXArray {
      // Use static_chunk_size when streaming, otherwise full context (0)
      let effectiveChunkSize = streaming ? staticChunkSize : 0

      var attnMask = addOptionalChunkMask(
        xs: hT,
        masks: maskDown.asType(.bool),
        useDynamicChunk: false,
        useDynamicLeftChunk: false,
        decodingChunkSize: 0,
        staticChunkSize: effectiveChunkSize,
        numDecodingLeftChunks: streaming ? numDecodingLeftChunks : -1
      )
      // Broadcast along query dimension for proper attention
      attnMask = MLX.broadcast(attnMask, to: [attnMask.shape[0], hT.shape[1], attnMask.shape[2]])
      // Convert boolean mask to additive bias
      var bias = maskToBias(attnMask, dtype: hT.dtype)
      // Add heads dimension for scaledDotProductAttention: (B, T, T) -> (B, 1, T, T)
      bias = bias.expandedDimensions(axis: 1)
      return bias
    }

    // Down blocks
    var hiddens: [MLXArray] = []
    var masks: [MLXArray] = [mask]
    for downBlock in downBlocks {
      let maskDown = masks.last!
      h = downBlock.resnet(h, mask: maskDown, timeEmb: tEmb)

      // Transformer blocks with attention mask
      var hT = h.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)
      let attnBias = computeAttentionBias(hT, maskDown)
      for transformer in downBlock.transformers {
        hT = transformer(hT, attentionMask: attnBias, timestep: tEmb)
      }
      h = hT.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)

      hiddens.append(h)

      // Apply downsample
      if let ds = downBlock.downsample as? Downsample1D {
        h = ds(h * maskDown)
      } else if let ds = downBlock.downsample as? CausalConv1d {
        h = ds(h * maskDown)
      }
      // Downsample mask
      masks.append(maskDown[0..., 0..., .stride(by: 2)])
    }

    masks.removeLast()
    let maskMid = masks.last!

    // Mid blocks
    for midBlock in midBlocks {
      h = midBlock.resnet(h, mask: maskMid, timeEmb: tEmb)

      var hT = h.swappedAxes(1, 2)
      let attnBias = computeAttentionBias(hT, maskMid)
      for transformer in midBlock.transformers {
        hT = transformer(hT, attentionMask: attnBias, timestep: tEmb)
      }
      h = hT.swappedAxes(1, 2)
    }

    // Up blocks
    var maskUp = mask
    for upBlock in upBlocks {
      maskUp = masks.removeLast()
      let skip = hiddens.removeLast()
      // Truncate h to match skip length
      h = MLX.concatenated([h[0..., 0..., 0 ..< skip.shape[2]], skip], axis: 1)
      h = upBlock.resnet(h, mask: maskUp, timeEmb: tEmb)

      var hT = h.swappedAxes(1, 2)
      let attnBias = computeAttentionBias(hT, maskUp)
      for transformer in upBlock.transformers {
        hT = transformer(hT, attentionMask: attnBias, timestep: tEmb)
      }
      h = hT.swappedAxes(1, 2)

      // Apply upsample
      if let us = upBlock.upsample as? MatchaUpsample1D {
        h = us(h * maskUp)
      } else if let us = upBlock.upsample as? CausalConv1d {
        h = us(h * maskUp)
      }
    }

    // Final layers
    h = finalBlock(h, mask: maskUp)
    var output = (h * maskUp).swappedAxes(1, 2)
    output = finalProj(output)
    output = output.swappedAxes(1, 2)
    return output * mask
  }
}
