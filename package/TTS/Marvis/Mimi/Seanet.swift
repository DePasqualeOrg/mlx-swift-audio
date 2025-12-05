import Foundation
import MLX
import MLXNN

// MARK: - Config

struct SeanetConfig {
  let dimension: Int
  let channels: Int
  let causal: Bool
  let nfilters: Int
  let nresidualLayers: Int
  let ratios: [Int]
  let ksize: Int
  let residualKsize: Int
  let lastKsize: Int
  let dilationBase: Int
  let padMode: PadMode
  let trueSkip: Bool
  let compress: Int

  init(
    dimension: Int,
    channels: Int,
    causal: Bool,
    nfilters: Int,
    nresidualLayers: Int,
    ratios: [Int],
    ksize: Int,
    residualKsize: Int,
    lastKsize: Int,
    dilationBase: Int,
    padMode: PadMode,
    trueSkip: Bool,
    compress: Int,
  ) {
    self.dimension = dimension
    self.channels = channels
    self.causal = causal
    self.nfilters = nfilters
    self.nresidualLayers = nresidualLayers
    self.ratios = ratios
    self.ksize = ksize
    self.residualKsize = residualKsize
    self.lastKsize = lastKsize
    self.dilationBase = dilationBase
    self.padMode = padMode
    self.trueSkip = trueSkip
    self.compress = compress
  }
}

// MARK: - StreamingAdd

final class StreamingAdd: Module {
  private var lhsHold: MLXArray?
  private var rhsHold: MLXArray?

  override init() {}

  func step(lhs: MLXArray, rhs: MLXArray) -> MLXArray {
    var l = lhs
    var r = rhs

    if let h = lhsHold {
      l = concatenated([h, l], axis: 2)
      lhsHold = nil
    }
    if let h = rhsHold {
      r = concatenated([h, r], axis: 2)
      rhsHold = nil
    }

    let ll = l.shape[2]
    let rl = r.shape[2]

    if ll == rl {
      return l + r
    } else if ll < rl {
      let parts = split(r, indices: [ll], axis: 2)
      rhsHold = parts.count > 1 ? parts[1] : nil
      return l + parts[0]
    } else {
      let parts = split(l, indices: [rl], axis: 2)
      lhsHold = parts.count > 1 ? parts[1] : nil
      return parts[0] + r
    }
  }
}

// MARK: - SeanetResnetBlock

final class SeanetResnetBlock: Module {
  @ModuleInfo var block: [StreamableConv1d]
  @ModuleInfo(key: "streaming_add") var streamingAdd: StreamingAdd
  @ModuleInfo var shortcut: StreamableConv1d?

  init(config: SeanetConfig, dim: Int, ksizesAndDilations: [(Int, Int)]) {
    var layers: [StreamableConv1d] = []
    let hidden = dim / config.compress
    for (i, kd) in ksizesAndDilations.enumerated() {
      let (ksize, dilation) = kd
      let inC = (i == 0) ? dim : hidden
      let outC = (i == ksizesAndDilations.count - 1) ? dim : hidden
      layers.append(StreamableConv1d(
        inChannels: inC, outChannels: outC, ksize: ksize,
        stride: 1, dilation: dilation, groups: 1, bias: true,
        causal: config.causal, padMode: config.padMode,
      ))
    }
    _block.wrappedValue = layers
    _streamingAdd.wrappedValue = StreamingAdd()

    if config.trueSkip {
      _shortcut.wrappedValue = nil
    } else {
      _shortcut.wrappedValue = StreamableConv1d(
        inChannels: dim, outChannels: dim, ksize: 1,
        stride: 1, dilation: 1, groups: 1, bias: true,
        causal: config.causal, padMode: config.padMode,
      )
    }
  }

  func resetState() {
    shortcut?.resetState()
    for b in block {
      b.resetState()
    }
  }

  func callAsFunction(_ xs: MLXArray) -> MLXArray {
    var x = xs
    for b in block {
      x = b(elu(x, alpha: 1.0))
    }
    if let sc = shortcut {
      x = x + sc(xs)
    } else {
      x = x + xs
    }
    return x
  }

  func step(_ xs: MLXArray) -> MLXArray {
    var x = xs
    for b in block {
      x = b.step(elu(x, alpha: 1.0))
    }
    if let sc = shortcut {
      return streamingAdd.step(lhs: x, rhs: sc.step(xs))
    } else {
      return streamingAdd.step(lhs: x, rhs: xs)
    }
  }
}

// MARK: - EncoderLayer

final class EncoderLayer: Module {
  @ModuleInfo var residuals: [SeanetResnetBlock]
  @ModuleInfo var downsample: StreamableConv1d

  init(config: SeanetConfig, ratio: Int, mult: Int) {
    var res: [SeanetResnetBlock] = []
    var dilation = 1
    for _ in 0 ..< config.nresidualLayers {
      res.append(SeanetResnetBlock(
        config: config,
        dim: mult * config.nfilters,
        ksizesAndDilations: [(config.residualKsize, dilation), (1, 1)],
      ))
      dilation *= config.dilationBase
    }
    _residuals.wrappedValue = res

    // NOTE: causal = true here (matches python)
    _downsample.wrappedValue = StreamableConv1d(
      inChannels: mult * config.nfilters,
      outChannels: mult * config.nfilters * 2,
      ksize: ratio * 2,
      stride: ratio,
      dilation: 1,
      groups: 1,
      bias: true,
      causal: true,
      padMode: config.padMode,
    )
  }

  func resetState() {
    downsample.resetState()
    for r in residuals {
      r.resetState()
    }
  }

  func callAsFunction(_ xs: MLXArray) -> MLXArray {
    var x = xs
    for r in residuals {
      x = r(x)
    }
    return downsample(elu(x, alpha: 1.0))
  }

  func step(_ xs: MLXArray) -> MLXArray {
    var x = xs
    for r in residuals {
      x = r.step(x)
    }
    return downsample.step(elu(x, alpha: 1.0))
  }
}

// MARK: - SeanetEncoder

final class SeanetEncoder: Module {
  @ModuleInfo(key: "init_conv1d") var initConv1d: StreamableConv1d
  @ModuleInfo var layers: [EncoderLayer]
  @ModuleInfo(key: "final_conv1d") var finalConv1d: StreamableConv1d

  init(config: SeanetConfig) {
    var mult = 1

    _initConv1d.wrappedValue = StreamableConv1d(
      inChannels: config.channels, outChannels: mult * config.nfilters,
      ksize: config.ksize, stride: 1, dilation: 1, groups: 1, bias: true,
      causal: config.causal, padMode: config.padMode,
    )

    var encLayers: [EncoderLayer] = []
    for ratio in config.ratios.reversed() {
      encLayers.append(EncoderLayer(config: config, ratio: ratio, mult: mult))
      mult *= 2
    }
    _layers.wrappedValue = encLayers

    _finalConv1d.wrappedValue = StreamableConv1d(
      inChannels: mult * config.nfilters, outChannels: config.dimension,
      ksize: config.lastKsize, stride: 1, dilation: 1, groups: 1, bias: true,
      causal: config.causal, padMode: config.padMode,
    )
  }

  func resetState() {
    initConv1d.resetState()
    finalConv1d.resetState()
    for l in layers {
      l.resetState()
    }
  }

  func callAsFunction(_ xs: MLXArray) -> MLXArray {
    var x = initConv1d(xs)
    for l in layers {
      x = l(x)
    }
    x = elu(x, alpha: 1.0)
    return finalConv1d(x)
  }

  func step(_ xs: MLXArray) -> MLXArray {
    var x = initConv1d.step(xs)
    for l in layers {
      x = l.step(x)
    }
    x = elu(x, alpha: 1.0)
    return finalConv1d.step(x)
  }
}

// MARK: - DecoderLayer

final class DecoderLayer: Module {
  @ModuleInfo var upsample: StreamableConvTranspose1d
  @ModuleInfo var residuals: [SeanetResnetBlock]

  init(config: SeanetConfig, ratio: Int, mult: Int) {
    _upsample.wrappedValue = StreamableConvTranspose1d(
      inChannels: mult * config.nfilters,
      outChannels: mult * config.nfilters / 2,
      ksize: ratio * 2,
      stride: ratio,
      groups: 1,
      bias: true,
      causal: config.causal,
    )

    var res: [SeanetResnetBlock] = []
    var dilation = 1
    for _ in 0 ..< config.nresidualLayers {
      res.append(SeanetResnetBlock(
        config: config,
        dim: mult * config.nfilters / 2,
        ksizesAndDilations: [(config.residualKsize, dilation), (1, 1)],
      ))
      dilation *= config.dilationBase
    }
    _residuals.wrappedValue = res
  }

  func resetState() {
    upsample.resetState()
    for r in residuals {
      r.resetState()
    }
  }

  func callAsFunction(_ xs: MLXArray) -> MLXArray {
    var x = upsample(elu(xs, alpha: 1.0))
    for r in residuals {
      x = r(x)
    }
    return x
  }

  func step(_ xs: MLXArray) -> MLXArray {
    var x = upsample.step(elu(xs, alpha: 1.0))
    for r in residuals {
      x = r.step(x)
    }
    return x
  }
}

// MARK: - SeanetDecoder

final class SeanetDecoder: Module {
  @ModuleInfo(key: "init_conv1d") var initConv1d: StreamableConv1d
  @ModuleInfo var layers: [DecoderLayer]
  @ModuleInfo(key: "final_conv1d") var finalConv1d: StreamableConv1d

  init(config: SeanetConfig) {
    var mult = 1 << config.ratios.count

    _initConv1d.wrappedValue = StreamableConv1d(
      inChannels: config.dimension, outChannels: mult * config.nfilters,
      ksize: config.ksize, stride: 1, dilation: 1, groups: 1, bias: true,
      causal: config.causal, padMode: config.padMode,
    )

    var decLayers: [DecoderLayer] = []
    for ratio in config.ratios {
      decLayers.append(DecoderLayer(config: config, ratio: ratio, mult: mult))
      mult /= 2
    }
    _layers.wrappedValue = decLayers

    _finalConv1d.wrappedValue = StreamableConv1d(
      inChannels: config.nfilters, outChannels: config.channels,
      ksize: config.lastKsize, stride: 1, dilation: 1, groups: 1, bias: true,
      causal: config.causal, padMode: config.padMode,
    )
  }

  func resetState() {
    initConv1d.resetState()
    finalConv1d.resetState()
    for l in layers {
      l.resetState()
    }
  }

  func callAsFunction(_ xs: MLXArray) -> MLXArray {
    var x = initConv1d(xs)
    for l in layers {
      x = l(x)
    }
    x = elu(x, alpha: 1.0)
    return finalConv1d(x)
  }

  func step(_ xs: MLXArray) -> MLXArray {
    var x = initConv1d.step(xs)
    for l in layers {
      x = l.step(x)
    }
    x = elu(x, alpha: 1.0)
    return finalConv1d.step(x)
  }
}

// MARK: - Seanet

final class Seanet: Module {
  @ModuleInfo var encoder: SeanetEncoder
  @ModuleInfo var decoder: SeanetDecoder

  init(config: SeanetConfig) {
    _encoder.wrappedValue = SeanetEncoder(config: config)
    _decoder.wrappedValue = SeanetDecoder(config: config)
  }

  // Optional convenience funcs if you want them:
  func encode(_ xs: MLXArray) -> MLXArray { encoder(xs) }
  func decode(_ zs: MLXArray) -> MLXArray { decoder(zs) }
}
