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
  @ModuleInfo(key: "streaming_add") var streamingAdd = StreamingAdd()
  @ModuleInfo var shortcut: StreamableConv1d?

  init(cfg: SeanetConfig, dim: Int, ksizesAndDilations: [(Int, Int)]) {
    var layers: [StreamableConv1d] = []
    let hidden = dim / cfg.compress
    for (i, kd) in ksizesAndDilations.enumerated() {
      let (ksize, dilation) = kd
      let inC = (i == 0) ? dim : hidden
      let outC = (i == ksizesAndDilations.count - 1) ? dim : hidden
      layers.append(StreamableConv1d(
        inChannels: inC, outChannels: outC, ksize: ksize,
        stride: 1, dilation: dilation, groups: 1, bias: true,
        causal: cfg.causal, padMode: cfg.padMode,
      ))
    }
    _block = ModuleInfo(wrappedValue: layers)

    if cfg.trueSkip {
      _shortcut = ModuleInfo(wrappedValue: nil)
    } else {
      _shortcut = ModuleInfo(wrappedValue: StreamableConv1d(
        inChannels: dim, outChannels: dim, ksize: 1,
        stride: 1, dilation: 1, groups: 1, bias: true,
        causal: cfg.causal, padMode: cfg.padMode,
      ))
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

  init(cfg: SeanetConfig, ratio: Int, mult: Int) {
    var res: [SeanetResnetBlock] = []
    var dilation = 1
    for _ in 0 ..< cfg.nresidualLayers {
      res.append(SeanetResnetBlock(
        cfg: cfg,
        dim: mult * cfg.nfilters,
        ksizesAndDilations: [(cfg.residualKsize, dilation), (1, 1)],
      ))
      dilation *= cfg.dilationBase
    }
    _residuals = ModuleInfo(wrappedValue: res)

    // NOTE: causal = true here (matches python)
    _downsample = ModuleInfo(wrappedValue: StreamableConv1d(
      inChannels: mult * cfg.nfilters,
      outChannels: mult * cfg.nfilters * 2,
      ksize: ratio * 2,
      stride: ratio,
      dilation: 1,
      groups: 1,
      bias: true,
      causal: true,
      padMode: cfg.padMode,
    ))
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
  @ModuleInfo var init_conv1d: StreamableConv1d
  @ModuleInfo var layers: [EncoderLayer]
  @ModuleInfo var final_conv1d: StreamableConv1d

  init(cfg: SeanetConfig) {
    var mult = 1

    _init_conv1d = ModuleInfo(wrappedValue: StreamableConv1d(
      inChannels: cfg.channels, outChannels: mult * cfg.nfilters,
      ksize: cfg.ksize, stride: 1, dilation: 1, groups: 1, bias: true,
      causal: cfg.causal, padMode: cfg.padMode,
    ))

    var encLayers: [EncoderLayer] = []
    for ratio in cfg.ratios.reversed() {
      encLayers.append(EncoderLayer(cfg: cfg, ratio: ratio, mult: mult))
      mult *= 2
    }
    _layers = ModuleInfo(wrappedValue: encLayers)

    _final_conv1d = ModuleInfo(wrappedValue: StreamableConv1d(
      inChannels: mult * cfg.nfilters, outChannels: cfg.dimension,
      ksize: cfg.lastKsize, stride: 1, dilation: 1, groups: 1, bias: true,
      causal: cfg.causal, padMode: cfg.padMode,
    ))
  }

  func resetState() {
    init_conv1d.resetState()
    final_conv1d.resetState()
    for l in layers {
      l.resetState()
    }
  }

  func callAsFunction(_ xs: MLXArray) -> MLXArray {
    var x = init_conv1d(xs)
    for l in layers {
      x = l(x)
    }
    x = elu(x, alpha: 1.0)
    return final_conv1d(x)
  }

  func step(_ xs: MLXArray) -> MLXArray {
    var x = init_conv1d.step(xs)
    for l in layers {
      x = l.step(x)
    }
    x = elu(x, alpha: 1.0)
    return final_conv1d.step(x)
  }
}

// MARK: - DecoderLayer

final class DecoderLayer: Module {
  @ModuleInfo var upsample: StreamableConvTranspose1d
  @ModuleInfo var residuals: [SeanetResnetBlock]

  init(cfg: SeanetConfig, ratio: Int, mult: Int) {
    _upsample = ModuleInfo(wrappedValue: StreamableConvTranspose1d(
      inChannels: mult * cfg.nfilters,
      outChannels: mult * cfg.nfilters / 2,
      ksize: ratio * 2,
      stride: ratio,
      groups: 1,
      bias: true,
      causal: cfg.causal,
    ))

    var res: [SeanetResnetBlock] = []
    var dilation = 1
    for _ in 0 ..< cfg.nresidualLayers {
      res.append(SeanetResnetBlock(
        cfg: cfg,
        dim: mult * cfg.nfilters / 2,
        ksizesAndDilations: [(cfg.residualKsize, dilation), (1, 1)],
      ))
      dilation *= cfg.dilationBase
    }
    _residuals = ModuleInfo(wrappedValue: res)
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
  @ModuleInfo var init_conv1d: StreamableConv1d
  @ModuleInfo var layers: [DecoderLayer]
  @ModuleInfo var final_conv1d: StreamableConv1d

  init(cfg: SeanetConfig) {
    var mult = 1 << cfg.ratios.count

    _init_conv1d = ModuleInfo(wrappedValue: StreamableConv1d(
      inChannels: cfg.dimension, outChannels: mult * cfg.nfilters,
      ksize: cfg.ksize, stride: 1, dilation: 1, groups: 1, bias: true,
      causal: cfg.causal, padMode: cfg.padMode,
    ))

    var decLayers: [DecoderLayer] = []
    for ratio in cfg.ratios {
      decLayers.append(DecoderLayer(cfg: cfg, ratio: ratio, mult: mult))
      mult /= 2
    }
    _layers = ModuleInfo(wrappedValue: decLayers)

    _final_conv1d = ModuleInfo(wrappedValue: StreamableConv1d(
      inChannels: cfg.nfilters, outChannels: cfg.channels,
      ksize: cfg.lastKsize, stride: 1, dilation: 1, groups: 1, bias: true,
      causal: cfg.causal, padMode: cfg.padMode,
    ))
  }

  func resetState() {
    init_conv1d.resetState()
    final_conv1d.resetState()
    for l in layers {
      l.resetState()
    }
  }

  func callAsFunction(_ xs: MLXArray) -> MLXArray {
    var x = init_conv1d(xs)
    for l in layers {
      x = l(x)
    }
    x = elu(x, alpha: 1.0)
    return final_conv1d(x)
  }

  func step(_ xs: MLXArray) -> MLXArray {
    var x = init_conv1d.step(xs)
    for l in layers {
      x = l.step(x)
    }
    x = elu(x, alpha: 1.0)
    return final_conv1d.step(x)
  }
}

// MARK: - Seanet

final class Seanet: Module {
  @ModuleInfo var encoder: SeanetEncoder
  @ModuleInfo var decoder: SeanetDecoder

  init(cfg: SeanetConfig) {
    _encoder = ModuleInfo(wrappedValue: SeanetEncoder(cfg: cfg))
    _decoder = ModuleInfo(wrappedValue: SeanetDecoder(cfg: cfg))
  }

  // Optional convenience funcs if you want them:
  func encode(_ xs: MLXArray) -> MLXArray { encoder(xs) }
  func decode(_ zs: MLXArray) -> MLXArray { decoder(zs) }
}
