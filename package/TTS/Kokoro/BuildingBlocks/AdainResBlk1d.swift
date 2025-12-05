//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AdainResBlk1d: Module {
  let actv: LeakyReLU
  let dimIn: Int
  let upsampleType: String
  let upsample: UpSample1d
  let learnedSc: Bool
  let hasPool: Bool

  @ModuleInfo var conv1: ConvWeighted
  @ModuleInfo var conv2: ConvWeighted
  @ModuleInfo var norm1: AdaIN1d
  @ModuleInfo var norm2: AdaIN1d
  @ModuleInfo var pool: ConvWeighted?
  @ModuleInfo var conv1x1: ConvWeighted?

  init(
    dimIn: Int,
    dimOut: Int,
    styleDim: Int = 64,
    actv: LeakyReLU = LeakyReLU(negativeSlope: 0.2),
    upsample: String = "none",
  ) {
    self.actv = actv
    self.dimIn = dimIn
    upsampleType = upsample
    self.upsample = UpSample1d(layerType: upsample)
    learnedSc = dimIn != dimOut
    hasPool = upsample != "none"

    _conv1.wrappedValue = ConvWeighted(
      inChannels: dimIn,
      outChannels: dimOut,
      kernelSize: 3,
      stride: 1,
      padding: 1,
    )

    _conv2.wrappedValue = ConvWeighted(
      inChannels: dimOut,
      outChannels: dimOut,
      kernelSize: 3,
      stride: 1,
      padding: 1,
    )

    _norm1.wrappedValue = AdaIN1d(styleDim: styleDim, numFeatures: dimIn)
    _norm2.wrappedValue = AdaIN1d(styleDim: styleDim, numFeatures: dimOut)

    _pool.wrappedValue = hasPool ? ConvWeighted(
      inChannels: dimIn,
      outChannels: dimIn,
      kernelSize: 3,
      stride: 2,
      padding: 1,
      groups: dimIn,
    ) : nil

    _conv1x1.wrappedValue = learnedSc ? ConvWeighted(
      inChannels: dimIn,
      outChannels: dimOut,
      kernelSize: 1,
      stride: 1,
      padding: 0,
      bias: false,
    ) : nil
  }

  func shortcut(_ x: MLXArray) -> MLXArray {
    var x = x.swappedAxes(1, 2)
    x = upsample(x)
    x = x.swappedAxes(1, 2)

    if let conv1x1 {
      x = x.swappedAxes(1, 2)
      x = conv1x1(x, conv: MLX.conv1d)
      x = x.swappedAxes(1, 2)
    }

    return x
  }

  func residual(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
    var x = norm1(x, s: s)
    x = actv(x)

    x = x.swappedAxes(1, 2)
    if upsampleType != "none" {
      if let pool {
        x = pool.callAsFunction(x, conv: { a, b, c, d, e, f, g in
          MLX.convTransposed1d(a, b, stride: c, padding: d, dilation: e, outputPadding: 0, groups: f, stream: g)
        })
      }
      x = MLX.padded(x, widths: [IntOrPair([0, 0]), IntOrPair([1, 0]), IntOrPair([0, 0])])
    }
    x = x.swappedAxes(1, 2)

    x = x.swappedAxes(1, 2)
    x = conv1(x, conv: MLX.conv1d)
    x = x.swappedAxes(1, 2)

    x = norm2(x, s: s)
    x = actv(x)

    x = x.swappedAxes(1, 2)
    x = conv2(x, conv: MLX.conv1d)
    x = x.swappedAxes(1, 2)

    return x
  }

  func callAsFunction(x: MLXArray, s: MLXArray) -> MLXArray {
    let out = residual(x, s)
    let result = (out + shortcut(x)) / sqrt(2.0)
    return result
  }
}
