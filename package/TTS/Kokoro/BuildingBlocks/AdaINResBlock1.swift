//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AdaINResBlock1: Module {
  @ModuleInfo var convs1: [ConvWeighted]
  @ModuleInfo var convs2: [ConvWeighted]
  @ModuleInfo var adain1: [AdaIN1d]
  @ModuleInfo var adain2: [AdaIN1d]
  @ModuleInfo var alpha1: [MLXArray]
  @ModuleInfo var alpha2: [MLXArray]

  let kernelSize: Int
  let dilations: [Int]

  private func getPadding(kernelSize: Int, dilation: Int = 1) -> Int {
    Int((kernelSize * dilation - dilation) / 2)
  }

  init(
    channels: Int,
    kernelSize: Int = 3,
    dilation: [Int] = [1, 3, 5],
    styleDim: Int = 64,
  ) {
    self.kernelSize = kernelSize
    dilations = dilation

    var c1: [ConvWeighted] = []
    var c2: [ConvWeighted] = []
    var a1: [AdaIN1d] = []
    var a2: [AdaIN1d] = []
    var al1: [MLXArray] = []
    var al2: [MLXArray] = []

    for i in 0 ..< 3 {
      let dilationValue = dilation[i]
      c1.append(ConvWeighted(
        inChannels: channels,
        outChannels: channels,
        kernelSize: kernelSize,
        stride: 1,
        padding: Int((kernelSize * dilationValue - dilationValue) / 2),
        dilation: dilationValue,
      ))

      c2.append(ConvWeighted(
        inChannels: channels,
        outChannels: channels,
        kernelSize: kernelSize,
        stride: 1,
        padding: Int((kernelSize - 1) / 2),
        dilation: 1,
      ))

      a1.append(AdaIN1d(styleDim: styleDim, numFeatures: channels))
      a2.append(AdaIN1d(styleDim: styleDim, numFeatures: channels))

      al1.append(MLXArray.ones([1]))
      al2.append(MLXArray.ones([1]))
    }

    _convs1.wrappedValue = c1
    _convs2.wrappedValue = c2
    _adain1.wrappedValue = a1
    _adain2.wrappedValue = a2
    _alpha1.wrappedValue = al1
    _alpha2.wrappedValue = al2
  }

  func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
    var result = x

    for i in 0 ..< convs1.count {
      let c1 = convs1[i]
      let c2 = convs2[i]
      let n1 = adain1[i]
      let n2 = adain2[i]
      let a1 = alpha1[i]
      let a2 = alpha2[i]

      var xt = n1(result, s: s)
      xt = xt + (1 / a1) * (MLX.sin(a1 * xt).pow(2))

      xt = xt.swappedAxes(1, 2)
      xt = c1(xt, conv: MLX.conv1d)
      xt = xt.swappedAxes(1, 2)

      xt = n2(xt, s: s)
      xt = xt + (1 / a2) * (MLX.sin(a2 * xt).pow(2))

      xt = xt.swappedAxes(1, 2)
      xt = c2(xt, conv: MLX.conv1d)
      xt = xt.swappedAxes(1, 2)

      result = xt + result
    }
    return result
  }
}
