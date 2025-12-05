//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class Conv1dInference: Module {
  @ModuleInfo var weight: MLXArray
  @ModuleInfo var bias: MLXArray?

  let padding: Int
  let dilation: Int
  let stride: Int
  let groups: Int

  init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    groups: Int = 1,
    bias: Bool = true,
  ) {
    self.padding = padding
    self.dilation = dilation
    self.stride = stride
    self.groups = groups

    // Initialize with zeros - will be replaced by weight loading
    _weight.wrappedValue = MLXArray.zeros([outChannels, kernelSize, inChannels / groups])
    _bias.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
  }

  open func callAsFunction(_ x: MLXArray) -> MLXArray {
    var y = conv1d(
      x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups,
    )

    if let bias {
      y = y + bias
    }
    return y
  }
}
