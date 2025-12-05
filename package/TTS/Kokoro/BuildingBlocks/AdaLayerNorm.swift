//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AdaLayerNorm: Module {
  let eps: Float
  @ModuleInfo var fc: Linear

  init(inputDim: Int = 0, outputDim: Int = 0, eps: Float = 1e-5) {
    self.eps = eps
    _fc.wrappedValue = Linear(inputDim, outputDim)
  }

  func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
    let h = fc(s)
    let reshaped = h.reshaped([h.shape[0], h.shape[1], 1])
    let split = reshaped.split(parts: 2, axis: 1)
    let gamma = split[0].transposed(2, 0, 1)
    let beta = split[1].transposed(2, 0, 1)

    let mean = MLX.mean(x, axes: [-1], keepDims: true)
    let variance = MLX.variance(x, axes: [-1], keepDims: true)
    let normalized = (x - mean) / MLX.sqrt(variance + eps)

    return (1 + gamma) * normalized + beta
  }
}
