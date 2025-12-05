//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AdaIN1d: Module {
  @ModuleInfo var norm: InstanceNorm1d
  @ModuleInfo var fc: Linear

  init(styleDim: Int = 0, numFeatures: Int = 0) {
    _norm.wrappedValue = InstanceNorm1d(numFeatures: numFeatures, affine: false)
    _fc.wrappedValue = Linear(styleDim, numFeatures * 2)
  }

  func callAsFunction(_ x: MLXArray, s: MLXArray) -> MLXArray {
    let h = fc(s)
    let hExpanded = h.expandedDimensions(axes: [2])
    let split = hExpanded.split(parts: 2, axis: 1)
    let gamma = split[0]
    let beta = split[1]

    let normalized = norm(x)
    return (1 + gamma) * normalized + beta
  }
}
