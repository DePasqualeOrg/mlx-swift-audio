//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertLayerGroup: Module {
  @ModuleInfo(key: "albert_layers") var albertLayers: [AlbertLayer]

  init(config: AlbertConfig) {
    _albertLayers.wrappedValue = (0 ..< config.innerGroupNum).map { _ in AlbertLayer(config: config) }
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil,
  ) -> MLXArray {
    var output = hiddenStates
    for layer in albertLayers {
      output = layer(output, attentionMask: attentionMask)
    }
    return output
  }
}
