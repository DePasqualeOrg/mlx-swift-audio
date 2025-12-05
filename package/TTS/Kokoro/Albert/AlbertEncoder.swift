//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertEncoder: Module {
  let config: AlbertConfig

  @ModuleInfo(key: "embedding_hidden_mapping_in") var embeddingHiddenMappingIn: Linear
  @ModuleInfo(key: "albert_layer_groups") var albertLayerGroups: [AlbertLayerGroup]

  init(config: AlbertConfig) {
    self.config = config

    _embeddingHiddenMappingIn.wrappedValue = Linear(config.embeddingSize, config.hiddenSize)
    _albertLayerGroups.wrappedValue = (0 ..< config.numHiddenGroups).map { _ in
      AlbertLayerGroup(config: config)
    }
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil,
  ) -> MLXArray {
    var output = embeddingHiddenMappingIn(hiddenStates)

    for i in 0 ..< config.numHiddenLayers {
      let groupIdx = i / (config.numHiddenLayers / config.numHiddenGroups)

      output = albertLayerGroups[groupIdx](output, attentionMask: attentionMask)
    }

    return output
  }
}
