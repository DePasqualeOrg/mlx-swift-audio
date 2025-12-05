//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertSelfAttention: Module {
  let numAttentionHeads: Int
  let attentionHeadSize: Int
  let allHeadSize: Int

  @ModuleInfo var query: Linear
  @ModuleInfo var key: Linear
  @ModuleInfo var value: Linear
  @ModuleInfo var dense: Linear
  @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
  let attentionDropout: Dropout

  init(config: AlbertConfig) {
    numAttentionHeads = config.numAttentionHeads
    attentionHeadSize = config.hiddenSize / config.numAttentionHeads
    allHeadSize = numAttentionHeads * attentionHeadSize
    attentionDropout = Dropout(p: config.attentionProbsDropoutProb)

    _query.wrappedValue = Linear(config.hiddenSize, allHeadSize)
    _key.wrappedValue = Linear(config.hiddenSize, allHeadSize)
    _value.wrappedValue = Linear(config.hiddenSize, allHeadSize)
    _dense.wrappedValue = Linear(allHeadSize, config.hiddenSize)
    _layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
  }

  func transposeForScores(_ x: MLXArray) -> MLXArray {
    let shape = x.shape
    var newShape: [Int] = []

    for i in 0 ..< (shape.count - 1) {
      newShape.append(shape[i])
    }

    newShape.append(numAttentionHeads)
    newShape.append(attentionHeadSize)

    let reshaped = x.reshaped(newShape)
    return reshaped.transposed(0, 2, 1, 3)
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil,
  ) -> MLXArray {
    let mixedQueryLayer = query(hiddenStates)
    let mixedKeyLayer = key(hiddenStates)
    let mixedValueLayer = value(hiddenStates)

    let queryLayer = transposeForScores(mixedQueryLayer)
    let keyLayer = transposeForScores(mixedKeyLayer)
    let valueLayer = transposeForScores(mixedValueLayer)

    let keyLayerTransposed = keyLayer.transposed(0, 1, 3, 2)
    var attentionScores = MLX.matmul(queryLayer, keyLayerTransposed)
    attentionScores = attentionScores / sqrt(Float(attentionHeadSize))

    if let attentionMask {
      attentionScores = attentionScores + attentionMask
    }

    var attentionProbs = MLX.softmax(attentionScores, axis: -1)
    attentionProbs = attentionDropout(attentionProbs)

    var contextLayer = MLX.matmul(attentionProbs, valueLayer)
    contextLayer = contextLayer.transposed(0, 2, 1, 3)

    var newContextLayerShape: [Int] = []
    let shape = contextLayer.shape

    for i in 0 ..< (shape.count - 2) {
      newContextLayerShape.append(shape[i])
    }

    newContextLayerShape.append(allHeadSize)

    contextLayer = contextLayer.reshaped(newContextLayerShape)
    contextLayer = dense(contextLayer)
    contextLayer = layerNorm(contextLayer + hiddenStates)

    return contextLayer
  }
}
