//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class CustomAlbert: Module {
  let config: AlbertConfig

  @ModuleInfo var embeddings: AlbertEmbeddings
  @ModuleInfo var encoder: AlbertEncoder
  @ModuleInfo var pooler: Linear

  init(config: AlbertConfig) {
    self.config = config

    _embeddings.wrappedValue = AlbertEmbeddings(config: config)
    _encoder.wrappedValue = AlbertEncoder(config: config)
    _pooler.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
  }

  func callAsFunction(
    _ inputIds: MLXArray,
    tokenTypeIds: MLXArray? = nil,
    attentionMask: MLXArray? = nil,
  ) -> (sequenceOutput: MLXArray, pooledOutput: MLXArray) {
    let embeddingOutput = embeddings(inputIds, tokenTypeIds: tokenTypeIds)

    var attentionMaskProcessed: MLXArray?
    if let attentionMask {
      let shape = attentionMask.shape
      let newDims = [shape[0], 1, 1, shape[1]]
      attentionMaskProcessed = attentionMask.reshaped(newDims)
      attentionMaskProcessed = (1.0 - attentionMaskProcessed!) * -10000.0
    }

    let sequenceOutput = encoder(embeddingOutput, attentionMask: attentionMaskProcessed)
    let firstTokenReshaped = sequenceOutput[0..., 0, 0...]
    let pooledOutput = MLX.tanh(pooler(firstTokenReshaped))

    return (sequenceOutput, pooledOutput)
  }
}
