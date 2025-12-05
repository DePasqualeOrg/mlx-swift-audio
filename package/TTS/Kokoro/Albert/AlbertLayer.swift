//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertLayer: Module {
  @ModuleInfo var attention: AlbertSelfAttention
  @ModuleInfo(key: "full_layer_layer_norm") var fullLayerLayerNorm: LayerNorm
  @ModuleInfo var ffn: Linear
  @ModuleInfo(key: "ffn_output") var ffnOutput: Linear
  let seqLenDim: Int

  init(config: AlbertConfig) {
    seqLenDim = 1

    _attention.wrappedValue = AlbertSelfAttention(config: config)
    _ffn.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
    _ffnOutput.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    _fullLayerLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
  }

  func ffChunk(_ attentionOutput: MLXArray) -> MLXArray {
    var ffnOutputArray = ffn(attentionOutput)
    ffnOutputArray = MLXNN.gelu(ffnOutputArray)
    ffnOutputArray = ffnOutput(ffnOutputArray)
    return ffnOutputArray
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil,
  ) -> MLXArray {
    let attentionOutput = attention(hiddenStates, attentionMask: attentionMask)
    let ffnOutput = ffChunk(attentionOutput)
    let output = fullLayerLayerNorm(ffnOutput + attentionOutput)
    return output
  }
}
