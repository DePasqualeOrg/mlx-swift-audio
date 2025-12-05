//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

/// A single CNN block containing ConvWeighted, LayerNormInference, activation, and dropout
/// Weight keys are remapped in KokoroWeightLoader: 0 → conv, 1 → norm
class TextEncoderCNNBlock: Module {
  @ModuleInfo var conv: ConvWeighted
  @ModuleInfo var norm: LayerNormInference
  let actv: LeakyReLU
  let dropout: Dropout

  init(channels: Int, kernelSize: Int, padding: Int) {
    actv = LeakyReLU(negativeSlope: 0.2)
    dropout = Dropout(p: 0.2)

    _conv.wrappedValue = ConvWeighted(
      inChannels: channels,
      outChannels: channels,
      kernelSize: kernelSize,
      stride: 1,
      padding: padding,
    )
    _norm.wrappedValue = LayerNormInference(dims: channels)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var features = x

    // ConvWeighted expects [batch, seq, channels], outputs same
    features = features.swappedAxes(1, 2)
    features = conv(features, conv: MLX.conv1d)
    features = features.swappedAxes(1, 2)

    // LayerNormInference expects [batch, seq, channels]
    features = features.swappedAxes(1, 2)
    features = norm(features)
    features = features.swappedAxes(1, 2)

    features = actv(features)
    features = dropout(features)

    return features
  }
}

class TextEncoder: Module {
  @ModuleInfo var embedding: Embedding
  @ModuleInfo var cnn: [TextEncoderCNNBlock]
  @ModuleInfo var lstm: BiLSTM

  init(channels: Int, kernelSize: Int, depth: Int, nSymbols: Int) {
    let padding = (kernelSize - 1) / 2

    _embedding.wrappedValue = Embedding(embeddingCount: nSymbols, dimensions: channels)
    _cnn.wrappedValue = (0 ..< depth).map { _ in
      TextEncoderCNNBlock(channels: channels, kernelSize: kernelSize, padding: padding)
    }
    _lstm.wrappedValue = BiLSTM(inputSize: channels, hiddenSize: channels / 2)
  }

  func callAsFunction(_ x: MLXArray, inputLengths _: MLXArray, m: MLXArray) -> MLXArray {
    var features = embedding(x)
    features = features.transposed(0, 2, 1)
    let mask = m.expandedDimensions(axis: 1)
    features = MLX.where(mask, 0.0, features)

    for block in cnn {
      features = block(features)
      features = MLX.where(mask, 0.0, features)
    }

    features = features.swappedAxes(1, 2)
    let (lstmOutput, _) = lstm(features)
    features = lstmOutput.swappedAxes(1, 2)

    // Pad output to match mask size
    let maskLen = m.shape[m.shape.count - 1]
    if features.shape[features.shape.count - 1] < maskLen {
      let featuresPad = MLXArray.zeros([features.shape[0], features.shape[1], maskLen])
      featuresPad[0..., 0..., 0 ..< features.shape[features.shape.count - 1]] = features
      features = featuresPad
    }

    return MLX.where(mask, 0.0, features)
  }
}
