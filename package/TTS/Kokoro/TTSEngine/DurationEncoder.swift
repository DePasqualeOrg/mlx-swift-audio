//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

/// DurationEncoder creates pairs of [LSTM, AdaLayerNorm]
/// The lstms array alternates: lstm at even indices, AdaLayerNorm at odd indices
/// Weight keys: predictor.text_encoder.lstms.0.*, predictor.text_encoder.lstms.1.fc.*, etc.
class DurationEncoder: Module {
  let nlayers: Int
  let dModel: Int
  let styDim: Int

  // Fixed 3 layers (nlayers=3 is hardcoded in Kokoro)
  // Even indices: LSTM, Odd indices: AdaLayerNorm
  @ModuleInfo var lstm0: BiLSTM
  @ModuleInfo var norm0: AdaLayerNorm
  @ModuleInfo var lstm1: BiLSTM
  @ModuleInfo var norm1: AdaLayerNorm
  @ModuleInfo var lstm2: BiLSTM
  @ModuleInfo var norm2: AdaLayerNorm

  init(dModel: Int, styDim: Int, nlayers: Int = 3) {
    precondition(nlayers == 3, "DurationEncoder currently only supports nlayers=3")
    self.nlayers = nlayers
    self.dModel = dModel
    self.styDim = styDim

    // LSTM layers (weight keys remapped in KokoroWeightLoader: lstms.0 → lstm0, etc.)
    _lstm0.wrappedValue = BiLSTM(inputSize: dModel + styDim, hiddenSize: dModel / 2)
    _lstm1.wrappedValue = BiLSTM(inputSize: dModel + styDim, hiddenSize: dModel / 2)
    _lstm2.wrappedValue = BiLSTM(inputSize: dModel + styDim, hiddenSize: dModel / 2)

    // AdaLayerNorm layers (weight keys remapped: lstms.1 → norm0, etc.)
    // Note: AdaLayerNorm's fc layer maps styDim -> dModel*2 (for gamma and beta)
    _norm0.wrappedValue = AdaLayerNorm(inputDim: styDim, outputDim: dModel * 2)
    _norm1.wrappedValue = AdaLayerNorm(inputDim: styDim, outputDim: dModel * 2)
    _norm2.wrappedValue = AdaLayerNorm(inputDim: styDim, outputDim: dModel * 2)
  }

  func callAsFunction(_ x: MLXArray, style: MLXArray, textLengths _: MLXArray, m: MLXArray) -> MLXArray {
    var x = x.transposed(2, 0, 1)
    let s = MLX.broadcast(style, to: [x.shape[0], x.shape[1], style.shape[style.shape.count - 1]])
    x = MLX.concatenated([x, s], axis: -1)
    x = MLX.where(m.expandedDimensions(axes: [-1]).transposed(1, 0, 2), MLXArray.zeros(like: x), x)
    x = x.transposed(1, 2, 0)

    // Process through alternating LSTM and AdaLayerNorm layers
    let lstms: [BiLSTM] = [lstm0, lstm1, lstm2]
    let norms: [AdaLayerNorm] = [norm0, norm1, norm2]

    for i in 0 ..< nlayers {
      // LSTM layer
      x = x.transposed(0, 2, 1)[0]
      let (lstmOutput, _) = lstms[i](x)
      x = lstmOutput.transposed(0, 2, 1)
      let xPad = MLXArray.zeros([x.shape[0], x.shape[1], m.shape[m.shape.count - 1]])
      xPad[0 ..< x.shape[0], 0 ..< x.shape[1], 0 ..< x.shape[2]] = x
      x = xPad

      // AdaLayerNorm layer
      x = norms[i](x.transposed(0, 2, 1), style).transposed(0, 2, 1)
      x = MLX.concatenated([x, s.transposed(1, 2, 0)], axis: 1)
      x = MLX.where(m.expandedDimensions(axes: [-1]).transposed(0, 2, 1), MLXArray.zeros(like: x), x)
    }

    return x.transposed(0, 2, 1)
  }
}
