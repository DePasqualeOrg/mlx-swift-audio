//
//  ChatterboxLSTM.swift
//  MLXAudio
//
//  Multi-layer unidirectional LSTM for Chatterbox VoiceEncoder.
//  Uses MLX weight format after sanitization from PyTorch.
//

import Foundation
import MLX
import MLXNN

// MARK: - LSTMCell

/// Single LSTM layer cell with MLX weight naming convention
class LSTMCell: Module {
  let hiddenSize: Int

  // MLX format weight names
  @ParameterInfo(key: "Wx") var Wx: MLXArray // Input weights
  @ParameterInfo(key: "Wh") var Wh: MLXArray // Hidden weights
  @ParameterInfo(key: "bias") var bias: MLXArray // Combined bias

  init(inputSize: Int, hiddenSize: Int) {
    self.hiddenSize = hiddenSize

    // Initialize weights (will be loaded from checkpoint)
    _Wx.wrappedValue = MLXArray.zeros([4 * hiddenSize, inputSize])
    _Wh.wrappedValue = MLXArray.zeros([4 * hiddenSize, hiddenSize])
    _bias.wrappedValue = MLXArray.zeros([4 * hiddenSize])
  }

  /// Process a single LSTM layer
  func callAsFunction(
    x: MLXArray,
    hidden: MLXArray?,
    cell: MLXArray?,
  ) -> (MLXArray, MLXArray, MLXArray) {
    let batchSize = x.shape[0]
    let seqLen = x.shape[1]

    var h = hidden ?? MLXArray.zeros([batchSize, hiddenSize])
    var c = cell ?? MLXArray.zeros([batchSize, hiddenSize])

    // Pre-compute input projections: (batch, seq_len, 4*hidden)
    let xProj = MLX.matmul(x, Wx.transposed()) + bias

    var allHidden: [MLXArray] = []

    for t in 0 ..< seqLen {
      let xT = xProj[0..., t, 0...]
      let hProj = MLX.matmul(h, Wh.transposed())
      let gates = xT + hProj

      let i = sigmoid(gates[0..., 0 ..< hiddenSize])
      let f = sigmoid(gates[0..., hiddenSize ..< 2 * hiddenSize])
      let g = tanh(gates[0..., 2 * hiddenSize ..< 3 * hiddenSize])
      let o = sigmoid(gates[0..., 3 * hiddenSize ..< 4 * hiddenSize])

      c = f * c + i * g
      h = o * tanh(c)

      allHidden.append(h)
    }

    let allH = MLX.stacked(allHidden, axis: 1)
    return (allH, h, c)
  }
}

// MARK: - ChatterboxLSTM

/// Multi-layer unidirectional LSTM that matches MLX's sanitized weight format
///
/// Weight naming follows MLX convention (after sanitization from PyTorch):
/// - layers.0.Wx, layers.0.Wh, layers.0.bias for layer 0
/// - layers.1.Wx, layers.1.Wh, layers.1.bias for layer 1
/// - layers.2.Wx, layers.2.Wh, layers.2.bias for layer 2
class ChatterboxLSTM: Module {
  let inputSize: Int
  let hiddenSize: Int
  let numLayers: Int

  // Use a layers array with MLX naming convention
  @ModuleInfo(key: "layers") var layers: [LSTMCell]

  init(inputSize: Int, hiddenSize: Int, numLayers: Int = 3) {
    precondition(numLayers == 3, "ChatterboxLSTM currently only supports 3 layers")

    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    self.numLayers = numLayers

    // Create layers: first layer takes inputSize, rest take hiddenSize
    var layerArray: [LSTMCell] = []
    for i in 0 ..< numLayers {
      let layerInputSize = i == 0 ? inputSize : hiddenSize
      layerArray.append(LSTMCell(inputSize: layerInputSize, hiddenSize: hiddenSize))
    }
    _layers.wrappedValue = layerArray
  }

  /// Process sequence through all LSTM layers
  /// - Parameters:
  ///   - x: Input tensor of shape (batch, seq_len, input_size)
  ///   - hidden: Optional tuple of (h0, c0) where each is (num_layers, batch, hidden_size)
  /// - Returns: Tuple of (output, (hN, cN))
  ///            output: (batch, seq_len, hidden_size)
  ///            hN, cN: (num_layers, batch, hidden_size)
  func callAsFunction(
    _ x: MLXArray,
    hidden: (MLXArray, MLXArray)? = nil,
  ) -> (MLXArray, (MLXArray, MLXArray)) {
    var hList: [MLXArray?]
    var cList: [MLXArray?]

    if let (h0, c0) = hidden {
      hList = (0 ..< numLayers).map { i in h0[i] }
      cList = (0 ..< numLayers).map { i in c0[i] }
    } else {
      hList = [MLXArray?](repeating: nil, count: numLayers)
      cList = [MLXArray?](repeating: nil, count: numLayers)
    }

    var currentOutput = x
    var finalHidden: [MLXArray] = []
    var finalCell: [MLXArray] = []

    for i in 0 ..< numLayers {
      let (out, hFinal, cFinal) = layers[i](
        x: currentOutput,
        hidden: hList[i],
        cell: cList[i],
      )
      currentOutput = out
      finalHidden.append(hFinal)
      finalCell.append(cFinal)
    }

    let hN = MLX.stacked(finalHidden, axis: 0)
    let cN = MLX.stacked(finalCell, axis: 0)

    return (currentOutput, (hN, cN))
  }
}
