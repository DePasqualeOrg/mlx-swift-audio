//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

/// Bidirectional LSTM module with proper @ModuleInfo weight mapping
class BiLSTM: Module {
  let inputSize: Int
  let hiddenSize: Int
  let hasBias: Bool
  let batchFirst: Bool

  // Forward direction weights and biases
  @ModuleInfo(key: "weight_ih_l0") var weightIhL0: MLXArray
  @ModuleInfo(key: "weight_hh_l0") var weightHhL0: MLXArray
  @ModuleInfo(key: "bias_ih_l0") var biasIhL0: MLXArray?
  @ModuleInfo(key: "bias_hh_l0") var biasHhL0: MLXArray?

  // Backward direction weights and biases
  @ModuleInfo(key: "weight_ih_l0_reverse") var weightIhL0Reverse: MLXArray
  @ModuleInfo(key: "weight_hh_l0_reverse") var weightHhL0Reverse: MLXArray
  @ModuleInfo(key: "bias_ih_l0_reverse") var biasIhL0Reverse: MLXArray?
  @ModuleInfo(key: "bias_hh_l0_reverse") var biasHhL0Reverse: MLXArray?

  init(inputSize: Int, hiddenSize: Int, bias: Bool = true, batchFirst: Bool = true) {
    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    hasBias = bias
    self.batchFirst = batchFirst

    // Initialize with zeros - will be replaced by weight loading
    let weightShape = [4 * hiddenSize, inputSize]
    let hiddenWeightShape = [4 * hiddenSize, hiddenSize]
    let biasShape = [4 * hiddenSize]

    _weightIhL0.wrappedValue = MLXArray.zeros(weightShape)
    _weightHhL0.wrappedValue = MLXArray.zeros(hiddenWeightShape)
    _biasIhL0.wrappedValue = bias ? MLXArray.zeros(biasShape) : nil
    _biasHhL0.wrappedValue = bias ? MLXArray.zeros(biasShape) : nil

    _weightIhL0Reverse.wrappedValue = MLXArray.zeros(weightShape)
    _weightHhL0Reverse.wrappedValue = MLXArray.zeros(hiddenWeightShape)
    _biasIhL0Reverse.wrappedValue = bias ? MLXArray.zeros(biasShape) : nil
    _biasHhL0Reverse.wrappedValue = bias ? MLXArray.zeros(biasShape) : nil
  }

  /// Process sequence in forward direction
  private func forwardDirection(
    _ x: MLXArray,
    hidden: MLXArray? = nil,
    cell: MLXArray? = nil,
  ) -> (MLXArray, MLXArray) {
    // Pre-compute input projections
    let xProj: MLXArray = if let biasIhL0, let biasHhL0 {
      MLX.addMM(
        biasIhL0 + biasHhL0,
        x,
        weightIhL0.transposed(),
      )
    } else {
      MLX.matmul(x, weightIhL0.transposed())
    }

    var allHidden: [MLXArray] = []
    var allCell: [MLXArray] = []

    let seqLen = x.shape[x.shape.count - 2]

    var currentHidden = hidden ?? MLXArray.zeros([x.shape[0], hiddenSize])
    var currentCell = cell ?? MLXArray.zeros([x.shape[0], hiddenSize])

    // Process sequence in forward direction (0 to seqLen-1)
    for idx in 0 ..< seqLen {
      var ifgo = xProj[0..., idx, 0...]
      ifgo = ifgo + MLX.matmul(currentHidden, weightHhL0.transposed())

      // Split gates
      let gates = MLX.split(ifgo, parts: 4, axis: -1)
      let i = MLX.sigmoid(gates[0])
      let f = MLX.sigmoid(gates[1])
      let g = MLX.tanh(gates[2])
      let o = MLX.sigmoid(gates[3])

      // Update cell and hidden states
      currentCell = f * currentCell + i * g
      currentHidden = o * MLX.tanh(currentCell)

      allCell.append(currentCell)
      allHidden.append(currentHidden)
    }

    return (MLX.stacked(allHidden, axis: -2), MLX.stacked(allCell, axis: -2))
  }

  /// Process sequence in backward direction
  private func backwardDirection(
    _ x: MLXArray,
    hidden: MLXArray? = nil,
    cell: MLXArray? = nil,
  ) -> (MLXArray, MLXArray) {
    let xProj: MLXArray = if let biasIhL0Reverse, let biasHhL0Reverse {
      MLX.addMM(
        biasIhL0Reverse + biasHhL0Reverse,
        x,
        weightIhL0Reverse.transposed(),
      )
    } else {
      MLX.matmul(x, weightIhL0Reverse.transposed())
    }

    var allHidden: [MLXArray] = []
    var allCell: [MLXArray] = []

    let seqLen = x.shape[x.shape.count - 2]

    var currentHidden = hidden ?? MLXArray.zeros([x.shape[0], hiddenSize])
    var currentCell = cell ?? MLXArray.zeros([x.shape[0], hiddenSize])

    // Process sequence in backward direction (seqLen-1 to 0)
    for idx in stride(from: seqLen - 1, through: 0, by: -1) {
      var ifgo = xProj[0..., idx, 0...]
      ifgo = ifgo + MLX.matmul(currentHidden, weightHhL0Reverse.transposed())

      // Split gates
      let gates = MLX.split(ifgo, parts: 4, axis: -1)
      let i = MLX.sigmoid(gates[0])
      let f = MLX.sigmoid(gates[1])
      let g = MLX.tanh(gates[2])
      let o = MLX.sigmoid(gates[3])

      // Update cell and hidden states
      currentCell = f * currentCell + i * g
      currentHidden = o * MLX.tanh(currentCell)

      // Insert at beginning to maintain original sequence order
      allCell.insert(currentCell, at: 0)
      allHidden.insert(currentHidden, at: 0)
    }

    return (MLX.stacked(allHidden, axis: -2), MLX.stacked(allCell, axis: -2))
  }

  func callAsFunction(
    _ x: MLXArray,
    hiddenForward: MLXArray? = nil,
    cellForward: MLXArray? = nil,
    hiddenBackward: MLXArray? = nil,
    cellBackward: MLXArray? = nil,
  ) -> (MLXArray, ((MLXArray, MLXArray), (MLXArray, MLXArray))) {
    let input: MLXArray = if x.ndim == 2 {
      x.expandedDimensions(axis: 0) // (1, seq_len, input_size)
    } else {
      x
    }

    let (forwardHidden, forwardCell) = forwardDirection(
      input,
      hidden: hiddenForward,
      cell: cellForward,
    )

    let (backwardHidden, backwardCell) = backwardDirection(
      input,
      hidden: hiddenBackward,
      cell: cellBackward,
    )

    let output = MLX.concatenated([forwardHidden, backwardHidden], axis: -1)

    return (
      output,
      (
        (forwardHidden[0..., -1, 0...], forwardCell[0..., -1, 0...]),
        (backwardHidden[0..., 0, 0...], backwardCell[0..., 0, 0...]),
      ),
    )
  }
}
