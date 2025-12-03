import Foundation
import MLX
import MLXNN

// MARK: - EuclideanCodebook

final class EuclideanCodebook: Module {
  private let epsilon: Float = 1e-5
  private let dim: Int

  var initialized: MLXArray
  var embedding_sum: MLXArray
  var cluster_usage: MLXArray

  private(set) var _embedding: MLXArray
  private(set) var _c2: MLXArray

  init(dim: Int, codebookSize: Int) {
    self.dim = dim
    initialized = MLXArray.zeros([1], dtype: .float32)
    embedding_sum = MLXArray.zeros([codebookSize, dim], dtype: .float32)
    cluster_usage = MLXArray.zeros([codebookSize], dtype: .float32)

    let cluster_usageSafe = maximum(cluster_usage, epsilon).reshaped([codebookSize, 1])
    _embedding = embedding_sum / cluster_usageSafe
    _c2 = _embedding.square().sum(axis: -1) / 2
  }

  func updateInPlace() {
    let cluster_usageSafe = maximum(cluster_usage, epsilon).reshaped([cluster_usage.shape[0], 1])
    _embedding = embedding_sum / cluster_usageSafe
    _c2 = _embedding.square().sum(axis: -1) / 2
  }

  override func update(parameters: ModuleParameters, verify: Module.VerifyUpdate, path: [String] = [], modulePath: [String] = []) throws -> Self {
    try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    updateInPlace()
    return self
  }

  func encode(_ xs: MLXArray) -> MLXArray {
    let targetShape = Array(xs.shape.dropLast())
    let flat = xs.reshaped([-1, dim])
    let dotProd = flat.matmul(swappedAxes(_embedding, -1, -2))
    let dists = _c2 - dotProd
    return argMin(dists, axis: -1).reshaped(targetShape)
  }

  func decode(_ xs: MLXArray) -> MLXArray {
    let targetShape = xs.shape + [dim]
    let taken = take(_embedding, xs.flattened(), axis: 0)
    return taken.reshaped(targetShape)
  }
}

// MARK: - VectorQuantization

final class VectorQuantization: Module {
  @ModuleInfo var project_in: Linear?
  @ModuleInfo var project_out: Linear?
  @ModuleInfo var codebook: EuclideanCodebook

  init(dim: Int, codebookSize: Int, codebookDim: Int?) {
    let cbDim = codebookDim ?? dim
    if dim == cbDim {
      _project_in = ModuleInfo(wrappedValue: nil)
      _project_out = ModuleInfo(wrappedValue: nil)
    } else {
      _project_in = ModuleInfo(wrappedValue: Linear(dim, cbDim))
      _project_out = ModuleInfo(wrappedValue: Linear(cbDim, dim))
    }
    _codebook = ModuleInfo(wrappedValue: EuclideanCodebook(dim: cbDim, codebookSize: codebookSize))
  }

  func encode(_ xs: MLXArray) -> MLXArray {
    var x = swappedAxes(xs, -1, -2)
    if let pin = project_in { x = pin(x) }
    return codebook.encode(x)
  }

  func decode(_ xs: MLXArray) -> MLXArray {
    var x = codebook.decode(xs)
    if let pout = project_out { x = pout(x) }
    return swappedAxes(x, -1, -2)
  }
}

// MARK: - ResidualVectorQuantization

final class ResidualVectorQuantization: Module {
  @ModuleInfo var layers: [VectorQuantization]

  init(nq: Int, dim: Int, codebookSize: Int, codebookDim: Int?) {
    var ls: [VectorQuantization] = []
    for _ in 0 ..< nq {
      ls.append(VectorQuantization(dim: dim, codebookSize: codebookSize, codebookDim: codebookDim))
    }
    _layers = ModuleInfo(wrappedValue: ls)
  }

  func encode(_ xs: MLXArray) -> MLXArray {
    var codes: [MLXArray] = []
    var residual = xs
    for layer in layers {
      let indices = layer.encode(residual)
      let quantized = layer.decode(indices)
      residual = residual - quantized
      codes.append(indices)
    }
    return stacked(codes, axis: 0)
  }

  func decode(_ xs: MLXArray) -> MLXArray {
    let seqLen = xs.shape[0]
    var quantized = layers[0].decode(xs[0])
    for i in 1 ..< seqLen {
      quantized = quantized + layers[i].decode(xs[i])
    }
    return quantized
  }
}

// MARK: - ResidualVectorQuantizer

final class ResidualVectorQuantizer: Module {
  @ModuleInfo var input_proj: Conv1d?
  @ModuleInfo var output_proj: Conv1d?
  @ModuleInfo var vq: ResidualVectorQuantization

  init(
    dim: Int,
    inputDim: Int?,
    outputDim: Int?,
    nq: Int,
    bins: Int,
    forceProjection: Bool,
  ) {
    let inDim = inputDim ?? dim
    let outDim = outputDim ?? dim
    if inDim == dim, !forceProjection {
      _input_proj = ModuleInfo(wrappedValue: nil)
    } else {
      _input_proj = ModuleInfo(wrappedValue: Conv1d(inChannels: inDim, outChannels: dim, ksize: 1, bias: false))
    }
    if outDim == dim, !forceProjection {
      _output_proj = ModuleInfo(wrappedValue: nil)
    } else {
      _output_proj = ModuleInfo(wrappedValue: Conv1d(inChannels: dim, outChannels: outDim, ksize: 1, bias: false))
    }
    _vq = ModuleInfo(wrappedValue: ResidualVectorQuantization(
      nq: nq, dim: dim, codebookSize: bins, codebookDim: nil,
    ))
  }

  func encode(_ xs: MLXArray) -> MLXArray {
    var x = xs
    if let ip = input_proj { x = ip(x) }
    return swappedAxes(vq.encode(x), 0, 1)
  }

  func decode(_ xs: MLXArray) -> MLXArray {
    let x = swappedAxes(xs, 0, 1)
    var quantized = vq.decode(x)
    if let op = output_proj { quantized = op(quantized) }
    return quantized
  }
}

// MARK: - SplitResidualVectorQuantizer

final class SplitResidualVectorQuantizer: Module {
  private let nq: Int
  @ModuleInfo var rvq_first: ResidualVectorQuantizer
  @ModuleInfo var rvq_rest: ResidualVectorQuantizer

  init(
    dim: Int,
    inputDim: Int?,
    outputDim: Int?,
    nq: Int,
    bins: Int,
  ) {
    self.nq = nq
    _rvq_first = ModuleInfo(wrappedValue: ResidualVectorQuantizer(
      dim: dim, inputDim: inputDim, outputDim: outputDim,
      nq: 1, bins: bins, forceProjection: true,
    ))
    _rvq_rest = ModuleInfo(wrappedValue: ResidualVectorQuantizer(
      dim: dim, inputDim: inputDim, outputDim: outputDim,
      nq: max(nq - 1, 0), bins: bins, forceProjection: true,
    ))
  }

  func encode(_ xs: MLXArray) -> MLXArray {
    var codes = rvq_first.encode(xs)
    if nq > 1 {
      let rest = rvq_rest.encode(xs)
      codes = concatenated([codes, rest], axis: 1)
    }
    return codes
  }

  func decode(_ xs: MLXArray) -> MLXArray {
    var quantized = rvq_first.decode(xs[0 ..< xs.shape[0], 0 ..< 1])
    if nq > 1 {
      let rest = rvq_rest.decode(xs[0 ..< xs.shape[0], 1...])
      quantized = quantized + rest
    }
    return quantized
  }
}
