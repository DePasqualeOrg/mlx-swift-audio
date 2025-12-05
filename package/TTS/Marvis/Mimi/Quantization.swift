import Foundation
import MLX
import MLXNN

// MARK: - EuclideanCodebook

final class EuclideanCodebook: Module {
  private let epsilon: Float = 1e-5
  private let dim: Int

  var initialized: MLXArray
  @ModuleInfo(key: "embedding_sum") var embeddingSum: MLXArray
  @ModuleInfo(key: "cluster_usage") var clusterUsage: MLXArray

  private(set) var _embedding: MLXArray
  private(set) var _c2: MLXArray

  init(dim: Int, codebookSize: Int) {
    self.dim = dim
    initialized = MLXArray.zeros([1], dtype: .float32)
    let embeddingSumInit = MLXArray.zeros([codebookSize, dim], dtype: .float32)
    let clusterUsageInit = MLXArray.zeros([codebookSize], dtype: .float32)
    _embeddingSum.wrappedValue = embeddingSumInit
    _clusterUsage.wrappedValue = clusterUsageInit

    let clusterUsageSafe = maximum(clusterUsageInit, epsilon).reshaped([codebookSize, 1])
    _embedding = embeddingSumInit / clusterUsageSafe
    _c2 = _embedding.square().sum(axis: -1) / 2
  }

  func updateInPlace() {
    let clusterUsageSafe = maximum(clusterUsage, epsilon).reshaped([clusterUsage.shape[0], 1])
    _embedding = embeddingSum / clusterUsageSafe
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
  @ModuleInfo(key: "project_in") var projectIn: Linear?
  @ModuleInfo(key: "project_out") var projectOut: Linear?
  @ModuleInfo var codebook: EuclideanCodebook

  init(dim: Int, codebookSize: Int, codebookDim: Int?) {
    let cbDim = codebookDim ?? dim
    if dim == cbDim {
      _projectIn.wrappedValue = nil
      _projectOut.wrappedValue = nil
    } else {
      _projectIn.wrappedValue = Linear(dim, cbDim)
      _projectOut.wrappedValue = Linear(cbDim, dim)
    }
    _codebook.wrappedValue = EuclideanCodebook(dim: cbDim, codebookSize: codebookSize)
  }

  func encode(_ xs: MLXArray) -> MLXArray {
    var x = swappedAxes(xs, -1, -2)
    if let pin = projectIn { x = pin(x) }
    return codebook.encode(x)
  }

  func decode(_ xs: MLXArray) -> MLXArray {
    var x = codebook.decode(xs)
    if let pout = projectOut { x = pout(x) }
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
    _layers.wrappedValue = ls
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
  @ModuleInfo(key: "input_proj") var inputProj: MimiConv1d?
  @ModuleInfo(key: "output_proj") var outputProj: MimiConv1d?
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
      _inputProj.wrappedValue = nil
    } else {
      _inputProj.wrappedValue = MimiConv1d(inChannels: inDim, outChannels: dim, ksize: 1, bias: false)
    }
    if outDim == dim, !forceProjection {
      _outputProj.wrappedValue = nil
    } else {
      _outputProj.wrappedValue = MimiConv1d(inChannels: dim, outChannels: outDim, ksize: 1, bias: false)
    }
    _vq.wrappedValue = ResidualVectorQuantization(
      nq: nq, dim: dim, codebookSize: bins, codebookDim: nil,
    )
  }

  func encode(_ xs: MLXArray) -> MLXArray {
    var x = xs
    if let ip = inputProj { x = ip(x) }
    return swappedAxes(vq.encode(x), 0, 1)
  }

  func decode(_ xs: MLXArray) -> MLXArray {
    let x = swappedAxes(xs, 0, 1)
    var quantized = vq.decode(x)
    if let op = outputProj { quantized = op(quantized) }
    return quantized
  }
}

// MARK: - SplitResidualVectorQuantizer

final class SplitResidualVectorQuantizer: Module {
  private let nq: Int
  @ModuleInfo(key: "rvq_first") var rvqFirst: ResidualVectorQuantizer
  @ModuleInfo(key: "rvq_rest") var rvqRest: ResidualVectorQuantizer

  init(
    dim: Int,
    inputDim: Int?,
    outputDim: Int?,
    nq: Int,
    bins: Int,
  ) {
    self.nq = nq
    _rvqFirst.wrappedValue = ResidualVectorQuantizer(
      dim: dim, inputDim: inputDim, outputDim: outputDim,
      nq: 1, bins: bins, forceProjection: true,
    )
    _rvqRest.wrappedValue = ResidualVectorQuantizer(
      dim: dim, inputDim: inputDim, outputDim: outputDim,
      nq: max(nq - 1, 0), bins: bins, forceProjection: true,
    )
  }

  func encode(_ xs: MLXArray) -> MLXArray {
    var codes = rvqFirst.encode(xs)
    if nq > 1 {
      let rest = rvqRest.encode(xs)
      codes = concatenated([codes, rest], axis: 1)
    }
    return codes
  }

  func decode(_ xs: MLXArray) -> MLXArray {
    var quantized = rvqFirst.decode(xs[0 ..< xs.shape[0], 0 ..< 1])
    if nq > 1 {
      let rest = rvqRest.decode(xs[0 ..< xs.shape[0], 1...])
      quantized = quantized + rest
    }
    return quantized
  }
}
