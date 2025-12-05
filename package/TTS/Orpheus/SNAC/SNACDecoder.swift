import Foundation
import Hub
import MLX
import MLXNN

// MARK: - SNAC Decoder Model

/// SNAC Decoder - converts audio codes to waveform
/// Uses standard MLX Module pattern with semantic weight key remapping
class SNACDecoder: Module {
  // Initial convolutions
  @ModuleInfo var depthwiseConv: WNConv1d
  @ModuleInfo var pointwiseConv: WNConv1d

  // Decoder blocks
  @ModuleInfo var block0: SNACDecoderBlock
  @ModuleInfo var block1: SNACDecoderBlock
  @ModuleInfo var block2: SNACDecoderBlock
  @ModuleInfo var block3: SNACDecoderBlock

  // Final layers
  @ModuleInfo var finalSnake: SNACSnake
  @ModuleInfo var finalConv: WNConv1d

  let config: SNACConfig
  private var quantizerWeights: [String: MLXArray] = [:]

  static let defaultRepoId = "mlx-community/snac_24khz"
  static let defaultWeightsFilename = "model.safetensors"

  init(config: SNACConfig) {
    self.config = config

    // Initial convolutions
    _depthwiseConv.wrappedValue = WNConv1d(
      inChannels: config.latentDim,
      outChannels: config.latentDim,
      kernelSize: 7,
      padding: 3,
      groups: config.latentDim,
      bias: true,
    )

    _pointwiseConv.wrappedValue = WNConv1d(
      inChannels: config.latentDim,
      outChannels: config.decoderDim,
      kernelSize: 1,
      padding: 0,
      bias: true,
    )

    // Decoder blocks - calculate dimensions for each
    var currentDim = config.decoderDim
    let dims = (0 ..< 4).map { i -> (input: Int, output: Int, stride: Int, groups: Int) in
      let inputDim = currentDim
      let outputDim = config.decoderDim / Int(pow(2.0, Double(i + 1)))
      let stride = config.decoderRates[i]
      let groups = config.depthwise ? outputDim : 1
      currentDim = outputDim
      return (inputDim, outputDim, stride, groups)
    }

    _block0.wrappedValue = SNACDecoderBlock(
      inputDim: dims[0].input, outputDim: dims[0].output, stride: dims[0].stride,
      groups: dims[0].groups, noise: config.noise,
    )
    _block1.wrappedValue = SNACDecoderBlock(
      inputDim: dims[1].input, outputDim: dims[1].output, stride: dims[1].stride,
      groups: dims[1].groups, noise: config.noise,
    )
    _block2.wrappedValue = SNACDecoderBlock(
      inputDim: dims[2].input, outputDim: dims[2].output, stride: dims[2].stride,
      groups: dims[2].groups, noise: config.noise,
    )
    _block3.wrappedValue = SNACDecoderBlock(
      inputDim: dims[3].input, outputDim: dims[3].output, stride: dims[3].stride,
      groups: dims[3].groups, noise: config.noise,
    )

    // Final layers
    let finalDim = config.decoderDim / Int(pow(2.0, Double(config.decoderRates.count)))
    _finalSnake.wrappedValue = SNACSnake(channels: finalDim)
    _finalConv.wrappedValue = WNConv1d(
      inChannels: finalDim,
      outChannels: 1,
      kernelSize: 7,
      padding: 3,
      bias: true,
    )
  }

  /// Remap weight keys from original structure to clean semantic names
  /// Original: decoder.model.layers.2.block.layers.0.alpha
  /// Remapped: block0.snake.alpha
  static func sanitizeWeights(_ weights: [String: MLXArray], noise: Bool = true) -> (decoder: [String: MLXArray], quantizer: [String: MLXArray]) {
    var decoderWeights: [String: MLXArray] = [:]
    var quantizerWeights: [String: MLXArray] = [:]

    let prefix = "decoder.model.layers."

    for (key, value) in weights {
      if key.hasPrefix(prefix) {
        let remainder = String(key.dropFirst(prefix.count))
        if let remappedKey = remapWeightKey(remainder, noise: noise) {
          decoderWeights[remappedKey] = value
        }
      } else if key.hasPrefix("quantizer.") {
        quantizerWeights[key] = value
      }
    }

    return (decoderWeights, quantizerWeights)
  }

  /// Remap a single weight key from indexed structure to semantic names
  private static func remapWeightKey(_ key: String, noise: Bool) -> String? {
    let parts = key.split(separator: ".").map(String.init)
    guard let firstIndex = Int(parts[0]) else { return nil }

    let rest = Array(parts.dropFirst())

    switch firstIndex {
      case 0:
        // depthwiseConv
        return "depthwiseConv." + rest.joined(separator: ".")

      case 1:
        // pointwiseConv
        return "pointwiseConv." + rest.joined(separator: ".")

      case 2, 3, 4, 5:
        // Decoder blocks (indices 2-5 map to block0-block3)
        let blockIndex = firstIndex - 2
        return remapBlockKey(rest, blockIndex: blockIndex, noise: noise)

      case 6:
        // finalSnake
        return "finalSnake." + rest.joined(separator: ".")

      case 7:
        // finalConv
        return "finalConv." + rest.joined(separator: ".")

      default:
        return nil
    }
  }

  /// Remap keys within a decoder block
  /// Input: ["block", "layers", "0", "alpha"] for block.layers.0.alpha
  /// Output: block0.snake.alpha
  private static func remapBlockKey(_ parts: [String], blockIndex: Int, noise: Bool) -> String? {
    // Expected structure: block.layers.<index>.<rest>
    guard parts.count >= 3,
          parts[0] == "block",
          parts[1] == "layers",
          let layerIndex = Int(parts[2])
    else {
      return nil
    }

    let rest = Array(parts.dropFirst(3))
    let blockPrefix = "block\(blockIndex)"

    // Layer indices within a block:
    // 0: snake
    // 1: convT
    // 2: noiseBlock (if noise=true) or residual0 (if noise=false)
    // 3: residual0 (if noise=true) or residual1 (if noise=false)
    // 4: residual1 (if noise=true) or residual2 (if noise=false)
    // 5: residual2 (if noise=true only)

    if noise {
      switch layerIndex {
        case 0:
          return "\(blockPrefix).snake." + rest.joined(separator: ".")
        case 1:
          return "\(blockPrefix).convT." + rest.joined(separator: ".")
        case 2:
          // noiseBlock has a .linear submodule
          return "\(blockPrefix).noiseBlock." + rest.joined(separator: ".")
        case 3, 4, 5:
          let residualIndex = layerIndex - 3
          return remapResidualKey(rest, blockPrefix: blockPrefix, residualIndex: residualIndex)
        default:
          return nil
      }
    } else {
      switch layerIndex {
        case 0:
          return "\(blockPrefix).snake." + rest.joined(separator: ".")
        case 1:
          return "\(blockPrefix).convT." + rest.joined(separator: ".")
        case 2, 3, 4:
          let residualIndex = layerIndex - 2
          return remapResidualKey(rest, blockPrefix: blockPrefix, residualIndex: residualIndex)
        default:
          return nil
      }
    }
  }

  /// Remap keys within a residual unit
  /// Input: ["block", "layers", "0", "alpha"] for block.layers.0.alpha
  /// Output: block0.residual0.snake1.alpha
  private static func remapResidualKey(_ parts: [String], blockPrefix: String, residualIndex: Int) -> String? {
    // Expected structure: block.layers.<index>.<rest>
    guard parts.count >= 3,
          parts[0] == "block",
          parts[1] == "layers",
          let layerIndex = Int(parts[2])
    else {
      return nil
    }

    let rest = Array(parts.dropFirst(3))
    let residualPrefix = "\(blockPrefix).residual\(residualIndex)"

    // Layer indices within a residual unit:
    // 0: snake1
    // 1: conv1
    // 2: snake2
    // 3: conv2

    switch layerIndex {
      case 0:
        return "\(residualPrefix).snake1." + rest.joined(separator: ".")
      case 1:
        return "\(residualPrefix).conv1." + rest.joined(separator: ".")
      case 2:
        return "\(residualPrefix).snake2." + rest.joined(separator: ".")
      case 3:
        return "\(residualPrefix).conv2." + rest.joined(separator: ".")
      default:
        return nil
    }
  }

  /// Set quantizer weights for embedCodes function
  func setQuantizerWeights(_ weights: [String: MLXArray]) {
    quantizerWeights = weights
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var y = x

    // Initial convolutions
    y = depthwiseConv(y)
    y = pointwiseConv(y)

    // Decoder blocks
    y = block0(y)
    y = block1(y)
    y = block2(y)
    y = block3(y)

    // Final snake activation
    // Ensure [B, C, T] format for snake input
    if y.shape[1] != finalSnake.alpha.shape[1], y.shape[2] == finalSnake.alpha.shape[1] {
      y = y.transposed(axes: [0, 2, 1])
    }
    // snake outputs [B, T, C], need to transpose back to [B, C, T] for conv
    y = finalSnake(y)
    y = y.transposed(axes: [0, 2, 1])

    // Final convolution
    y = finalConv(y)

    // Final tanh activation
    y = MLX.tanh(y)

    return y
  }

  func decode(codes: [[Int]]) -> MLXArray {
    // 1. Convert codes to embeddings
    var x = embedCodes(codes: codes)

    // 2. Apply decoder
    x = self(x)

    return x
  }

  static func loadWeights(
    repoId: String = defaultRepoId,
    filename: String = defaultWeightsFilename,
    progressHandler: @escaping (Progress) -> Void = { _ in },
  ) async throws -> [String: MLXArray] {
    let modelDirectoryURL = try await Hub.snapshot(
      from: repoId,
      matching: [filename],
      progressHandler: progressHandler,
    )
    let weightFileURL = modelDirectoryURL.appending(path: filename)
    return try loadWeights(from: weightFileURL)
  }

  static func loadWeights(from url: URL) throws -> [String: MLXArray] {
    try MLX.loadArrays(url: url)
  }

  static func loadConfig(
    repoId: String = defaultRepoId,
    filename: String = "config.json",
    progressHandler: @escaping (Progress) -> Void = { _ in },
  ) async throws -> SNACConfig {
    let modelDirectoryURL = try await Hub.snapshot(
      from: repoId,
      matching: [filename],
      progressHandler: progressHandler,
    )
    let configFileURL = modelDirectoryURL.appending(path: filename)
    return try loadConfig(from: configFileURL)
  }

  static func loadConfig(from url: URL) throws -> SNACConfig {
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(SNACConfig.self, from: data)
  }

  private func embedCodes(codes: [[Int]]) -> MLXArray {
    let dInput = config.latentDim
    var maxExpandedLength = 0

    for i in 0 ..< config.vqStrides.count {
      if i < codes.count, !codes[i].isEmpty {
        maxExpandedLength = max(maxExpandedLength, codes[i].count * config.vqStrides[i])
      }
    }

    if maxExpandedLength == 0, !codes.isEmpty {
      Log.tts.warning("maxExpandedLength is 0, but codes exist. This might happen if all code chunks are empty.")
    } else if maxExpandedLength == 0 {
      Log.tts.warning("maxExpandedLength is 0 (no codes). Decoder might not produce output.")
    }

    var zQSum = MLXArray.zeros([dInput, maxExpandedLength])

    for i in 0 ..< config.vqStrides.count {
      guard i < codes.count else {
        Log.tts.warning("Not enough code layers for VQ \(i). Skipping.")
        continue
      }

      let currentCodes = codes[i]
      if currentCodes.isEmpty {
        Log.tts.warning("Empty codes for VQ \(i). Skipping.")
        continue
      }

      guard let codeEmbeddings = quantizerWeights["quantizer.quantizers.\(i).codebook.weight"] else {
        Log.model.error("quantizer.quantizers.\(i).codebook.weight not found")
        continue
      }

      let codeIndices = MLXArray(currentCodes)
      let decodedZPI = codeEmbeddings[codeIndices]

      guard let outProjWeightG = quantizerWeights["quantizer.quantizers.\(i).out_proj.weight_g"],
            let outProjWeightV = quantizerWeights["quantizer.quantizers.\(i).out_proj.weight_v"],
            let outProjBias = quantizerWeights["quantizer.quantizers.\(i).out_proj.bias"]
      else {
        Log.model.error("quantizer.quantizers.\(i) output projection weights not found")
        continue
      }

      let weightG = outProjWeightG.squeezed()
      let weightV = outProjWeightV.squeezed()
      let normV = MLX.sqrt(MLX.sum(weightV * weightV, axis: 1, keepDims: true))
      let effectiveWeight = weightG.reshaped([dInput, 1]) * weightV / (normV + 1e-12)
      let projectedZQI = MLX.matmul(decodedZPI, effectiveWeight.transposed()) + outProjBias
      let projectedZQIT = projectedZQI.transposed()

      var expandedZQI = projectedZQIT
      let currentStride = config.vqStrides[i]

      if currentStride > 1 {
        let timestepsBeforeStride = projectedZQIT.shape[1]
        let expandedLen = timestepsBeforeStride * currentStride
        let expanded = MLXArray.zeros([dInput, expandedLen])

        for t in 0 ..< timestepsBeforeStride {
          let val = projectedZQIT[0 ..< dInput, t]
          for s in 0 ..< currentStride {
            expanded[0 ..< dInput, t * currentStride + s] = val
          }
        }

        expandedZQI = expanded
      }

      if expandedZQI.shape == zQSum.shape {
        zQSum = zQSum + expandedZQI
      } else {
        Log.tts.warning("Shape mismatch for VQ \(i). Expected \(zQSum.shape), got \(expandedZQI.shape)")
      }
    }

    return zQSum
  }

  static func snake(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
    let xPermuted = x
    let alphaReshaped = alpha.reshaped([1, alpha.shape[1], 1])
    let sinSquared = MLX.pow(MLX.sin(alphaReshaped * xPermuted), 2)
    let term = (1.0 / (alphaReshaped + 1e-9)) * sinSquared
    let out = xPermuted + term
    return out.transposed(axes: [0, 2, 1])
  }
}

// MARK: - Decoder Block

/// DecoderBlock for SNAC decoder
class SNACDecoderBlock: Module {
  @ModuleInfo var snake: SNACSnake
  @ModuleInfo var convT: WNConvTranspose1d
  @ModuleInfo var noiseBlock: SNACNoiseBlock?
  @ModuleInfo var residual0: SNACResidualUnit
  @ModuleInfo var residual1: SNACResidualUnit
  @ModuleInfo var residual2: SNACResidualUnit

  let hasNoise: Bool

  init(inputDim: Int, outputDim: Int, stride: Int, groups: Int, noise: Bool) {
    hasNoise = noise

    let paddingT = Int(ceil(Double(stride) / 2.0))
    let outputPaddingT = stride % 2

    _snake.wrappedValue = SNACSnake(channels: inputDim)
    _convT.wrappedValue = WNConvTranspose1d(
      inChannels: inputDim,
      outChannels: outputDim,
      kernelSize: stride * 2,
      stride: stride,
      padding: paddingT,
      outputPadding: outputPaddingT,
      bias: true,
    )

    if noise {
      _noiseBlock.wrappedValue = SNACNoiseBlock(dim: outputDim)
    } else {
      _noiseBlock.wrappedValue = nil
    }

    _residual0.wrappedValue = SNACResidualUnit(
      dim: outputDim, dilation: 1, kernelSize: 7, groups: groups,
    )
    _residual1.wrappedValue = SNACResidualUnit(
      dim: outputDim, dilation: 3, kernelSize: 7, groups: groups,
    )
    _residual2.wrappedValue = SNACResidualUnit(
      dim: outputDim, dilation: 9, kernelSize: 7, groups: groups,
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var y = x
    // Ensure input is [B, C, T] format for snake
    if y.shape[1] != snake.alpha.shape[1], y.shape[2] == snake.alpha.shape[1] {
      y = y.transposed(axes: [0, 2, 1])
    }

    // snake outputs [B, T, C], which is correct for convT (MLX native format)
    y = snake(y)
    // convT expects [B, T, C] and outputs [B, T, C]
    y = convT(y)
    // Transpose to [B, C, T] for noiseBlock and residuals (which use WNConv1d)
    y = y.transposed(axes: [0, 2, 1])

    if let noiseBlock {
      y = noiseBlock(y)
    }

    y = residual0(y)
    y = residual1(y)
    y = residual2(y)

    return y
  }
}
