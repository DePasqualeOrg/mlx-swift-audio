//  CAM++ speaker embedding model

import Foundation
import MLX
import MLXNN

// MARK: - Helper Functions

/// Create a Povey window (used by Kaldi)
func poveyWindow(size: Int) -> MLXArray {
  let n = MLXArray(0 ..< size).asType(.float32)
  let hann = 0.5 - 0.5 * MLX.cos(2 * Float.pi * n / Float(size - 1))
  return MLX.pow(hann, 0.85)
}

/// Return next power of 2 >= n
func nextPowerOf2(_ n: Int) -> Int {
  if n <= 1 { return 1 }
  var power = 1
  while power < n {
    power *= 2
  }
  return power
}

/// Extract Kaldi-compatible filterbank features
func kaldiFbankCAMPPlus(
  audio: MLXArray,
  sampleRate: Int = 16000,
  numMelBins: Int = 80,
  frameLength: Float = 25.0,
  frameShift: Float = 10.0,
) -> MLXArray {
  // Calculate frame parameters
  let winLength = Int(Float(sampleRate) * frameLength / 1000) // 400 for 25ms @ 16kHz
  let hopLength = Int(Float(sampleRate) * frameShift / 1000) // 160 for 10ms @ 16kHz
  let nFft = nextPowerOf2(winLength) // 400 -> 512

  // Ensure 1D
  var audioData = audio
  if audioData.ndim > 1 {
    audioData = audioData.squeezed()
  }

  // Calculate number of frames (snip_edges=True)
  let signalLen = audioData.shape[0]
  var numFrames = (signalLen - winLength) / hopLength + 1
  if numFrames < 1 { numFrames = 1 }

  // Create Povey window
  let window = poveyWindow(size: winLength)

  // Vectorized frame extraction
  let frameStarts = MLXArray(0 ..< numFrames) * Int32(hopLength)
  let frameOffsets = MLXArray(0 ..< winLength)
  let indices = frameStarts[0..., .newAxis] + frameOffsets[.newAxis, 0...]

  var frames = MLX.take(audioData, indices.flattened(), axis: 0).reshaped([numFrames, winLength])

  // Remove DC offset per frame
  frames = frames - MLX.mean(frames, axis: 1, keepDims: true)

  // Apply pre-emphasis: frame[1:] - 0.97 * frame[:-1]
  let preemph: Float = 0.97
  let firstSample = frames[0..., 0 ..< 1]
  let restSamples = frames[0..., 1...] - preemph * frames[0..., 0 ..< (winLength - 1)]
  frames = MLX.concatenated([firstSample, restSamples], axis: 1)

  // Apply window
  frames = frames * window

  // Zero-pad to n_fft
  if winLength < nFft {
    let padAmount = nFft - winLength
    frames = MLX.concatenated([frames, MLXArray.zeros([numFrames, padAmount])], axis: 1)
  }

  // FFT - using rfft for real input
  let spec = MLXFFT.rfft(frames, axis: 1)

  // Power spectrum
  let powerSpec = MLX.pow(MLX.abs(spec), 2)

  // Mel filterbank (HTK mel scale)
  let filters = melFiltersHTK(
    sampleRate: sampleRate,
    nFft: nFft,
    nMels: numMelBins,
    fMin: 20.0,
    fMax: Float(sampleRate) / 2,
  )

  // Apply filterbank: (T, F) @ (F, M) -> (T, M)
  // melFiltersHTK returns (nFft/2+1, nMels) = (F, M), so no transpose needed
  let melSpec = MLX.matmul(powerSpec, filters)

  // Log compression
  let fbank = MLX.log(MLX.maximum(melSpec, MLXArray(1.1920929e-07)))

  return fbank
}

// MARK: - Mel Filter Cache

/// Thread-safe cache for precomputed mel filterbanks
private final class MelFilterCache: @unchecked Sendable {
  private var cache: [String: MLXArray] = [:]
  private let lock = NSLock()

  static let shared = MelFilterCache()

  func get(sampleRate: Int, nFft: Int, nMels: Int, fMin: Float, fMax: Float) -> MLXArray {
    let key = "\(sampleRate)_\(nFft)_\(nMels)_\(fMin)_\(fMax)"

    lock.lock()
    defer { lock.unlock() }

    if let cached = cache[key] {
      return cached
    }

    let filters = computeMelFiltersHTK(sampleRate: sampleRate, nFft: nFft, nMels: nMels, fMin: fMin, fMax: fMax)
    cache[key] = filters
    return filters
  }
}

/// Compute HTK mel filterbank (internal, called once per configuration)
private func computeMelFiltersHTK(sampleRate: Int, nFft: Int, nMels: Int, fMin: Float, fMax: Float) -> MLXArray {
  // HTK mel scale
  func hzToMel(_ hz: Float) -> Float {
    2595.0 * log10(1.0 + hz / 700.0)
  }

  func melToHz(_ mel: Float) -> Float {
    700.0 * (pow(10.0, mel / 2595.0) - 1.0)
  }

  let melMin = hzToMel(fMin)
  let melMax = hzToMel(fMax)
  let melPoints = (0 ... nMels + 1).map { i in
    melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
  }
  let hzPoints = melPoints.map { melToHz($0) }
  let binPoints = hzPoints.map { Int(round($0 * Float(nFft) / Float(sampleRate))) }

  // Build filters directly as a flat array for efficiency
  let filterSize = (nFft / 2 + 1) * nMels
  var filters = [Float](repeating: 0, count: filterSize)

  for m in 1 ... nMels {
    let fmMinus = binPoints[m - 1]
    let fm = binPoints[m]
    let fmPlus = binPoints[m + 1]

    for k in fmMinus ..< fm where k >= 0 && k < nFft / 2 + 1 && fm != fmMinus {
      filters[k * nMels + (m - 1)] = Float(k - fmMinus) / Float(fm - fmMinus)
    }
    for k in fm ..< fmPlus where k >= 0 && k < nFft / 2 + 1 && fmPlus != fm {
      filters[k * nMels + (m - 1)] = Float(fmPlus - k) / Float(fmPlus - fm)
    }
  }

  return MLXArray(filters).reshaped([nFft / 2 + 1, nMels])
}

/// HTK mel filterbank (cached)
func melFiltersHTK(sampleRate: Int, nFft: Int, nMels: Int, fMin: Float, fMax: Float) -> MLXArray {
  MelFilterCache.shared.get(sampleRate: sampleRate, nFft: nFft, nMels: nMels, fMin: fMin, fMax: fMax)
}

// MARK: - BasicResBlock

/// Basic residual block for 2D convolution
class BasicResBlock: Module {
  static let expansion = 1

  @ModuleInfo(key: "conv1") var conv1: Conv2d
  @ModuleInfo(key: "bn1") var bn1: BatchNorm
  @ModuleInfo(key: "conv2") var conv2: Conv2d
  @ModuleInfo(key: "bn2") var bn2: BatchNorm
  @ModuleInfo(key: "shortcut") var shortcut: [Module]

  init(inPlanes: Int, planes: Int, stride: Int = 1) {
    _conv1.wrappedValue = Conv2d(
      inputChannels: inPlanes,
      outputChannels: planes,
      kernelSize: IntOrPair(3),
      stride: IntOrPair((stride, 1)),
      padding: IntOrPair(1),
      bias: false,
    )
    _bn1.wrappedValue = BatchNorm(featureCount: planes)
    _conv2.wrappedValue = Conv2d(
      inputChannels: planes,
      outputChannels: planes,
      kernelSize: IntOrPair(3),
      stride: IntOrPair(1),
      padding: IntOrPair(1),
      bias: false,
    )
    _bn2.wrappedValue = BatchNorm(featureCount: planes)

    var sc: [Module] = []
    if stride != 1 || inPlanes != Self.expansion * planes {
      sc = [
        Conv2d(
          inputChannels: inPlanes,
          outputChannels: Self.expansion * planes,
          kernelSize: IntOrPair(1),
          stride: IntOrPair((stride, 1)),
          bias: false,
        ),
        BatchNorm(featureCount: Self.expansion * planes),
      ]
    }
    _shortcut.wrappedValue = sc
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = relu(bn1(conv1(x)))
    out = bn2(conv2(out))

    var shortcutOut = x
    for layer in shortcut {
      if let conv = layer as? Conv2d {
        shortcutOut = conv(shortcutOut)
      } else if let bn = layer as? BatchNorm {
        shortcutOut = bn(shortcutOut)
      }
    }
    out = out + shortcutOut

    return relu(out)
  }
}

// MARK: - FCM

/// Feature Channel Module - processes input features
class FCM: Module {
  var inPlanes: Int
  let outChannels: Int

  @ModuleInfo(key: "conv1") var conv1: Conv2d
  @ModuleInfo(key: "bn1") var bn1: BatchNorm
  @ModuleInfo(key: "layer1") var layer1: [BasicResBlock]
  @ModuleInfo(key: "layer2") var layer2: [BasicResBlock]
  @ModuleInfo(key: "conv2") var conv2: Conv2d
  @ModuleInfo(key: "bn2") var bn2: BatchNorm

  init(mChannels: Int = 32, featDim: Int = 80) {
    inPlanes = mChannels

    _conv1.wrappedValue = Conv2d(
      inputChannels: 1,
      outputChannels: mChannels,
      kernelSize: IntOrPair(3),
      stride: IntOrPair(1),
      padding: IntOrPair(1),
      bias: false,
    )
    _bn1.wrappedValue = BatchNorm(featureCount: mChannels)

    // Create layer1
    var l1: [BasicResBlock] = []
    l1.append(BasicResBlock(inPlanes: inPlanes, planes: mChannels, stride: 2))
    inPlanes = mChannels
    l1.append(BasicResBlock(inPlanes: inPlanes, planes: mChannels, stride: 1))
    _layer1.wrappedValue = l1

    // Create layer2
    var l2: [BasicResBlock] = []
    l2.append(BasicResBlock(inPlanes: inPlanes, planes: mChannels, stride: 2))
    inPlanes = mChannels
    l2.append(BasicResBlock(inPlanes: inPlanes, planes: mChannels, stride: 1))
    _layer2.wrappedValue = l2

    _conv2.wrappedValue = Conv2d(
      inputChannels: mChannels,
      outputChannels: mChannels,
      kernelSize: IntOrPair(3),
      stride: IntOrPair((2, 1)),
      padding: IntOrPair(1),
      bias: false,
    )
    _bn2.wrappedValue = BatchNorm(featureCount: mChannels)

    outChannels = mChannels * (featDim / 8)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Input x: (B, F, T) - add channel dimension
    var out = x.expandedDimensions(axis: -1) // (B, F, T, 1) = (B, H, W, C)
    out = relu(bn1(conv1(out)))

    for layer in layer1 {
      out = layer(out)
    }
    for layer in layer2 {
      out = layer(out)
    }

    out = relu(bn2(conv2(out)))

    // Reshape: (B, H, W, C) -> (B, C*H, W)
    let shape = out.shape
    let B = shape[0]
    let H = shape[1]
    let W = shape[2]
    let C = shape[3]

    // Permute to (B, C, H, W) then reshape
    out = out.transposed(0, 3, 1, 2) // (B, C, H, W)
    out = out.reshaped([B, C * H, W])
    return out
  }
}

// MARK: - Statistics Pooling

/// Compute mean and std statistics along axis
func statisticsPooling(_ x: MLXArray, axis: Int = -1) -> MLXArray {
  let mean = MLX.mean(x, axis: axis, keepDims: false)
  let variance = MLX.variance(x, axis: axis, keepDims: false)
  let std = MLX.sqrt(variance + 1e-5)
  return MLX.concatenated([mean, std], axis: -1)
}

/// Statistics pooling layer
class StatsPool: Module {
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    statisticsPooling(x)
  }
}

// MARK: - TDNNLayer

/// Time-Delay Neural Network layer
class TDNNLayer: Module {
  @ModuleInfo(key: "linear") var linear: Conv1d
  @ModuleInfo(key: "nonlinear") var nonlinear: [Module]

  init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    bias: Bool = false,
    configStr: String = "batchnorm-relu",
  ) {
    var pad = padding
    if padding < 0 {
      precondition(kernelSize % 2 == 1, "Expected odd kernel size")
      pad = (kernelSize - 1) / 2 * dilation
    }

    _linear.wrappedValue = Conv1d(
      inputChannels: inChannels,
      outputChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: pad,
      dilation: dilation,
      bias: bias,
    )
    _nonlinear.wrappedValue = getNonlinear(configStr: configStr, channels: outChannels)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Input x: (B, C, T) - PyTorch format
    var out = x.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)
    out = linear(out)
    // Apply nonlinear in MLX format (B, T, C)
    for layer in nonlinear {
      if let bn = layer as? BatchNorm {
        out = bn(out)
      } else if layer is ReLUActivation {
        out = relu(out)
      }
    }
    out = out.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)
    return out
  }
}

/// ReLU activation as a Module for use in heterogeneous layer arrays
class ReLUActivation: Module {
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    relu(x)
  }
}

/// Get nonlinear layers based on config string
func getNonlinear(configStr: String, channels: Int) -> [Module] {
  var layers: [Module] = []
  for name in configStr.split(separator: "-") {
    switch String(name) {
      case "relu":
        layers.append(ReLUActivation())
      case "batchnorm", "batchnorm_":
        layers.append(BatchNorm(featureCount: channels))
      default:
        fatalError("Unexpected module: \(name)")
    }
  }
  return layers
}

// MARK: - CAMLayer

/// Context Attentive Module layer
class CAMLayer: Module {
  @ModuleInfo(key: "linear_local") var linearLocal: Conv1d
  @ModuleInfo(key: "linear1") var linear1: Conv1d
  @ModuleInfo(key: "linear2") var linear2: Conv1d

  init(
    bnChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int,
    padding: Int,
    dilation: Int,
    bias: Bool,
    reduction: Int = 2,
  ) {
    _linearLocal.wrappedValue = Conv1d(
      inputChannels: bnChannels,
      outputChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: padding,
      dilation: dilation,
      bias: bias,
    )
    _linear1.wrappedValue = Conv1d(
      inputChannels: bnChannels,
      outputChannels: bnChannels / reduction,
      kernelSize: 1,
    )
    _linear2.wrappedValue = Conv1d(
      inputChannels: bnChannels / reduction,
      outputChannels: outChannels,
      kernelSize: 1,
    )
  }

  /// Apply Conv1d to input in PyTorch format (B, C, T)
  private func conv1dPytorchFormat(_ x: MLXArray, conv: Conv1d) -> MLXArray {
    var out = x.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)
    out = conv(out)
    out = out.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)
    return out
  }

  /// Segment pooling
  private func segPooling(_ x: MLXArray, segLen: Int = 100, stype: String = "avg") -> MLXArray {
    let shape = x.shape
    let B = shape[0]
    let C = shape[1]
    let T = shape[2]

    let nSegs = (T + segLen - 1) / segLen
    let padLen = nSegs * segLen - T

    var xPadded = x
    if padLen > 0 {
      xPadded = MLX.concatenated([x, MLXArray.zeros([B, C, padLen])], axis: -1)
    }

    let xReshaped = xPadded.reshaped([B, C, nSegs, segLen])

    var seg: MLXArray = if stype == "avg" {
      MLX.mean(xReshaped, axis: -1)
    } else {
      MLX.max(xReshaped, axis: -1)
    }

    seg = seg.expandedDimensions(axis: -1)
    seg = MLX.broadcast(seg, to: [B, C, nSegs, segLen])
    seg = seg.reshaped([B, C, -1])
    seg = seg[0..., 0..., 0 ..< T]

    return seg
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let y = conv1dPytorchFormat(x, conv: linearLocal)
    var context = MLX.mean(x, axis: -1, keepDims: true) + segPooling(x)
    context = relu(conv1dPytorchFormat(context, conv: linear1))
    let m = sigmoid(conv1dPytorchFormat(context, conv: linear2))
    return y * m
  }
}

// MARK: - CAMDenseTDNNLayer

/// CAM Dense TDNN layer
class CAMDenseTDNNLayer: Module {
  @ModuleInfo(key: "nonlinear1") var nonlinear1: [Module]
  @ModuleInfo(key: "linear1") var linear1: Conv1d
  @ModuleInfo(key: "nonlinear2") var nonlinear2: [Module]
  @ModuleInfo(key: "cam_layer") var camLayer: CAMLayer

  init(
    inChannels: Int,
    outChannels: Int,
    bnChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    dilation: Int = 1,
    bias: Bool = false,
    configStr: String = "batchnorm-relu",
  ) {
    precondition(kernelSize % 2 == 1, "Expected odd kernel size")
    let padding = (kernelSize - 1) / 2 * dilation

    _nonlinear1.wrappedValue = getNonlinear(configStr: configStr, channels: inChannels)
    _linear1.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: bnChannels, kernelSize: 1, bias: false)
    _nonlinear2.wrappedValue = getNonlinear(configStr: configStr, channels: bnChannels)
    _camLayer.wrappedValue = CAMLayer(
      bnChannels: bnChannels,
      outChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: padding,
      dilation: dilation,
      bias: bias,
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Input x: (B, C, T) - PyTorch format
    var out = x.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)

    for layer in nonlinear1 {
      if let bn = layer as? BatchNorm {
        out = bn(out)
      } else if layer is ReLUActivation {
        out = relu(out)
      }
    }

    out = linear1(out)

    for layer in nonlinear2 {
      if let bn = layer as? BatchNorm {
        out = bn(out)
      } else if layer is ReLUActivation {
        out = relu(out)
      }
    }

    out = out.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)
    out = camLayer(out)
    return out
  }
}

// MARK: - CAMDenseTDNNBlock

/// CAM Dense TDNN block with multiple layers
class CAMDenseTDNNBlock: Module {
  @ModuleInfo(key: "layers") var layers: [CAMDenseTDNNLayer]

  init(
    numLayers: Int,
    inChannels: Int,
    outChannels: Int,
    bnChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    dilation: Int = 1,
    bias: Bool = false,
    configStr: String = "batchnorm-relu",
  ) {
    var layersList: [CAMDenseTDNNLayer] = []
    for i in 0 ..< numLayers {
      layersList.append(CAMDenseTDNNLayer(
        inChannels: inChannels + i * outChannels,
        outChannels: outChannels,
        bnChannels: bnChannels,
        kernelSize: kernelSize,
        stride: stride,
        dilation: dilation,
        bias: bias,
        configStr: configStr,
      ))
    }
    _layers.wrappedValue = layersList
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = x
    for layer in layers {
      out = MLX.concatenated([out, layer(out)], axis: 1)
    }
    return out
  }
}

// MARK: - TransitLayer

/// Transition layer between dense blocks
class TransitLayer: Module {
  @ModuleInfo(key: "nonlinear") var nonlinear: [Module]
  @ModuleInfo(key: "linear") var linear: Conv1d

  init(inChannels: Int, outChannels: Int, bias: Bool = true, configStr: String = "batchnorm-relu") {
    _nonlinear.wrappedValue = getNonlinear(configStr: configStr, channels: inChannels)
    _linear.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: bias)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = x.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)

    for layer in nonlinear {
      if let bn = layer as? BatchNorm {
        out = bn(out)
      } else if layer is ReLUActivation {
        out = relu(out)
      }
    }

    out = linear(out)
    out = out.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)
    return out
  }
}

// MARK: - DenseLayer

/// Dense layer for final embedding
class DenseLayer: Module {
  @ModuleInfo(key: "linear") var linear: Conv1d
  @ModuleInfo(key: "nonlinear") var nonlinear: [BatchNorm]

  init(inChannels: Int, outChannels: Int, bias: Bool = false, configStr: String = "batchnorm-relu") {
    _linear.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: bias)

    var batchNorms: [BatchNorm] = []
    for name in configStr.split(separator: "-") {
      if String(name) == "batchnorm" {
        batchNorms.append(BatchNorm(featureCount: outChannels, affine: true))
      } else if String(name) == "batchnorm_" {
        // batchnorm_ means no affine parameters (no weight/bias)
        batchNorms.append(BatchNorm(featureCount: outChannels, affine: false))
      }
    }
    _nonlinear.wrappedValue = batchNorms
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out: MLXArray

    if x.ndim == 2 {
      out = x.expandedDimensions(axis: 1)
      out = linear(out)
      for bn in nonlinear {
        out = bn(out)
      }
      out = out.squeezed(axis: 1)
    } else {
      out = x.swappedAxes(1, 2)
      out = linear(out)
      for bn in nonlinear {
        out = bn(out)
      }
      out = out.swappedAxes(1, 2)
    }

    return out
  }
}

// MARK: - CAMPPlus

/// CAM++ speaker embedding model
class CAMPPlus: Module {
  let outputLevel: String

  @ModuleInfo(key: "head") var head: FCM
  @ModuleInfo(key: "tdnn") var tdnn: TDNNLayer
  @ModuleInfo(key: "blocks") var blocks: [CAMDenseTDNNBlock]
  @ModuleInfo(key: "transits") var transits: [TransitLayer]
  @ModuleInfo(key: "out_nonlinear") var outNonlinear: [Module]
  @ModuleInfo(key: "stats") var stats: StatsPool
  @ModuleInfo(key: "dense") var dense: DenseLayer

  init(
    featDim: Int = 80,
    embeddingSize: Int = 192,
    growthRate: Int = 32,
    bnSize: Int = 4,
    initChannels: Int = 128,
    configStr: String = "batchnorm-relu",
    outputLevel: String = "segment",
  ) {
    self.outputLevel = outputLevel

    _head.wrappedValue = FCM(mChannels: 32, featDim: featDim)
    var channels = 32 * (featDim / 8) // head.outChannels

    _tdnn.wrappedValue = TDNNLayer(
      inChannels: channels,
      outChannels: initChannels,
      kernelSize: 5,
      stride: 2,
      padding: -1,
      dilation: 1,
      configStr: configStr,
    )
    channels = initChannels

    // Dense TDNN blocks
    var blocksArray: [CAMDenseTDNNBlock] = []
    var transitsArray: [TransitLayer] = []

    let configs: [(Int, Int, Int)] = [(12, 3, 1), (24, 3, 2), (16, 3, 2)]
    for (numLayers, kernelSize, dilation) in configs {
      blocksArray.append(CAMDenseTDNNBlock(
        numLayers: numLayers,
        inChannels: channels,
        outChannels: growthRate,
        bnChannels: bnSize * growthRate,
        kernelSize: kernelSize,
        dilation: dilation,
        configStr: configStr,
      ))
      channels = channels + numLayers * growthRate

      transitsArray.append(TransitLayer(
        inChannels: channels,
        outChannels: channels / 2,
        bias: false,
        configStr: configStr,
      ))
      channels /= 2
    }

    _blocks.wrappedValue = blocksArray
    _transits.wrappedValue = transitsArray
    _outNonlinear.wrappedValue = getNonlinear(configStr: configStr, channels: channels)
    _stats.wrappedValue = StatsPool()
    _dense.wrappedValue = DenseLayer(inChannels: channels * 2, outChannels: embeddingSize, configStr: "batchnorm_")
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = x.swappedAxes(1, 2) // (B, T, F) -> (B, F, T)
    out = head(out)
    out = tdnn(out)

    // Dense blocks with transitions
    for (block, transit) in zip(blocks, transits) {
      out = block(out)
      out = transit(out)
    }

    // Output nonlinearity
    out = out.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)
    for layer in outNonlinear {
      if let bn = layer as? BatchNorm {
        out = bn(out)
      } else if layer is ReLUActivation {
        out = relu(out)
      }
    }
    out = out.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)

    if outputLevel == "segment" {
      out = stats(out)
      out = dense(out)
      if out.ndim == 3, out.shape[out.ndim - 1] == 1 {
        out = out.squeezed(axis: -1)
      }
    }

    return out
  }

  /// Inference on raw audio waveform
  func inference(_ audio: MLXArray) -> MLXArray {
    var audioData = audio
    if audioData.ndim == 1 {
      audioData = audioData.expandedDimensions(axis: 0)
    }

    // Extract fbank features for each audio in batch (independent - can be parallelized)
    let features = (0 ..< audioData.shape[0]).map { i -> MLXArray in
      var fbank = kaldiFbankCAMPPlus(audio: audioData[i])
      // Mean normalization
      fbank = fbank - MLX.mean(fbank, axis: 0, keepDims: true)
      return fbank
    }

    // Pad to same length
    let maxLen = features.map { $0.shape[0] }.max() ?? 0
    let paddedFeatures = features.map { f -> MLXArray in
      if f.shape[0] < maxLen {
        let pad = MLXArray.zeros([maxLen - f.shape[0], f.shape[1]])
        return MLX.concatenated([f, pad], axis: 0)
      }
      return f
    }

    // Stack to batch: (B, T, F)
    let batchFeatures = MLX.stacked(paddedFeatures, axis: 0)

    return callAsFunction(batchFeatures)
  }
}
