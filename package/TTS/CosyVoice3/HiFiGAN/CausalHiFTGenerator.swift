// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

import Foundation
import MLX
import MLXNN

// MARK: - Causal Convolution Modules

/// Causal 1D convolution supporting both left (past) and right (future) causal types
/// - 'left': Standard causal convolution (past context only)
/// - 'right': Lookahead convolution (future context for streaming)
class CosyVoice3CausalConv1d: Module {
  enum CausalType {
    case left
    case right
  }

  @ModuleInfo(key: "conv") var conv: Conv1d
  let causalPadding: Int
  let causalType: CausalType

  init(
    inputChannels: Int,
    outputChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    dilation: Int = 1,
    groups: Int = 1,
    bias: Bool = true,
    causalType: CausalType = .left
  ) {
    precondition(stride == 1, "CosyVoice3CausalConv1d only supports stride=1")

    causalPadding = ((kernelSize * dilation - dilation) / 2) * 2 + (kernelSize + 1) % 2
    self.causalType = causalType

    _conv.wrappedValue = Conv1d(
      inputChannels: inputChannels,
      outputChannels: outputChannels,
      kernelSize: kernelSize,
      stride: 1,
      padding: 0,
      dilation: dilation,
      groups: groups,
      bias: bias
    )
  }

  func callAsFunction(_ x: MLXArray, cache: MLXArray? = nil) -> MLXArray {
    var h = x.swappedAxes(1, 2)

    let cacheData: MLXArray = if let cache, cache.size > 0 {
      cache.swappedAxes(1, 2)
    } else {
      MLXArray.zeros([h.shape[0], causalPadding, h.shape[2]])
    }

    switch causalType {
      case .left:
        h = MLX.concatenated([cacheData, h], axis: 1)
      case .right:
        h = MLX.concatenated([h, cacheData], axis: 1)
    }

    h = conv(h)
    return h.swappedAxes(1, 2)
  }
}

/// Causal 1D convolution with downsampling
class CosyVoice3CausalConv1dDownSample: Module {
  @ModuleInfo(key: "conv") var conv: Conv1d
  let causalPadding: Int
  let stride: Int

  init(
    inputChannels: Int,
    outputChannels: Int,
    kernelSize: Int,
    stride: Int,
    dilation: Int = 1,
    groups: Int = 1,
    bias: Bool = true
  ) {
    precondition(stride != 1 && dilation == 1, "CosyVoice3CausalConv1dDownSample requires stride != 1 and dilation == 1")
    precondition(kernelSize % stride == 0, "kernel_size must be divisible by stride")

    causalPadding = stride - 1
    self.stride = stride

    _conv.wrappedValue = Conv1d(
      inputChannels: inputChannels,
      outputChannels: outputChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: 0,
      dilation: dilation,
      groups: groups,
      bias: bias
    )
  }

  func callAsFunction(_ x: MLXArray, cache: MLXArray? = nil) -> MLXArray {
    var h = x.swappedAxes(1, 2)

    if let cache, cache.size > 0 {
      let cacheData = cache.swappedAxes(1, 2)
      h = MLX.concatenated([cacheData, h], axis: 1)
    } else {
      h = MLX.padded(h, widths: [IntOrPair((0, 0)), IntOrPair((causalPadding, 0)), IntOrPair((0, 0))])
    }

    h = conv(h)
    return h.swappedAxes(1, 2)
  }
}

/// Causal 1D convolution with upsampling
class CosyVoice3CausalConv1dUpsample: Module {
  @ModuleInfo(key: "conv") var conv: Conv1d
  let causalPadding: Int
  let upsampleFactor: Int

  init(
    inputChannels: Int,
    outputChannels: Int,
    kernelSize: Int,
    stride: Int,
    dilation: Int = 1,
    groups: Int = 1,
    bias: Bool = true
  ) {
    precondition(dilation == 1, "CosyVoice3CausalConv1dUpsample requires dilation == 1")

    causalPadding = kernelSize - 1
    upsampleFactor = stride

    _conv.wrappedValue = Conv1d(
      inputChannels: inputChannels,
      outputChannels: outputChannels,
      kernelSize: kernelSize,
      stride: 1,
      padding: 0,
      dilation: dilation,
      groups: groups,
      bias: bias
    )
  }

  func callAsFunction(_ x: MLXArray, cache: MLXArray? = nil) -> MLXArray {
    var h = x.swappedAxes(1, 2)
    h = MLX.repeated(h, count: upsampleFactor, axis: 1)

    if let cache, cache.size > 0 {
      let cacheData = cache.swappedAxes(1, 2)
      h = MLX.concatenated([cacheData, h], axis: 1)
    } else {
      h = MLX.padded(h, widths: [IntOrPair((0, 0)), IntOrPair((causalPadding, 0)), IntOrPair((0, 0))])
    }

    h = conv(h)
    return h.swappedAxes(1, 2)
  }
}

// MARK: - Snake Activation

/// Snake activation function: x + sin^2(alpha * x) / alpha
class CosyVoice3Snake: Module {
  var alpha: MLXArray
  let alphaLogscale: Bool
  let noZeroDivision: Float = 1e-9

  init(channels: Int, alphaLogscale: Bool = false) {
    self.alphaLogscale = alphaLogscale
    alpha = MLXArray.ones([1, channels, 1])
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var a = alpha
    if alphaLogscale {
      a = MLX.exp(a)
    }
    if a.ndim == 1 {
      a = a.expandedDimensions(axes: [0, -1])
    }
    return x + (MLX.pow(MLX.sin(a * x), 2)) / (a + noZeroDivision)
  }
}

// MARK: - ResBlock

/// Residual block with Snake activation for HiFi-GAN
class CosyVoice3CausalResBlock: Module {
  @ModuleInfo(key: "convs1") var convs1: [CosyVoice3CausalConv1d]
  @ModuleInfo(key: "convs2") var convs2: [CosyVoice3CausalConv1d]
  @ModuleInfo(key: "activations1") var activations1: [CosyVoice3Snake]
  @ModuleInfo(key: "activations2") var activations2: [CosyVoice3Snake]

  init(
    channels: Int,
    kernelSize: Int = 3,
    dilations: [Int] = [1, 3, 5]
  ) {
    var convs1Array: [CosyVoice3CausalConv1d] = []
    var convs2Array: [CosyVoice3CausalConv1d] = []
    var acts1Array: [CosyVoice3Snake] = []
    var acts2Array: [CosyVoice3Snake] = []

    for dilation in dilations {
      convs1Array.append(CosyVoice3CausalConv1d(
        inputChannels: channels,
        outputChannels: channels,
        kernelSize: kernelSize,
        stride: 1,
        dilation: dilation,
        causalType: .left
      ))
      convs2Array.append(CosyVoice3CausalConv1d(
        inputChannels: channels,
        outputChannels: channels,
        kernelSize: kernelSize,
        stride: 1,
        dilation: 1,
        causalType: .left
      ))
      acts1Array.append(CosyVoice3Snake(channels: channels, alphaLogscale: false))
      acts2Array.append(CosyVoice3Snake(channels: channels, alphaLogscale: false))
    }

    _convs1.wrappedValue = convs1Array
    _convs2.wrappedValue = convs2Array
    _activations1.wrappedValue = acts1Array
    _activations2.wrappedValue = acts2Array
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = x
    for i in 0 ..< convs1.count {
      var xt = activations1[i](h)
      xt = convs1[i](xt)
      xt = activations2[i](xt)
      xt = convs2[i](xt)
      h = xt + h
    }
    return h
  }
}

// MARK: - SineGen2 (Causal)

/// Causal sine generator for 24kHz
class CosyVoice3SineGen2: Module {
  let sineAmp: Float
  let noiseStd: Float
  let harmonicNum: Int
  let dim: Int
  let samplingRate: Int
  let voicedThreshold: Float
  let upsampleScale: Int

  init(
    samplingRate: Int,
    upsampleScale: Int,
    harmonicNum: Int = 0,
    sineAmp: Float = 0.1,
    noiseStd: Float = 0.003,
    voicedThreshold: Float = 0
  ) {
    self.sineAmp = sineAmp
    self.noiseStd = noiseStd
    self.harmonicNum = harmonicNum
    dim = harmonicNum + 1
    self.samplingRate = samplingRate
    self.voicedThreshold = voicedThreshold
    self.upsampleScale = upsampleScale
  }

  private func f02uv(_ f0: MLXArray) -> MLXArray {
    (f0 .> voicedThreshold).asType(.float32)
  }

  private func f02sine(_ f0Values: MLXArray) -> MLXArray {
    var radValues = (f0Values / Float(samplingRate)) % 1
    let (B, T, D) = (radValues.shape[0], radValues.shape[1], radValues.shape[2])

    var randIni = MLXRandom.uniform(low: Float(0), high: Float(1), [B, D])
    let zeroCol = MLXArray.zeros([B, 1])
    randIni = MLX.concatenated([zeroCol, randIni[0..., 1...]], axis: 1)

    let firstStep = radValues[0..., 0 ..< 1, 0...] + randIni.expandedDimensions(axis: 1)
    radValues = MLX.concatenated([firstStep, radValues[0..., 1..., 0...]], axis: 1)

    let radDownsampled = linearInterpolate1d(radValues, scaleFactor: 1.0 / Float(upsampleScale))
    var phase = MLX.cumsum(radDownsampled, axis: 1) * 2 * Float.pi

    phase = MLX.repeated(phase, count: upsampleScale, axis: 1)
    phase = phase * Float(upsampleScale)
    phase = phase[0..., 0 ..< T, 0...]

    return MLX.sin(phase)
  }

  func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let harmonics = MLXArray(1 ... Int32(harmonicNum + 1)).asType(.float32)
    let fn = f0 * harmonics.reshaped(1, 1, -1)

    var sineWaves = f02sine(fn) * sineAmp
    let uv = f02uv(f0)

    let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3
    let noise = noiseAmp * MLXRandom.normal(sineWaves.shape)

    sineWaves = sineWaves * uv + noise

    return (sineWaves, uv, noise)
  }
}

// MARK: - SourceModuleHnNSF2 (Causal)

/// Causal Neural Source Filter module for 24kHz
class CosyVoice3SourceModuleHnNSF2: Module {
  let sineAmp: Float
  let noiseStd: Float

  @ModuleInfo(key: "l_sin_gen") var lSinGen: CosyVoice3SineGen2
  @ModuleInfo(key: "l_linear") var lLinear: Linear

  init(
    samplingRate: Int,
    upsampleScale: Int,
    harmonicNum: Int = 0,
    sineAmp: Float = 0.1,
    addNoiseStd: Float = 0.003,
    voicedThreshold: Float = 0
  ) {
    self.sineAmp = sineAmp
    noiseStd = addNoiseStd

    _lSinGen.wrappedValue = CosyVoice3SineGen2(
      samplingRate: samplingRate,
      upsampleScale: upsampleScale,
      harmonicNum: harmonicNum,
      sineAmp: sineAmp,
      noiseStd: addNoiseStd,
      voicedThreshold: voicedThreshold
    )

    _lLinear.wrappedValue = Linear(harmonicNum + 1, 1)
  }

  func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let (sineWavs, uv, _) = lSinGen(x)
    let sineMerge = tanh(lLinear(sineWavs))
    let noise = MLXRandom.normal(uv.shape) * sineAmp / 3
    return (sineMerge, noise, uv)
  }
}

// MARK: - CausalConvRNNF0Predictor

/// Causal F0 predictor for streaming inference
class CosyVoice3F0Predictor: Module {
  @ModuleInfo(key: "condnet_0") var condnet0: CosyVoice3CausalConv1d
  @ModuleInfo(key: "condnet_2") var condnet2: CosyVoice3CausalConv1d
  @ModuleInfo(key: "condnet_4") var condnet4: CosyVoice3CausalConv1d
  @ModuleInfo(key: "condnet_6") var condnet6: CosyVoice3CausalConv1d
  @ModuleInfo(key: "condnet_8") var condnet8: CosyVoice3CausalConv1d
  @ModuleInfo(key: "classifier") var classifier: Linear

  var condnet0CausalPadding: Int { condnet0.causalPadding }

  init(inChannels: Int = 80, condChannels: Int = 512) {
    _condnet0.wrappedValue = CosyVoice3CausalConv1d(
      inputChannels: inChannels, outputChannels: condChannels, kernelSize: 4, causalType: .right
    )
    _condnet2.wrappedValue = CosyVoice3CausalConv1d(
      inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, causalType: .left
    )
    _condnet4.wrappedValue = CosyVoice3CausalConv1d(
      inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, causalType: .left
    )
    _condnet6.wrappedValue = CosyVoice3CausalConv1d(
      inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, causalType: .left
    )
    _condnet8.wrappedValue = CosyVoice3CausalConv1d(
      inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, causalType: .left
    )
    _classifier.wrappedValue = Linear(condChannels, 1)
  }

  func callAsFunction(_ x: MLXArray, finalize: Bool = true) -> MLXArray {
    var h: MLXArray

    if finalize {
      h = condnet0(x)
    } else {
      let causalPadding = condnet0.causalPadding
      let xMain = x[0..., 0..., 0 ..< (x.shape[2] - causalPadding)]
      let xContext = x[0..., 0..., (x.shape[2] - causalPadding)...]
      h = condnet0(xMain, cache: xContext)
    }

    h = elu(h)
    h = condnet2(h)
    h = elu(h)
    h = condnet4(h)
    h = elu(h)
    h = condnet6(h)
    h = elu(h)
    h = condnet8(h)
    h = elu(h)

    h = h.swappedAxes(1, 2)
    var f0 = classifier(h).squeezed(axis: -1)
    f0 = MLX.abs(f0)

    return f0
  }
}

// MARK: - STFT/iSTFT Functions

/// Create a periodic Hann window for CosyVoice3
func cosyVoice3HannWindowPeriodic(size: Int) -> MLXArray {
  let n = MLXArray(0 ..< Int32(size)).asType(.float32)
  return 0.5 - 0.5 * MLX.cos(2 * Float.pi * n / Float(size))
}

/// Compute STFT for CosyVoice3 (vectorized using asStrided)
func cosyVoice3Stft(x: MLXArray, nFft: Int, hopLength: Int, window: MLXArray) -> (MLXArray, MLXArray) {
  let B = x.shape[0]
  let padLen = nFft / 2
  let padded = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((padLen, padLen))])

  let nFrames = (padded.shape[1] - nFft) / hopLength + 1

  // Use asStrided for efficient vectorized frame extraction (matching Python)
  var framesList: [MLXArray] = []
  for b in 0 ..< B {
    let batchSlice = padded[b]
    // Extract all frames at once using strided view: shape (nFrames, nFft), stride (hopLength, 1)
    let frames = asStrided(batchSlice, [nFrames, nFft], strides: [hopLength, 1])
    framesList.append(frames * window)
  }
  let allFrames = MLX.stacked(framesList, axis: 0)

  let spec = MLX.rfft(allFrames, axis: -1)
  var real = spec.realPart()
  var imag = spec.imaginaryPart()

  real = real.swappedAxes(1, 2)
  imag = imag.swappedAxes(1, 2)

  return (real, imag)
}

/// Compute inverse STFT for CosyVoice3 (vectorized implementation using scatter_add)
func cosyVoice3Istft(magnitude: MLXArray, phase: MLXArray, nFft: Int, hopLength: Int, window: MLXArray) -> MLXArray {
  let mag = MLX.clip(magnitude, min: Float(0.0), max: Float(1e2))

  let real = mag * MLX.cos(phase)
  let imag = mag * MLX.sin(phase)

  let realT = real.swappedAxes(1, 2)
  let imagT = imag.swappedAxes(1, 2)

  // Combine into complex: spectrum = real + 1j * imag
  let oneJ = MLXArray(real: 0, imaginary: 1)
  let spec = realT + oneJ * imagT
  var frames = MLXFFT.irfft(spec, n: nFft, axis: -1)
  frames = frames * window // (B, nFrames, nFft)

  let B = frames.shape[0]
  let nFrames = frames.shape[1]
  let outputLen = nFft + (nFrames - 1) * hopLength

  // Compute indices for scatter_add: for each frame f, indices are [f*hopLength, f*hopLength+1, ..., f*hopLength+nFft-1]
  // Frame offsets: [0, hopLength, 2*hopLength, ...]
  let frameOffsets = MLXArray(0 ..< Int32(nFrames)) * Int32(hopLength)
  // Indices within frame: [0, 1, ..., nFft-1]
  let frameIndices = MLXArray(0 ..< Int32(nFft))
  // Broadcast to get all indices: (nFrames, nFft)
  let indices = frameOffsets.expandedDimensions(axis: 1) + frameIndices.expandedDimensions(axis: 0)
  let indicesFlat = indices.flattened()

  // Compute window sum using scatter_add (same for all batches)
  let windowSq = window * window
  let windowUpdates = MLX.tiled(windowSq, repetitions: [nFrames])
  var windowSum = MLXArray.zeros([outputLen])
  windowSum = windowSum.at[indicesFlat].add(windowUpdates)
  windowSum = MLX.maximum(windowSum, MLXArray(Float(1e-8)))

  // Overlap-add for each batch element using scatter_add
  var outputList: [MLXArray] = []
  for b in 0 ..< B {
    let framesFlat = frames[b].flattened()
    var outB = MLXArray.zeros([outputLen])
    outB = outB.at[indicesFlat].add(framesFlat)
    outputList.append(outB)
  }
  var output = MLX.stacked(outputList, axis: 0)

  output = output / windowSum.expandedDimensions(axis: 0)

  let padLen = nFft / 2
  output = output[0..., padLen ..< (outputLen - padLen)]

  return output
}

// MARK: - CausalHiFTGenerator

/// Causal HiFi-GAN with Neural Source Filter for CosyVoice3
class CausalHiFTGenerator: Module {
  let outChannels: Int = 1
  let nbHarmonics: Int
  let samplingRate: Int
  let istftParams: [String: Int]
  let lreluSlope: Float
  let audioLimit: Float
  let numKernels: Int
  let numUpsamples: Int
  let f0UpsampleScale: Int
  let convPreLookRight: Int
  let upsampleRates: [Int]

  @ModuleInfo(key: "f0_predictor") var f0Predictor: CosyVoice3F0Predictor
  @ModuleInfo(key: "m_source") var mSource: CosyVoice3SourceModuleHnNSF2
  @ModuleInfo(key: "conv_pre") var convPre: CosyVoice3CausalConv1d
  @ModuleInfo(key: "ups") var ups: [CosyVoice3CausalConv1dUpsample]
  @ModuleInfo(key: "source_downs") var sourceDowns: [Module]
  @ModuleInfo(key: "source_resblocks") var sourceResblocks: [CosyVoice3CausalResBlock]
  @ModuleInfo(key: "resblocks") var resblocks: [CosyVoice3CausalResBlock]
  @ModuleInfo(key: "conv_post") var convPost: CosyVoice3CausalConv1d

  var stftWindow: MLXArray

  init(
    inChannels: Int = 80,
    baseChannels: Int = 512,
    nbHarmonics: Int = 8,
    samplingRate: Int = 24000,
    nsfAlpha: Float = 0.1,
    nsfSigma: Float = 0.003,
    nsfVoicedThreshold: Float = 10,
    upsampleRates: [Int] = [8, 5, 3],
    upsampleKernelSizes: [Int] = [16, 11, 7],
    istftParams: [String: Int] = ["n_fft": 16, "hop_len": 4],
    resblockKernelSizes: [Int] = [3, 7, 11],
    resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    sourceResblockKernelSizes: [Int] = [7, 11],
    sourceResblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5]],
    lreluSlope: Float = 0.1,
    audioLimit: Float = 0.99,
    convPreLookRight: Int = 4
  ) {
    self.nbHarmonics = nbHarmonics
    self.samplingRate = samplingRate
    self.istftParams = istftParams
    self.lreluSlope = lreluSlope
    self.audioLimit = audioLimit
    self.convPreLookRight = convPreLookRight
    self.upsampleRates = upsampleRates

    numKernels = resblockKernelSizes.count
    numUpsamples = upsampleRates.count

    let upsampleScale = upsampleRates.reduce(1, *) * istftParams["hop_len"]!
    f0UpsampleScale = upsampleScale

    _f0Predictor.wrappedValue = CosyVoice3F0Predictor(inChannels: inChannels, condChannels: baseChannels)

    _mSource.wrappedValue = CosyVoice3SourceModuleHnNSF2(
      samplingRate: samplingRate,
      upsampleScale: upsampleScale,
      harmonicNum: nbHarmonics,
      sineAmp: nsfAlpha,
      addNoiseStd: nsfSigma,
      voicedThreshold: nsfVoicedThreshold
    )

    _convPre.wrappedValue = CosyVoice3CausalConv1d(
      inputChannels: inChannels,
      outputChannels: baseChannels,
      kernelSize: convPreLookRight + 1,
      causalType: .right
    )

    var upsArray: [CosyVoice3CausalConv1dUpsample] = []
    for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
      upsArray.append(CosyVoice3CausalConv1dUpsample(
        inputChannels: baseChannels / (1 << i),
        outputChannels: baseChannels / (1 << (i + 1)),
        kernelSize: k,
        stride: u
      ))
    }
    _ups.wrappedValue = upsArray

    var sourceDownsArray: [Module] = []
    var sourceResArray: [CosyVoice3CausalResBlock] = []
    let downsampleRates = [1] + Array(upsampleRates.reversed().dropLast())
    var downsampleCumRates: [Int] = []
    var cumProd = 1
    for rate in downsampleRates {
      cumProd *= rate
      downsampleCumRates.append(cumProd)
    }

    let reversedCumRates = Array(downsampleCumRates.reversed())
    for (i, ((rate, kernelSize), dilations)) in zip(
      zip(reversedCumRates, sourceResblockKernelSizes),
      sourceResblockDilationSizes
    ).enumerated() {
      let nFft = istftParams["n_fft"]!
      if rate == 1 {
        sourceDownsArray.append(CosyVoice3CausalConv1d(
          inputChannels: nFft + 2,
          outputChannels: baseChannels / (1 << (i + 1)),
          kernelSize: 1,
          causalType: .left
        ))
      } else {
        sourceDownsArray.append(CosyVoice3CausalConv1dDownSample(
          inputChannels: nFft + 2,
          outputChannels: baseChannels / (1 << (i + 1)),
          kernelSize: rate * 2,
          stride: rate
        ))
      }
      sourceResArray.append(CosyVoice3CausalResBlock(
        channels: baseChannels / (1 << (i + 1)),
        kernelSize: kernelSize,
        dilations: dilations
      ))
    }
    _sourceDowns.wrappedValue = sourceDownsArray
    _sourceResblocks.wrappedValue = sourceResArray

    var resArray: [CosyVoice3CausalResBlock] = []
    for i in 0 ..< upsampleRates.count {
      let ch = baseChannels / (1 << (i + 1))
      for (k, d) in zip(resblockKernelSizes, resblockDilationSizes) {
        resArray.append(CosyVoice3CausalResBlock(channels: ch, kernelSize: k, dilations: d))
      }
    }
    _resblocks.wrappedValue = resArray

    let finalCh = baseChannels / (1 << upsampleRates.count)
    _convPost.wrappedValue = CosyVoice3CausalConv1d(
      inputChannels: finalCh,
      outputChannels: istftParams["n_fft"]! + 2,
      kernelSize: 7,
      causalType: .left
    )

    stftWindow = cosyVoice3HannWindowPeriodic(size: istftParams["n_fft"]!)
  }

  private func f0Upsample(_ f0: MLXArray) -> MLXArray {
    MLX.repeated(f0, count: f0UpsampleScale, axis: 2)
  }

  func decode(x: MLXArray, s: MLXArray, finalize: Bool = true) -> MLXArray {
    var (sStftReal, sStftImag) = cosyVoice3Stft(
      x: s.squeezed(axis: 1),
      nFft: istftParams["n_fft"]!,
      hopLength: istftParams["hop_len"]!,
      window: stftWindow
    )

    var h: MLXArray

    if finalize {
      h = convPre(x)
    } else {
      let causalPadding = convPre.causalPadding
      let xMain = x[0..., 0..., 0 ..< (x.shape[2] - causalPadding)]
      let xContext = x[0..., 0..., (x.shape[2] - causalPadding)...]
      h = convPre(xMain, cache: xContext)
      let trimLen = upsampleRates.reduce(1, *) * convPreLookRight
      sStftReal = sStftReal[0..., 0..., 0 ..< (sStftReal.shape[2] - trimLen)]
      sStftImag = sStftImag[0..., 0..., 0 ..< (sStftImag.shape[2] - trimLen)]
    }

    let sStft = MLX.concatenated([sStftReal, sStftImag], axis: 1)

    for i in 0 ..< numUpsamples {
      h = leakyRelu(h, negativeSlope: lreluSlope)
      h = ups[i](h)

      if i == numUpsamples - 1 {
        h = MLX.concatenated([h[0..., 0..., 1 ..< 2], h], axis: 2)
      }

      let si: MLXArray
      if let downConv = sourceDowns[i] as? CosyVoice3CausalConv1d {
        si = sourceResblocks[i](downConv(sStft))
      } else if let downConv = sourceDowns[i] as? CosyVoice3CausalConv1dDownSample {
        si = sourceResblocks[i](downConv(sStft))
      } else {
        fatalError("Unknown source down type")
      }
      h = h + si

      var xs: MLXArray?
      for j in 0 ..< numKernels {
        if xs == nil {
          xs = resblocks[i * numKernels + j](h)
        } else {
          xs = xs! + resblocks[i * numKernels + j](h)
        }
      }
      h = xs! / Float(numKernels)
    }

    // Note: Python uses default leaky_relu slope (0.01) here, not lreluSlope (0.1)
    h = leakyRelu(h)
    h = convPost(h)

    let nFftHalf = istftParams["n_fft"]! / 2 + 1
    let magnitude = MLX.exp(h[0..., 0 ..< nFftHalf, 0...])
    let phase = MLX.sin(h[0..., nFftHalf..., 0...])

    var output = cosyVoice3Istft(
      magnitude: magnitude,
      phase: phase,
      nFft: istftParams["n_fft"]!,
      hopLength: istftParams["hop_len"]!,
      window: stftWindow
    )

    if !finalize {
      let trimLen = upsampleRates.reduce(1, *) * istftParams["hop_len"]!
      output = output[0..., 0 ..< (output.shape[1] - trimLen)]
    }

    output = MLX.clip(output, min: MLXArray(-audioLimit), max: MLXArray(audioLimit))
    return output
  }

  func callAsFunction(_ speechFeat: MLXArray, finalize: Bool = true) -> (MLXArray, MLXArray) {
    let f0 = f0Predictor(speechFeat, finalize: finalize)
    var s = f0Upsample(f0.expandedDimensions(axis: 1))
    s = s.swappedAxes(1, 2)

    let (sineMerge, _, _) = mSource(s)
    s = sineMerge.swappedAxes(1, 2)

    let generatedSpeech: MLXArray
    if finalize {
      generatedSpeech = decode(x: speechFeat, s: s, finalize: finalize)
    } else {
      let causalPadding = f0Predictor.condnet0CausalPadding
      let melTrimmed = speechFeat[0..., 0..., 0 ..< (speechFeat.shape[2] - causalPadding)]
      generatedSpeech = decode(x: melTrimmed, s: s, finalize: finalize)
    }

    return (generatedSpeech, s)
  }

  func inference(_ speechFeat: MLXArray, finalize: Bool = true) -> (MLXArray, MLXArray) {
    callAsFunction(speechFeat, finalize: finalize)
  }
}
