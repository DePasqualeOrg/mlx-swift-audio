// CosyVoice2 HiFi-GAN vocoder (24kHz)
// Ported from mlx-audio-plus cosyvoice2/hifigan.py

import Foundation
import MLX
import MLXNN

// MARK: - Linear Interpolation

/// Linear interpolation for 1D signals matching PyTorch's F.interpolate behavior
/// - Parameters:
///   - x: Input array (B, T, C) or (B, T)
///   - scaleFactor: Scale factor for interpolation
/// - Returns: Interpolated array
func linearInterpolate1d(_ x: MLXArray, scaleFactor: Float) -> MLXArray {
  var input = x
  var squeeze = false

  if input.ndim == 2 {
    input = input.expandedDimensions(axis: -1)
    squeeze = true
  }

  let T = input.shape[1]
  var newT = Int(Float(T) * scaleFactor)

  if newT == 0 {
    newT = 1
  }

  // PyTorch F.interpolate with align_corners=False uses:
  // source_index = (dest_index + 0.5) * (input_size / output_size) - 0.5
  var indices = (MLXArray(0 ..< Int32(newT)).asType(.float32) + 0.5) * (Float(T) / Float(newT)) - 0.5
  indices = MLX.clip(indices, min: 0, max: Float(T) - 1.001)

  let idxLow = MLX.floor(indices).asType(.int32)
  let idxHigh = MLX.minimum(idxLow + 1, MLXArray(Int32(T - 1)))
  var weightHigh = indices - idxLow.asType(.float32)
  var weightLow = 1.0 - weightHigh

  // Vectorized gather and interpolate
  let lowVals = input[0..., idxLow, 0...] // (B, newT, C)
  let highVals = input[0..., idxHigh, 0...] // (B, newT, C)

  // Reshape weights for broadcasting: (newT,) -> (1, newT, 1)
  weightLow = weightLow.reshaped(1, -1, 1)
  weightHigh = weightHigh.reshaped(1, -1, 1)

  var out = lowVals * weightLow + highVals * weightHigh

  if squeeze {
    out = out.squeezed(axis: -1)
  }

  return out
}

// MARK: - SineGen2

/// Sine generator for CosyVoice2 (24kHz version)
/// Uses interpolation for phase calculation, required for 24kHz sampling rate
class SineGen2: Module {
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

  /// Generate UV (unvoiced) signal from F0
  private func f02uv(_ f0: MLXArray) -> MLXArray {
    (f0 .> voicedThreshold).asType(.float32)
  }

  /// Convert F0 values to sine waveforms with interpolation
  private func f02sine(_ f0Values: MLXArray) -> MLXArray {
    // Convert to normalized frequency (rad values)
    var radValues = (f0Values / Float(samplingRate)) % 1

    let (B, T, D) = (radValues.shape[0], radValues.shape[1], radValues.shape[2])

    // Initial phase noise (no noise for fundamental component)
    var randIni = MLXRandom.uniform(low: Float(0), high: Float(1), [B, D])
    // Zero out fundamental frequency phase
    let zeroCol = MLXArray.zeros([B, 1])
    randIni = MLX.concatenated([zeroCol, randIni[0..., 1...]], axis: 1)

    // Add random initial phase to first time step
    let firstStep = radValues[0..., 0 ..< 1, 0...] + randIni.expandedDimensions(axis: 1)
    radValues = MLX.concatenated([firstStep, radValues[0..., 1..., 0...]], axis: 1)

    // Downsample, accumulate phase, then upsample
    let radDownsampled = linearInterpolate1d(radValues, scaleFactor: 1.0 / Float(upsampleScale))

    // Cumulative sum for phase
    var phase = MLX.cumsum(radDownsampled, axis: 1) * 2 * Float.pi

    // Upsample phase back
    phase = linearInterpolate1d(phase * Float(upsampleScale), scaleFactor: Float(upsampleScale))

    // Trim to match original length
    phase = phase[0..., 0 ..< T, 0...]

    // Generate sine
    return MLX.sin(phase)
  }

  /// Generate sine waveforms from F0
  /// - Parameter f0: F0 values (B, T, 1)
  /// - Returns: Tuple of (sine_waves, uv, noise)
  func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    // Create harmonics: f0 * [1, 2, 3, ..., harmonic_num+1]
    let harmonics = MLXArray(1 ... Int32(harmonicNum + 1)).asType(.float32)
    let fn = f0 * harmonics.reshaped(1, 1, -1) // (B, T, dim)

    // Generate sine waveforms
    var sineWaves = f02sine(fn) * sineAmp

    // Generate UV signal
    let uv = f02uv(f0)

    // Noise: for unvoiced similar to sine_amp, for voiced use noise_std
    let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3
    let noise = noiseAmp * MLXRandom.normal(sineWaves.shape)

    // Apply UV masking and add noise
    sineWaves = sineWaves * uv + noise

    return (sineWaves, uv, noise)
  }
}

// MARK: - SourceModuleHnNSF2

/// Source module for CosyVoice2 (24kHz version)
/// Uses SineGen2 with interpolation for proper 24kHz generation
class SourceModuleHnNSF2: Module {
  let sineAmp: Float
  let noiseStd: Float

  @ModuleInfo(key: "l_sin_gen") var lSinGen: SineGen2
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

    _lSinGen.wrappedValue = SineGen2(
      samplingRate: samplingRate,
      upsampleScale: upsampleScale,
      harmonicNum: harmonicNum,
      sineAmp: sineAmp,
      noiseStd: addNoiseStd,
      voicedThreshold: voicedThreshold
    )

    _lLinear.wrappedValue = Linear(harmonicNum + 1, 1)
  }

  /// Generate source signal from F0
  /// - Parameter x: F0 values (B, T, 1)
  /// - Returns: Tuple of (source, noise, uv)
  func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    // Generate sine waveforms
    let (sineWavs, uv, _) = lSinGen(x)

    // Merge harmonics with tanh activation
    let sineMerge = tanh(lLinear(sineWavs))

    // Generate noise for noise branch
    let noise = MLXRandom.normal(uv.shape) * sineAmp / 3

    return (sineMerge, noise, uv)
  }
}

// MARK: - CosyF0Predictor

/// F0 predictor for CosyVoice2 HiFTGenerator (ConvRNNF0Predictor)
/// Takes mel spectrogram and predicts F0 (fundamental frequency)
class CosyF0Predictor: Module {
  @ModuleInfo(key: "condnet_0") var condnet0: Conv1d
  @ModuleInfo(key: "condnet_2") var condnet2: Conv1d
  @ModuleInfo(key: "condnet_4") var condnet4: Conv1d
  @ModuleInfo(key: "condnet_6") var condnet6: Conv1d
  @ModuleInfo(key: "condnet_8") var condnet8: Conv1d
  @ModuleInfo(key: "classifier") var classifier: Linear

  init(
    inChannels: Int = 80,
    hiddenChannels: Int = 512,
    numLayers _: Int = 5,
    kernelSize: Int = 3
  ) {
    let padding = kernelSize / 2

    _condnet0.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: hiddenChannels, kernelSize: kernelSize, padding: padding)
    _condnet2.wrappedValue = Conv1d(inputChannels: hiddenChannels, outputChannels: hiddenChannels, kernelSize: kernelSize, padding: padding)
    _condnet4.wrappedValue = Conv1d(inputChannels: hiddenChannels, outputChannels: hiddenChannels, kernelSize: kernelSize, padding: padding)
    _condnet6.wrappedValue = Conv1d(inputChannels: hiddenChannels, outputChannels: hiddenChannels, kernelSize: kernelSize, padding: padding)
    _condnet8.wrappedValue = Conv1d(inputChannels: hiddenChannels, outputChannels: hiddenChannels, kernelSize: kernelSize, padding: padding)

    _classifier.wrappedValue = Linear(hiddenChannels, 1)
  }

  /// Predict F0 from mel spectrogram
  /// - Parameter x: Mel spectrogram (B, C, T)
  /// - Returns: F0 prediction (B, T)
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x is (B, C, T), transpose to (B, T, C) for Conv1d
    var h = x.swappedAxes(1, 2)

    // Apply condnet layers with ELU activation
    h = elu(condnet0(h))
    h = elu(condnet2(h))
    h = elu(condnet4(h))
    h = elu(condnet6(h))
    h = elu(condnet8(h))

    // Classifier: (B, T, hidden_channels) -> (B, T, 1) -> (B, T)
    var f0 = classifier(h).squeezed(axis: -1)

    // Use abs() to ensure non-negative F0
    f0 = MLX.abs(f0)

    return f0
  }
}

// MARK: - CosyHiFTGenerator

/// HiFi-GAN with Neural Source Filter (HiFT-Net) for CosyVoice2
/// This version has F0 predictor built-in as a submodule and operates at 24kHz
class CosyHiFTGenerator: Module {
  let outChannels: Int = 1
  let nbHarmonics: Int
  let samplingRate: Int
  let istftParams: [String: Int]
  let lreluSlope: Float
  let audioLimit: Float
  let numKernels: Int
  let numUpsamples: Int
  let f0UpsampleScale: Int

  @ModuleInfo(key: "f0_predictor") var f0Predictor: CosyF0Predictor
  @ModuleInfo(key: "m_source") var mSource: SourceModuleHnNSF2
  @ModuleInfo(key: "conv_pre") var convPre: Conv1d
  @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
  @ModuleInfo(key: "source_downs") var sourceDowns: [Conv1d]
  @ModuleInfo(key: "source_resblocks") var sourceResblocks: [HiFiGANResBlock]
  @ModuleInfo(key: "resblocks") var resblocks: [HiFiGANResBlock]
  @ModuleInfo(key: "conv_post") var convPost: Conv1d

  /// STFT window
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
    sourceResblockKernelSizes: [Int] = [7, 7, 11],
    sourceResblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    lreluSlope: Float = 0.1,
    audioLimit: Float = 0.99
  ) {
    self.nbHarmonics = nbHarmonics
    self.samplingRate = samplingRate
    self.istftParams = istftParams
    self.lreluSlope = lreluSlope
    self.audioLimit = audioLimit

    numKernels = resblockKernelSizes.count
    numUpsamples = upsampleRates.count

    let upsampleScale = upsampleRates.reduce(1, *) * istftParams["hop_len"]!
    f0UpsampleScale = upsampleScale

    // Built-in F0 predictor
    _f0Predictor.wrappedValue = CosyF0Predictor(
      inChannels: inChannels,
      hiddenChannels: baseChannels
    )

    // Neural Source Filter - use SourceModuleHnNSF2 for 24kHz
    _mSource.wrappedValue = SourceModuleHnNSF2(
      samplingRate: samplingRate,
      upsampleScale: upsampleScale,
      harmonicNum: nbHarmonics,
      sineAmp: nsfAlpha,
      addNoiseStd: nsfSigma,
      voicedThreshold: nsfVoicedThreshold
    )

    // Pre-convolution
    _convPre.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: baseChannels, kernelSize: 7, stride: 1, padding: 3)

    // Upsampling layers
    var upsArray: [ConvTransposed1d] = []
    for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
      upsArray.append(ConvTransposed1d(
        inputChannels: baseChannels / (1 << i),
        outputChannels: baseChannels / (1 << (i + 1)),
        kernelSize: k,
        stride: u,
        padding: (k - u) / 2
      ))
    }
    _ups.wrappedValue = upsArray

    // Source downsampling and resblocks
    var sourceDownsArray: [Conv1d] = []
    var sourceResArray: [HiFiGANResBlock] = []
    let downsampleRates = [1] + Array(upsampleRates.reversed().dropLast())
    var downsampleCumRates: [Int] = []
    var cumProd = 1
    for rate in downsampleRates {
      cumProd *= rate
      downsampleCumRates.append(cumProd)
    }

    let reversedCumRates = Array(downsampleCumRates.reversed())
    for i in 0 ..< sourceResblockKernelSizes.count {
      let u = reversedCumRates[i]
      let k = sourceResblockKernelSizes[i]
      let d = sourceResblockDilationSizes[i]
      let nFft = istftParams["n_fft"]!
      if u == 1 {
        sourceDownsArray.append(Conv1d(
          inputChannels: nFft + 2,
          outputChannels: baseChannels / (1 << (i + 1)),
          kernelSize: 1,
          stride: 1
        ))
      } else {
        sourceDownsArray.append(Conv1d(
          inputChannels: nFft + 2,
          outputChannels: baseChannels / (1 << (i + 1)),
          kernelSize: u * 2,
          stride: u,
          padding: u / 2
        ))
      }
      sourceResArray.append(HiFiGANResBlock(
        channels: baseChannels / (1 << (i + 1)),
        kernelSize: k,
        dilations: d
      ))
    }
    _sourceDowns.wrappedValue = sourceDownsArray
    _sourceResblocks.wrappedValue = sourceResArray

    // Residual blocks after each upsampling
    var resArray: [HiFiGANResBlock] = []
    for i in 0 ..< upsampleRates.count {
      let ch = baseChannels / (1 << (i + 1))
      for (k, d) in zip(resblockKernelSizes, resblockDilationSizes) {
        resArray.append(HiFiGANResBlock(channels: ch, kernelSize: k, dilations: d))
      }
    }
    _resblocks.wrappedValue = resArray

    // Post-convolution
    let finalCh = baseChannels / (1 << upsampleRates.count)
    _convPost.wrappedValue = Conv1d(inputChannels: finalCh, outputChannels: istftParams["n_fft"]! + 2, kernelSize: 7, stride: 1, padding: 3)

    // STFT window
    stftWindow = hannWindowPeriodic(size: istftParams["n_fft"]!)
  }

  /// Upsample F0 using nearest neighbor interpolation
  private func f0Upsample(_ f0: MLXArray) -> MLXArray {
    MLX.repeated(f0, count: f0UpsampleScale, axis: 2)
  }

  /// Decode mel-spectrogram to waveform
  func decode(x: MLXArray, s: MLXArray) -> MLXArray {
    // STFT of source signal
    let (sStftReal, sStftImag) = stftHiFiGAN(
      x: s.squeezed(axis: 1),
      nFft: istftParams["n_fft"]!,
      hopLength: istftParams["hop_len"]!,
      window: stftWindow
    )
    let sStft = MLX.concatenated([sStftReal, sStftImag], axis: 1)

    // Pre-convolution
    var h = x.swappedAxes(1, 2)
    h = convPre(h)
    h = h.swappedAxes(1, 2)

    for i in 0 ..< numUpsamples {
      h = leakyRelu(h, negativeSlope: lreluSlope)
      // ConvTranspose1d
      h = h.swappedAxes(1, 2)
      h = ups[i](h)
      h = h.swappedAxes(1, 2)

      if i == numUpsamples - 1 {
        // Reflection pad: pad 1 sample on left
        h = MLX.concatenated([h[0..., 0..., 1 ..< 2], h], axis: 2)
      }

      // Source fusion
      var si = sStft.swappedAxes(1, 2)
      si = sourceDowns[i](si)
      si = si.swappedAxes(1, 2)
      si = sourceResblocks[i](si)
      h = h + si

      // Apply residual blocks and average
      let baseIdx = i * numKernels
      let resOutputs = (0 ..< numKernels).map { resblocks[baseIdx + $0](h) }
      h = resOutputs.dropFirst().reduce(resOutputs[0], +) / Float(numKernels)
    }

    // Note: PyTorch CosyVoice2 uses default negative_slope=0.01 here (not 0.1)
    h = leakyRelu(h, negativeSlope: 0.01)
    h = h.swappedAxes(1, 2)
    h = convPost(h)
    h = h.swappedAxes(1, 2)

    // Split into magnitude and phase
    let nFftHalf = istftParams["n_fft"]! / 2 + 1
    let magnitude = MLX.exp(h[0..., 0 ..< nFftHalf, 0...])
    let phase = MLX.sin(h[0..., nFftHalf..., 0...])

    // Inverse STFT
    var output = istftHiFiGAN(
      magnitude: magnitude,
      phase: phase,
      nFft: istftParams["n_fft"]!,
      hopLength: istftParams["hop_len"]!,
      window: stftWindow
    )

    output = MLX.clip(output, min: MLXArray(-audioLimit), max: MLXArray(audioLimit))

    return output
  }

  /// Generate waveform from mel-spectrogram
  /// - Parameters:
  ///   - speechFeat: Mel-spectrogram (B, C, T)
  ///   - cacheSource: Cached source for streaming
  /// - Returns: Tuple of (waveform, source)
  func callAsFunction(_ speechFeat: MLXArray, cacheSource: MLXArray? = nil) -> (MLXArray, MLXArray) {
    let cache = cacheSource ?? MLXArray.zeros([1, 1, 0])

    // Predict F0 from mel
    let f0 = f0Predictor(speechFeat)

    // Upsample F0
    var s = f0Upsample(f0.expandedDimensions(axis: 1))
    s = s.swappedAxes(1, 2) // (B, T, 1)

    // Generate source from F0
    let (sineMerge, _, _) = mSource(s)
    s = sineMerge.swappedAxes(1, 2) // (B, 1, T)

    // Use cache to avoid glitch in streaming
    if cache.shape[2] != 0 {
      let cacheLen = cache.shape[2]
      s = MLX.concatenated([cache, s[0..., 0..., cacheLen...]], axis: 2)
    }

    // Decode mel + source to audio
    let generatedSpeech = decode(x: speechFeat, s: s)

    return (generatedSpeech, s)
  }

  /// Inference-mode forward pass
  func inference(_ speechFeat: MLXArray, cacheSource: MLXArray? = nil) -> (MLXArray, MLXArray) {
    callAsFunction(speechFeat, cacheSource: cacheSource)
  }
}
