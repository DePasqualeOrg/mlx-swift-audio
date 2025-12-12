//  HiFi-GAN vocoder with Neural Source Filter for speech synthesis

import Foundation
import MLX
import MLXNN

// MARK: - Helper Functions

/// Create periodic Hann window matching scipy fftbins=True
func hannWindowPeriodic(size: Int) -> MLXArray {
  let values = (0 ..< size).map { n in
    0.5 * (1 - cos(2 * Float.pi * Float(n) / Float(size)))
  }
  return MLXArray(values)
}

/// Calculate padding for 'same' convolution
func getPadding(kernelSize: Int, dilation: Int = 1) -> Int {
  (kernelSize * dilation - dilation) / 2
}

// MARK: - Snake Activation

/// Snake activation function: x + (1/α) * sin²(αx)
class Snake: Module {
  let inFeatures: Int
  let alphaLogscale: Bool
  var alpha: MLXArray
  let noDivByZero: Float = 1e-9

  init(inFeatures: Int, alpha: Float = 1.0, alphaLogscale: Bool = false) {
    self.inFeatures = inFeatures
    self.alphaLogscale = alphaLogscale

    if alphaLogscale {
      self.alpha = MLXArray.zeros([inFeatures]) * alpha
    } else {
      self.alpha = MLXArray.ones([inFeatures]) * alpha
    }
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Reshape alpha to align with x: (C,) -> (1, C, 1)
    var a = alpha.reshaped([1, -1, 1])

    if alphaLogscale {
      a = MLX.exp(a)
    }

    // Snake activation: x + (1/α) * sin²(αx)
    // For fp16 safety, clamp |alpha| to prevent overflow when |1/alpha| > 65504
    // min_alpha = 1e-4 means 1/min_alpha = 10000, safe for fp16
    let minAlpha: Float = 1e-4
    let alphaSign = MLX.sign(a)
    let alphaAbs = MLX.abs(a)

    // Clamp magnitude while preserving sign
    var alphaClamped = alphaSign * MLX.maximum(alphaAbs, MLXArray(minAlpha))

    // Handle case where alpha is extremely close to zero (sign might be 0)
    alphaClamped = MLX.where(alphaAbs .< noDivByZero, MLXArray(minAlpha), alphaClamped)

    return x + (1.0 / alphaClamped) * MLX.pow(MLX.sin(x * a), 2)
  }
}

// MARK: - ResBlock

/// Residual block with Snake activation for HiFi-GAN
class HiFiGANResBlock: Module {
  let channels: Int
  @ModuleInfo(key: "convs1") var convs1: [Conv1d]
  @ModuleInfo(key: "convs2") var convs2: [Conv1d]
  @ModuleInfo(key: "activations1") var activations1: [Snake]
  @ModuleInfo(key: "activations2") var activations2: [Snake]

  init(channels: Int = 512, kernelSize: Int = 3, dilations: [Int] = [1, 3, 5]) {
    self.channels = channels

    var c1: [Conv1d] = []
    var c2: [Conv1d] = []
    var a1: [Snake] = []
    var a2: [Snake] = []

    for dilation in dilations {
      c1.append(Conv1d(
        inputChannels: channels,
        outputChannels: channels,
        kernelSize: kernelSize,
        stride: 1,
        padding: getPadding(kernelSize: kernelSize, dilation: dilation),
        dilation: dilation,
      ))
      c2.append(Conv1d(
        inputChannels: channels,
        outputChannels: channels,
        kernelSize: kernelSize,
        stride: 1,
        padding: getPadding(kernelSize: kernelSize, dilation: 1),
      ))
      a1.append(Snake(inFeatures: channels, alphaLogscale: false))
      a2.append(Snake(inFeatures: channels, alphaLogscale: false))
    }

    _convs1.wrappedValue = c1
    _convs2.wrappedValue = c2
    _activations1.wrappedValue = a1
    _activations2.wrappedValue = a2
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var result = x
    for i in 0 ..< convs1.count {
      var xt = activations1[i](result)
      // Conv1d: (B, C, T) -> transpose -> conv -> transpose back
      xt = xt.swappedAxes(1, 2)
      xt = convs1[i](xt)
      xt = xt.swappedAxes(1, 2)
      xt = activations2[i](xt)
      xt = xt.swappedAxes(1, 2)
      xt = convs2[i](xt)
      xt = xt.swappedAxes(1, 2)
      result = xt + result // Residual connection
    }
    return result
  }
}

// MARK: - SineGen

/// Sine wave generator for harmonic synthesis
class SineGen: Module {
  let sineAmp: Float
  let noiseStd: Float
  let harmonicNum: Int
  let samplingRate: Int
  let voicedThreshold: Float

  init(
    sampRate: Int,
    harmonicNum: Int = 0,
    sineAmp: Float = 0.1,
    noiseStd: Float = 0.003,
    voicedThreshold: Float = 0,
  ) {
    self.sineAmp = sineAmp
    self.noiseStd = noiseStd
    self.harmonicNum = harmonicNum
    samplingRate = sampRate
    self.voicedThreshold = voicedThreshold
  }

  /// Convert F0 to voiced/unvoiced signal
  private func f02uv(_ f0: MLXArray) -> MLXArray {
    (f0 .> voicedThreshold).asType(.float32)
  }

  /// Generate sine waves from F0
  /// - Parameter f0: Fundamental frequency tensor (B, 1, T) in Hz
  /// - Returns: Tuple of (sine_waves, uv, noise)
  func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let shape = f0.shape
    let B = shape[0]

    // Create frequency matrix for harmonics
    let harmonicMultipliers = MLXArray(1 ... (harmonicNum + 1)).reshaped([1, -1, 1])
    let FMat = f0 * harmonicMultipliers.asType(.float32) / Float(samplingRate)

    // Calculate phase
    let thetaMat = 2 * Float.pi * (MLX.cumsum(FMat, axis: -1) % 1)

    // Random phase offset for each harmonic
    var phaseVec = MLXRandom.uniform(
      low: -Float.pi,
      high: Float.pi,
      [B, harmonicNum + 1, 1],
    )
    // Zero out fundamental frequency phase offset (index 0)
    let mask = MLXArray(0 ..< (harmonicNum + 1)).reshaped([1, -1, 1]) .> 0
    phaseVec = MLX.where(mask, phaseVec, MLXArray(0.0))

    // Generate sine waveforms
    var sineWaves = sineAmp * MLX.sin(thetaMat + phaseVec)

    // Generate voiced/unvoiced signal
    let uv = f02uv(f0)

    // Noise: larger for unvoiced, smaller for voiced
    let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3
    let noise = noiseAmp * MLXRandom.normal(sineWaves.shape)

    // Voiced regions use sine, unvoiced use noise
    sineWaves = sineWaves * uv + noise

    return (sineWaves, uv, noise)
  }
}

// MARK: - SourceModuleHnNSF

/// Neural Source Filter (NSF) module for harmonic and noise generation
class SourceModuleHnNSF: Module {
  let sineAmp: Float
  let noiseStd: Float

  @ModuleInfo(key: "l_sin_gen") var lSinGen: SineGen
  @ModuleInfo(key: "l_linear") var lLinear: Linear

  init(
    samplingRate: Int,
    upsampleScale _: Int,
    harmonicNum: Int = 0,
    sineAmp: Float = 0.1,
    addNoiseStd: Float = 0.003,
    voicedThreshold: Float = 0,
  ) {
    self.sineAmp = sineAmp
    noiseStd = addNoiseStd

    _lSinGen.wrappedValue = SineGen(
      sampRate: samplingRate,
      harmonicNum: harmonicNum,
      sineAmp: sineAmp,
      noiseStd: addNoiseStd,
      voicedThreshold: voicedThreshold,
    )
    _lLinear.wrappedValue = Linear(harmonicNum + 1, 1)
  }

  /// Generate harmonic and noise sources from F0
  /// - Parameter x: F0 tensor (B, T, 1)
  /// - Returns: Tuple of (sine_merge, noise, uv)
  func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    // Generate sine harmonics
    let (sineWavs, uv, _) = lSinGen(x.swappedAxes(1, 2))
    let sineWavsT = sineWavs.swappedAxes(1, 2)
    let uvT = uv.swappedAxes(1, 2)

    // Merge harmonics
    let sineMerge = tanh(lLinear(sineWavsT))

    // Source for noise branch
    let noise = MLXRandom.normal(uvT.shape) * sineAmp / 3

    return (sineMerge, noise, uvT)
  }
}

// MARK: - STFT Functions

/// Short-Time Fourier Transform
func stftHiFiGAN(x: MLXArray, nFft: Int, hopLength: Int, window: MLXArray) -> (MLXArray, MLXArray) {
  let shape = x.shape
  let B = shape[0]
  let T = shape[1]

  // Reflect padding to match PyTorch's torch.stft behavior
  // Left pad: reverse of x[:, 1:pad_length+1]
  // Right pad: reverse of x[:, -(pad_length+1):-1]
  let padLength = nFft / 2
  let leftPad = reverseAlongAxis(x[0..., 1 ..< (padLength + 1)], axis: 1)
  let rightPad = reverseAlongAxis(x[0..., (T - padLength - 1) ..< (T - 1)], axis: 1)
  let xPadded = MLX.concatenated([leftPad, x, rightPad], axis: 1)

  // Calculate number of frames
  let numFrames = (xPadded.shape[1] - nFft) / hopLength + 1

  // Create frames using vectorized slicing
  let frameStarts = MLXArray(0 ..< numFrames) * Int32(hopLength)
  let sampleOffsets = MLXArray(0 ..< nFft)
  let allIndices = frameStarts.expandedDimensions(axis: 1) + sampleOffsets.expandedDimensions(axis: 0)

  // Gather frames
  var frames = MLX.take(xPadded, allIndices.flattened(), axis: 1).reshaped([B, numFrames, nFft])
  frames = frames.swappedAxes(1, 2) // (B, nFft, numFrames)

  // Apply window
  let windowExpanded = window.reshaped([1, -1, 1])
  frames = frames * windowExpanded

  // FFT
  let fftResult = MLXFFT.fft(frames, axis: 1)

  // Take positive frequencies
  let fftSlice = fftResult[0..., 0 ..< (nFft / 2 + 1), 0...]
  let real = fftSlice.realPart()
  let imag = fftSlice.imaginaryPart()

  return (real, imag)
}

/// Inverse Short-Time Fourier Transform
func istftHiFiGAN(magnitude: MLXArray, phase: MLXArray, nFft: Int, hopLength: Int, window: MLXArray) -> MLXArray {
  // Clip magnitude
  let magClipped = MLX.clip(magnitude, max: Float(1e2))

  // Convert to complex
  let real = magClipped * MLX.cos(phase)
  let imag = magClipped * MLX.sin(phase)

  // Create full spectrum (conjugate symmetric)
  let B = real.shape[0]
  let F = real.shape[1]
  let numFrames = real.shape[2]

  // Mirror the spectrum for negative frequencies: [:, 1:-1, :][:, ::-1, :]
  let realSlice = real[0..., 1 ..< (F - 1), 0...]
  let imagSlice = imag[0..., 1 ..< (F - 1), 0...]
  let realMirror = reverseAlongAxis(realSlice, axis: 1)
  let imagMirror = reverseAlongAxis(imagSlice, axis: 1)
  let realFull = MLX.concatenated([real, realMirror], axis: 1)
  let imagFull = MLX.concatenated([imag, -imagMirror], axis: 1)

  // Combine into complex: spectrum = real_full + 1j * imag_full
  let oneJ = MLXArray(real: 0, imaginary: 1)
  let spectrum = realFull + oneJ * imagFull

  // Inverse FFT
  var frames = MLXFFT.ifft(spectrum, axis: 1)
  frames = frames.realPart() // mx.real(frames)

  // Apply window
  let windowExpanded = window.reshaped([1, -1, 1])
  frames = frames * windowExpanded

  // Overlap-add
  let outputLength = (numFrames - 1) * hopLength + nFft

  // Create index arrays for scatter-add
  let frameOffsets = MLXArray(0 ..< numFrames) * Int32(hopLength)
  let sampleIndices = MLXArray(0 ..< nFft)
  let indices = frameOffsets.expandedDimensions(axis: 1) + sampleIndices.expandedDimensions(axis: 0)
  let indicesFlat = indices.flattened()

  // Window squared for normalization
  let windowSq = MLX.pow(window, 2)
  var windowSum = MLXArray.zeros([outputLength])
  let windowUpdates = MLX.tiled(windowSq, repetitions: [numFrames])
  windowSum = windowSum.at[indicesFlat].add(windowUpdates)
  windowSum = MLX.maximum(windowSum, MLXArray(1e-8))

  // Vectorized overlap-add for all batch items
  let frameData = frames.swappedAxes(1, 2) // (B, num_frames, n_fft)
  let updates = frameData.reshaped([B, -1]) // (B, num_frames * n_fft)

  var output = MLXArray.zeros([B, outputLength])
  let batchIndices = MLX.repeated(MLXArray(0 ..< Int32(B)), count: numFrames * nFft, axis: 0)
  let flatIndices = MLX.tiled(indicesFlat, repetitions: [B])
  var flatOutput = output.flattened()
  let linearIndices = batchIndices * Int32(outputLength) + flatIndices
  flatOutput = flatOutput.at[linearIndices].add(updates.flattened())
  output = flatOutput.reshaped([B, outputLength])

  // Normalize with window sum
  output = output / windowSum

  // Remove padding
  let padLength = nFft / 2
  output = output[0..., padLength ..< (outputLength - padLength)]

  return output
}

// MARK: - HiFTGenerator

/// HiFi-GAN with Neural Source Filter (HiFT-Net) generator
class HiFTGenerator: Module {
  let outChannels: Int = 1
  let nbHarmonics: Int
  let samplingRate: Int
  let istftParams: [String: Int]
  let lreluSlope: Float
  let audioLimit: Float
  let numKernels: Int
  let numUpsamples: Int
  let f0UpsampleScale: Int

  @ModuleInfo(key: "m_source") var mSource: SourceModuleHnNSF
  @ModuleInfo(key: "conv_pre") var convPre: Conv1d
  @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
  @ModuleInfo(key: "source_downs") var sourceDowns: [Conv1d]
  @ModuleInfo(key: "source_resblocks") var sourceResblocks: [HiFiGANResBlock]
  @ModuleInfo(key: "resblocks") var resblocks: [HiFiGANResBlock]
  @ModuleInfo(key: "conv_post") var convPost: Conv1d
  @ModuleInfo(key: "f0_predictor") var f0Predictor: ConvRNNF0Predictor
  /// STFT window - underscore prefix excludes from parameter validation
  var _stftWindow: MLXArray

  init(
    inChannels: Int = 80,
    baseChannels: Int = 512,
    nbHarmonics: Int = 8,
    samplingRate: Int = 22050,
    nsfAlpha: Float = 0.1,
    nsfSigma: Float = 0.003,
    nsfVoicedThreshold: Float = 10,
    upsampleRates: [Int] = [8, 8],
    upsampleKernelSizes: [Int] = [16, 16],
    istftParams: [String: Int] = ["n_fft": 16, "hop_len": 4],
    resblockKernelSizes: [Int] = [3, 7, 11],
    resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    sourceResblockKernelSizes: [Int] = [7, 11],
    sourceResblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5]],
    lreluSlope: Float = 0.1,
    audioLimit: Float = 0.99,
    f0Predictor: ConvRNNF0Predictor? = nil,
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

    // Neural Source Filter
    _mSource.wrappedValue = SourceModuleHnNSF(
      samplingRate: samplingRate,
      upsampleScale: upsampleScale,
      harmonicNum: nbHarmonics,
      sineAmp: nsfAlpha,
      addNoiseStd: nsfSigma,
      voicedThreshold: nsfVoicedThreshold,
    )

    // Pre-convolution
    _convPre.wrappedValue = Conv1d(
      inputChannels: inChannels,
      outputChannels: baseChannels,
      kernelSize: 7,
      stride: 1,
      padding: 3,
    )

    // Upsampling layers
    var upsArray: [ConvTransposed1d] = []
    for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
      upsArray.append(ConvTransposed1d(
        inputChannels: baseChannels / (1 << i),
        outputChannels: baseChannels / (1 << (i + 1)),
        kernelSize: k,
        stride: u,
        padding: (k - u) / 2,
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

    for (i, (u, (k, d))) in zip(
      downsampleCumRates.reversed(),
      zip(sourceResblockKernelSizes, sourceResblockDilationSizes),
    ).enumerated() {
      let nFft = istftParams["n_fft"]!
      if u == 1 {
        sourceDownsArray.append(Conv1d(
          inputChannels: nFft + 2,
          outputChannels: baseChannels / (1 << (i + 1)),
          kernelSize: 1,
          stride: 1,
        ))
      } else {
        sourceDownsArray.append(Conv1d(
          inputChannels: nFft + 2,
          outputChannels: baseChannels / (1 << (i + 1)),
          kernelSize: u * 2,
          stride: u,
          padding: u / 2,
        ))
      }
      sourceResArray.append(HiFiGANResBlock(
        channels: baseChannels / (1 << (i + 1)),
        kernelSize: k,
        dilations: d,
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
    _convPost.wrappedValue = Conv1d(
      inputChannels: finalCh,
      outputChannels: istftParams["n_fft"]! + 2,
      kernelSize: 7,
      stride: 1,
      padding: 3,
    )

    // STFT window
    _stftWindow = hannWindowPeriodic(size: istftParams["n_fft"]!)

    // F0 predictor
    _f0Predictor.wrappedValue = f0Predictor ?? ConvRNNF0Predictor()
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
      window: _stftWindow,
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

      // Apply residual blocks - compute all independently then sum
      // (MLX can parallelize independent operations in the computation graph)
      let baseIdx = i * numKernels
      let resOutputs = (0 ..< numKernels).map { resblocks[baseIdx + $0](h) }
      h = resOutputs.dropFirst().reduce(resOutputs[0], +) / Float(numKernels)
    }

    h = leakyRelu(h, negativeSlope: lreluSlope)
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
      window: _stftWindow,
    )
    output = MLX.clip(output, min: MLXArray(-audioLimit), max: MLXArray(audioLimit))

    return output
  }

  /// Generate waveform from mel-spectrogram
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
