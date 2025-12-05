//
//  S3Gen.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/s3gen/s3gen.py
//  S3Gen decoder: converts speech tokens to waveforms
//

import Foundation
import MLX
import MLXNN

// MARK: - Constants

/// S3Gen sample rate (24kHz for output)
let S3GenSr = 24000

/// S3 sample rate (16kHz for tokenizer)
let S3Sr = 16000

/// Speech vocabulary size (3^8 = 6561)
let SpeechVocabSize = 6561

// MARK: - Reference Dictionary

/// Container for reference audio embeddings
struct S3GenRefDict: @unchecked Sendable {
  /// Reference speech tokens (1, T_tok)
  var promptToken: MLXArray

  /// Reference token lengths (1,)
  var promptTokenLen: MLXArray

  /// Reference mel features (1, T, D)
  var promptFeat: MLXArray

  /// Reference mel feature lengths (1,)
  var promptFeatLen: MLXArray

  /// Speaker embedding from x-vector model
  var embedding: MLXArray

  init(
    promptToken: MLXArray,
    promptTokenLen: MLXArray,
    promptFeat: MLXArray,
    promptFeatLen: MLXArray,
    embedding: MLXArray,
  ) {
    self.promptToken = promptToken
    self.promptTokenLen = promptTokenLen
    self.promptFeat = promptFeat
    self.promptFeatLen = promptFeatLen
    self.embedding = embedding
  }
}

// MARK: - Audio Resampling

/// Resample audio using vectorized linear interpolation
/// Uses MLX array operations for efficient GPU-accelerated resampling
func resampleAudio(_ audio: MLXArray, origSr: Int, targetSr: Int) -> MLXArray {
  if origSr == targetSr {
    return audio
  }

  // Find GCD for rational resampling
  func gcd(_ a: Int, _ b: Int) -> Int {
    b == 0 ? a : gcd(b, a % b)
  }

  let g = gcd(origSr, targetSr)
  let up = targetSr / g
  let down = origSr / g

  let inputLen = audio.shape[0]
  let outputLen = (inputLen * up) / down

  // Vectorized linear interpolation using MLX operations
  // Compute source positions for all output samples at once
  let outputIndices = MLXArray(0 ..< outputLen).asType(.float32)
  let srcPositions = outputIndices * Float(down) / Float(up)

  // Get floor indices (left samples)
  let srcIndicesFloat = MLX.floor(srcPositions)
  let srcIndices = srcIndicesFloat.asType(.int32)

  // Compute fractional parts for interpolation
  let fracs = srcPositions - srcIndicesFloat

  // Clamp indices to valid range
  let maxIdx = MLXArray(Int32(inputLen - 1))
  let idx0 = MLX.minimum(srcIndices, maxIdx)
  let idx1 = MLX.minimum(srcIndices + 1, maxIdx)

  // Gather values at left and right indices
  let v0 = audio[idx0]
  let v1 = audio[idx1]

  // Linear interpolation: result = v0 * (1 - frac) + v1 * frac = v0 + frac * (v1 - v0)
  let result = v0 + fracs * (v1 - v0)

  return result
}

// MARK: - S3Token2Mel

/// S3Gen CFM decoder: maps S3 speech tokens to mel-spectrograms
///
/// This is the flow matching component that converts speech tokens to mel spectrograms
/// using a Conformer encoder and flow matching decoder with speaker conditioning.
class S3Token2Mel: Module {
  @ModuleInfo(key: "speaker_encoder") var speakerEncoder: CAMPPlus
  @ModuleInfo(key: "flow") var flow: CausalMaskedDiffWithXvec

  override init() {
    // Speaker encoder (CAM++ for x-vector extraction)
    _speakerEncoder.wrappedValue = CAMPPlus()

    // Conformer encoder with upsampling
    let encoder = UpsampleConformerEncoder(
      inputSize: 512,
      outputSize: 512,
      attentionHeads: 8,
      linearUnits: 2048,
      numBlocks: 6,
      dropoutRate: 0.1,
      positionalDropoutRate: 0.1,
      attentionDropoutRate: 0.1,
      inputLayer: "linear",
      posEncLayerType: "rel_pos_espnet",
      normalizeBefore: true,
      macaronStyle: false,
      selfattentionLayerType: "rel_selfattn",
      useCnnModule: false,
    )

    // Flow matching decoder (ConditionalDecoder as estimator)
    let estimator = ConditionalDecoder(
      inChannels: 320,
      outChannels: 80,
      causal: true,
      channels: [256],
      dropout: 0.0,
      attentionHeadDim: 64,
      nBlocks: 4,
      numMidBlocks: 12,
      numHeads: 8,
      actFn: "gelu",
    )

    let cfmParams = CFMParams()
    let decoder = CausalConditionalCFM(
      inChannels: 240,
      cfmParams: cfmParams,
      nSpks: 1,
      spkEmbDim: 80,
      estimator: estimator,
    )

    // Integration wrapper
    _flow.wrappedValue = CausalMaskedDiffWithXvec(
      inputSize: 512,
      outputSize: 80,
      spkEmbedDim: 192,
      vocabSize: SpeechVocabSize,
      inputFrameRate: 25,
      onlyMaskLoss: true,
      tokenMelRatio: 2,
      preLookaheadLen: 3,
      encoder: encoder,
      decoder: decoder,
    )

    super.init()
  }

  /// Embed reference audio for speaker conditioning
  func embedRef(
    refWav: MLXArray,
    refSr: Int,
    refSpeechTokens: MLXArray,
    refSpeechTokenLens _: MLXArray,
  ) -> S3GenRefDict {
    var wavArray = refWav
    if wavArray.ndim == 1 {
      wavArray = wavArray.expandedDimensions(axis: 0)
    }

    // Resample to 24kHz for mel extraction if needed
    var refWav24: MLXArray
    if refSr == S3GenSr {
      refWav24 = wavArray
    } else {
      refWav24 = resampleAudio(wavArray.squeezed(), origSr: refSr, targetSr: S3GenSr)
      refWav24 = refWav24.expandedDimensions(axis: 0)
    }

    // Extract mel features at 24kHz
    var refMels24 = s3genMelSpectrogram(
      y: refWav24,
      nFft: 1920,
      numMels: 80,
      samplingRate: S3GenSr,
      hopSize: 480,
      winSize: 1920,
      fmin: 0,
      fmax: 8000,
      center: false,
    )
    refMels24 = refMels24.transposed(0, 2, 1) // (B, T, D)

    // Speaker embedding (expects 16kHz audio)
    var refWav16: MLXArray
    if refSr == S3Sr {
      refWav16 = wavArray
    } else {
      refWav16 = resampleAudio(wavArray.squeezed(), origSr: refSr, targetSr: S3Sr)
      refWav16 = refWav16.expandedDimensions(axis: 0)
    }

    let refXVector = speakerEncoder.inference(refWav16)

    // Align mel frames and tokens (mel_frames = 2 * num_tokens)
    var tokens = refSpeechTokens
    let actualTokenLen = tokens.shape[1]
    let expectedTokenLen = refMels24.shape[1] / 2

    if actualTokenLen != expectedTokenLen {
      if actualTokenLen < expectedTokenLen {
        // Tokens are shorter - truncate mel to match
        let expectedMelLen = 2 * actualTokenLen
        refMels24 = refMels24[0..., 0 ..< expectedMelLen, 0...]
      } else {
        // Tokens are longer - truncate tokens to match
        tokens = tokens[0..., 0 ..< expectedTokenLen]
      }
    }

    let tokenLen = MLXArray([Int32(tokens.shape[1])])

    return S3GenRefDict(
      promptToken: tokens,
      promptTokenLen: tokenLen,
      promptFeat: refMels24,
      promptFeatLen: MLXArray([Int32(refMels24.shape[1])]),
      embedding: refXVector,
    )
  }

  /// Generate mel-spectrograms from S3 speech tokens
  func callAsFunction(
    speechTokens: MLXArray,
    refDict: S3GenRefDict,
    finalize: Bool = false,
  ) -> MLXArray {
    var tokens = speechTokens
    if tokens.ndim == 1 {
      tokens = tokens.expandedDimensions(axis: 0)
    }

    let speechTokenLens = MLXArray([Int32(tokens.shape[1])])

    let (outputMels, _) = flow.inference(
      token: tokens,
      tokenLen: speechTokenLens,
      promptToken: refDict.promptToken,
      promptTokenLen: refDict.promptTokenLen,
      promptFeat: refDict.promptFeat,
      promptFeatLen: refDict.promptFeatLen,
      embedding: refDict.embedding,
      finalize: finalize,
    )

    return outputMels
  }
}

// MARK: - S3Token2Wav

/// Full S3Gen decoder: token-to-mel (CFM) + mel-to-waveform (HiFiGAN)
///
/// This combines the flow matching decoder with the HiFi-GAN vocoder to
/// generate high-quality waveforms from speech tokens.
class S3Token2Wav: S3Token2Mel {
  @ModuleInfo(key: "mel2wav") var mel2wav: HiFTGenerator
  /// Fade-in window buffer - underscore prefix excludes from parameter validation
  var _trimFade: MLXArray

  override init() {
    // F0 predictor for vocoder
    let f0Predictor = ConvRNNF0Predictor()

    // HiFi-GAN vocoder
    _mel2wav.wrappedValue = HiFTGenerator(
      samplingRate: S3GenSr,
      upsampleRates: [8, 5, 3],
      upsampleKernelSizes: [16, 11, 7],
      sourceResblockKernelSizes: [7, 7, 11],
      sourceResblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
      f0Predictor: f0Predictor,
    )

    // Fade-in window to reduce artifacts (20ms)
    let nTrim = S3GenSr / 50
    let fadeCos = (MLX.cos(MLX.linspace(Float.pi, Float(0), count: nTrim)) + 1) / 2
    _trimFade = MLX.concatenated([MLXArray.zeros([nTrim]), fadeCos])

    super.init()
  }

  /// Generate waveforms from S3 speech tokens
  override func callAsFunction(
    speechTokens: MLXArray,
    refDict: S3GenRefDict,
    finalize: Bool = false,
  ) -> MLXArray {
    // Generate mel-spectrograms
    let outputMels = super.callAsFunction(
      speechTokens: speechTokens,
      refDict: refDict,
      finalize: finalize,
    )

    // Generate waveform from mels
    let hiftCacheSource = MLXArray.zeros([1, 1, 0])
    let (outputWavs, _) = mel2wav.inference(outputMels, cacheSource: hiftCacheSource)

    // Apply fade-in to reduce spillover artifacts
    var result = outputWavs
    let fadeLen = _trimFade.shape[0]
    if result.shape[1] >= fadeLen {
      result[0..., 0 ..< fadeLen] = result[0..., 0 ..< fadeLen] * _trimFade
    }

    return result
  }

  /// Run only the flow matching (token-to-mel) inference
  func flowInference(
    speechTokens: MLXArray,
    refDict: S3GenRefDict,
    finalize: Bool = false,
  ) -> MLXArray {
    super.callAsFunction(
      speechTokens: speechTokens,
      refDict: refDict,
      finalize: finalize,
    )
  }

  /// Run only the HiFi-GAN (mel-to-wav) inference
  func hiftInference(
    speechFeat: MLXArray,
    cacheSource: MLXArray? = nil,
  ) -> (MLXArray, MLXArray) {
    let cache = cacheSource ?? MLXArray.zeros([1, 1, 0])
    return mel2wav.inference(speechFeat, cacheSource: cache)
  }

  /// Full inference pipeline with separate flow and vocoder steps
  func inference(
    speechTokens: MLXArray,
    refDict: S3GenRefDict,
    cacheSource: MLXArray? = nil,
    finalize: Bool = true,
  ) -> (MLXArray, MLXArray) {
    let outputMels = flowInference(
      speechTokens: speechTokens,
      refDict: refDict,
      finalize: finalize,
    )

    let (outputWavs, outputSources) = hiftInference(
      speechFeat: outputMels,
      cacheSource: cacheSource,
    )

    // Apply fade-in to reduce spillover artifacts
    var result = outputWavs
    let fadeLen = _trimFade.shape[0]
    if result.shape[1] >= fadeLen {
      result[0..., 0 ..< fadeLen] = result[0..., 0 ..< fadeLen] * _trimFade
    }

    return (result, outputSources)
  }
}
