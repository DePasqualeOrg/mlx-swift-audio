// CosyVoice2-specific Conditional Flow Matching module
// Ported from mlx-audio-plus cosyvoice2/flow_matching.py

import Foundation
import MLX
import MLXNN

// MARK: - CosyVoice2 CFM Parameters

/// CosyVoice2-specific CFM configuration
struct CosyVoice2CFMConfig: Sendable {
  var sigmaMin: Float = 1e-6
  var solver: String = "euler"
  var tScheduler: String = "cosine"
  var trainingCfgRate: Float = 0.2
  var inferenceCfgRate: Float = 0.7
  var regLossType: String = "l1"

  init() {}
}

/// Default CFM parameters for CosyVoice2
let CosyVoice2DefaultCFMConfig = CosyVoice2CFMConfig()

// MARK: - CosyVoice2ConditionalCFM

/// Conditional Flow Matching with Classifier-Free Guidance for CosyVoice2
/// This version supports streaming attention mode for chunk-based generation
class CosyVoice2ConditionalCFM: Module {
  /// Mel output channels
  static let melChannels: Int = 80

  let nFeats: Int
  let nSpks: Int
  let spkEmbDim: Int
  let tScheduler: String
  let trainingCfgRate: Float
  let inferenceCfgRate: Float
  let sigmaMin: Float

  /// Estimator module (ConditionalDecoder)
  @ModuleInfo(key: "estimator") var estimator: ConditionalDecoder?

  init(
    inChannels: Int = 240,
    cfmConfig: CosyVoice2CFMConfig = CosyVoice2DefaultCFMConfig,
    nSpks: Int = 1,
    spkEmbDim: Int = 80,
    estimator: ConditionalDecoder? = nil
  ) {
    nFeats = inChannels
    self.nSpks = nSpks
    self.spkEmbDim = spkEmbDim
    tScheduler = cfmConfig.tScheduler
    trainingCfgRate = cfmConfig.trainingCfgRate
    inferenceCfgRate = cfmConfig.inferenceCfgRate
    sigmaMin = cfmConfig.sigmaMin

    _estimator.wrappedValue = estimator
  }

  /// Forward diffusion with dynamically generated noise
  /// - Parameters:
  ///   - mu: Encoder output (B, C, T)
  ///   - mask: Mask (B, 1, T)
  ///   - nTimesteps: Number of diffusion steps
  ///   - temperature: Noise temperature
  ///   - spks: Speaker embeddings (B, spk_dim)
  ///   - cond: Conditioning (B, C, T)
  ///   - streaming: Whether to use streaming (chunk-based) attention
  /// - Returns: Tuple of (generated_mel, nil)
  func callAsFunction(
    mu: MLXArray,
    mask: MLXArray,
    nTimesteps: Int,
    temperature: Float = 1.0,
    spks: MLXArray? = nil,
    cond: MLXArray? = nil,
    streaming: Bool = false
  ) -> (MLXArray, MLXArray?) {
    let B = mu.shape[0]
    let T = mu.shape[2]
    // Generate noise dynamically with the correct shape
    let z = MLXRandom.normal([B, Self.melChannels, T]) * temperature

    // Time span with cosine schedule
    var tSpan = MLX.linspace(Float32(0), Float32(1), count: nTimesteps + 1)
    if tScheduler == "cosine" {
      tSpan = 1 - MLX.cos(tSpan * 0.5 * Float.pi)
    }

    let result = solveEuler(
      x: z,
      tSpan: tSpan,
      mu: mu,
      mask: mask,
      spks: spks,
      cond: cond,
      streaming: streaming
    )
    return (result, nil)
  }

  /// Euler solver with Classifier-Free Guidance
  /// - Parameters:
  ///   - x: Starting noise (B, C, T)
  ///   - tSpan: Time steps
  ///   - mu: Encoder output (B, C, T)
  ///   - mask: Mask (B, 1, T)
  ///   - spks: Speaker embeddings (B, spk_dim)
  ///   - cond: Conditioning (B, C, T)
  ///   - streaming: Whether to use streaming attention
  /// - Returns: Generated mel spectrogram (B, C, T)
  private func solveEuler(
    x: MLXArray,
    tSpan: MLXArray,
    mu: MLXArray,
    mask: MLXArray,
    spks: MLXArray?,
    cond: MLXArray?,
    streaming: Bool
  ) -> MLXArray {
    guard let est = estimator else {
      fatalError("Estimator not set")
    }

    var currentX = x
    var t = tSpan[0].expandedDimensions(axis: 0)
    var dt = tSpan[1] - tSpan[0]

    let B = x.shape[0]
    let TLen = x.shape[2]

    // Pre-compute static inputs for CFG (these don't change during iteration)
    let maskIn = MLX.concatenated([mask, mask], axis: 0)
    let zerosMu = MLXArray.zeros(mu.shape)

    // Pre-compute spks_in
    let spksIn: MLXArray = if let s = spks {
      MLX.concatenated([s, MLXArray.zeros(s.shape)], axis: 0)
    } else {
      MLXArray.zeros([2, spkEmbDim])
    }

    // Pre-compute cond_in
    let condIn: MLXArray = if let c = cond {
      MLX.concatenated([c, MLXArray.zeros(c.shape)], axis: 0)
    } else {
      MLXArray.zeros([2, nFeats, TLen])
    }

    let numSteps = tSpan.shape[0]
    for step in 1 ..< numSteps {
      // Prepare inputs for CFG - only x and t change each iteration
      let xIn = MLX.concatenated([currentX, currentX], axis: 0)
      let muIn = MLX.concatenated([mu, zerosMu], axis: 0)
      let tIn = MLX.concatenated([t, t], axis: 0)

      // Forward estimator with streaming parameter
      let dphiDt = est(
        x: xIn,
        mask: maskIn,
        mu: muIn,
        t: tIn,
        spks: spksIn,
        cond: condIn,
        streaming: streaming
      )

      // Split and apply CFG
      let dphiDtCond = dphiDt[0 ..< B]
      let dphiDtUncond = dphiDt[B...]

      let dphiDtCombined = (1.0 + inferenceCfgRate) * dphiDtCond - inferenceCfgRate * dphiDtUncond

      currentX = currentX + dt * dphiDtCombined
      t = t + dt

      if step < numSteps - 1 {
        dt = tSpan[step + 1] - t
      }
    }

    return currentX
  }
}

// MARK: - CosyVoice2 Flow Model

/// Flow matching model that wraps encoder + CFM for CosyVoice2
class CosyVoice2FlowModel: Module {
  let inputSize: Int
  let outputSize: Int
  let spkEmbDim: Int
  let vocabSize: Int
  let tokenMelRatio: Int
  let nTimesteps: Int

  /// Input embedding for speech tokens
  @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding

  /// Encoder (UpsampleConformerEncoder)
  @ModuleInfo(key: "encoder") var encoder: UpsampleConformerEncoder

  /// CFM module
  @ModuleInfo(key: "cfm") var cfm: CosyVoice2ConditionalCFM

  init(
    inputSize: Int = 512,
    outputSize: Int = 80,
    spkEmbDim: Int = 192,
    vocabSize: Int = 6561,
    tokenMelRatio: Int = 2,
    nTimesteps: Int = 10,
    encoderConfig: FlowConfig = FlowConfig(),
    cfmConfig: CosyVoice2CFMConfig = CosyVoice2DefaultCFMConfig
  ) {
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.spkEmbDim = spkEmbDim
    self.vocabSize = vocabSize
    self.tokenMelRatio = tokenMelRatio
    self.nTimesteps = nTimesteps

    // Speech token embedding
    _inputEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: inputSize)

    // Encoder - using available parameters from UpsampleConformerEncoder
    _encoder.wrappedValue = UpsampleConformerEncoder(
      inputSize: inputSize,
      outputSize: inputSize,
      attentionHeads: encoderConfig.encoderAttentionHeads,
      linearUnits: encoderConfig.encoderLinearUnits,
      numBlocks: encoderConfig.encoderNumBlocks,
      dropoutRate: encoderConfig.encoderDropoutRate,
      positionalDropoutRate: encoderConfig.encoderPositionalDropoutRate,
      attentionDropoutRate: encoderConfig.encoderAttentionDropoutRate,
      normalizeBefore: encoderConfig.encoderNormalizeBefore,
      staticChunkSize: encoderConfig.encoderStaticChunkSize,
      macaronStyle: encoderConfig.encoderMacaronStyle,
      useCnnModule: encoderConfig.encoderUseCnnModule,
      cnnModuleKernel: encoderConfig.encoderCnnModuleKernel,
      causal: encoderConfig.encoderCausal
    )

    // CFM
    let decoderInChannels = outputSize * 4 // x + mu + spks + cond
    _cfm.wrappedValue = CosyVoice2ConditionalCFM(
      inChannels: decoderInChannels,
      cfmConfig: cfmConfig,
      nSpks: 1,
      spkEmbDim: outputSize // spks gets projected to outputSize
    )
  }

  /// Generate mel spectrogram from speech tokens
  /// - Parameters:
  ///   - speechTokens: Speech token IDs (B, T_tokens)
  ///   - speechTokenLens: Token lengths (B,)
  ///   - refMel: Reference mel for conditioning (B, C, T_mel)
  ///   - refMelLens: Reference mel lengths (B,)
  ///   - spkEmb: Speaker embedding (B, spk_dim)
  ///   - temperature: Noise temperature for CFM
  ///   - streaming: Whether to use streaming mode
  /// - Returns: Generated mel spectrogram (B, C, T_out)
  func callAsFunction(
    speechTokens: MLXArray,
    speechTokenLens: MLXArray,
    refMel: MLXArray,
    refMelLens _: MLXArray,
    spkEmb: MLXArray,
    temperature: Float = 1.0,
    streaming: Bool = false
  ) -> MLXArray {
    // Embed speech tokens
    let tokenEmb = inputEmbedding(speechTokens)

    // Calculate output length (token_len * token_mel_ratio)
    let melLens = speechTokenLens * Int32(tokenMelRatio)

    // Encode tokens
    let (encoderOut, _) = encoder(tokenEmb, xsLens: melLens.asType(.int32))

    // Create mask
    let maxLen = encoderOut.shape[1]
    let positions = MLXArray(0 ..< Int32(maxLen))
    let mask = (positions .< melLens.expandedDimensions(axis: 1)).asType(.float32)
    let maskExpanded = mask.expandedDimensions(axis: 1) // (B, 1, T)

    // Transpose encoder output to (B, C, T)
    let mu = encoderOut.swappedAxes(1, 2)

    // Project speaker embedding to mel dimension
    let spksProjected = spkEmb[0..., 0 ..< outputSize] // Take first outputSize dims

    // Prepare conditioning from reference mel
    let cond = refMel[0..., 0..., 0 ..< maxLen] // Truncate or pad to match

    // Run CFM
    let (mel, _) = cfm(
      mu: mu,
      mask: maskExpanded,
      nTimesteps: nTimesteps,
      temperature: temperature,
      spks: spksProjected,
      cond: cond,
      streaming: streaming
    )

    return mel
  }
}
