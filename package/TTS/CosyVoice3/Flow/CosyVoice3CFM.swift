// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

import Foundation
import MLX
import MLXNN

// MARK: - Helper Functions

/// Create a padding mask from sequence lengths
/// - Parameters:
///   - lengths: Sequence lengths (B,)
///   - maxLen: Maximum length (if nil, use max of lengths)
/// - Returns: Boolean mask (B, max_len) where True indicates padding
func makePadMask(lengths: MLXArray, maxLen: Int? = nil) -> MLXArray {
  let actualMaxLen = maxLen ?? Int(MLX.max(lengths).item(Int32.self))
  let positions = MLXArray(0 ..< actualMaxLen)
  let mask = positions.expandedDimensions(axis: 0) .>= lengths.expandedDimensions(axis: 1)
  return mask
}

// MARK: - PreLookaheadLayer

/// Pre-lookahead layer for CosyVoice3 flow processing
/// This is a simpler version than the full UpsampleConformerEncoder used in CosyVoice2
class CosyVoice3PreLookaheadLayer: Module {
  let inChannels: Int
  let channels: Int
  let preLookaheadLen: Int

  @ModuleInfo(key: "conv1") var conv1: Conv1d
  @ModuleInfo(key: "conv2") var conv2: Conv1d

  init(inChannels: Int, channels: Int, preLookaheadLen: Int = 3) {
    self.inChannels = inChannels
    self.channels = channels
    self.preLookaheadLen = preLookaheadLen

    _conv1.wrappedValue = Conv1d(
      inputChannels: inChannels,
      outputChannels: channels,
      kernelSize: preLookaheadLen + 1,
      stride: 1,
      padding: 0
    )
    _conv2.wrappedValue = Conv1d(
      inputChannels: channels,
      outputChannels: inChannels,
      kernelSize: 3,
      stride: 1,
      padding: 0
    )
  }

  /// Apply pre-lookahead processing
  /// - Parameters:
  ///   - inputs: Input (B, T, C) - sequence first
  ///   - context: Optional lookahead context for streaming
  /// - Returns: Output (B, T, C)
  func callAsFunction(_ inputs: MLXArray, context: MLXArray? = nil) -> MLXArray {
    var outputs = inputs

    // Handle lookahead padding based on whether context is provided
    if let ctx = context {
      // Streaming: concatenate context (context serves as the padding)
      outputs = MLX.concatenated([outputs, ctx], axis: 1)
    } else {
      // Non-streaming: pad with zeros on right
      outputs = MLX.padded(outputs, widths: [IntOrPair(0), IntOrPair((0, preLookaheadLen)), IntOrPair(0)])
    }

    // First convolution + leaky relu
    outputs = leakyRelu(conv1(outputs))

    // Left causal padding for second conv (kernel_size - 1 = 2)
    outputs = MLX.padded(outputs, widths: [IntOrPair(0), IntOrPair((2, 0)), IntOrPair(0)])
    outputs = conv2(outputs)

    // Truncate if context was provided (restore original length)
    if context != nil {
      outputs = outputs[0..., 0 ..< inputs.shape[1], 0...]
    }

    // Residual connection
    return outputs + inputs
  }
}

// MARK: - CosyVoice3ConditionalCFM

/// Conditional Flow Matching module for CosyVoice3 with DiT estimator
/// Implements the ODE-based flow matching algorithm with classifier-free guidance
class CosyVoice3ConditionalCFM: Module {
  let sigmaMin: Float
  let tScheduler: String
  let inferenceCfgRate: Float

  @ModuleInfo(key: "estimator") var estimator: DiT?

  private var randNoise: MLXArray?

  init(
    estimator: DiT?,
    sigmaMin: Float = 1e-6,
    tScheduler: String = "cosine",
    inferenceCfgRate: Float = 0.7,
    randNoise: MLXArray? = nil
  ) {
    self.sigmaMin = sigmaMin
    self.tScheduler = tScheduler
    self.inferenceCfgRate = inferenceCfgRate
    self.randNoise = randNoise

    _estimator.wrappedValue = estimator
  }

  /// Apply cosine schedule to timestep
  private func cosineSchedule(_ t: MLXArray) -> MLXArray {
    1 - MLX.cos(t * Float.pi / 2)
  }

  /// Generate mel-spectrogram using flow matching
  /// - Parameters:
  ///   - mu: Condition/text embedding (B, mel_dim, N)
  ///   - mask: Mask (B, 1, N)
  ///   - spks: Speaker embedding (B, D)
  ///   - cond: Conditioning audio (B, mel_dim, N)
  ///   - nTimesteps: Number of ODE solver steps
  ///   - streaming: Whether in streaming mode
  /// - Returns: Generated mel-spectrogram and nil
  func callAsFunction(
    mu: MLXArray,
    mask: MLXArray,
    spks: MLXArray,
    cond: MLXArray,
    nTimesteps: Int = 10,
    streaming: Bool = false
  ) -> (MLXArray, MLXArray?) {
    let B = mu.shape[0]
    let melDim = mu.shape[1]
    let N = mu.shape[2]

    // Use pre-computed noise or generate
    let z: MLXArray
    if let noise = randNoise {
      z = noise[0..., 0..., 0 ..< N].asType(mu.dtype)
    } else {
      MLXRandom.seed(0)
      z = MLXRandom.normal([B, melDim, N])
    }

    let output = solveEuler(
      z: z,
      mu: mu,
      mask: mask,
      spks: spks,
      cond: cond,
      nTimesteps: nTimesteps,
      streaming: streaming
    )

    return (output, nil)
  }

  /// Solve the ODE using Euler method with classifier-free guidance.
  ///
  /// Uses batched computation for efficiency: conditional and unconditional
  /// paths are batched together in a single forward pass through the estimator.
  private func solveEuler(
    z: MLXArray,
    mu: MLXArray,
    mask: MLXArray,
    spks: MLXArray,
    cond: MLXArray,
    nTimesteps: Int = 10,
    streaming: Bool = false
  ) -> MLXArray {
    guard let est = estimator else {
      fatalError("Estimator not set")
    }

    // Create time span with cosine schedule
    var tSpan = MLX.linspace(Float32(0), Float32(1), count: nTimesteps + 1)
    if tScheduler == "cosine" {
      tSpan = 1 - MLX.cos(tSpan * 0.5 * Float.pi)
    }

    var x = z
    let B = mu.shape[0]

    print("[CFM] Starting ODE solver with \(nTimesteps) steps, shape: \(x.shape)")
    let cfmStart = CFAbsoluteTimeGetCurrent()

    // Squeeze mask for DiT
    let maskSqueeze = mask.squeezed(axis: 1) // (B, N)

    // Pre-allocate batched tensors for CFG (batch size 2*B)
    // First B samples: conditional, Last B samples: unconditional
    let muZeros = MLXArray.zeros(like: mu)
    let spksZeros = MLXArray.zeros(like: spks)
    let condZeros = MLXArray.zeros(like: cond)

    for step in 1 ... nTimesteps {
      let stepStart = CFAbsoluteTimeGetCurrent()
      let t = tSpan[step - 1]
      let dt = tSpan[step] - tSpan[step - 1]

      // Batch conditional and unconditional inputs together
      // This matches PyTorch's efficient batched CFG computation
      let xBatched = MLX.concatenated([x, x], axis: 0) // (2B, mel_dim, N)
      let maskBatched = MLX.concatenated([maskSqueeze, maskSqueeze], axis: 0)
      let muBatched = MLX.concatenated([mu, muZeros], axis: 0) // cond, then uncond
      let spksBatched = MLX.concatenated([spks, spksZeros], axis: 0)
      let condBatched = MLX.concatenated([cond, condZeros], axis: 0)
      let tBatched = MLX.broadcast(t, to: [2 * B])

      // Single batched forward pass through estimator
      let dphiDtBatched = est(
        x: xBatched,
        mask: maskBatched,
        mu: muBatched,
        t: tBatched,
        spks: spksBatched,
        cond: condBatched,
        streaming: streaming
      )

      // Split back into conditional and unconditional
      let dphiDtCond = dphiDtBatched[0 ..< B]
      let dphiDtUncond = dphiDtBatched[B...]

      // Classifier-free guidance
      let dphiDt = (1.0 + inferenceCfgRate) * dphiDtCond - inferenceCfgRate * dphiDtUncond

      // Euler step
      x = x + dt * dphiDt

      // Force evaluation to prevent computation graph explosion
      x.eval()

      let stepTime = CFAbsoluteTimeGetCurrent() - stepStart
      print("[CFM] Step \(step)/\(nTimesteps), elapsed: \(String(format: "%.2f", stepTime))s")
    }

    let totalTime = CFAbsoluteTimeGetCurrent() - cfmStart
    print("[CFM] ODE solver complete in \(String(format: "%.2f", totalTime))s")

    return x.asType(.float32)
  }
}

// MARK: - CausalMaskedDiffWithDiT

/// CosyVoice3 flow model with DiT-based decoder
/// This replaces the encoder + U-Net decoder of CosyVoice2 with:
/// - PreLookaheadLayer for simple token processing
/// - DiT-based CFM decoder for mel generation
class CausalMaskedDiffWithDiT: Module {
  let inputSize: Int
  let outputSize: Int
  let vocabSize: Int
  let inputFrameRate: Int
  let tokenMelRatio: Int
  let preLookaheadLen: Int
  let nTimesteps: Int

  @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
  @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffineLayer: Linear
  @ModuleInfo(key: "pre_lookahead_layer") var preLookaheadLayer: CosyVoice3PreLookaheadLayer
  @ModuleInfo(key: "decoder") var decoder: CosyVoice3ConditionalCFM?

  init(
    inputSize: Int = 512,
    outputSize: Int = 80,
    spkEmbedDim: Int = 192,
    vocabSize: Int = 6561,
    inputFrameRate: Int = 25,
    tokenMelRatio: Int = 2,
    preLookaheadLen: Int = 3,
    preLookaheadLayer: CosyVoice3PreLookaheadLayer? = nil,
    decoder: CosyVoice3ConditionalCFM? = nil,
    nTimesteps: Int = 10
  ) {
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.vocabSize = vocabSize
    self.inputFrameRate = inputFrameRate
    self.tokenMelRatio = tokenMelRatio
    self.preLookaheadLen = preLookaheadLen
    self.nTimesteps = nTimesteps

    _inputEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: inputSize)
    _spkEmbedAffineLayer.wrappedValue = Linear(spkEmbedDim, outputSize)
    _preLookaheadLayer.wrappedValue = preLookaheadLayer ?? CosyVoice3PreLookaheadLayer(
      inChannels: inputSize,
      channels: inputSize,
      preLookaheadLen: preLookaheadLen
    )
    _decoder.wrappedValue = decoder
  }

  /// Generate mel-spectrogram from speech tokens
  /// - Parameters:
  ///   - token: Speech token IDs (1, T_token)
  ///   - tokenLen: Token length (1,)
  ///   - promptToken: Prompt speech tokens (1, T_prompt)
  ///   - promptTokenLen: Prompt token length (1,)
  ///   - promptFeat: Prompt mel features (1, T_mel, mel_dim)
  ///   - promptFeatLen: Prompt feature length (1,)
  ///   - embedding: Speaker embedding (1, spk_dim)
  ///   - streaming: Whether in streaming mode
  ///   - finalize: Whether this is the final chunk
  /// - Returns: Generated mel-spectrogram and nil
  func inference(
    token: MLXArray,
    tokenLen: MLXArray,
    promptToken: MLXArray,
    promptTokenLen: MLXArray,
    promptFeat: MLXArray,
    promptFeatLen _: MLXArray,
    embedding: MLXArray,
    streaming: Bool = false,
    finalize: Bool = true
  ) -> (MLXArray, MLXArray?) {
    precondition(token.shape[0] == 1, "Batch size must be 1 for inference")

    // Normalize and project speaker embedding
    let embNorm = embedding / MLX.sqrt(MLX.sum(embedding * embedding, axis: -1, keepDims: true) + 1e-8)
    let embProj = spkEmbedAffineLayer(embNorm)

    // Concatenate prompt and input tokens
    let combinedToken = MLX.concatenated([promptToken, token], axis: 1)
    let combinedTokenLen = promptTokenLen + tokenLen

    // Create mask and embed tokens
    let mask = MLX.logicalNot(makePadMask(lengths: combinedTokenLen, maxLen: combinedToken.shape[1]))
    let maskFloat = mask.expandedDimensions(axis: -1).asType(.float32)
    let clippedToken = MLX.clip(combinedToken, min: 0, max: vocabSize - 1)
    let tokenEmb = inputEmbedding(clippedToken) * maskFloat

    // Apply pre-lookahead layer
    let h: MLXArray
    if finalize {
      h = preLookaheadLayer(tokenEmb)
    } else {
      // Split for streaming: main tokens and lookahead context
      let mainTokens = tokenEmb[0..., 0 ..< (tokenEmb.shape[1] - preLookaheadLen), 0...]
      let context = tokenEmb[0..., (tokenEmb.shape[1] - preLookaheadLen)..., 0...]
      h = preLookaheadLayer(mainTokens, context: context)
    }

    // Upsample by token_mel_ratio
    let hUpsampled = MLX.repeated(h, count: tokenMelRatio, axis: 1)

    let melLen1 = promptFeat.shape[1]
    let melLen2 = hUpsampled.shape[1] - melLen1

    // Prepare conditioning: concatenate prompt features with zeros
    let zerosPart = MLXArray.zeros([1, melLen2, outputSize])
    let condSeq = MLX.concatenated([promptFeat, zerosPart], axis: 1) // (1, T, D)

    // Transpose to channel-first for decoder
    let cond = condSeq.swappedAxes(1, 2) // (B, mel_dim, N)
    let mu = hUpsampled.swappedAxes(1, 2) // (B, input_size, N)

    // Create mask for decoder
    let totalLen = melLen1 + melLen2
    let decoderMask = MLXArray.ones([1, 1, totalLen])

    // Generate mel using flow matching
    guard let dec = decoder else {
      fatalError("Decoder not set")
    }

    let (feat, _) = dec(
      mu: mu,
      mask: decoderMask,
      spks: embProj,
      cond: cond,
      nTimesteps: nTimesteps,
      streaming: streaming
    )

    // Remove prompt portion
    let outputFeat = feat[0..., 0..., melLen1...]
    precondition(outputFeat.shape[2] == melLen2, "Output feature length mismatch")

    return (outputFeat.asType(.float32), nil)
  }
}

// MARK: - Flow Model Builder

/// Build the complete flow model for CosyVoice3
func buildCosyVoice3FlowModel(
  inputSize: Int = 512,
  outputSize: Int = 80,
  spkEmbedDim: Int = 192,
  vocabSize: Int = 6561,
  inputFrameRate: Int = 25,
  tokenMelRatio: Int = 2,
  preLookaheadLen: Int = 3,
  ditDim: Int = 1024,
  ditDepth: Int = 22,
  ditHeads: Int = 16,
  ditDimHead: Int = 64,
  ditFfMult: Int = 2,
  ditDropout: Float = 0.1,
  cfmSigmaMin: Float = 1e-6,
  cfmTScheduler: String = "cosine",
  cfmInferenceCfgRate: Float = 0.7,
  nTimesteps: Int = 10,
  staticChunkSize: Int = 50,
  randNoise: MLXArray? = nil
) -> CausalMaskedDiffWithDiT {
  // Build pre-lookahead layer
  let preLookaheadLayer = CosyVoice3PreLookaheadLayer(
    inChannels: inputSize,
    channels: ditDim,
    preLookaheadLen: preLookaheadLen
  )

  // Build DiT estimator
  let dit = DiT(
    dim: ditDim,
    depth: ditDepth,
    heads: ditHeads,
    dimHead: ditDimHead,
    dropout: ditDropout,
    ffMult: ditFfMult,
    melDim: outputSize,
    muDim: inputSize,
    spkDim: outputSize,
    outChannels: outputSize,
    staticChunkSize: staticChunkSize
  )

  // Build CFM decoder
  let decoder = CosyVoice3ConditionalCFM(
    estimator: dit,
    sigmaMin: cfmSigmaMin,
    tScheduler: cfmTScheduler,
    inferenceCfgRate: cfmInferenceCfgRate,
    randNoise: randNoise
  )

  // Build full flow model
  let flowModel = CausalMaskedDiffWithDiT(
    inputSize: inputSize,
    outputSize: outputSize,
    spkEmbedDim: spkEmbedDim,
    vocabSize: vocabSize,
    inputFrameRate: inputFrameRate,
    tokenMelRatio: tokenMelRatio,
    preLookaheadLen: preLookaheadLen,
    preLookaheadLayer: preLookaheadLayer,
    decoder: decoder,
    nTimesteps: nTimesteps
  )

  return flowModel
}
