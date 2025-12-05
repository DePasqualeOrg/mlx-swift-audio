//
//  FlowMatching.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/s3gen/flow_matching.py
//  and mlx_audio/tts/models/chatterbox/s3gen/matcha/flow_matching.py
//  Conditional Flow Matching for speech synthesis
//

import Foundation
import MLX
import MLXNN

// MARK: - CFMParams

/// Configuration for Conditional Flow Matching
struct CFMParams: Sendable {
  var sigmaMin: Float = 1e-06
  var solver: String = "euler"
  var tScheduler: String = "cosine"
  var trainingCfgRate: Float = 0.2
  var inferenceCfgRate: Float = 0.7
  var regLossType: String = "l1"

  init() {}
}

/// Default CFM parameters for Chatterbox
let DefaultCFMParams = CFMParams()

// MARK: - BASECFM

/// Base Conditional Flow Matching module
class BASECFM: Module {
  let nFeats: Int
  let nSpks: Int
  let spkEmbDim: Int
  let solver: String
  let sigmaMin: Float

  init(nFeats: Int, cfmParams: CFMParams, nSpks: Int = 1, spkEmbDim: Int = 128) {
    self.nFeats = nFeats
    self.nSpks = nSpks
    self.spkEmbDim = spkEmbDim
    solver = cfmParams.solver
    sigmaMin = cfmParams.sigmaMin
  }

  /// Forward diffusion
  func callAsFunction(
    mu: MLXArray,
    mask: MLXArray,
    nTimesteps: Int,
    temperature: Float = 1.0,
    spks: MLXArray? = nil,
    cond: MLXArray? = nil,
    estimator: ((MLXArray, MLXArray, MLXArray, MLXArray, MLXArray?, MLXArray?) -> MLXArray)? = nil,
  ) -> MLXArray {
    let z = MLXRandom.normal(mu.shape) * temperature
    let tSpan = MLX.linspace(Float32(0), Float32(1), count: nTimesteps + 1)
    return solveEuler(x: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond, estimator: estimator)
  }

  /// Fixed Euler solver for ODEs
  func solveEuler(
    x: MLXArray,
    tSpan: MLXArray,
    mu: MLXArray,
    mask: MLXArray,
    spks: MLXArray?,
    cond: MLXArray?,
    estimator: ((MLXArray, MLXArray, MLXArray, MLXArray, MLXArray?, MLXArray?) -> MLXArray)?,
  ) -> MLXArray {
    guard let est = estimator else {
      fatalError("Estimator not provided")
    }

    var currentX = x
    // Keep t as scalar - no expansion needed until estimator call
    var t = tSpan[0]
    // Pre-compute all dt values to avoid per-step indexing
    let numSteps = tSpan.shape[0]

    for step in 1 ..< numSteps {
      // Compute dt for this step (constant for uniform spacing, but we handle non-uniform too)
      let dt = tSpan[step] - tSpan[step - 1]
      let dphiDt = est(currentX, mask, mu, t.expandedDimensions(axis: 0), spks, cond)
      currentX = currentX + dt * dphiDt
      t = tSpan[step]
    }

    return currentX
  }
}

// MARK: - ConditionalCFM

/// Conditional Flow Matching with Classifier-Free Guidance
class ConditionalCFM: BASECFM {
  let tScheduler: String
  let trainingCfgRate: Float
  let inferenceCfgRate: Float

  @ModuleInfo(key: "estimator") var estimatorModule: ConditionalDecoder?

  init(
    inChannels: Int,
    cfmParams: CFMParams,
    nSpks: Int = 1,
    spkEmbDim: Int = 64,
    estimator: ConditionalDecoder? = nil,
  ) {
    tScheduler = cfmParams.tScheduler
    trainingCfgRate = cfmParams.trainingCfgRate
    inferenceCfgRate = cfmParams.inferenceCfgRate
    _estimatorModule.wrappedValue = estimator
    super.init(nFeats: inChannels, cfmParams: cfmParams, nSpks: nSpks, spkEmbDim: spkEmbDim)
  }

  func callAsFunction(
    mu: MLXArray,
    mask: MLXArray,
    nTimesteps: Int,
    temperature: Float = 1.0,
    spks: MLXArray? = nil,
    cond: MLXArray? = nil,
    promptLen: Int = 0,
    flowCache: MLXArray? = nil,
  ) -> (MLXArray, MLXArray) {
    var cache = flowCache ?? MLXArray.zeros([1, 80, 0, 2])
    var z = MLXRandom.normal(mu.shape) * temperature
    var muVar = mu
    let cacheSize = cache.shape[2]

    // Fix prompt and overlap part
    if cacheSize != 0 {
      z = MLX.concatenated([cache[0..., 0..., 0..., 0], z[0..., 0..., cacheSize...]], axis: 2)
      muVar = MLX.concatenated([cache[0..., 0..., 0..., 1], muVar[0..., 0..., cacheSize...]], axis: 2)
    }

    let zLen = z.shape[2]
    let zCache = MLX.concatenated([z[0..., 0..., 0 ..< promptLen], z[0..., 0..., (zLen - 34)...]], axis: 2)
    let muCache = MLX.concatenated([muVar[0..., 0..., 0 ..< promptLen], muVar[0..., 0..., (zLen - 34)...]], axis: 2)
    cache = MLX.stacked([zCache, muCache], axis: -1)

    // Time span
    var tSpan = MLX.linspace(Float32(0), Float32(1), count: nTimesteps + 1)
    if tScheduler == "cosine" {
      tSpan = 1 - MLX.cos(tSpan * 0.5 * Float.pi)
    }

    let result = solveEulerCFG(z: z, tSpan: tSpan, mu: muVar, mask: mask, spks: spks, cond: cond)
    return (result, cache)
  }

  /// Euler solver with Classifier-Free Guidance
  private func solveEulerCFG(
    z: MLXArray,
    tSpan: MLXArray,
    mu: MLXArray,
    mask: MLXArray,
    spks: MLXArray?,
    cond: MLXArray?,
  ) -> MLXArray {
    guard let estimator = estimatorModule else {
      fatalError("Estimator not set")
    }

    var x = z
    // Keep t as shape (1,) for CFG concatenation
    var t = tSpan[0].expandedDimensions(axis: 0)

    // Pre-compute zero arrays outside loop to avoid repeated allocation
    let zeroMu = MLXArray.zeros(mu.shape)
    let zeroSpks: MLXArray? = spks != nil ? MLXArray.zeros(spks!.shape) : nil
    let zeroCond: MLXArray? = cond != nil ? MLXArray.zeros(cond!.shape) : nil

    let numSteps = tSpan.shape[0]
    for step in 1 ..< numSteps {
      // Compute dt for this step directly from tSpan (avoids accumulation error)
      let dt = tSpan[step] - tSpan[step - 1]

      // Prepare inputs for CFG
      let xIn = MLX.concatenated([x, x], axis: 0)
      let maskIn = MLX.concatenated([mask, mask], axis: 0)
      let muIn = MLX.concatenated([mu, zeroMu], axis: 0)
      let tIn = MLX.concatenated([t, t], axis: 0)

      let spksIn: MLXArray? = spks != nil ? MLX.concatenated([spks!, zeroSpks!], axis: 0) : nil
      let condIn: MLXArray? = cond != nil ? MLX.concatenated([cond!, zeroCond!], axis: 0) : nil

      // Forward estimator
      let dphiDt = estimator(
        x: xIn,
        mask: maskIn,
        mu: muIn,
        t: tIn,
        spks: spksIn,
        cond: condIn,
      )

      // Split and apply CFG
      let batchSize = x.shape[0]
      let dphiDtCond = dphiDt[0 ..< batchSize]
      let dphiDtUncond = dphiDt[batchSize...]

      let dphiDtCombined = (1.0 + inferenceCfgRate) * dphiDtCond - inferenceCfgRate * dphiDtUncond

      x = x + dt * dphiDtCombined
      // Update t directly from tSpan to avoid indexing t[0]
      t = tSpan[step].expandedDimensions(axis: 0)
    }

    return x
  }
}

// MARK: - CausalConditionalCFM

/// Causal Conditional Flow Matching with fixed noise
class CausalConditionalCFM: ConditionalCFM {
  /// Pre-generated random noise - underscore prefix excludes from parameter validation
  var _randNoise: MLXArray

  override init(
    inChannels: Int = 240,
    cfmParams: CFMParams = DefaultCFMParams,
    nSpks: Int = 1,
    spkEmbDim: Int = 80,
    estimator: ConditionalDecoder? = nil,
  ) {
    // Pre-generate random noise for deterministic generation
    _randNoise = MLXRandom.normal([1, 80, 50 * 300])
    super.init(inChannels: inChannels, cfmParams: cfmParams, nSpks: nSpks, spkEmbDim: spkEmbDim, estimator: estimator)
  }

  func callAsFunction(
    mu: MLXArray,
    mask: MLXArray,
    nTimesteps: Int,
    temperature: Float = 1.0,
    spks: MLXArray? = nil,
    cond: MLXArray? = nil,
  ) -> (MLXArray, MLXArray?) {
    let T = mu.shape[2]
    let z = _randNoise[0..., 0..., 0 ..< T] * temperature

    // Time span
    var tSpan = MLX.linspace(Float32(0), Float32(1), count: nTimesteps + 1)
    if tScheduler == "cosine" {
      tSpan = 1 - MLX.cos(tSpan * 0.5 * Float.pi)
    }

    let result = solveEulerCausal(z: z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond)
    return (result, nil)
  }

  /// Euler solver for causal CFM
  private func solveEulerCausal(
    z: MLXArray,
    tSpan: MLXArray,
    mu: MLXArray,
    mask: MLXArray,
    spks: MLXArray?,
    cond: MLXArray?,
  ) -> MLXArray {
    guard let estimator = estimatorModule else {
      fatalError("Estimator not set")
    }

    var x = z
    // Keep t as shape (1,) for CFG concatenation
    var t = tSpan[0].expandedDimensions(axis: 0)

    // Pre-compute zero arrays outside loop to avoid repeated allocation
    let zeroMu = MLXArray.zeros(mu.shape)
    let zeroSpks: MLXArray? = spks != nil ? MLXArray.zeros(spks!.shape) : nil
    let zeroCond: MLXArray? = cond != nil ? MLXArray.zeros(cond!.shape) : nil

    let numSteps = tSpan.shape[0]
    for step in 1 ..< numSteps {
      // Compute dt for this step directly from tSpan (avoids accumulation error)
      let dt = tSpan[step] - tSpan[step - 1]

      // Prepare inputs for CFG
      let xIn = MLX.concatenated([x, x], axis: 0)
      let maskIn = MLX.concatenated([mask, mask], axis: 0)
      let muIn = MLX.concatenated([mu, zeroMu], axis: 0)
      let tIn = MLX.concatenated([t, t], axis: 0)

      let spksIn: MLXArray? = spks != nil ? MLX.concatenated([spks!, zeroSpks!], axis: 0) : nil
      let condIn: MLXArray? = cond != nil ? MLX.concatenated([cond!, zeroCond!], axis: 0) : nil

      // Forward estimator
      let dphiDt = estimator(
        x: xIn,
        mask: maskIn,
        mu: muIn,
        t: tIn,
        spks: spksIn,
        cond: condIn,
      )

      // Split and apply CFG
      let batchSize = x.shape[0]
      let dphiDtCond = dphiDt[0 ..< batchSize]
      let dphiDtUncond = dphiDt[batchSize...]

      let dphiDtCombined = (1.0 + inferenceCfgRate) * dphiDtCond - inferenceCfgRate * dphiDtUncond

      x = x + dt * dphiDtCombined
      // Update t directly from tSpan to avoid indexing t[0]
      t = tSpan[step].expandedDimensions(axis: 0)
    }

    return x
  }
}
