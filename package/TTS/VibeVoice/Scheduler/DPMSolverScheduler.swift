// Copyright © Microsoft (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/microsoft/VibeVoice
// License: licenses/vibevoice.txt

import Foundation
import MLX

// MARK: - SchedulerOutput

/// Output from scheduler step
struct SchedulerOutput {
  let prevSample: MLXArray
  let x0Pred: MLXArray?
}

// MARK: - Beta Schedule

/// Create a beta schedule based on alpha_bar function
func betasForAlphaBar(
  numDiffusionTimesteps: Int,
  maxBeta: Float = 0.999,
  alphaTransformType: String = "cosine"
) -> MLXArray {
  func alphaBarFn(_ t: Float) -> Float {
    switch alphaTransformType {
      case "cosine":
        pow(cos((t + 0.008) / 1.008 * Float.pi / 2), 2)
      case "exp":
        exp(t * -12.0)
      default:
        fatalError("Unsupported alphaTransformType: \(alphaTransformType)")
    }
  }

  var betas: [Float] = []
  for i in 0 ..< numDiffusionTimesteps {
    let t1 = Float(i) / Float(numDiffusionTimesteps)
    let t2 = Float(i + 1) / Float(numDiffusionTimesteps)
    betas.append(min(1 - alphaBarFn(t2) / alphaBarFn(t1), maxBeta))
  }

  return MLXArray(betas)
}

// MARK: - DPMSolverMultistepScheduler

/// DPM-Solver multistep scheduler for fast diffusion sampling
/// Implements DPM-Solver++ algorithm for efficient sampling with multi-order updates
class DPMSolverMultistepScheduler {
  let numTrainTimesteps: Int
  let betaStart: Float
  let betaEnd: Float
  let betaSchedule: String
  let predictionType: String
  let solverOrder: Int
  let lowerOrderFinal: Bool
  let finalSigmasType: String

  // Precomputed schedule values
  let betas: MLXArray
  let alphas: MLXArray
  let alphasCumprod: MLXArray
  let alphaT: MLXArray
  let sigmaT: MLXArray
  let lambdaT: MLXArray
  let sigmas: MLXArray
  let initNoiseSigma: Float

  // State
  var numInferenceSteps: Int?
  var timesteps: MLXArray?
  var modelOutputs: [MLXArray?]
  var lowerOrderNums: Int
  var stepIndex: Int?

  // Precomputed values for inference
  var cachedAlphaT: [Float] = []
  var cachedSigmaT: [Float] = []
  var cachedLambda: [Float] = []

  init(
    numTrainTimesteps: Int = 1000,
    betaStart: Float = 0.0001,
    betaEnd: Float = 0.02,
    betaSchedule: String = "scaled_linear",
    predictionType: String = "v_prediction",
    solverOrder: Int = 2,
    lowerOrderFinal: Bool = true,
    finalSigmasType: String = "zero"
  ) {
    self.numTrainTimesteps = numTrainTimesteps
    self.betaStart = betaStart
    self.betaEnd = betaEnd
    self.betaSchedule = betaSchedule
    self.predictionType = predictionType
    self.solverOrder = solverOrder
    self.lowerOrderFinal = lowerOrderFinal
    self.finalSigmasType = finalSigmasType

    // Create beta schedule
    switch betaSchedule {
      case "linear":
        betas = MLXArray(stride(from: betaStart, to: betaEnd, by: (betaEnd - betaStart) / Float(numTrainTimesteps - 1)))
      case "scaled_linear", "squaredcos_cap_v2", "cosine":
        betas = betasForAlphaBar(numDiffusionTimesteps: numTrainTimesteps, alphaTransformType: "cosine")
      default:
        fatalError("Unknown betaSchedule: \(betaSchedule)")
    }

    alphas = 1.0 - betas
    alphasCumprod = MLX.cumprod(alphas, axis: 0)

    // DPM-Solver parameters
    alphaT = MLX.sqrt(alphasCumprod)
    sigmaT = MLX.sqrt(1 - alphasCumprod)
    lambdaT = MLX.log(alphaT) - MLX.log(sigmaT)
    sigmas = MLX.sqrt((1 - alphasCumprod) / alphasCumprod)

    initNoiseSigma = 1.0

    // State initialization
    modelOutputs = Array(repeating: nil, count: solverOrder)
    lowerOrderNums = 0
    stepIndex = nil
  }

  /// Set the number of inference steps
  func setTimesteps(_ numInferenceSteps: Int) {
    self.numInferenceSteps = numInferenceSteps

    // Create timesteps - linspace from max to 0
    var timestepValues: [Int] = []
    for i in 0 ..< numInferenceSteps {
      let t = Float(numTrainTimesteps - 1) * (1.0 - Float(i) / Float(numInferenceSteps))
      timestepValues.append(Int(round(t)))
    }

    timesteps = MLXArray(timestepValues.map { Int32($0) })

    // Precompute values for each inference timestep
    cachedAlphaT = []
    cachedSigmaT = []
    cachedLambda = []

    let alphaTArray = alphaT.asArray(Float.self)
    // sigmaT array not needed directly - we compute from alphaT

    for t in timestepValues {
      let sigma = sqrt((1 - alphaTArray[t] * alphaTArray[t]) / (alphaTArray[t] * alphaTArray[t]))
      let alphaTVal = 1.0 / sqrt(sigma * sigma + 1.0)
      let sigmaTVal = sigma * alphaTVal
      let lambdaVal = log(alphaTVal) - log(sigmaTVal)

      cachedAlphaT.append(alphaTVal)
      cachedSigmaT.append(sigmaTVal)
      cachedLambda.append(lambdaVal)
    }

    // Add final step values
    cachedAlphaT.append(1.0)
    cachedSigmaT.append(0.0)
    cachedLambda.append(Float.infinity)

    // Reset state
    modelOutputs = Array(repeating: nil, count: solverOrder)
    lowerOrderNums = 0
    stepIndex = nil
  }

  /// Convert model output to x0 prediction based on prediction type
  private func convertModelOutput(
    modelOutput: MLXArray,
    sample: MLXArray,
    stepIdx: Int
  ) -> MLXArray {
    let alphaTVal = cachedAlphaT[stepIdx]
    let sigmaTVal = cachedSigmaT[stepIdx]

    switch predictionType {
      case "epsilon":
        // model predicts noise
        return (sample - sigmaTVal * modelOutput) / alphaTVal
      case "v_prediction":
        // model predicts v = alpha_t * noise - sigma_t * x0
        return alphaTVal * sample - sigmaTVal * modelOutput
      case "sample":
        return modelOutput
      default:
        fatalError("Unknown predictionType: \(predictionType)")
    }
  }

  /// First order DPM-Solver++ update
  private func dpmSolverFirstOrderUpdate(
    x0Pred: MLXArray,
    sample: MLXArray,
    stepIdx: Int
  ) -> MLXArray {
    let alphaTVal = cachedAlphaT[stepIdx + 1]
    let sigmaTNext = cachedSigmaT[stepIdx + 1]
    let sigmaTCurr = cachedSigmaT[stepIdx]

    let lambdaTVal = cachedLambda[stepIdx + 1]
    let lambdaS = cachedLambda[stepIdx]
    let h = lambdaTVal - lambdaS

    let sigmaRatio: Float = sigmaTCurr > 0 ? sigmaTNext / sigmaTCurr : 0.0
    let expNegH = exp(-h)

    return sigmaRatio * sample - alphaTVal * (expNegH - 1.0) * x0Pred
  }

  /// Second order DPM-Solver++ update
  private func dpmSolverSecondOrderUpdate(
    x0Pred: MLXArray,
    prevX0: MLXArray,
    sample: MLXArray,
    stepIdx: Int
  ) -> MLXArray {
    let alphaTVal = cachedAlphaT[stepIdx + 1]
    let sigmaTNext = cachedSigmaT[stepIdx + 1]
    let sigmaTCurr = cachedSigmaT[stepIdx]

    let lambdaTVal = cachedLambda[stepIdx + 1]
    let lambdaS0 = cachedLambda[stepIdx]
    let lambdaS1 = stepIdx > 0 ? cachedLambda[stepIdx - 1] : lambdaS0

    let h = lambdaTVal - lambdaS0
    let h0 = lambdaS0 - lambdaS1
    let r0: Float = h != 0 ? h0 / h : 1.0

    let d0 = x0Pred
    let d1: MLXArray = r0 != 0 ? (1.0 / r0) * (x0Pred - prevX0) : MLXArray.zeros(like: x0Pred)

    let sigmaRatio: Float = sigmaTCurr > 0 ? sigmaTNext / sigmaTCurr : 0.0
    let expNegH = exp(-h)

    return sigmaRatio * sample
      - alphaTVal * (expNegH - 1.0) * d0
      - 0.5 * alphaTVal * (expNegH - 1.0) * d1
  }

  /// Perform one step of the DPM-Solver
  /// - Parameters:
  ///   - modelOutput: Direct output from the model
  ///   - timestep: Current timestep
  ///   - sample: Current sample
  ///   - prevX0: Previous x0 prediction for multi-order updates
  /// - Returns: SchedulerOutput with prevSample and x0Pred
  func step(
    modelOutput: MLXArray,
    timestep _: Int,
    sample: MLXArray,
    prevX0: MLXArray? = nil
  ) -> SchedulerOutput {
    if stepIndex == nil {
      stepIndex = 0
    }

    let stepIdx = stepIndex!

    // Convert model output to x0 prediction
    let x0Pred = convertModelOutput(modelOutput: modelOutput, sample: sample, stepIdx: stepIdx)

    // Store model output for multi-order updates
    for i in stride(from: solverOrder - 1, through: 1, by: -1) {
      modelOutputs[i] = modelOutputs[i - 1]
    }
    modelOutputs[0] = x0Pred

    // Determine order for this step
    let lowerOrderFinalFlag = (stepIdx == (numInferenceSteps! - 1)) &&
      ((lowerOrderFinal && numInferenceSteps! < 15) || finalSigmasType == "zero")

    let order: Int
    if lowerOrderNums < 1 || lowerOrderFinalFlag {
      order = 1
    } else if solverOrder == 2 || lowerOrderNums < 2 {
      order = 2
    } else {
      order = solverOrder
    }

    // Compute previous sample based on order
    let prevSample: MLXArray
    if order == 1 {
      prevSample = dpmSolverFirstOrderUpdate(x0Pred: x0Pred, sample: sample, stepIdx: stepIdx)
    } else if order == 2 {
      if let prev = prevX0 {
        prevSample = dpmSolverSecondOrderUpdate(x0Pred: x0Pred, prevX0: prev, sample: sample, stepIdx: stepIdx)
      } else if let modelOut = modelOutputs[1] {
        prevSample = dpmSolverSecondOrderUpdate(x0Pred: x0Pred, prevX0: modelOut, sample: sample, stepIdx: stepIdx)
      } else {
        // Fall back to first order
        prevSample = dpmSolverFirstOrderUpdate(x0Pred: x0Pred, sample: sample, stepIdx: stepIdx)
      }
    } else {
      // Higher orders not implemented, fall back to second order
      if let prev = prevX0 {
        prevSample = dpmSolverSecondOrderUpdate(x0Pred: x0Pred, prevX0: prev, sample: sample, stepIdx: stepIdx)
      } else {
        prevSample = dpmSolverFirstOrderUpdate(x0Pred: x0Pred, sample: sample, stepIdx: stepIdx)
      }
    }

    // Update lower order count
    if lowerOrderNums < solverOrder - 1 {
      lowerOrderNums += 1
    }

    // Update step index
    stepIndex! += 1

    return SchedulerOutput(prevSample: prevSample, x0Pred: x0Pred)
  }

  /// Reset scheduler state for new generation
  func reset() {
    modelOutputs = Array(repeating: nil, count: solverOrder)
    lowerOrderNums = 0
    stepIndex = nil
  }

  /// Scale model input (identity for DPM-Solver)
  func scaleModelInput(_ sample: MLXArray, timestep _: Int? = nil) -> MLXArray {
    sample
  }

  /// Add noise to samples
  func addNoise(
    originalSamples: MLXArray,
    noise: MLXArray,
    timesteps: MLXArray
  ) -> MLXArray {
    var ts = timesteps
    if ts.ndim == 0 {
      ts = ts.expandedDimensions(axis: 0)
    }

    var alphaTVal = alphaT[ts]
    var sigmaTVal = sigmaT[ts]

    // Reshape for broadcasting
    while alphaTVal.ndim < originalSamples.ndim {
      alphaTVal = alphaTVal.expandedDimensions(axis: -1)
      sigmaTVal = sigmaTVal.expandedDimensions(axis: -1)
    }

    return alphaTVal * originalSamples + sigmaTVal * noise
  }
}
