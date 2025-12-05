//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import MLXRandom

class KokoroSourceModuleHnNSF: Module {
  private let sineAmp: Float
  private let noiseStd: Float
  private let lSinGen: KokoroSineGen

  @ModuleInfo(key: "l_linear") var lLinear: Linear

  init(
    samplingRate: Int,
    upsampleScale: Float,
    harmonicNum: Int = 0,
    sineAmp: Float = 0.1,
    addNoiseStd: Float = 0.003,
    voicedThreshold: Float = 0,
  ) {
    self.sineAmp = sineAmp
    noiseStd = addNoiseStd

    // To produce sine waveforms
    lSinGen = KokoroSineGen(
      sampRate: samplingRate,
      upsampleScale: upsampleScale,
      harmonicNum: harmonicNum,
      sineAmp: sineAmp,
      noiseStd: addNoiseStd,
      voicedThreshold: voicedThreshold,
    )

    // To merge source harmonics into a single excitation
    // Input: harmonicNum + 1, Output: 1
    _lLinear.wrappedValue = Linear(harmonicNum + 1, 1)

    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let (sineWavs, uv, _) = lSinGen(x)
    let sineMerge = tanh(lLinear(sineWavs))

    let noise = MLXRandom.normal(uv.shape) * (sineAmp / 3)

    // Note: Turns out we don't need noise or uv for that matter
    return (sineMerge, noise, uv)
  }
}
