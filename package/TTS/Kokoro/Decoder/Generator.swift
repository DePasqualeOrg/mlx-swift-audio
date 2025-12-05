//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class Generator: Module {
  let numKernels: Int
  let numUpsamples: Int
  let postNFFt: Int

  @ModuleInfo(key: "m_source") var mSource: KokoroSourceModuleHnNSF
  let f0Upsample: Upsample
  @ModuleInfo(key: "noise_convs") var noiseConvs: [Conv1dInference]
  @ModuleInfo(key: "noise_res") var noiseRes: [AdaINResBlock1]
  @ModuleInfo var ups: [ConvWeighted]
  @ModuleInfo var resblocks: [AdaINResBlock1]
  @ModuleInfo(key: "conv_post") var convPost: ConvWeighted
  let reflectionPad: ReflectionPad1d
  let stft: MLXSTFT

  init(
    styleDim: Int,
    resblockKernelSizes: [Int],
    upsampleRates: [Int],
    upsampleInitialChannel: Int,
    resblockDilationSizes: [[Int]],
    upsampleKernelSizes: [Int],
    genIstftNFft: Int,
    genIstftHopSize: Int,
  ) {
    numKernels = resblockKernelSizes.count
    numUpsamples = upsampleRates.count
    postNFFt = genIstftNFft

    let upsampleScaleNum = MLX.product(MLXArray(upsampleRates)) * genIstftHopSize
    let upsampleScaleNumVal: Int = upsampleScaleNum.item()

    _mSource.wrappedValue = KokoroSourceModuleHnNSF(
      samplingRate: 24000,
      upsampleScale: upsampleScaleNum.item(),
      harmonicNum: 8,
      voicedThreshold: 10,
    )

    f0Upsample = Upsample(scaleFactor: .float(Float(upsampleScaleNumVal)))

    var upsArray: [ConvWeighted] = []
    for (u, k) in zip(upsampleRates, upsampleKernelSizes) {
      upsArray.append(ConvWeighted(
        inChannels: upsampleInitialChannel / Int(pow(2.0, Double(upsArray.count))),
        outChannels: upsampleInitialChannel / Int(pow(2.0, Double(upsArray.count + 1))),
        kernelSize: k,
        stride: u,
        padding: (k - u) / 2,
      ))
    }
    _ups.wrappedValue = upsArray

    var resBlocksArray: [AdaINResBlock1] = []
    var noiseConvsArray: [Conv1dInference] = []
    var noiseResArray: [AdaINResBlock1] = []

    for i in 0 ..< upsArray.count {
      let ch = upsampleInitialChannel / Int(pow(2.0, Double(i + 1)))
      for (k, d) in zip(resblockKernelSizes, resblockDilationSizes) {
        resBlocksArray.append(AdaINResBlock1(
          channels: ch,
          kernelSize: k,
          dilation: d,
          styleDim: styleDim,
        ))
      }

      let cCur = ch
      if i + 1 < upsampleRates.count {
        let strideF0: Int = MLX.product(MLXArray(upsampleRates)[(i + 1)...]).item()
        noiseConvsArray.append(Conv1dInference(
          inChannels: genIstftNFft + 2,
          outChannels: cCur,
          kernelSize: strideF0 * 2,
          stride: strideF0,
          padding: (strideF0 + 1) / 2,
        ))
        noiseResArray.append(AdaINResBlock1(
          channels: cCur,
          kernelSize: 7,
          dilation: [1, 3, 5],
          styleDim: styleDim,
        ))
      } else {
        noiseConvsArray.append(Conv1dInference(
          inChannels: genIstftNFft + 2,
          outChannels: cCur,
          kernelSize: 1,
        ))
        noiseResArray.append(AdaINResBlock1(
          channels: cCur,
          kernelSize: 11,
          dilation: [1, 3, 5],
          styleDim: styleDim,
        ))
      }
    }

    _resblocks.wrappedValue = resBlocksArray
    _noiseConvs.wrappedValue = noiseConvsArray
    _noiseRes.wrappedValue = noiseResArray

    // conv_post output channels = (genIstftNFft / 2 + 1) * 2 for spec and phase
    let lastCh = upsampleInitialChannel / Int(pow(2.0, Double(upsampleRates.count)))
    _convPost.wrappedValue = ConvWeighted(
      inChannels: lastCh,
      outChannels: (genIstftNFft / 2 + 1) * 2,
      kernelSize: 7,
      stride: 1,
      padding: 3,
    )

    reflectionPad = ReflectionPad1d(padding: (1, 0))

    stft = MLXSTFT(
      filterLength: genIstftNFft,
      hopLength: genIstftHopSize,
      winLength: genIstftNFft,
    )
  }

  func callAsFunction(_ x: MLXArray, _ s: MLXArray, _ F0Curve: MLXArray) -> MLXArray {
    Task { await BenchmarkTimer.shared.create(id: "GeneratorStart", parent: "Decoder") }

    // F0Curve: [B, seq] → expandedDimensions(axis: 1) → [B, 1, seq] → transpose → [B, seq, 1]
    var f0New = F0Curve.expandedDimensions(axis: 1).transposed(0, 2, 1)
    f0New = f0Upsample(f0New)

    var (harSource, _, _) = mSource(f0New)

    harSource = MLX.squeezed(harSource.transposed(0, 2, 1), axis: 1)
    let (harSpec, harPhase) = stft.transform(inputData: harSource)
    var har = MLX.concatenated([harSpec, harPhase], axis: 1)
    har = har.swappedAxes(1, 2)

    var newX = x
    for i in 0 ..< numUpsamples {
      newX = LeakyReLU(negativeSlope: 0.1)(newX)
      var xSource = noiseConvs[i](har)
      xSource = xSource.swappedAxes(1, 2)
      xSource = noiseRes[i](xSource, s)

      newX = newX.swappedAxes(1, 2)
      let upsi = ups[i]
      newX = upsi.callAsFunction(newX, conv: { a, b, c, d, e, f, g in
        MLX.convTransposed1d(a, b, stride: c, padding: d, dilation: e, outputPadding: 0, groups: f, stream: g)
      })
      newX = newX.swappedAxes(1, 2)

      if i == numUpsamples - 1 {
        newX = reflectionPad(newX)
      }
      newX = newX + xSource

      var xs: MLXArray?
      for j in 0 ..< numKernels {
        if xs == nil {
          xs = resblocks[i * numKernels + j](newX, s)
        } else {
          let temp = resblocks[i * numKernels + j](newX, s)
          xs = xs! + temp
        }
      }
      newX = xs! / numKernels
    }

    newX = LeakyReLU(negativeSlope: 0.01)(newX)

    newX = newX.swappedAxes(1, 2)
    newX = convPost(newX, conv: MLX.conv1d)
    newX = newX.swappedAxes(1, 2)

    let spec = MLX.exp(newX[0..., 0 ..< (postNFFt / 2 + 1), 0...])
    let phase = MLX.sin(newX[0..., (postNFFt / 2 + 1)..., 0...])

    spec.eval()
    phase.eval()

    Task { await BenchmarkTimer.shared.stop(id: "GeneratorStart") }

    Task { await BenchmarkTimer.shared.create(id: "InverseSTFT", parent: "Decoder") }
    let result = stft.inverse(magnitude: spec, phase: phase)
    result.eval()
    Task { await BenchmarkTimer.shared.stop(id: "InverseSTFT") }

    return result
  }
}
