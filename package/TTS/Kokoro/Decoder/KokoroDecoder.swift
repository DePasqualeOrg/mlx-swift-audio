import Foundation
import MLX
import MLXNN

class KokoroDecoder: Module {
  @ModuleInfo var encode: AdainResBlk1d
  @ModuleInfo var decode: [AdainResBlk1d]
  @ModuleInfo(key: "F0_conv") var F0Conv: ConvWeighted
  @ModuleInfo(key: "N_conv") var NConv: ConvWeighted
  @ModuleInfo(key: "asr_res") var asrRes: [ConvWeighted]
  @ModuleInfo var generator: Generator

  init(
    dimIn: Int,
    styleDim: Int,
    dimOut _: Int,
    resblockKernelSizes: [Int],
    upsampleRates: [Int],
    upsampleInitialChannel: Int,
    resblockDilationSizes: [[Int]],
    upsampleKernelSizes: [Int],
    genIstftNFft: Int,
    genIstftHopSize: Int,
  ) {
    _encode.wrappedValue = AdainResBlk1d(
      dimIn: dimIn + 2,
      dimOut: 1024,
      styleDim: styleDim,
    )

    _decode.wrappedValue = [
      AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim),
      AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim),
      AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim),
      AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 512, styleDim: styleDim, upsample: "true"),
    ]

    _F0Conv.wrappedValue = ConvWeighted(
      inChannels: 1,
      outChannels: 1,
      kernelSize: 3,
      stride: 2,
      padding: 1,
      groups: 1,
    )
    _NConv.wrappedValue = ConvWeighted(
      inChannels: 1,
      outChannels: 1,
      kernelSize: 3,
      stride: 2,
      padding: 1,
      groups: 1,
    )

    _asrRes.wrappedValue = [
      ConvWeighted(
        inChannels: dimIn,
        outChannels: 64,
        kernelSize: 1,
        stride: 1,
        padding: 0,
      ),
    ]

    _generator.wrappedValue = Generator(
      styleDim: styleDim,
      resblockKernelSizes: resblockKernelSizes,
      upsampleRates: upsampleRates,
      upsampleInitialChannel: upsampleInitialChannel,
      resblockDilationSizes: resblockDilationSizes,
      upsampleKernelSizes: upsampleKernelSizes,
      genIstftNFft: genIstftNFft,
      genIstftHopSize: genIstftHopSize,
    )
  }

  func callAsFunction(asr: MLXArray, F0Curve: MLXArray, N: MLXArray, s: MLXArray) -> MLXArray {
    let F0CurveSwapped = F0Curve.reshaped([F0Curve.shape[0], 1, F0Curve.shape[1]]).swappedAxes(1, 2)
    let F0 = F0Conv(F0CurveSwapped, conv: MLX.conv1d).swappedAxes(1, 2)

    let NSwapped = N.reshaped([N.shape[0], 1, N.shape[1]]).swappedAxes(1, 2)
    let NProcessed = NConv(NSwapped, conv: MLX.conv1d).swappedAxes(1, 2)

    var x = MLX.concatenated([asr, F0, NProcessed], axis: 1)
    x = encode(x: x, s: s)

    let asrResidual = asrRes[0](asr.swappedAxes(1, 2), conv: MLX.conv1d).swappedAxes(1, 2)
    var res = true

    x.eval()

    for block in decode {
      if res {
        x = MLX.concatenated([x, asrResidual, F0, NProcessed], axis: 1)
      }
      x = block(x: x, s: s)

      if block.upsampleType != "none" {
        res = false
      }
    }

    x.eval()

    return generator(x, s, F0Curve)
  }
}
