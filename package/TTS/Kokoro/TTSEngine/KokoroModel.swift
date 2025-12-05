//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

/// Predictor module containing duration encoder, duration LSTM, and prosody predictor components
/// Weight keys: predictor.text_encoder.*, predictor.lstm.*, predictor.duration_proj.*, predictor.shared.*, etc.
class Predictor: Module {
  // Duration prediction components
  @ModuleInfo(key: "text_encoder") var textEncoder: DurationEncoder
  @ModuleInfo var lstm: BiLSTM
  @ModuleInfo(key: "duration_proj") var durationProj: Linear

  // Prosody prediction components (previously in ProsodyPredictor)
  @ModuleInfo var shared: BiLSTM
  @ModuleInfo var F0: [AdainResBlk1d]
  @ModuleInfo var N: [AdainResBlk1d]
  @ModuleInfo(key: "F0_proj") var F0Proj: Conv1dInference
  @ModuleInfo(key: "N_proj") var NProj: Conv1dInference

  init(dModel: Int = 512, styleDim: Int = 128) {
    // Duration encoder
    _textEncoder.wrappedValue = DurationEncoder(dModel: dModel, styDim: styleDim, nlayers: 3)

    // Duration prediction LSTM
    _lstm.wrappedValue = BiLSTM(inputSize: dModel + styleDim, hiddenSize: dModel / 2)

    // Duration projection
    _durationProj.wrappedValue = Linear(dModel, 1)

    // Prosody predictor components
    _shared.wrappedValue = BiLSTM(inputSize: dModel + styleDim, hiddenSize: dModel / 2)

    _F0.wrappedValue = [
      AdainResBlk1d(dimIn: dModel, dimOut: dModel, styleDim: styleDim),
      AdainResBlk1d(dimIn: dModel, dimOut: dModel / 2, styleDim: styleDim, upsample: "true"),
      AdainResBlk1d(dimIn: dModel / 2, dimOut: dModel / 2, styleDim: styleDim),
    ]

    _N.wrappedValue = [
      AdainResBlk1d(dimIn: dModel, dimOut: dModel, styleDim: styleDim),
      AdainResBlk1d(dimIn: dModel, dimOut: dModel / 2, styleDim: styleDim, upsample: "true"),
      AdainResBlk1d(dimIn: dModel / 2, dimOut: dModel / 2, styleDim: styleDim),
    ]

    _F0Proj.wrappedValue = Conv1dInference(
      inChannels: dModel / 2,
      outChannels: 1,
      kernelSize: 1,
      padding: 0,
    )

    _NProj.wrappedValue = Conv1dInference(
      inChannels: dModel / 2,
      outChannels: 1,
      kernelSize: 1,
      padding: 0,
    )
  }

  /// Predict F0 and N prosody features
  func F0NTrain(x: MLXArray, s: MLXArray) -> (MLXArray, MLXArray) {
    let (x1, _) = shared(x.transposed(0, 2, 1))

    // F0 prediction
    var F0Val = x1.transposed(0, 2, 1)
    for block in F0 {
      F0Val = block(x: F0Val, s: s)
    }
    F0Val = F0Val.swappedAxes(1, 2)
    F0Val = F0Proj(F0Val)
    F0Val = F0Val.swappedAxes(1, 2)

    // N prediction
    var NVal = x1.transposed(0, 2, 1)
    for block in N {
      NVal = block(x: NVal, s: s)
    }
    NVal = NVal.swappedAxes(1, 2)
    NVal = NProj(NVal)
    NVal = NVal.swappedAxes(1, 2)

    return (F0Val.squeezed(axis: 1), NVal.squeezed(axis: 1))
  }
}

/// Top-level Kokoro TTS model containing all components
/// Weight keys match the safetensor structure: bert.*, bert_encoder.*, text_encoder.*, predictor.*, decoder.*
class KokoroModel: Module {
  let config: AlbertConfig

  @ModuleInfo var bert: CustomAlbert
  @ModuleInfo(key: "bert_encoder") var bertEncoder: Linear
  @ModuleInfo(key: "text_encoder") var textEncoder: TextEncoder
  @ModuleInfo var predictor: Predictor
  @ModuleInfo var decoder: KokoroDecoder

  init(config: AlbertConfig = AlbertConfig()) {
    self.config = config

    _bert.wrappedValue = CustomAlbert(config: config)
    _bertEncoder.wrappedValue = Linear(config.hiddenSize, 512)

    _textEncoder.wrappedValue = TextEncoder(
      channels: 512,
      kernelSize: 5,
      depth: 3,
      nSymbols: 178,
    )

    _predictor.wrappedValue = Predictor(dModel: 512, styleDim: 128)

    _decoder.wrappedValue = KokoroDecoder(
      dimIn: 512,
      styleDim: 128,
      dimOut: 80,
      resblockKernelSizes: [3, 7, 11],
      upsampleRates: [10, 6],
      upsampleInitialChannel: 512,
      resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
      upsampleKernelSizes: [20, 12],
      genIstftNFft: 20,
      genIstftHopSize: 5,
    )
  }
}
