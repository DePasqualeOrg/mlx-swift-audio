import Foundation
import MLX
import MLXNN
import Synchronization

/// Kokoro TTS actor providing thread-safe text-to-speech generation.
///
/// Use the static `load()` factory method to create an initialized instance.
actor KokoroTTS {
  enum KokoroTTSError: LocalizedError {
    case tooManyTokens
    case sentenceSplitError
    case audioGenerationError
    case voiceLoadFailed

    var errorDescription: String? {
      switch self {
        case .tooManyTokens:
          "Input text exceeds maximum token limit"
        case .sentenceSplitError:
          "Failed to split text into sentences"
        case .audioGenerationError:
          "Failed to generate audio"
        case .voiceLoadFailed:
          "Failed to load voice"
      }
    }
  }

  // MARK: - Constants

  private static let maxTokenCount = 510
  private static let sampleRate = 24000

  // MARK: - Properties

  private let model: KokoroModel
  private let eSpeakEngine: ESpeakNGEngine
  private let kokoroTokenizer: KokoroTokenizer
  private let repoId: String
  private let progressHandler: @Sendable (Progress) -> Void

  private var chosenVoice: KokoroEngine.Voice?
  private var voice: MLXArray?

  // MARK: - Initialization

  private init(
    model: KokoroModel,
    eSpeakEngine: ESpeakNGEngine,
    kokoroTokenizer: KokoroTokenizer,
    repoId: String,
    progressHandler: @escaping @Sendable (Progress) -> Void,
  ) {
    self.model = model
    self.eSpeakEngine = eSpeakEngine
    self.kokoroTokenizer = kokoroTokenizer
    self.repoId = repoId
    self.progressHandler = progressHandler
  }

  /// Load and initialize a KokoroTTS instance.
  ///
  /// - Parameters:
  ///   - repoId: Hugging Face repository ID for the model
  ///   - progressHandler: Callback for download progress
  /// - Returns: A fully initialized KokoroTTS actor
  static func load(
    repoId: String = KokoroWeightLoader.defaultRepoId,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in },
  ) async throws -> KokoroTTS {
    // Initialize text processing components
    let eSpeakEngine = try ESpeakNGEngine()
    let kokoroTokenizer = KokoroTokenizer(engine: eSpeakEngine)

    // Load lexicons from GitHub (cached on disk)
    async let usLexicon = LexiconLoader.loadUSLexicon()
    async let gbLexicon = LexiconLoader.loadGBLexicon()
    try await kokoroTokenizer.setLexicons(us: usLexicon, gb: gbLexicon)

    // Load weights from Hugging Face
    let weights = try await KokoroWeightLoader.loadWeights(
      repoId: repoId,
      progressHandler: progressHandler,
    )

    // Create model and load weights
    let model = KokoroModel()
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: .noUnusedKeys)

    return KokoroTTS(
      model: model,
      eSpeakEngine: eSpeakEngine,
      kokoroTokenizer: kokoroTokenizer,
      repoId: repoId,
      progressHandler: progressHandler,
    )
  }

  // MARK: - Public API

  func generate(
    text: String,
    voice: KokoroEngine.Voice,
    speed: Float = 1.0,
    chunkCallback: @escaping @Sendable ([Float]) -> Void,
  ) async throws {
    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    if sentences.isEmpty {
      throw KokoroTTSError.sentenceSplitError
    }

    self.voice = nil

    for sentence in sentences {
      let audio = try await generateAudioForSentence(text: sentence, voice: voice, speed: speed)
      chunkCallback(audio)
      MLX.GPU.clearCache()
    }
  }

  func generateStream(
    text: String,
    voice: KokoroEngine.Voice,
    speed: Float = 1.0,
  ) async throws -> AsyncThrowingStream<[Float], Error> {
    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    if sentences.isEmpty {
      throw KokoroTTSError.sentenceSplitError
    }

    self.voice = nil
    let index = Atomic<Int>(0)

    return AsyncThrowingStream {
      let i = index.wrappingAdd(1, ordering: .relaxed).oldValue
      guard i < sentences.count else { return nil }

      let audio = try await self.generateAudioForSentence(text: sentences[i], voice: voice, speed: speed)
      MLX.GPU.clearCache()
      return audio
    }
  }

  // MARK: - Private Methods

  private func generateAudioForSentence(
    text: String,
    voice: KokoroEngine.Voice,
    speed: Float,
  ) async throws -> [Float] {
    if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      return [0.0]
    }

    // Load voice if it changed or if it was cleared
    if chosenVoice != voice || self.voice == nil {
      self.voice = try await VoiceLoader.loadVoice(
        voice,
        repoId: repoId,
        progressHandler: progressHandler,
      )
      self.voice?.eval()

      try kokoroTokenizer.setLanguage(for: voice)
      chosenVoice = voice
    }

    let phonemizedResult = try kokoroTokenizer.phonemize(text)

    let inputIds = PhonemeTokenizer.tokenize(phonemizedText: phonemizedResult.phonemes)
    guard inputIds.count <= Self.maxTokenCount else {
      throw KokoroTTSError.tooManyTokens
    }

    return try generateAudioForTokens(inputIds: inputIds, speed: speed)
  }

  private func generateAudioForTokens(
    inputIds: [Int],
    speed: Float,
  ) throws -> [Float] {
    let paddedInputIdsBase = [0] + inputIds + [0]
    let paddedInputIds = MLXArray(paddedInputIdsBase).expandedDimensions(axes: [0])
    paddedInputIds.eval()

    let inputLengths = MLXArray(paddedInputIds.dim(-1))
    inputLengths.eval()

    let inputLengthMax: Int = MLX.max(inputLengths).item()
    var textMask = MLXArray(0 ..< inputLengthMax)
    textMask.eval()

    textMask = textMask + 1 .> inputLengths
    textMask.eval()

    textMask = textMask.expandedDimensions(axes: [0])
    textMask.eval()

    let swiftTextMask: [Bool] = textMask.asArray(Bool.self)
    let swiftTextMaskInt = swiftTextMask.map { !$0 ? 1 : 0 }
    let attentionMask = MLXArray(swiftTextMaskInt).reshaped(textMask.shape)
    attentionMask.eval()

    let (bertDur, _) = model.bert(paddedInputIds, attentionMask: attentionMask)
    bertDur.eval()

    let dEn = model.bertEncoder(bertDur).transposed(0, 2, 1)
    dEn.eval()

    guard let voice else {
      throw KokoroTTSError.voiceLoadFailed
    }

    // Voice shape is [510, 1, 256], index by phoneme length to get [1, 256]
    let voiceIdx = min(inputIds.count - 1, voice.shape[0] - 1)
    let refS = voice[voiceIdx]
    refS.eval()

    // Extract style vector: columns 128+ for duration/prosody prediction
    let s = refS[0..., 128...]
    s.eval()

    let d = model.predictor.textEncoder(dEn, style: s, textLengths: inputLengths, m: textMask)
    d.eval()

    let (x, _) = model.predictor.lstm(d)
    x.eval()

    let duration = model.predictor.durationProj(x)
    duration.eval()

    let durationSigmoid = MLX.sigmoid(duration).sum(axis: -1) / speed
    durationSigmoid.eval()

    let predDur = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]
    predDur.eval()

    // Index and matrix generation
    // Build indices in chunks to reduce memory
    var allIndices: [MLXArray] = []
    let chunkSize = 50

    for startIdx in stride(from: 0, to: predDur.shape[0], by: chunkSize) {
      let endIdx = min(startIdx + chunkSize, predDur.shape[0])
      let chunkIndices = predDur[startIdx ..< endIdx]

      let indices = MLX.concatenated(
        chunkIndices.enumerated().map { i, n in
          let nSize: Int = n.item()
          let arrayIndex = MLXArray([i + startIdx])
          arrayIndex.eval()
          let repeated = MLX.repeated(arrayIndex, count: nSize)
          repeated.eval()
          return repeated
        },
      )
      indices.eval()
      allIndices.append(indices)
    }

    let indices = MLX.concatenated(allIndices)
    indices.eval()

    allIndices.removeAll()

    let indicesShape = indices.shape[0]
    let inputIdsShape = paddedInputIds.shape[1]

    // Create sparse matrix using COO format
    var rowIndices: [Int] = []
    var colIndices: [Int] = []

    // Reserve capacity to avoid reallocations
    let estimatedNonZeros = min(indicesShape, inputIdsShape * 5)
    rowIndices.reserveCapacity(estimatedNonZeros)
    colIndices.reserveCapacity(estimatedNonZeros)

    // Process in batches
    let batchSize = 256
    for startIdx in stride(from: 0, to: indicesShape, by: batchSize) {
      let endIdx = min(startIdx + batchSize, indicesShape)
      for i in startIdx ..< endIdx {
        let indiceValue: Int = indices[i].item()
        if indiceValue < inputIdsShape {
          rowIndices.append(indiceValue)
          colIndices.append(i)
        }
      }
    }

    // Create dense matrix from COO data
    var swiftPredAlnTrg = [Float](repeating: 0.0, count: inputIdsShape * indicesShape)
    let matrixBatchSize = 1000
    for startIdx in stride(from: 0, to: rowIndices.count, by: matrixBatchSize) {
      let endIdx = min(startIdx + matrixBatchSize, rowIndices.count)
      for i in startIdx ..< endIdx {
        let row = rowIndices[i]
        let col = colIndices[i]
        if row < inputIdsShape, col < indicesShape {
          swiftPredAlnTrg[row * indicesShape + col] = 1.0
        }
      }
    }

    // Create MLXArray from the dense matrix
    let predAlnTrg = MLXArray(swiftPredAlnTrg).reshaped([inputIdsShape, indicesShape])
    predAlnTrg.eval()

    // Clear Swift arrays
    swiftPredAlnTrg = []
    rowIndices = []
    colIndices = []

    let predAlnTrgBatched = predAlnTrg.expandedDimensions(axis: 0)
    predAlnTrgBatched.eval()

    let en = d.transposed(0, 2, 1).matmul(predAlnTrgBatched)
    en.eval()

    let (F0Pred, NPred) = model.predictor.F0NTrain(x: en, s: s)
    F0Pred.eval()
    NPred.eval()

    let tEn = model.textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
    tEn.eval()

    let asr = MLX.matmul(tEn, predAlnTrg)
    asr.eval()

    // Extract style vector: columns 0-127 for decoder
    let voiceS = refS[0..., ..<128]
    voiceS.eval()

    let audio = model.decoder(asr: asr, F0Curve: F0Pred, N: NPred, s: voiceS)[0]
    audio.eval()

    let audioShape = audio.shape

    // Check if the audio shape is valid
    let totalSamples: Int = if audioShape.count == 1 {
      audioShape[0]
    } else if audioShape.count == 2 {
      audioShape[1]
    } else {
      0
    }

    if totalSamples <= 1 {
      Log.tts.error("KokoroTTS: Invalid audio shape - totalSamples: \(totalSamples), shape: \(audioShape)")
      throw KokoroTTSError.audioGenerationError
    }

    return audio.asArray(Float.self)
  }
}
