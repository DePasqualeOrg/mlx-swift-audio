//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import Synchronization

public actor KokoroTTS {
  enum KokoroTTSError: LocalizedError {
    case tooManyTokens
    case sentenceSplitError
    case modelNotInitialized
    case audioGenerationError

    var errorDescription: String? {
      switch self {
        case .tooManyTokens:
          "Input text exceeds maximum token limit"
        case .sentenceSplitError:
          "Failed to split text into sentences"
        case .modelNotInitialized:
          "Model has not been initialized"
        case .audioGenerationError:
          "Failed to generate audio"
      }
    }
  }

  // MARK: - Constants

  private static let maxTokenCount = 510
  private static let sampleRate = 24000

  // MARK: - Properties

  private var model: KokoroModel!
  private var eSpeakEngine: ESpeakNGEngine!
  private var kokoroTokenizer: KokoroTokenizer!
  private var chosenVoice: KokoroEngine.Voice?
  private var voice: MLXArray!

  // Flag to indicate if model components are initialized
  private var isModelInitialized = false

  // Hugging Face repo configuration
  private var repoId: String
  private var progressHandler: @Sendable (Progress) -> Void

  /// Initializes with optional Hugging Face repo configuration.
  ///
  /// Models are downloaded from Hugging Face Hub on first use.
  init(
    repoId: String = KokoroWeightLoader.defaultRepoId,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in },
  ) {
    self.repoId = repoId
    self.progressHandler = progressHandler
  }

  // Reset the model to free up memory
  func resetModel(preserveTextProcessing: Bool = true) {
    // Reset heavy ML model components
    model = nil
    voice = nil
    chosenVoice = nil
    isModelInitialized = false

    // Optionally preserve text processing components for faster restart
    if !preserveTextProcessing {
      if let _ = eSpeakEngine {
        // Ensure eSpeakEngine is terminated properly
        eSpeakEngine = nil
      }
      kokoroTokenizer = nil
    }
  }

  // Initialize model on demand
  private func ensureModelInitialized() async throws {
    if isModelInitialized {
      return
    }

    // Initialize text processing components first (less expensive)
    if eSpeakEngine == nil {
      eSpeakEngine = try ESpeakNGEngine()
    }

    if kokoroTokenizer == nil {
      kokoroTokenizer = KokoroTokenizer(engine: eSpeakEngine)
    }

    // Load lexicons from GitHub (cached on disk)
    if !kokoroTokenizer.lexiconsLoaded {
      async let usLexicon = LexiconLoader.loadUSLexicon()
      async let gbLexicon = LexiconLoader.loadGBLexicon()
      try await kokoroTokenizer.setLexicons(us: usLexicon, gb: gbLexicon)
    }

    // Load weights from Hugging Face
    let weights = try await KokoroWeightLoader.loadWeights(
      repoId: repoId,
      progressHandler: progressHandler,
    )

    // Create model and load weights using standard MLX pattern
    model = KokoroModel()
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: .noUnusedKeys)

    isModelInitialized = true
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

    // Ensure model is initialized
    guard let model else {
      throw KokoroTTSError.modelNotInitialized
    }

    let (bertDur, _) = model.bert(paddedInputIds, attentionMask: attentionMask)
    bertDur.eval()

    let dEn = model.bertEncoder(bertDur).transposed(0, 2, 1)
    dEn.eval()

    guard let voice else {
      throw KokoroTTSError.modelNotInitialized
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

  func generateAudio(voice: KokoroEngine.Voice, text: String, speed: Float = 1.0, chunkCallback: @escaping @Sendable ([Float]) -> Void) async throws {
    try await ensureModelInitialized()

    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    if sentences.isEmpty {
      throw KokoroTTSError.sentenceSplitError
    }

    self.voice = nil

    for sentence in sentences {
      let audio = try await generateAudioForSentence(voice: voice, text: sentence, speed: speed)
      chunkCallback(audio)
      MLX.GPU.clearCache()
    }

    // Reset model after completing a long text to free memory
    if sentences.count > 5 {
      resetModel()
    }
  }

  func generateAudioStream(voice: KokoroEngine.Voice, text: String, speed: Float = 1.0) async throws -> AsyncThrowingStream<[Float], Error> {
    try await ensureModelInitialized()

    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    if sentences.isEmpty {
      throw KokoroTTSError.sentenceSplitError
    }

    self.voice = nil
    let index = Atomic<Int>(0)

    return AsyncThrowingStream {
      let i = index.wrappingAdd(1, ordering: .relaxed).oldValue
      guard i < sentences.count else { return nil }

      let audio = try await self.generateAudioForSentence(voice: voice, text: sentences[i], speed: speed)
      MLX.GPU.clearCache()
      return audio
    }
  }

  private func generateAudioForSentence(voice: KokoroEngine.Voice, text: String, speed: Float) async throws -> [Float] {
    try await ensureModelInitialized()

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
      self.voice?.eval() // Force immediate evaluation

      try kokoroTokenizer.setLanguage(for: voice)
      chosenVoice = voice
    }

    do {
      let phonemizedResult = try kokoroTokenizer.phonemize(text)

      let inputIds = PhonemeTokenizer.tokenize(phonemizedText: phonemizedResult.phonemes)
      guard inputIds.count <= Self.maxTokenCount else {
        throw KokoroTTSError.tooManyTokens
      }

      // Continue with normal audio generation
      return try processTokensToAudio(inputIds: inputIds, speed: speed)
    } catch {
      // Re-throw the error instead of silently returning a beep
      // This allows proper error handling up the call stack
      Log.tts.error("KokoroTTS: Error generating audio for sentence - \(error)")
      throw error
    }
  }

  // Common processing method to convert tokens to audio - used by streaming methods
  private func processTokensToAudio(inputIds: [Int], speed: Float) throws -> [Float] {
    // Use the token processing method
    try generateAudioForTokens(
      inputIds: inputIds,
      speed: speed,
    )
  }
}
