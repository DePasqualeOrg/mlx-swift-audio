import Foundation
import MLX
import Synchronization

/// Actor wrapper for Whisper model that provides thread-safe transcription
actor WhisperSTT {
  // MARK: - Properties

  // Model is nonisolated(unsafe) because it contains non-Sendable types (MLXArray)
  // but is only accessed within the actor's methods
  nonisolated(unsafe) let model: WhisperModel
  nonisolated(unsafe) let tokenizer: WhisperTokenizer

  // MARK: - Initialization

  private init(model: WhisperModel, tokenizer: WhisperTokenizer) {
    self.model = model
    self.tokenizer = tokenizer
  }

  /// Load WhisperSTT from Hugging Face Hub
  ///
  /// - Parameters:
  ///   - modelSize: Model size to load
  ///   - progressHandler: Optional callback for download/load progress
  /// - Returns: Initialized WhisperSTT instance
  static func load(
    modelSize: WhisperModelSize,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> WhisperSTT {
    // Load model first (the slow operation with progress)
    let model = try await WhisperModel.load(
      modelSize: modelSize,
      progressHandler: progressHandler
    )

    // Then load tokenizer (fast operation) with correct vocabulary for model type
    let tokenizer = try await WhisperTokenizer.load(isMultilingual: model.isMultilingual)

    // Validate tokenizer configuration matches model expectations
    // This catches critical bugs like off-by-one errors in special token IDs
    let modelVocabSize = model.dims.n_vocab

    // Verify key special tokens are in valid range
    let maxTokenId = max(
      tokenizer.eot,
      tokenizer.sot,
      tokenizer.translate,
      tokenizer.transcribe,
      tokenizer.noSpeech,
      tokenizer.timestampBegin
    )

    if maxTokenId >= modelVocabSize {
      throw STTError.invalidArgument(
        """
        Tokenizer misconfiguration: token ID \(maxTokenId) >= model vocab size \(modelVocabSize). \
        This indicates a critical bug in tokenizer setup.
        """
      )
    }

    // Verify critical token IDs match expected values (universal across all Whisper models)
    assert(tokenizer.eot == 50257, "EOT token must be 50257")
    assert(tokenizer.sot == 50258, "SOT token must be 50258")
    assert(tokenizer.transcribe == 50360, "Transcribe token must be 50360")

    return WhisperSTT(model: model, tokenizer: tokenizer)
  }

  // MARK: - Transcription

  /// Transcribe audio to text
  ///
  /// This runs on the actor's background executor, not blocking the main thread.
  /// Long audio is automatically split into 30-second segments and processed.
  ///
  /// - Parameters:
  ///   - audio: Audio waveform (T,) in 16 kHz
  ///   - language: Optional language code (e.g., "en", "zh"), nil for auto-detect
  ///   - task: Transcription task (transcribe or translate)
  ///   - temperature: Sampling temperature (0.0 for greedy)
  ///   - timestamps: Timestamp granularity
  /// - Returns: Transcription result
  func transcribe(
    audio: MLXArray,
    language: String?,
    task: TranscriptionTask,
    temperature: Float,
    timestamps: TimestampGranularity
  ) -> TranscriptionResult {
    let startTime = CFAbsoluteTimeGetCurrent()

    // Segment audio into 30-second chunks if needed
    let segments = segmentAudio(audio)

    Log.model.info("Transcribing \(segments.count) audio segment(s)")

    var allSegments: [TranscriptionSegment] = []
    var detectedLanguage: String?

    for (segmentIdx, audioSegment) in segments.enumerated() {
      // Pad or trim to exactly 30 seconds
      let paddedAudio = padOrTrim(audioSegment)
      eval(paddedAudio) // Ensure padding is evaluated before computing mel

      // Compute mel spectrogram with the model's n_mels
      let mel = whisperLogMelSpectrogram(audio: paddedAudio, nMels: model.dims.n_mels)

      // Transpose from (n_mels, n_frames) to (n_frames, n_mels) for MLX Conv1d
      // MLX Conv1d expects (batch, length, channels) format - channels LAST (unlike PyTorch)
      let melTransposed = mel.transposed()
      let batchedMel = melTransposed.expandedDimensions(axis: 0) // (1, n_frames, n_mels)

      // Detect language on first segment if not specified
      if detectedLanguage == nil, language == nil {
        let (lang, prob) = detectLanguageFromMel(batchedMel)
        detectedLanguage = lang
        Log.model.info("Detected language: \(lang) (probability: \(String(format: "%.2f", prob)))")
      }

      let languageToUse = language ?? detectedLanguage ?? "en"

      // Create decoding options
      let options = DecodingOptions(
        task: task,
        language: languageToUse,
        temperature: temperature,
        maxTokens: 448,
        timestamps: timestamps
      )

      // Create decoder and decode
      let decoder = GreedyDecoder(
        model: model,
        tokenizer: tokenizer,
        options: options
      )

      let decodingResult = decoder.decode(batchedMel)

      // Create segment with timestamps
      let startTime = Float(segmentIdx) * Float(WhisperAudio.chunkLength)
      let endTime = startTime + Float(WhisperAudio.chunkLength)

      let segment = TranscriptionSegment(
        text: decodingResult.text,
        start: TimeInterval(startTime),
        end: TimeInterval(endTime),
        tokens: decodingResult.tokens,
        avgLogProb: decodingResult.avgLogProb,
        noSpeechProb: decodingResult.noSpeechProb,
        words: nil // Word-level timestamps not implemented yet
      )

      allSegments.append(segment)
    }

    // Combine all segment texts
    let fullText = allSegments.map { $0.text }.joined(separator: " ").trimmingCharacters(in: CharacterSet.whitespaces)

    let endTime = CFAbsoluteTimeGetCurrent()
    let totalTime = endTime - startTime
    let audioDuration = Double(audio.shape[0]) / Double(WhisperAudio.sampleRate)

    Log.model.info("Transcription complete: \(String(format: "%.2f", totalTime))s for \(String(format: "%.2f", audioDuration))s audio (RTF: \(String(format: "%.2f", totalTime / audioDuration)))")

    return TranscriptionResult(
      text: fullText,
      language: detectedLanguage ?? language ?? "en",
      segments: allSegments,
      processingTime: totalTime,
      duration: audioDuration
    )
  }

  // MARK: - Language Detection

  /// Detect the language of audio
  ///
  /// - Parameter audio: Audio waveform (T,) in 16 kHz
  /// - Returns: Tuple of (language_code, probability)
  func detectLanguage(audio: MLXArray) -> (String, Float) {
    // Pad or trim to 30 seconds
    let paddedAudio = padOrTrim(audio)
    eval(paddedAudio)

    // Compute mel spectrogram
    let mel = whisperLogMelSpectrogram(audio: paddedAudio, nMels: model.dims.n_mels)
    let melTransposed = mel.transposed()
    let batchedMel = melTransposed.expandedDimensions(axis: 0)

    return detectLanguageFromMel(batchedMel)
  }

  /// Detect language from mel spectrogram
  ///
  /// - Parameter mel: Mel spectrogram (batch=1 or unbatched)
  /// - Returns: Tuple of (language_code, probability)
  private func detectLanguageFromMel(_ mel: MLXArray) -> (String, Float) {
    // Add batch dimension if needed
    var melBatched = mel
    if mel.ndim == 2 {
      melBatched = mel.expandedDimensions(axis: 0)
    }

    // Encode audio
    let audioFeatures = model.encode(melBatched)

    // Create SOT token
    let sotToken = MLXArray([Int32(tokenizer.sot)]).expandedDimensions(axis: 0)

    // Get logits for first token after SOT
    let (logits, _, _) = model.decode(sotToken, audioFeatures: audioFeatures)

    // Extract language token logits (50259-50357)
    let languageTokenStart = 50259
    let languageTokenEnd = 50358 // Exclusive
    let languageLogits = logits[0, 0, languageTokenStart ..< languageTokenEnd]

    // Find language with highest probability
    let probs = MLX.softmax(languageLogits, axis: -1)
    let maxIdx = MLX.argMax(probs).item(Int32.self)
    let maxProb = probs[Int(maxIdx)].item(Float.self)

    // Map index to language code
    let languageIdx = Int(maxIdx)
    let languageCode = Self.languageCodes[languageIdx] ?? "en"

    return (languageCode, maxProb)
  }

  /// Language codes indexed by position (token offset from 50259)
  private static let languageCodes: [Int: String] = [
    0: "en", 1: "zh", 2: "de", 3: "es", 4: "ru", 5: "ko",
    6: "fr", 7: "ja", 8: "pt", 9: "tr", 10: "pl", 11: "ca",
    12: "nl", 13: "ar", 14: "sv", 15: "it", 16: "id", 17: "hi",
    18: "fi", 19: "vi", 20: "he", 21: "uk", 22: "el", 23: "ms",
    24: "cs", 25: "ro", 26: "da", 27: "hu", 28: "ta", 29: "no",
    30: "th", 31: "ur", 32: "hr", 33: "bg", 34: "lt", 35: "la",
    36: "mi", 37: "ml", 38: "cy", 39: "sk", 40: "te", 41: "fa",
    42: "lv", 43: "bn", 44: "sr", 45: "az", 46: "sl", 47: "kn",
    48: "et", 49: "mk", 50: "br", 51: "eu", 52: "is", 53: "hy",
    54: "ne", 55: "mn", 56: "bs", 57: "kk", 58: "sq", 59: "sw",
    60: "gl", 61: "mr", 62: "pa", 63: "si", 64: "km", 65: "sn",
    66: "yo", 67: "so", 68: "af", 69: "oc", 70: "ka", 71: "be",
    72: "tg", 73: "sd", 74: "gu", 75: "am", 76: "yi", 77: "lo",
    78: "uz", 79: "fo", 80: "ht", 81: "ps", 82: "tk", 83: "nn",
    84: "mt", 85: "sa", 86: "lb", 87: "my", 88: "bo", 89: "tl",
    90: "mg", 91: "as", 92: "tt", 93: "haw", 94: "ln", 95: "ha",
    96: "ba", 97: "jw", 98: "su",
  ]

  // MARK: - Audio Segmentation

  /// Segment long audio into 30-second chunks
  ///
  /// - Parameter audio: Audio waveform (T,)
  /// - Returns: Array of audio segments
  private func segmentAudio(_ audio: MLXArray) -> [MLXArray] {
    let audioLength = audio.shape[0]
    let chunkSamples = WhisperAudio.nSamples // 480,000 samples (30s at 16kHz)

    // If audio is shorter than or equal to 30 seconds, return as single segment
    if audioLength <= chunkSamples {
      return [audio]
    }

    // Split into 30-second chunks
    var segments: [MLXArray] = []
    var start = 0

    while start < audioLength {
      let end = min(start + chunkSamples, audioLength)
      let segment = audio[start ..< end]
      segments.append(segment)
      start = end
    }

    return segments
  }
}
