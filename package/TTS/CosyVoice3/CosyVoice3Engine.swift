// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN

// MARK: - Speaker

/// Prepared speaker profile for CosyVoice3 TTS
///
/// Create using `CosyVoice3Engine.prepareSpeaker(from:)` methods.
/// Can be reused across multiple `say()` or `generate()` calls for efficient multi-speaker scenarios.
///
/// If the speaker has a transcription, zero-shot mode is used automatically for better voice alignment.
/// Otherwise, cross-lingual mode is used (works across languages).
///
/// ```swift
/// // Prepare speaker (auto-transcribes with Whisper by default)
/// let speaker = try await engine.prepareSpeaker(from: url)
/// try await engine.say("Hello world", speaker: speaker)
///
/// // With style instruction
/// try await engine.say("Hello world", speaker: speaker, instruction: "Speak slowly and calmly")
/// ```
public struct CosyVoice3Speaker: Sendable {
  /// The pre-computed conditionals for generation
  let conditionals: CosyVoice3Conditionals

  /// Sample rate of the original reference audio
  public let sampleRate: Int

  /// Duration of the reference audio in seconds
  public let duration: TimeInterval

  /// Description for display purposes
  public let description: String

  /// Whether this speaker has transcription (enables zero-shot mode)
  public let hasTranscription: Bool

  /// The transcription text (if available)
  public let transcription: String?

  init(
    conditionals: CosyVoice3Conditionals,
    sampleRate: Int,
    sampleCount: Int,
    description: String,
    transcription: String?
  ) {
    self.conditionals = conditionals
    self.sampleRate = sampleRate
    duration = Double(sampleCount) / Double(sampleRate)
    self.description = description
    hasTranscription = transcription != nil
    self.transcription = transcription
  }
}

/// Default reference audio URL - LibriVox public domain reading
/// "The Dead Boche" by Robert Graves, ~38 seconds (public domain)
public let cosyVoice3DefaultReferenceAudioURL = URL(
  string:
  "https://archive.org/download/short_poetry_001_librivox/dead_boche_graves_sm.mp3"
)!

// MARK: - CosyVoice3 Engine

/// CosyVoice3 TTS engine - Voice matching with DiT-based flow matching
///
/// CosyVoice3 is a state-of-the-art TTS model that supports:
/// - **Zero-shot voice matching**: Match any voice with just a few seconds of reference audio
/// - **Cross-lingual synthesis**: Generate speech in different languages while preserving voice characteristics
/// - **High-quality output**: 24kHz audio with natural prosody
@Observable
@MainActor
public final class CosyVoice3Engine: TTSEngine {
  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .cosyVoice3
  public let streamingGranularity: StreamingGranularity = .sentence
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - CosyVoice3-Specific Properties

  /// Generation mode for CosyVoice3
  public enum GenerationMode: String, CaseIterable, Sendable {
    /// Cross-lingual: Uses speaker embedding only (no reference text needed)
    case crossLingual = "Cross-lingual"

    /// Zero-shot: Uses reference text for better voice alignment
    case zeroShot = "Zero-shot"

    /// Instruct: Uses style instructions to control speech generation
    case instruct = "Instruct"

    /// Voice Conversion: Converts source audio to target speaker voice
    case voiceConversion = "Voice Conversion"

    public var description: String {
      switch self {
        case .crossLingual:
          "Match voice without needing reference transcription. Works across languages."
        case .zeroShot:
          "Match voice with reference transcription for better alignment."
        case .instruct:
          "Control speech style with text instructions (e.g., \"Speak slowly\")."
        case .voiceConversion:
          "Convert source audio to target speaker's voice."
      }
    }
  }

  /// Current generation mode
  public var generationMode: GenerationMode = .crossLingual

  /// Style instruction for instruct mode
  public var instructText: String = ""

  /// Top-K sampling parameter for LLM
  public var sampling: Int = 25

  /// Number of flow matching timesteps
  public var nTimesteps: Int = 10

  /// Description of the loaded source audio
  public private(set) var sourceAudioDescription: String = "No source audio"

  /// Whether source audio is loaded for voice conversion
  public private(set) var isSourceAudioLoaded: Bool = false

  /// Whether to auto-transcribe reference audio for zero-shot mode
  public var autoTranscribe: Bool = true

  // MARK: - Private Properties

  @ObservationIgnored private var cosyVoice3TTS: CosyVoice3TTS?
  @ObservationIgnored private var s3Tokenizer: S3TokenizerV3?
  @ObservationIgnored private var whisperSTT: WhisperSTT?
  @ObservationIgnored private let playback = TTSPlaybackController(sampleRate: 24000)
  @ObservationIgnored private var defaultSpeaker: CosyVoice3Speaker?
  @ObservationIgnored private var cachedSourceAudioURL: URL?

  /// HuggingFace repo ID for S3 tokenizer
  private static let s3TokenizerRepoId = "mlx-community/S3TokenizerV3"

  // MARK: - Initialization

  public init() {
    Log.tts.debug("CosyVoice3Engine initialized")
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.tts.debug("CosyVoice3Engine already loaded")
      return
    }

    Log.model.info("Loading CosyVoice3 TTS model...")

    do {
      // Load CosyVoice3 model
      cosyVoice3TTS = try await CosyVoice3TTS.load(
        progressHandler: progressHandler ?? { _ in }
      )

      // Load S3TokenizerV3
      Log.model.info("Loading S3TokenizerV3...")
      s3Tokenizer = try await Self.loadS3Tokenizer()

      isLoaded = true
      Log.model.info("CosyVoice3 TTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load CosyVoice3 model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  /// Load S3TokenizerV3 from HuggingFace
  private static func loadS3Tokenizer() async throws -> S3TokenizerV3 {
    let modelDirectory = try await HubConfiguration.shared.snapshot(from: s3TokenizerRepoId)
    let weightURL = modelDirectory.appendingPathComponent("model.safetensors")
    let weights = try MLX.loadArrays(url: weightURL)

    let tokenizer = S3TokenizerV3()

    // Load weights into tokenizer
    let parameters = ModuleParameters.unflattened(weights)
    try tokenizer.update(parameters: parameters, verify: [.noUnusedKeys])

    // Set to eval mode
    tokenizer.train(false)
    eval(tokenizer)

    return tokenizer
  }

  /// Load WhisperSTT for transcription (lazy-loaded when needed)
  private func loadWhisperSTT() async throws -> WhisperSTT {
    if let existing = whisperSTT {
      return existing
    }

    Log.model.info("Loading Whisper for transcription...")
    let whisper = try await WhisperSTT.load(
      modelSize: .largeTurbo,
      quantization: .q4
    )
    whisperSTT = whisper
    Log.model.info("Whisper loaded successfully")
    return whisper
  }

  /// Transcribe audio using Whisper
  public func transcribe(samples: [Float], sampleRate: Int) async throws -> String {
    let result = try await transcribeWithTimestamps(samples: samples, sampleRate: sampleRate, wordTimestamps: false)
    return result.text
  }

  private func transcribeWithTimestamps(
    samples: [Float],
    sampleRate: Int,
    wordTimestamps: Bool
  ) async throws -> TranscriptionResult {
    let whisper = try await loadWhisperSTT()

    // Resample to 16kHz for Whisper if needed
    let whisperSampleRate = 16000
    let resampledSamples: [Float] = if sampleRate != whisperSampleRate {
      resampleAudio(samples, fromRate: sampleRate, toRate: whisperSampleRate)
    } else {
      samples
    }

    let audio = MLXArray(resampledSamples)
    let result = await whisper.transcribe(
      audio: audio,
      language: nil,
      task: .transcribe,
      temperature: 0.0,
      timestamps: wordTimestamps ? .word : .segment,
      hallucinationSilenceThreshold: wordTimestamps ? 2.0 : nil
    )

    Log.tts.debug("Transcribed reference audio: \(result.text)")
    return result
  }

  public func stop() async {
    await playback.stop(
      setGenerating: { self.isGenerating = $0 },
      setPlaying: { self.isPlaying = $0 }
    )
    Log.tts.debug("CosyVoice3Engine stopped")
  }

  public func unload() async {
    await stop()
    cosyVoice3TTS = nil
    s3Tokenizer = nil
    isLoaded = false
    Log.tts.debug("CosyVoice3Engine unloaded (reference audio preserved)")
  }

  public func cleanup() async throws {
    await unload()
    defaultSpeaker = nil
  }

  // MARK: - Playback

  public func play(_ audio: AudioResult) async {
    await playback.play(audio, setPlaying: { self.isPlaying = $0 })
  }

  // MARK: - Speaker Preparation

  /// Prepare a speaker profile from a URL (local file or remote)
  ///
  /// - Parameters:
  ///   - url: URL to audio file (local file path or remote URL)
  ///   - transcription: Optional transcription of the reference audio (enables zero-shot mode).
  ///                    If nil and `autoTranscribe` is true, Whisper will be used to transcribe.
  /// - Returns: Prepared speaker ready for generation
  public func prepareSpeaker(
    from url: URL,
    transcription: String? = nil
  ) async throws -> CosyVoice3Speaker {
    if !isLoaded {
      try await load()
    }

    guard let cosyVoice3TTS, let s3Tokenizer else {
      throw TTSError.modelNotLoaded
    }

    let (samples, sampleRate) = try await loadAudioSamples(from: url)
    let baseDescription = url.lastPathComponent

    return try await prepareSpeakerFromSamples(
      samples,
      sampleRate: sampleRate,
      transcription: transcription,
      baseDescription: baseDescription,
      tts: cosyVoice3TTS,
      tokenizer: s3Tokenizer
    )
  }

  /// Prepare a speaker profile from raw samples
  public func prepareSpeaker(
    from samples: [Float],
    sampleRate: Int,
    transcription: String? = nil
  ) async throws -> CosyVoice3Speaker {
    if !isLoaded {
      try await load()
    }

    guard let cosyVoice3TTS, let s3Tokenizer else {
      throw TTSError.modelNotLoaded
    }

    let duration = Double(samples.count) / Double(sampleRate)
    let baseDescription = String(format: "Custom audio (%.1f sec.)", duration)

    return try await prepareSpeakerFromSamples(
      samples,
      sampleRate: sampleRate,
      transcription: transcription,
      baseDescription: baseDescription,
      tts: cosyVoice3TTS,
      tokenizer: s3Tokenizer
    )
  }

  /// Prepare the default speaker (LibriVox public domain sample)
  public func prepareDefaultSpeaker(
    transcription: String? = nil
  ) async throws -> CosyVoice3Speaker {
    try await prepareSpeaker(from: cosyVoice3DefaultReferenceAudioURL, transcription: transcription)
  }

  // MARK: - Private Speaker Preparation

  private func prepareSpeakerFromSamples(
    _ samples: [Float],
    sampleRate: Int,
    transcription: String?,
    baseDescription: String,
    tts: CosyVoice3TTS,
    tokenizer: S3TokenizerV3
  ) async throws -> CosyVoice3Speaker {
    Log.tts.debug("Preparing speaker: \(baseDescription)")

    // Resample to 24kHz if needed
    let targetSampleRate = 24000
    var processedSamples: [Float] = if sampleRate != targetSampleRate {
      resampleAudio(samples, fromRate: sampleRate, toRate: targetSampleRate)
    } else {
      samples
    }

    let originalDuration = Float(processedSamples.count) / Float(targetSampleRate)
    Log.tts.debug("Original reference audio duration: \(originalDuration)s")

    // Trim silence
    processedSamples = AudioTrimmer.trimSilence(
      processedSamples,
      sampleRate: targetSampleRate,
      config: .cosyVoice2
    )

    let silenceTrimmedDuration = Float(processedSamples.count) / Float(targetSampleRate)
    Log.tts.debug("After silence trimming: \(silenceTrimmedDuration)s")

    // Handle max duration (30 seconds)
    let maxDuration: Float = 30.0
    let maxSamples = Int(maxDuration) * targetSampleRate

    var finalTranscription = transcription
    var clippedAtWordBoundary = false

    if processedSamples.count > maxSamples {
      if autoTranscribe {
        Log.tts.debug("Audio exceeds \(maxDuration)s, using word-boundary clipping...")

        let result = try await transcribeWithTimestamps(
          samples: processedSamples,
          sampleRate: targetSampleRate,
          wordTimestamps: true
        )

        var allWords = result.segments.flatMap { $0.words ?? [] }

        if !allWords.isEmpty {
          allWords = AudioTrimmer.dropUnreliableTrailingWords(
            allWords,
            audioDuration: silenceTrimmedDuration,
            config: .cosyVoice2
          )

          if let (clipSample, validWords) = AudioTrimmer.findWordBoundaryClipPoint(
            words: allWords,
            maxDuration: maxDuration,
            sampleRate: targetSampleRate
          ) {
            processedSamples = Array(processedSamples.prefix(clipSample))
            finalTranscription = validWords.map(\.word).joined().trimmingCharacters(in: .whitespaces)
            clippedAtWordBoundary = true
            let newDuration = Float(processedSamples.count) / Float(targetSampleRate)
            Log.tts.debug("Clipped at word boundary: \(newDuration)s")
          } else {
            processedSamples = Array(processedSamples.prefix(maxSamples))
          }
        } else {
          processedSamples = Array(processedSamples.prefix(maxSamples))
        }
      } else {
        processedSamples = Array(processedSamples.prefix(maxSamples))
      }
    }

    // Auto-transcribe if enabled
    if finalTranscription == nil, autoTranscribe, !clippedAtWordBoundary {
      Log.tts.debug("Auto-transcribing reference audio...")
      finalTranscription = try await transcribe(samples: processedSamples, sampleRate: targetSampleRate)
    }

    finalTranscription = finalTranscription?.trimmingCharacters(in: .whitespacesAndNewlines)
    if finalTranscription?.isEmpty == true {
      finalTranscription = nil
    }

    let refWav = MLXArray(processedSamples)

    nonisolated(unsafe) let tokenizerUnsafe = tokenizer

    let conditionals = await tts.prepareConditionals(
      refWav: refWav,
      refText: finalTranscription,
      s3Tokenizer: { mel, melLen in tokenizerUnsafe(mel, melLen: melLen) }
    )

    let description = finalTranscription != nil
      ? "\(baseDescription) (with transcription)"
      : baseDescription

    Log.tts.debug("Speaker prepared: \(description)")

    return CosyVoice3Speaker(
      conditionals: conditionals,
      sampleRate: targetSampleRate,
      sampleCount: processedSamples.count,
      description: description,
      transcription: finalTranscription
    )
  }

  // MARK: - Audio Loading Helpers

  private func resampleAudio(_ samples: [Float], fromRate: Int, toRate: Int) -> [Float] {
    if fromRate == toRate { return samples }

    let ratio = Float(toRate) / Float(fromRate)
    let newLength = Int(Float(samples.count) * ratio)
    var resampled = [Float](repeating: 0, count: newLength)

    for i in 0 ..< newLength {
      let srcIdx = Float(i) / ratio
      let idx0 = Int(srcIdx)
      let idx1 = min(idx0 + 1, samples.count - 1)
      let frac = srcIdx - Float(idx0)
      resampled[i] = samples[idx0] * (1 - frac) + samples[idx1] * frac
    }

    return resampled
  }

  private func loadAudioSamples(from url: URL) async throws -> (samples: [Float], sampleRate: Int) {
    if url.isFileURL {
      try await loadAudioFromFile(url)
    } else {
      try await loadAudioFromRemoteURL(url)
    }
  }

  private func loadAudioFromRemoteURL(_ url: URL) async throws -> (samples: [Float], sampleRate: Int) {
    Log.tts.debug("Downloading reference audio from URL: \(url)")

    let (data, response) = try await URLSession.shared.data(from: url)

    guard let httpResponse = response as? HTTPURLResponse,
          (200 ... 299).contains(httpResponse.statusCode)
    else {
      throw TTSError.invalidArgument("Failed to download reference audio from URL")
    }

    let tempURL = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString)
      .appendingPathExtension(url.pathExtension.isEmpty ? "mp3" : url.pathExtension)

    try data.write(to: tempURL)
    defer { try? FileManager.default.removeItem(at: tempURL) }

    return try await loadAudioFromFile(tempURL)
  }

  private func loadAudioFromFile(_ url: URL) async throws -> (samples: [Float], sampleRate: Int) {
    Log.tts.debug("Loading reference audio from file: \(url.path)")

    let audioFile = try AVAudioFile(forReading: url)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
      throw TTSError.invalidArgument("Failed to create audio buffer")
    }

    try audioFile.read(into: buffer)

    guard let floatData = buffer.floatChannelData else {
      throw TTSError.invalidArgument("Failed to read audio data")
    }

    let samples: [Float]
    if format.channelCount == 1 {
      samples = Array(UnsafeBufferPointer(start: floatData[0], count: Int(buffer.frameLength)))
    } else {
      let left = UnsafeBufferPointer(start: floatData[0], count: Int(buffer.frameLength))
      let right = UnsafeBufferPointer(start: floatData[1], count: Int(buffer.frameLength))
      samples = zip(left, right).map { ($0 + $1) / 2.0 }
    }

    return (samples, Int(format.sampleRate))
  }

  // MARK: - Generation

  /// Generate audio from text
  ///
  /// Automatically selects zero-shot or cross-lingual mode based on whether the speaker
  /// has a transcription.
  public func generate(
    _ text: String,
    speaker: CosyVoice3Speaker? = nil,
    instruction: String? = nil
  ) async throws -> AudioResult {
    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw TTSError.invalidArgument("Text cannot be empty")
    }

    if !isLoaded {
      try await load()
    }

    guard let cosyVoice3TTS else {
      throw TTSError.modelNotLoaded
    }

    isGenerating = true
    let startTime = CFAbsoluteTimeGetCurrent()

    defer {
      isGenerating = false
      generationTime = CFAbsoluteTimeGetCurrent() - startTime
    }

    // Prepare speaker if needed
    let spk: CosyVoice3Speaker
    if let speaker {
      spk = speaker
    } else {
      if defaultSpeaker == nil {
        defaultSpeaker = try await prepareDefaultSpeaker()
      }
      spk = defaultSpeaker!
    }

    // Tokenize input text
    let textTokens = await cosyVoice3TTS.encode(text: text)

    // Generate based on mode or instruction
    let result: TTSGenerationResult

    let effectiveInstruction = instruction ?? (generationMode == .instruct ? instructText : nil)

    if let instruction = effectiveInstruction,
       !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    {
      // Instruct mode
      result = try await cosyVoice3TTS.generateInstruct(
        text: text,
        textTokens: textTokens,
        instructText: instruction,
        conditionals: spk.conditionals,
        sampling: sampling,
        nTimesteps: nTimesteps
      )
    } else if generationMode == .voiceConversion {
      throw TTSError.invalidArgument(
        "Voice conversion requires source audio. Use convertVoice(from:to:) instead."
      )
    } else if spk.hasTranscription, generationMode != .crossLingual {
      // Zero-shot mode
      result = try await cosyVoice3TTS.generateZeroShot(
        text: text,
        textTokens: textTokens,
        conditionals: spk.conditionals,
        sampling: sampling,
        nTimesteps: nTimesteps
      )
    } else {
      // Cross-lingual mode
      result = try await cosyVoice3TTS.generateCrossLingual(
        text: text,
        textTokens: textTokens,
        conditionals: spk.conditionals,
        sampling: sampling,
        nTimesteps: nTimesteps
      )
    }

    Log.tts.timing("CosyVoice3 generation", duration: result.processingTime)
    lastGeneratedAudioURL = playback.saveAudioFile(samples: result.audio, sampleRate: result.sampleRate)

    return .samples(
      data: result.audio,
      sampleRate: result.sampleRate,
      processingTime: result.processingTime
    )
  }

  /// Generate and immediately play audio
  public func say(
    _ text: String,
    speaker: CosyVoice3Speaker? = nil,
    instruction: String? = nil
  ) async throws {
    let audio = try await generate(text, speaker: speaker, instruction: instruction)
    await play(audio)
  }

  // MARK: - Voice Conversion

  /// Convert source audio to target speaker's voice
  public func convertVoice(
    from sourceURL: URL,
    to speaker: CosyVoice3Speaker? = nil
  ) async throws -> AudioResult {
    if !isLoaded {
      try await load()
    }

    guard let cosyVoice3TTS, let s3Tokenizer else {
      throw TTSError.modelNotLoaded
    }

    isGenerating = true
    let startTime = CFAbsoluteTimeGetCurrent()

    defer {
      isGenerating = false
      generationTime = CFAbsoluteTimeGetCurrent() - startTime
    }

    let (samples, sampleRate) = try await loadAudioSamples(from: sourceURL)

    let targetSampleRate = 24000
    let resampledSamples: [Float] = if sampleRate != targetSampleRate {
      resampleAudio(samples, fromRate: sampleRate, toRate: targetSampleRate)
    } else {
      samples
    }

    let sourceWav = MLXArray(resampledSamples)

    nonisolated(unsafe) let tokenizerUnsafe = s3Tokenizer
    nonisolated(unsafe) let sourceWavUnsafe = sourceWav
    await cosyVoice3TTS.prepareSourceAudioForVC(
      audio: sourceWavUnsafe,
      s3Tokenizer: { mel, melLen in tokenizerUnsafe(mel, melLen: melLen) }
    )

    let spk: CosyVoice3Speaker
    if let speaker {
      spk = speaker
    } else {
      if defaultSpeaker == nil {
        defaultSpeaker = try await prepareDefaultSpeaker()
      }
      spk = defaultSpeaker!
    }

    let result = try await cosyVoice3TTS.generateVoiceConversionFromPrepared(
      conditionals: spk.conditionals,
      nTimesteps: nTimesteps
    )

    Log.tts.timing("CosyVoice3 voice conversion", duration: result.processingTime)
    lastGeneratedAudioURL = playback.saveAudioFile(samples: result.audio, sampleRate: result.sampleRate)

    return .samples(
      data: result.audio,
      sampleRate: result.sampleRate,
      processingTime: result.processingTime
    )
  }

  /// Prepare source audio for voice conversion (caches URL for later use)
  ///
  /// - Parameter url: URL to source audio file (local or remote)
  public func prepareSourceAudio(from url: URL) async throws {
    // Validate the audio can be loaded
    _ = try await loadAudioSamples(from: url)

    cachedSourceAudioURL = url
    isSourceAudioLoaded = true
    sourceAudioDescription = url.lastPathComponent

    Log.tts.debug("Prepared source audio for voice conversion: \(url.lastPathComponent)")
  }

  /// Clear cached source audio
  public func clearSourceAudio() async {
    cachedSourceAudioURL = nil
    isSourceAudioLoaded = false
    sourceAudioDescription = "No source audio"

    Log.tts.debug("Cleared source audio for voice conversion")
  }

  /// Generate voice conversion using cached source audio
  ///
  /// - Parameter speaker: Target speaker (uses default if nil)
  /// - Returns: Converted audio result
  public func generateVoiceConversion(
    speaker: CosyVoice3Speaker? = nil
  ) async throws -> AudioResult {
    guard let sourceURL = cachedSourceAudioURL else {
      throw TTSError.invalidArgument("No source audio prepared. Call prepareSourceAudio(from:) first.")
    }

    return try await convertVoice(from: sourceURL, to: speaker)
  }

  // MARK: - Streaming

  /// Generate audio as a stream of chunks
  public func generateStreaming(
    _ text: String,
    speaker: CosyVoice3Speaker? = nil
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Text cannot be empty")) }
    }

    let sampleRate = 24000
    let currentSampling = sampling
    let currentNTimesteps = nTimesteps
    let currentGenerationMode = generationMode

    return playback.createGenerationStream(
      setGenerating: { self.isGenerating = $0 },
      setGenerationTime: { self.generationTime = $0 }
    ) { [weak self] in
      guard let self else {
        return AsyncThrowingStream { $0.finish() }
      }

      if !isLoaded {
        try await load()
      }

      guard let cosyVoice3TTS else {
        throw TTSError.modelNotLoaded
      }

      let spk: CosyVoice3Speaker
      if let speaker {
        spk = speaker
      } else {
        if defaultSpeaker == nil {
          defaultSpeaker = try await prepareDefaultSpeaker()
        }
        spk = defaultSpeaker!
      }

      let sentences = Self.splitIntoSentences(trimmedText)

      let startTime = Date()
      return AsyncThrowingStream { continuation in
        Task {
          do {
            for sentence in sentences {
              guard !Task.isCancelled else { break }

              let textTokens = await cosyVoice3TTS.encode(text: sentence)
              let useZeroShot = spk.hasTranscription && currentGenerationMode != .crossLingual

              let result: TTSGenerationResult = if useZeroShot {
                try await cosyVoice3TTS.generateZeroShot(
                  text: sentence,
                  textTokens: textTokens,
                  conditionals: spk.conditionals,
                  sampling: currentSampling,
                  nTimesteps: currentNTimesteps
                )
              } else {
                try await cosyVoice3TTS.generateCrossLingual(
                  text: sentence,
                  textTokens: textTokens,
                  conditionals: spk.conditionals,
                  sampling: currentSampling,
                  nTimesteps: currentNTimesteps
                )
              }

              let chunk = AudioChunk(
                samples: result.audio,
                sampleRate: sampleRate,
                processingTime: Date().timeIntervalSince(startTime)
              )
              continuation.yield(chunk)
            }
            continuation.finish()
          } catch is CancellationError {
            continuation.finish()
          } catch {
            continuation.finish(throwing: error)
          }
        }
      }
    }
  }

  private static func splitIntoSentences(_ text: String) -> [String] {
    let pattern = #"[.!?]+\s*"#
    let regex = try? NSRegularExpression(pattern: pattern, options: [])
    let nsText = text as NSString
    let matches = regex?.matches(in: text, options: [], range: NSRange(location: 0, length: nsText.length)) ?? []

    var sentences: [String] = []
    var lastEnd = 0

    for match in matches {
      let range = NSRange(location: lastEnd, length: match.range.location + match.range.length - lastEnd)
      let sentence = nsText.substring(with: range).trimmingCharacters(in: .whitespaces)
      if !sentence.isEmpty {
        sentences.append(sentence)
      }
      lastEnd = match.range.location + match.range.length
    }

    if lastEnd < nsText.length {
      let remaining = nsText.substring(from: lastEnd).trimmingCharacters(in: .whitespaces)
      if !remaining.isEmpty {
        sentences.append(remaining)
      }
    }

    if sentences.isEmpty, !text.isEmpty {
      sentences.append(text)
    }

    return sentences
  }

  /// Play audio with streaming
  @discardableResult
  public func sayStreaming(
    _ text: String,
    speaker: CosyVoice3Speaker? = nil
  ) async throws -> AudioResult {
    let (samples, processingTime) = try await playback.playStream(
      generateStreaming(text, speaker: speaker),
      setPlaying: { self.isPlaying = $0 }
    )

    lastGeneratedAudioURL = playback.saveAudioFile(samples: samples, sampleRate: 24000)

    return .samples(
      data: samples,
      sampleRate: 24000,
      processingTime: processingTime
    )
  }
}
