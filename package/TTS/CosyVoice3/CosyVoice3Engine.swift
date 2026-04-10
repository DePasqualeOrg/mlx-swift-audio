// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

import AVFoundation
import Foundation
import MLX
import MLXLMCommon
import MLXLMHFAPI
import MLXLMTokenizers
import MLXNN

// MARK: - Speaker

/// Prepared speaker profile for CosyVoice3 TTS
///
/// Create using `CosyVoice3Engine.prepareSpeaker(from:)` methods.
/// Can be reused across multiple `say()` or `generate()` calls for efficient multi-speaker scenarios.
///
/// If the speaker has an explicit reference transcription, zero-shot mode is used automatically
/// for better voice alignment. Otherwise, cross-lingual mode is used.
///
/// This matches the standard frontend behavior in the original PyTorch
/// implementation (`FunAudioLLM/CosyVoice`) in:
/// - `cosyvoice/cli/frontend.py:_extract_speech_token`
/// - `cosyvoice/cli/frontend.py:frontend_zero_shot`
///
/// The original PyTorch zero-shot path expects the caller to provide both:
/// - a reference clip no longer than 30 seconds
/// - the exact transcript for that same clip
///
/// ```swift
/// // Prepare speaker. Auto-transcribed text is metadata only and does not
/// // enable the zero-shot mode used by the original PyTorch implementation;
/// // pass `transcription:` explicitly for that.
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

  /// Whether this speaker has any transcription metadata available.
  public let hasTranscription: Bool

  /// Whether the transcription was explicitly supplied by the caller.
  /// Upstream CosyVoice3 zero-shot requires an explicit prompt transcript.
  public let hasExplicitTranscription: Bool

  /// The transcription text (if available)
  public let transcription: String?

  init(
    conditionals: CosyVoice3Conditionals,
    sampleRate: Int,
    sampleCount: Int,
    description: String,
    transcription: String?,
    hasExplicitTranscription: Bool = false
  ) {
    self.conditionals = conditionals
    self.sampleRate = sampleRate
    duration = Double(sampleCount) / Double(sampleRate)
    self.description = description
    hasTranscription = transcription != nil
    self.hasExplicitTranscription = transcription != nil && hasExplicitTranscription
    self.transcription = transcription
  }
}

/// Default reference audio URL - LJ Speech Dataset sample
/// ~7 seconds (public domain)
public let cosyVoice3DefaultReferenceAudioURL = URL(
  string:
  "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav"
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
  public let supportedStreamingGranularities: Set<StreamingGranularity> = [.sentence, .token]
  public let defaultStreamingGranularity: StreamingGranularity = .token
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

  /// Whether to auto-transcribe reference audio for metadata/display when the
  /// caller did not provide a transcript.
  ///
  /// This is a Swift convenience only. The standard zero-shot path in the
  /// original PyTorch implementation, in
  /// `cosyvoice/cli/frontend.py:frontend_zero_shot`, uses explicit
  /// caller-supplied `prompt_text`; auto-transcribed text is not used here as a
  /// zero-shot prompt because misalignment between STT text and prompt audio can
  /// produce unstable outputs.
  public var autoTranscribe: Bool = true

  // MARK: - Private Properties

  @ObservationIgnored private var cosyVoice3TTS: CosyVoice3TTS?
  @ObservationIgnored private var s3Tokenizer: S3TokenizerV3?
  @ObservationIgnored private var whisperSTT: WhisperSTT?
  @ObservationIgnored private let playback = TTSPlaybackController(sampleRate: CosyVoice3Constants.sampleRate)
  @ObservationIgnored private var defaultSpeaker: CosyVoice3Speaker?
  @ObservationIgnored private var cachedSourceAudioURL: URL?
  @ObservationIgnored private let downloader: any Downloader
  @ObservationIgnored private let tokenizerLoader: any TokenizerLoader

  /// Repo ID for S3 tokenizer
  private static let s3TokenizerRepoId = "mlx-community/S3TokenizerV3"

  /// Standard frontend limit from the original PyTorch implementation in
  /// `cosyvoice/cli/frontend.py:_extract_speech_token`, which asserts that
  /// prompt audio must be 30 seconds or shorter before speech-token extraction.
  private nonisolated static let maxReferenceDurationSeconds = 30

  // MARK: - Initialization

  public init(
    from downloader: any Downloader = HubClient.default,
    using tokenizerLoader: any TokenizerLoader = TokenizersLoader()
  ) {
    self.downloader = downloader
    self.tokenizerLoader = tokenizerLoader
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
        from: downloader,
        using: tokenizerLoader,
        progressHandler: progressHandler ?? { _ in }
      )

      // Load S3TokenizerV3
      Log.model.info("Loading S3TokenizerV3...")
      s3Tokenizer = try await Self.loadS3Tokenizer(from: downloader)

      isLoaded = true
      Log.model.info("CosyVoice3 TTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load CosyVoice3 model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  /// Download and load S3TokenizerV3
  private static func loadS3Tokenizer(from downloader: any Downloader) async throws -> S3TokenizerV3 {
    let modelDirectory = try await downloader.download(
      id: s3TokenizerRepoId,
      revision: nil,
      matching: ["*.safetensors"],
      useLatest: false,
      progressHandler: { _ in }
    )
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
      modelSize: .base,
      quantization: .q4,
      from: downloader
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
  ///   - transcription: Optional explicit transcription of the reference audio.
  ///                    Providing this enables the standard zero-shot mode used
  ///                    by the original PyTorch implementation.
  ///                    This should match the exact audio segment after any
  ///                    user-side trimming, with no cut-off words at either end.
  ///                    If nil and `autoTranscribe` is true, Whisper may still
  ///                    produce a transcript for metadata, but Swift will not use
  ///                    that auto-transcribed text as a zero-shot prompt.
  ///                    Reference: `cosyvoice/cli/frontend.py:frontend_zero_shot`.
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

  nonisolated static func validateReferenceDuration(sampleCount: Int, sampleRate: Int) throws {
    let maxSamples = maxReferenceDurationSeconds * sampleRate
    guard sampleCount <= maxSamples else {
      throw TTSError.invalidArgument(
        "CosyVoice3 reference audio longer than \(maxReferenceDurationSeconds) seconds is not supported. Trim the reference clip to \(maxReferenceDurationSeconds) seconds or less."
      )
    }
  }

  nonisolated static func shouldUseZeroShot(
    hasExplicitTranscription: Bool,
    generationMode: GenerationMode,
    hasInstruction: Bool
  ) -> Bool {
    hasExplicitTranscription && generationMode != .crossLingual && !hasInstruction
  }

  nonisolated static func shouldUseZeroShot(
    speaker: CosyVoice3Speaker,
    generationMode: GenerationMode,
    hasInstruction: Bool
  ) -> Bool {
    shouldUseZeroShot(
      hasExplicitTranscription: speaker.hasExplicitTranscription,
      generationMode: generationMode,
      hasInstruction: hasInstruction
    )
  }

  // MARK: - Private Speaker Preparation

  private func validateZeroShotAvailability(
    for speaker: CosyVoice3Speaker,
    generationMode: GenerationMode,
    hasInstruction: Bool
  ) throws {
    // Match the standard zero-shot contract from the original PyTorch
    // implementation in
    // `cosyvoice/cli/frontend.py:frontend_zero_shot`: zero-shot uses explicit
    // caller-provided `prompt_text` paired with the reference audio. When the
    // transcript is not explicit, Swift falls back to cross-lingual instead of
    // inventing a zero-shot prompt from STT output.
    guard generationMode != .zeroShot || hasInstruction || speaker.hasExplicitTranscription else {
      throw TTSError.invalidArgument(
        "CosyVoice3 zero-shot mode requires an explicit reference transcription. Auto-transcribed reference text is not used as zero-shot prompt text."
      )
    }
  }

  private func prepareSpeakerFromSamples(
    _ samples: [Float],
    sampleRate: Int,
    transcription: String?,
    baseDescription: String,
    tts: CosyVoice3TTS,
    tokenizer: S3TokenizerV3
  ) async throws -> CosyVoice3Speaker {
    Log.tts.debug("Preparing speaker: \(baseDescription)")

    try Self.validateReferenceDuration(sampleCount: samples.count, sampleRate: sampleRate)

    let referenceDuration = Float(samples.count) / Float(sampleRate)
    Log.tts.debug("Reference audio duration: \(referenceDuration)s")

    let explicitTranscription = transcription?.trimmingCharacters(in: .whitespacesAndNewlines)
    let hasExplicitTranscription = !(explicitTranscription?.isEmpty ?? true)
    var finalTranscription = hasExplicitTranscription ? explicitTranscription : nil

    if finalTranscription == nil, autoTranscribe {
      Log.tts.debug("Auto-transcribing reference audio for metadata...")
      finalTranscription = try await transcribe(samples: samples, sampleRate: sampleRate)
    }

    finalTranscription = finalTranscription?.trimmingCharacters(in: .whitespacesAndNewlines)
    if finalTranscription?.isEmpty == true {
      finalTranscription = nil
    }

    let conditionalsWav = MLXArray(samples)

    nonisolated(unsafe) let tokenizerUnsafe = tokenizer

    let conditionals = try await tts.prepareConditionals(
      refWav: conditionalsWav,
      refSampleRate: sampleRate,
      refText: hasExplicitTranscription ? finalTranscription : nil,
      s3Tokenizer: { mel, melLen in tokenizerUnsafe(mel, melLen: melLen) }
    )

    let description: String = if hasExplicitTranscription {
      "\(baseDescription) (with transcription)"
    } else if finalTranscription != nil {
      "\(baseDescription) (auto-transcribed)"
    } else {
      baseDescription
    }

    Log.tts.debug("Speaker prepared: \(description)")

    return CosyVoice3Speaker(
      conditionals: conditionals,
      sampleRate: sampleRate,
      sampleCount: samples.count,
      description: description,
      transcription: finalTranscription,
      hasExplicitTranscription: hasExplicitTranscription
    )
  }

  // MARK: - Audio Loading Helpers

  private func resampleAudio(_ samples: [Float], fromRate: Int, toRate: Int) -> [Float] {
    AudioResampler.resample(MLXArray(samples), from: fromRate, to: toRate).asArray(Float.self)
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
  /// has an explicit reference transcription.
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
    if speaker == nil, defaultSpeaker == nil {
      defaultSpeaker = try await prepareDefaultSpeaker()
    }
    guard let speaker = speaker ?? defaultSpeaker else {
      throw TTSError.modelNotLoaded
    }

    // Generate based on mode or instruction
    let textChunks = await splitTextForInference(text, using: cosyVoice3TTS)
    let effectiveInstruction = instruction ?? (generationMode == .instruct ? instructText : nil)
    let hasInstruction = !(effectiveInstruction?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)

    if generationMode == .voiceConversion {
      throw TTSError.invalidArgument(
        "Voice conversion requires source audio. Use convertVoice(from:to:) instead."
      )
    }

    var combinedAudio: [Float] = []
    var totalProcessingTime: TimeInterval = 0

    try validateZeroShotAvailability(for: speaker, generationMode: generationMode, hasInstruction: hasInstruction)

    for textChunk in textChunks {
      let textTokens = await cosyVoice3TTS.encode(text: textChunk)
      let result: TTSGenerationResult = if let instruction = effectiveInstruction, hasInstruction {
        try await cosyVoice3TTS.generateInstruct(
          text: textChunk,
          textTokens: textTokens,
          instructText: instruction,
          conditionals: speaker.conditionals,
          sampling: sampling,
          nTimesteps: nTimesteps
        )
      } else if Self.shouldUseZeroShot(
        speaker: speaker,
        generationMode: generationMode,
        hasInstruction: hasInstruction
      ) {
        try await cosyVoice3TTS.generateZeroShot(
          text: textChunk,
          textTokens: textTokens,
          conditionals: speaker.conditionals,
          sampling: sampling,
          nTimesteps: nTimesteps
        )
      } else {
        try await cosyVoice3TTS.generateCrossLingual(
          text: textChunk,
          textTokens: textTokens,
          conditionals: speaker.conditionals,
          sampling: sampling,
          nTimesteps: nTimesteps
        )
      }

      combinedAudio.append(contentsOf: result.audio)
      totalProcessingTime += result.processingTime
    }

    Log.tts.timing("CosyVoice3 generation", duration: totalProcessingTime)
    lastGeneratedAudioURL = playback.saveAudioFile(
      samples: combinedAudio,
      sampleRate: CosyVoice3Constants.sampleRate
    )

    return .samples(
      data: combinedAudio,
      sampleRate: CosyVoice3Constants.sampleRate,
      processingTime: totalProcessingTime
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

    let targetSampleRate = CosyVoice3Constants.sampleRate
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

    if speaker == nil, defaultSpeaker == nil {
      defaultSpeaker = try await prepareDefaultSpeaker()
    }
    guard let speaker = speaker ?? defaultSpeaker else {
      throw TTSError.modelNotLoaded
    }

    let result = try await cosyVoice3TTS.generateVoiceConversionFromPrepared(
      conditionals: speaker.conditionals,
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
  ///
  /// Supports multiple streaming granularities:
  /// - `.token` (default): Low-latency streaming that yields audio as speech tokens are generated.
  ///   Each chunk contains audio for approximately `chunkSize` tokens.
  /// - `.sentence`: Higher-latency streaming that yields complete sentences.
  ///   More natural break points but slower time to first audio.
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Prepared speaker profile (if nil, uses default)
  ///   - instruction: Optional one-shot style instruction for instruct-mode synthesis
  ///   - granularity: Streaming granularity (if nil, uses `defaultStreamingGranularity`)
  ///   - chunkSize: Number of tokens per audio chunk (only used for `.token` granularity, default: 25).
  ///     Smaller values give faster time-to-first-audio but more processing overhead.
  ///     Each token produces ~0.04s of audio (at tokenMelRatio=2, 50Hz mel rate).
  /// - Returns: An async stream of audio chunks
  /// - Throws: `TTSError.unsupportedStreamingGranularity` if the requested granularity is not supported
  public func generateStreaming(
    _ text: String,
    speaker: CosyVoice3Speaker? = nil,
    instruction: String? = nil,
    granularity: StreamingGranularity? = nil,
    chunkSize: Int = 25
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    let effectiveGranularity = granularity ?? defaultStreamingGranularity

    // Validate granularity is supported
    guard supportedStreamingGranularities.contains(effectiveGranularity) else {
      return AsyncThrowingStream { continuation in
        continuation.finish(throwing: TTSError.unsupportedStreamingGranularity(
          requested: effectiveGranularity,
          supported: supportedStreamingGranularities
        ))
      }
    }

    // Validate chunkSize for token-level streaming
    guard chunkSize > 0 else {
      return AsyncThrowingStream { continuation in
        continuation.finish(throwing: TTSError.invalidArgument("chunkSize must be positive"))
      }
    }

    switch effectiveGranularity {
      case .token:
        return generateStreamingTokenLevel(text, speaker: speaker, instruction: instruction, chunkSize: chunkSize)
      case .sentence:
        return generateStreamingSentenceLevel(text, speaker: speaker, instruction: instruction)
      case .frame:
        // Unreachable: guard above rejects unsupported granularities
        return generateStreamingSentenceLevel(text, speaker: speaker, instruction: instruction)
    }
  }

  /// Play audio with streaming (plays chunks as they arrive)
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Prepared speaker profile (if nil, uses default)
  ///   - instruction: Optional one-shot style instruction for instruct-mode synthesis
  ///   - granularity: Streaming granularity (if nil, uses `defaultStreamingGranularity`)
  ///   - chunkSize: Number of tokens per audio chunk (only used for `.token` granularity, default: 25)
  /// - Returns: The complete audio result after playback
  /// - Throws: `TTSError.unsupportedStreamingGranularity` if the requested granularity is not supported
  @discardableResult
  public func sayStreaming(
    _ text: String,
    speaker: CosyVoice3Speaker? = nil,
    instruction: String? = nil,
    granularity: StreamingGranularity? = nil,
    chunkSize: Int = 25
  ) async throws -> AudioResult {
    let (samples, processingTime) = try await playback.playStream(
      generateStreaming(
        text,
        speaker: speaker,
        instruction: instruction,
        granularity: granularity,
        chunkSize: chunkSize
      ),
      setPlaying: { self.isPlaying = $0 }
    )

    lastGeneratedAudioURL = playback.saveAudioFile(samples: samples, sampleRate: CosyVoice3Constants.sampleRate)

    return .samples(
      data: samples,
      sampleRate: CosyVoice3Constants.sampleRate,
      processingTime: processingTime
    )
  }

  // MARK: - Private Streaming Implementations

  /// Token-level streaming implementation
  private func generateStreamingTokenLevel(
    _ text: String,
    speaker: CosyVoice3Speaker?,
    instruction: String?,
    chunkSize: Int
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Text cannot be empty")) }
    }

    let sampleRate = CosyVoice3Constants.sampleRate
    let currentSampling = sampling
    let currentNTimesteps = nTimesteps
    let currentGenerationMode = generationMode
    let currentInstructText = instructText
    let currentInstruction = instruction

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

      // Prepare speaker if needed
      if speaker == nil, defaultSpeaker == nil {
        defaultSpeaker = try await prepareDefaultSpeaker()
      }
      guard let speaker = speaker ?? defaultSpeaker else {
        throw TTSError.modelNotLoaded
      }

      let textChunks = await splitTextForInference(trimmedText, using: cosyVoice3TTS)
      let effectiveInstruction = currentInstruction ?? (currentGenerationMode == .instruct ? currentInstructText : "")
      let hasInstruction = !effectiveInstruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      let startTime = Date()
      return AsyncThrowingStream { continuation in
        let task = Task {
          do {
            for textChunk in textChunks {
              guard !Task.isCancelled else { break }

              let textTokens = await cosyVoice3TTS.encode(text: textChunk)
              try self.validateZeroShotAvailability(
                for: speaker,
                generationMode: currentGenerationMode,
                hasInstruction: hasInstruction
              )
              let useZeroShot = Self.shouldUseZeroShot(
                speaker: speaker,
                generationMode: currentGenerationMode,
                hasInstruction: hasInstruction
              )

              let audioStream: AsyncThrowingStream<[Float], Error> = if hasInstruction {
                await cosyVoice3TTS.generateInstructStreaming(
                  textTokens: textTokens,
                  instructText: effectiveInstruction,
                  conditionals: speaker.conditionals,
                  sampling: currentSampling,
                  nTimesteps: currentNTimesteps,
                  chunkSize: chunkSize
                )
              } else if useZeroShot {
                await cosyVoice3TTS.generateZeroShotStreaming(
                  textTokens: textTokens,
                  conditionals: speaker.conditionals,
                  sampling: currentSampling,
                  nTimesteps: currentNTimesteps,
                  chunkSize: chunkSize
                )
              } else {
                await cosyVoice3TTS.generateCrossLingualStreaming(
                  text: textChunk,
                  conditionals: speaker.conditionals,
                  sampling: currentSampling,
                  nTimesteps: currentNTimesteps,
                  chunkSize: chunkSize
                )
              }

              for try await samples in audioStream {
                continuation.yield(AudioChunk(
                  samples: samples,
                  sampleRate: sampleRate,
                  processingTime: Date().timeIntervalSince(startTime)
                ))
              }
            }
            continuation.finish()
          } catch is CancellationError {
            continuation.finish()
          } catch {
            continuation.finish(throwing: error)
          }
        }
        continuation.onTermination = { _ in
          task.cancel()
        }
      }
    }
  }

  /// Sentence-level streaming implementation
  private func generateStreamingSentenceLevel(
    _ text: String,
    speaker: CosyVoice3Speaker?,
    instruction: String?
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Text cannot be empty")) }
    }

    let sampleRate = CosyVoice3Constants.sampleRate
    let currentSampling = sampling
    let currentNTimesteps = nTimesteps
    let currentGenerationMode = generationMode
    let currentInstructText = instructText
    let currentInstruction = instruction

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

      if speaker == nil, defaultSpeaker == nil {
        defaultSpeaker = try await prepareDefaultSpeaker()
      }
      guard let speaker = speaker ?? defaultSpeaker else {
        throw TTSError.modelNotLoaded
      }

      let textChunks = await splitTextForInference(trimmedText, using: cosyVoice3TTS)

      let startTime = Date()
      return AsyncThrowingStream { continuation in
        let task = Task {
          do {
            for textChunk in textChunks {
              guard !Task.isCancelled else { break }

              let textTokens = await cosyVoice3TTS.encode(text: textChunk)
              let effectiveInstruction = currentInstruction ?? (currentGenerationMode == .instruct ? currentInstructText : "")
              let hasInstruction = !effectiveInstruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
              try self.validateZeroShotAvailability(
                for: speaker,
                generationMode: currentGenerationMode,
                hasInstruction: hasInstruction
              )
              let useZeroShot = Self.shouldUseZeroShot(
                speaker: speaker,
                generationMode: currentGenerationMode,
                hasInstruction: hasInstruction
              )

              let result: TTSGenerationResult = if hasInstruction {
                try await cosyVoice3TTS.generateInstruct(
                  text: textChunk,
                  textTokens: textTokens,
                  instructText: effectiveInstruction,
                  conditionals: speaker.conditionals,
                  sampling: currentSampling,
                  nTimesteps: currentNTimesteps
                )
              } else if useZeroShot {
                try await cosyVoice3TTS.generateZeroShot(
                  text: textChunk,
                  textTokens: textTokens,
                  conditionals: speaker.conditionals,
                  sampling: currentSampling,
                  nTimesteps: currentNTimesteps
                )
              } else {
                try await cosyVoice3TTS.generateCrossLingual(
                  text: textChunk,
                  textTokens: textTokens,
                  conditionals: speaker.conditionals,
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
        continuation.onTermination = { _ in
          task.cancel()
        }
      }
    }
  }

  private func splitTextForInference(_ text: String, using tts: CosyVoice3TTS) async -> [String] {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedText.isEmpty else { return [] }

    // Match the original PyTorch frontend behavior: bypass splitting when inline control tags are present.
    if trimmedText.contains("<|"), trimmedText.contains("|>") {
      return [trimmedText]
    }

    let isChinese = Self.containsChinese(trimmedText)
    let punctuation: Set<Character> = if isChinese {
      ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
    } else {
      [".", "?", "!", ";", ":"]
    }

    var normalizedText = trimmedText
    if let lastChar = normalizedText.last, !punctuation.contains(lastChar) {
      normalizedText.append(isChinese ? "。" : ".")
    }

    var utterances: [String] = []
    var startIndex = normalizedText.startIndex
    var index = normalizedText.startIndex

    while index < normalizedText.endIndex {
      let character = normalizedText[index]
      if punctuation.contains(character) {
        var endIndex = normalizedText.index(after: index)
        if endIndex < normalizedText.endIndex, ["\"", "”"].contains(normalizedText[endIndex]) {
          endIndex = normalizedText.index(after: endIndex)
        }

        let utterance = String(normalizedText[startIndex ..< endIndex])
          .trimmingCharacters(in: .whitespacesAndNewlines)
        if !utterance.isEmpty {
          utterances.append(utterance)
        }
        startIndex = endIndex
      }
      index = normalizedText.index(after: index)
    }

    func chunkLength(_ value: String) async -> Int {
      if value.isEmpty { return 0 }
      if isChinese { return value.count }
      return await tts.encode(text: value).count
    }

    var finalChunks: [String] = []
    var currentChunk = ""

    for utterance in utterances {
      let combined = currentChunk + utterance
      if await chunkLength(combined) > 80, await chunkLength(currentChunk) > 60 {
        if !Self.isOnlyPunctuation(currentChunk) {
          finalChunks.append(currentChunk)
        }
        currentChunk = ""
      }
      currentChunk += utterance
    }

    if !currentChunk.isEmpty {
      if await chunkLength(currentChunk) < 20, !finalChunks.isEmpty {
        finalChunks[finalChunks.count - 1] += currentChunk
      } else if !Self.isOnlyPunctuation(currentChunk) {
        finalChunks.append(currentChunk)
      }
    }

    return finalChunks.isEmpty ? [trimmedText] : finalChunks
  }

  private static func containsChinese(_ text: String) -> Bool {
    text.unicodeScalars.contains { scalar in
      (0x4E00 ... 0x9FFF).contains(scalar.value)
    }
  }

  private static func isOnlyPunctuation(_ text: String) -> Bool {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return true }
    return trimmed.unicodeScalars.allSatisfy {
      CharacterSet.punctuationCharacters.contains($0) || CharacterSet.symbols.contains($0)
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
}
