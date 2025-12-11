import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN

// MARK: - Speaker

/// Prepared speaker profile for CosyVoice2 TTS
///
/// Create using `CosyVoice2Engine.prepareSpeaker(from:)` methods.
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
public struct CosyVoice2Speaker: Sendable {
  /// The pre-computed conditionals for generation
  let conditionals: CosyVoice2Conditionals

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
    conditionals: CosyVoice2Conditionals,
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
public let cosyVoice2DefaultReferenceAudioURL = URL(
  string:
  "https://archive.org/download/short_poetry_001_librivox/dead_boche_graves_sm.mp3"
)!

// MARK: - CosyVoice2 Engine

/// CosyVoice2 TTS engine - Voice matching with zero-shot and cross-lingual modes
///
/// CosyVoice2 is a state-of-the-art TTS model that supports:
/// - **Zero-shot voice matching**: Match any voice with just a few seconds of reference audio
/// - **Cross-lingual synthesis**: Generate speech in different languages while preserving voice characteristics
/// - **High-quality output**: 24kHz audio with natural prosody
@Observable
@MainActor
public final class CosyVoice2Engine: TTSEngine {
  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .cosyVoice2
  public let streamingGranularity: StreamingGranularity = .sentence
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - CosyVoice2-Specific Properties

  /// Generation mode for CosyVoice2
  ///
  /// CosyVoice2 supports 4 inference modes:
  /// - **Voice Conversion**: Converts source audio to target speaker voice (no text generation)
  /// - **Zero-shot**: Uses reference audio + transcription for semantic alignment
  /// - **Cross-lingual**: Uses reference audio only (works across languages)
  /// - **Instruct**: Uses style instructions to control speech generation
  public enum GenerationMode: String, CaseIterable, Sendable {
    /// Cross-lingual: Uses speaker embedding only (no reference text needed)
    /// Good for synthesizing text in a different language than the reference.
    case crossLingual = "Cross-lingual"

    /// Zero-shot: Uses reference text for better voice alignment
    /// Requires transcription of the reference audio for semantic alignment.
    case zeroShot = "Zero-shot"

    /// Instruct: Uses style instructions to control speech generation
    /// e.g., "Speak slowly and calmly", "Read with excitement"
    case instruct = "Instruct"

    /// Voice Conversion: Converts source audio to target speaker voice
    /// Requires source audio to be converted.
    case voiceConversion = "Voice Conversion"

    /// Short description of the mode
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

  /// Style instruction for instruct mode (e.g., "Speak slowly and calmly")
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

  @ObservationIgnored private var cosyVoice2TTS: CosyVoice2TTS?
  @ObservationIgnored private var s3Tokenizer: S3TokenizerV2?
  @ObservationIgnored private var whisperSTT: WhisperSTT?
  @ObservationIgnored private let playback = TTSPlaybackController(sampleRate: 24000)
  @ObservationIgnored private var defaultSpeaker: CosyVoice2Speaker?

  /// HuggingFace repo ID for S3 tokenizer
  private static let s3TokenizerRepoId = "mlx-community/S3TokenizerV2"

  // MARK: - Initialization

  public init() {
    Log.tts.debug("CosyVoice2Engine initialized")
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.tts.debug("CosyVoice2Engine already loaded")
      return
    }

    Log.model.info("Loading CosyVoice2 TTS model...")

    do {
      // Load CosyVoice2 model
      cosyVoice2TTS = try await CosyVoice2TTS.load(
        progressHandler: progressHandler ?? { _ in }
      )

      // Load S3TokenizerV2
      Log.model.info("Loading S3TokenizerV2...")
      s3Tokenizer = try await Self.loadS3Tokenizer()

      isLoaded = true
      Log.model.info("CosyVoice2 TTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load CosyVoice2 model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  /// Load S3TokenizerV2 from HuggingFace
  private static func loadS3Tokenizer() async throws -> S3TokenizerV2 {
    let modelDirectory = try await HubConfiguration.shared.snapshot(from: s3TokenizerRepoId)
    let weightURL = modelDirectory.appendingPathComponent("model.safetensors")
    let weights = try MLX.loadArrays(url: weightURL)

    let tokenizer = S3TokenizerV2()

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
      quantization: .q4
    )
    whisperSTT = whisper
    Log.model.info("Whisper loaded successfully")
    return whisper
  }

  /// Transcribe audio using Whisper
  ///
  /// - Parameters:
  ///   - samples: Audio samples at any sample rate
  ///   - sampleRate: Sample rate of the audio
  /// - Returns: Transcribed text
  public func transcribe(samples: [Float], sampleRate: Int) async throws -> String {
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
      language: nil, // Auto-detect
      task: .transcribe,
      temperature: 0.0,
      timestamps: .segment
    )

    Log.tts.debug("Transcribed reference audio: \(result.text)")
    return result.text
  }

  public func stop() async {
    await playback.stop(
      setGenerating: { self.isGenerating = $0 },
      setPlaying: { self.isPlaying = $0 }
    )
    Log.tts.debug("CosyVoice2Engine stopped")
  }

  public func unload() async {
    await stop()

    // Clear model but preserve prepared reference audio (expensive to recompute)
    cosyVoice2TTS = nil
    s3Tokenizer = nil
    isLoaded = false

    Log.tts.debug("CosyVoice2Engine unloaded (reference audio preserved)")
  }

  public func cleanup() async throws {
    await unload()

    // Also clear prepared speaker
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
  ) async throws -> CosyVoice2Speaker {
    if !isLoaded {
      try await load()
    }

    guard let cosyVoice2TTS, let s3Tokenizer else {
      throw TTSError.modelNotLoaded
    }

    let (samples, sampleRate) = try await loadAudioSamples(from: url)

    // Description will be updated with transcription status in prepareSpeakerFromSamples
    let baseDescription = url.lastPathComponent

    return try await prepareSpeakerFromSamples(
      samples,
      sampleRate: sampleRate,
      transcription: transcription,
      baseDescription: baseDescription,
      tts: cosyVoice2TTS,
      tokenizer: s3Tokenizer
    )
  }

  /// Prepare a speaker profile from raw samples
  ///
  /// - Parameters:
  ///   - samples: Audio samples as Float array
  ///   - sampleRate: Sample rate of the audio
  ///   - transcription: Optional transcription of the reference audio (enables zero-shot mode).
  ///                    If nil and `autoTranscribe` is true, Whisper will be used to transcribe.
  /// - Returns: Prepared speaker ready for generation
  public func prepareSpeaker(
    from samples: [Float],
    sampleRate: Int,
    transcription: String? = nil
  ) async throws -> CosyVoice2Speaker {
    if !isLoaded {
      try await load()
    }

    guard let cosyVoice2TTS, let s3Tokenizer else {
      throw TTSError.modelNotLoaded
    }

    let duration = Double(samples.count) / Double(sampleRate)
    let baseDescription = String(format: "Custom audio (%.1f sec.)", duration)

    return try await prepareSpeakerFromSamples(
      samples,
      sampleRate: sampleRate,
      transcription: transcription,
      baseDescription: baseDescription,
      tts: cosyVoice2TTS,
      tokenizer: s3Tokenizer
    )
  }

  /// Prepare the default speaker (LibriVox public domain sample)
  ///
  /// - Parameter transcription: Optional transcription of the default audio
  /// - Returns: Prepared speaker ready for generation
  public func prepareDefaultSpeaker(
    transcription: String? = nil
  ) async throws -> CosyVoice2Speaker {
    try await prepareSpeaker(from: cosyVoice2DefaultReferenceAudioURL, transcription: transcription)
  }

  // MARK: - Private Audio Loading

  private func prepareSpeakerFromSamples(
    _ samples: [Float],
    sampleRate: Int,
    transcription: String?,
    baseDescription: String,
    tts: CosyVoice2TTS,
    tokenizer: S3TokenizerV2
  ) async throws -> CosyVoice2Speaker {
    Log.tts.debug("Preparing speaker: \(baseDescription)")

    // Resample to 24kHz if needed (CosyVoice2 expects 24kHz)
    let targetSampleRate = 24000
    var processedSamples: [Float] = if sampleRate != targetSampleRate {
      resampleAudio(samples, fromRate: sampleRate, toRate: targetSampleRate)
    } else {
      samples
    }

    // Truncate to max 30 seconds (matching prepareConditionals behavior)
    // This ensures the transcription matches the audio actually used for conditioning
    let maxSamples = 30 * targetSampleRate
    if processedSamples.count > maxSamples {
      processedSamples = Array(processedSamples.prefix(maxSamples))
      Log.tts.debug("Truncated reference audio to 30 seconds")
    }

    // Auto-transcribe if enabled and no transcription provided
    // IMPORTANT: Transcribe AFTER truncation so the transcription matches the clipped audio
    var finalTranscription = transcription
    if transcription == nil, autoTranscribe {
      Log.tts.debug("Auto-transcribing reference audio (after truncation)...")
      finalTranscription = try await transcribe(samples: processedSamples, sampleRate: targetSampleRate)
    }

    let refWav = MLXArray(processedSamples)

    // Create a sendable wrapper for the tokenizer
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

    return CosyVoice2Speaker(
      conditionals: conditionals,
      sampleRate: targetSampleRate,
      sampleCount: processedSamples.count,
      description: description,
      transcription: finalTranscription
    )
  }

  /// Simple audio resampling using linear interpolation
  private func resampleAudio(_ samples: [Float], fromRate: Int, toRate: Int) -> [Float] {
    if fromRate == toRate {
      return samples
    }

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

    // Save to temporary file and load
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

    // Convert to mono if stereo
    let samples: [Float]
    if format.channelCount == 1 {
      samples = Array(UnsafeBufferPointer(start: floatData[0], count: Int(buffer.frameLength)))
    } else {
      // Mix stereo to mono
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
  /// has a transcription. If the speaker has a transcription, zero-shot mode is used
  /// for better voice alignment. Otherwise, cross-lingual mode is used.
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Prepared speaker profile (if nil, uses default)
  ///   - instruction: Optional style instruction (e.g., "Speak slowly and calmly")
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    speaker: CosyVoice2Speaker? = nil,
    instruction: String? = nil
  ) async throws -> AudioResult {
    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw TTSError.invalidArgument("Text cannot be empty")
    }

    // Auto-load if needed
    if !isLoaded {
      try await load()
    }

    guard let cosyVoice2TTS else {
      throw TTSError.modelNotLoaded
    }

    isGenerating = true
    let startTime = CFAbsoluteTimeGetCurrent()

    defer {
      isGenerating = false
      generationTime = CFAbsoluteTimeGetCurrent() - startTime
    }

    // Prepare speaker if needed
    let spk: CosyVoice2Speaker
    if let speaker {
      spk = speaker
    } else {
      if defaultSpeaker == nil {
        defaultSpeaker = try await prepareDefaultSpeaker()
      }
      spk = defaultSpeaker!
    }

    // Tokenize input text
    let textTokens = await cosyVoice2TTS.encode(text: text)

    // Generate based on mode or instruction
    let result: TTSGenerationResult

    // Check for instruction parameter first
    let effectiveInstruction = instruction ?? (generationMode == .instruct ? instructText : nil)

    if let instruction = effectiveInstruction,
       !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    {
      // Instruct mode
      result = try await cosyVoice2TTS.generateInstruct(
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
      // Zero-shot mode (speaker has transcription)
      result = try await cosyVoice2TTS.generateZeroShot(
        text: text,
        textTokens: textTokens,
        conditionals: spk.conditionals,
        sampling: sampling,
        nTimesteps: nTimesteps
      )
    } else {
      // Cross-lingual mode (no transcription or explicitly set)
      result = try await cosyVoice2TTS.generateCrossLingual(
        text: text,
        textTokens: textTokens,
        conditionals: spk.conditionals,
        sampling: sampling,
        nTimesteps: nTimesteps
      )
    }

    Log.tts.timing("CosyVoice2 generation", duration: result.processingTime)
    lastGeneratedAudioURL = playback.saveAudioFile(samples: result.audio, sampleRate: result.sampleRate)

    return .samples(
      data: result.audio,
      sampleRate: result.sampleRate,
      processingTime: result.processingTime
    )
  }

  /// Generate and immediately play audio
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Prepared speaker profile (if nil, uses default)
  ///   - instruction: Optional style instruction (e.g., "Speak slowly and calmly")
  public func say(
    _ text: String,
    speaker: CosyVoice2Speaker? = nil,
    instruction: String? = nil
  ) async throws {
    let audio = try await generate(text, speaker: speaker, instruction: instruction)
    await play(audio)
  }

  // MARK: - Voice Conversion

  /// Convert source audio to target speaker's voice
  ///
  /// This is a one-step voice conversion that loads the source audio and converts it
  /// to sound like the target speaker.
  ///
  /// - Parameters:
  ///   - sourceURL: URL to source audio file (the audio to convert)
  ///   - speaker: Target speaker profile (if nil, uses default)
  /// - Returns: Generated audio result in the target speaker's voice
  public func convertVoice(
    from sourceURL: URL,
    to speaker: CosyVoice2Speaker? = nil
  ) async throws -> AudioResult {
    // Auto-load if needed
    if !isLoaded {
      try await load()
    }

    guard let cosyVoice2TTS, let s3Tokenizer else {
      throw TTSError.modelNotLoaded
    }

    isGenerating = true
    let startTime = CFAbsoluteTimeGetCurrent()

    defer {
      isGenerating = false
      generationTime = CFAbsoluteTimeGetCurrent() - startTime
    }

    // Load and prepare source audio
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
    await cosyVoice2TTS.prepareSourceAudioForVC(
      audio: sourceWavUnsafe,
      s3Tokenizer: { mel, melLen in tokenizerUnsafe(mel, melLen: melLen) }
    )

    // Prepare speaker if needed
    let spk: CosyVoice2Speaker
    if let speaker {
      spk = speaker
    } else {
      if defaultSpeaker == nil {
        defaultSpeaker = try await prepareDefaultSpeaker()
      }
      spk = defaultSpeaker!
    }

    let result = try await cosyVoice2TTS.generateVoiceConversionFromPrepared(
      conditionals: spk.conditionals,
      nTimesteps: nTimesteps
    )

    Log.tts.timing("CosyVoice2 voice conversion", duration: result.processingTime)
    lastGeneratedAudioURL = playback.saveAudioFile(samples: result.audio, sampleRate: result.sampleRate)

    return .samples(
      data: result.audio,
      sampleRate: result.sampleRate,
      processingTime: result.processingTime
    )
  }

  /// Prepare source audio for voice conversion (for UI-driven workflows)
  ///
  /// Call this to pre-load source audio, then use `generateVoiceConversion(speaker:)`.
  /// For simple one-step conversion, use `convertVoice(from:to:)` instead.
  ///
  /// - Parameter url: URL to source audio file
  public func prepareSourceAudio(from url: URL) async throws {
    if !isLoaded {
      try await load()
    }

    guard let cosyVoice2TTS, let s3Tokenizer else {
      throw TTSError.modelNotLoaded
    }

    let (samples, sampleRate) = try await loadAudioSamples(from: url)

    let targetSampleRate = 24000
    let resampledSamples: [Float] = if sampleRate != targetSampleRate {
      resampleAudio(samples, fromRate: sampleRate, toRate: targetSampleRate)
    } else {
      samples
    }

    let sourceWav = MLXArray(resampledSamples)

    nonisolated(unsafe) let tokenizerUnsafe = s3Tokenizer
    nonisolated(unsafe) let sourceWavUnsafe = sourceWav
    await cosyVoice2TTS.prepareSourceAudioForVC(
      audio: sourceWavUnsafe,
      s3Tokenizer: { mel, melLen in tokenizerUnsafe(mel, melLen: melLen) }
    )

    isSourceAudioLoaded = true
    let duration = Double(resampledSamples.count) / Double(targetSampleRate)
    sourceAudioDescription = String(format: "%@ (%.1fs)", url.lastPathComponent, duration)

    Log.tts.debug("Source audio prepared")
  }

  /// Clear source audio
  public func clearSourceAudio() async {
    await cosyVoice2TTS?.clearSourceAudio()
    isSourceAudioLoaded = false
    sourceAudioDescription = "No source audio"
  }

  /// Generate voice conversion from pre-loaded source audio
  ///
  /// Requires `prepareSourceAudio(from:)` to be called first.
  /// For simple one-step conversion, use `convertVoice(from:to:)` instead.
  ///
  /// - Parameter speaker: Target speaker profile (if nil, uses default)
  /// - Returns: Generated audio result
  public func generateVoiceConversion(
    speaker: CosyVoice2Speaker? = nil
  ) async throws -> AudioResult {
    guard isSourceAudioLoaded else {
      throw TTSError.invalidArgument("Voice conversion requires source audio. Call prepareSourceAudio(from:) first, or use convertVoice(from:to:).")
    }

    // Auto-load if needed
    if !isLoaded {
      try await load()
    }

    guard let cosyVoice2TTS else {
      throw TTSError.modelNotLoaded
    }

    isGenerating = true
    let startTime = CFAbsoluteTimeGetCurrent()

    defer {
      isGenerating = false
      generationTime = CFAbsoluteTimeGetCurrent() - startTime
    }

    // Prepare speaker if needed
    let spk: CosyVoice2Speaker
    if let speaker {
      spk = speaker
    } else {
      if defaultSpeaker == nil {
        defaultSpeaker = try await prepareDefaultSpeaker()
      }
      spk = defaultSpeaker!
    }

    let result = try await cosyVoice2TTS.generateVoiceConversionFromPrepared(
      conditionals: spk.conditionals,
      nTimesteps: nTimesteps
    )

    Log.tts.timing("CosyVoice2 voice conversion", duration: result.processingTime)
    lastGeneratedAudioURL = playback.saveAudioFile(samples: result.audio, sampleRate: result.sampleRate)

    return .samples(
      data: result.audio,
      sampleRate: result.sampleRate,
      processingTime: result.processingTime
    )
  }

  // MARK: - Streaming

  /// Generate audio as a stream of chunks
  ///
  /// Note: CosyVoice2 generates audio sentence-by-sentence for streaming.
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Prepared speaker profile (if nil, uses default)
  /// - Returns: An async stream of audio chunks
  public func generateStreaming(
    _ text: String,
    speaker: CosyVoice2Speaker? = nil
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Text cannot be empty")) }
    }

    let sampleRate = 24000

    // Capture current parameter values
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

      // Auto-load if needed
      if !isLoaded {
        try await load()
      }

      guard let cosyVoice2TTS else {
        throw TTSError.modelNotLoaded
      }

      // Prepare speaker if needed
      let spk: CosyVoice2Speaker
      if let speaker {
        spk = speaker
      } else {
        if defaultSpeaker == nil {
          defaultSpeaker = try await prepareDefaultSpeaker()
        }
        spk = defaultSpeaker!
      }

      // Split text into sentences for streaming
      let sentences = Self.splitIntoSentences(trimmedText)

      let startTime = Date()
      return AsyncThrowingStream { continuation in
        Task {
          do {
            for sentence in sentences {
              guard !Task.isCancelled else { break }

              let textTokens = await cosyVoice2TTS.encode(text: sentence)
              let useZeroShot = spk.hasTranscription && currentGenerationMode != .crossLingual

              let result: TTSGenerationResult = if useZeroShot {
                try await cosyVoice2TTS.generateZeroShot(
                  text: sentence,
                  textTokens: textTokens,
                  conditionals: spk.conditionals,
                  sampling: currentSampling,
                  nTimesteps: currentNTimesteps
                )
              } else {
                try await cosyVoice2TTS.generateCrossLingual(
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

  /// Split text into sentences for streaming
  private static func splitIntoSentences(_ text: String) -> [String] {
    // Simple sentence splitting on common punctuation
    let pattern = #"[.!?。！？]+\s*"#
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

    // Add remaining text if any
    if lastEnd < nsText.length {
      let remaining = nsText.substring(from: lastEnd).trimmingCharacters(in: .whitespaces)
      if !remaining.isEmpty {
        sentences.append(remaining)
      }
    }

    // If no sentences found, return the whole text
    if sentences.isEmpty, !text.isEmpty {
      sentences.append(text)
    }

    return sentences
  }

  /// Play audio with streaming (plays as chunks arrive)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Prepared speaker profile (if nil, uses default)
  @discardableResult
  public func sayStreaming(
    _ text: String,
    speaker: CosyVoice2Speaker? = nil
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
