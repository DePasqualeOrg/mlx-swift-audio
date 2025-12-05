import AVFoundation
import Foundation
import MLX

// MARK: - Reference Audio

/// Prepared reference audio for Chatterbox TTS
///
/// Create using `ChatterboxEngine.prepareReferenceAudio(from:)` methods.
/// Can be reused across multiple `say()` or `generate()` calls for efficient multi-speaker scenarios.
///
/// ```swift
/// let speakerA = try await engine.prepareReferenceAudio(from: urlA)
/// let speakerB = try await engine.prepareReferenceAudio(from: urlB)
///
/// try await engine.say("Hello from A", referenceAudio: speakerA)
/// try await engine.say("Hello from B", referenceAudio: speakerB)  // instant switch
/// ```
public struct ChatterboxReferenceAudio: Sendable {
  /// The pre-computed conditionals for generation
  let conditionals: ChatterboxConditionals

  /// Sample rate of the original reference audio
  public let sampleRate: Int

  /// Duration of the reference audio in seconds
  public let duration: TimeInterval

  /// Description for display purposes
  public let description: String

  init(conditionals: ChatterboxConditionals, sampleRate: Int, sampleCount: Int, description: String) {
    self.conditionals = conditionals
    self.sampleRate = sampleRate
    duration = Double(sampleCount) / Double(sampleRate)
    self.description = description
  }
}

/// Default reference audio URL - LibriVox public domain reading
/// "The Dead Boche" by Robert Graves, ~38 seconds (public domain)
public let defaultReferenceAudioURL = URL(
  string:
  "https://archive.org/download/short_poetry_001_librivox/dead_boche_graves_sm.mp3",
)!

/// Chatterbox TTS engine - TTS with reference audio
///
/// Supports generating speech using reference audio clips.
@Observable
@MainActor
public final class ChatterboxEngine: TTSEngine {
  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .chatterbox
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - Chatterbox-Specific Properties

  /// Temperature for sampling (higher = more variation)
  public var temperature: Float = 0.8

  /// Top-p (nucleus) sampling threshold
  public var topP: Float = 1.0

  /// Minimum probability threshold for sampling
  public var minP: Float = 0.05

  /// Repetition penalty
  public var repetitionPenalty: Float = 1.2

  /// Classifier-free guidance weight
  public var cfgWeight: Float = 0.5

  /// Emotion exaggeration factor (0-1)
  public var exaggeration: Float = 0.1

  /// Maximum number of tokens to generate
  public var maxNewTokens: Int = 1000

  // MARK: - Private Properties

  @ObservationIgnored private var chatterboxTTS: ChatterboxTTS?
  @ObservationIgnored private let audioPlayer = AudioSamplePlayer(sampleRate: TTSProvider.chatterbox.sampleRate)
  @ObservationIgnored private var generationTask: Task<Void, Never>?
  @ObservationIgnored private var defaultReferenceAudio: ChatterboxReferenceAudio?

  // MARK: - Initialization

  public init() {
    Log.tts.debug("ChatterboxEngine initialized")
  }

  deinit {
    generationTask?.cancel()
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.tts.debug("ChatterboxEngine already loaded")
      return
    }

    Log.model.info("Loading Chatterbox TTS model...")

    do {
      // Load TTS actor from Hub with weights
      chatterboxTTS = try await ChatterboxTTS.load(
        progressHandler: progressHandler ?? { _ in },
      )

      isLoaded = true
      Log.model.info("Chatterbox TTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load Chatterbox model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  public func stop() async {
    generationTask?.cancel()
    generationTask = nil
    isGenerating = false

    await audioPlayer.stop()
    isPlaying = false

    Log.tts.debug("ChatterboxEngine stopped")
  }

  public func unload() async {
    await stop()

    // Clear model but preserve prepared reference audio (expensive to recompute)
    chatterboxTTS = nil
    isLoaded = false

    Log.tts.debug("ChatterboxEngine unloaded (reference audio preserved)")
  }

  public func cleanup() async throws {
    await unload()

    // Also clear prepared reference audio (expensive to recompute, but full cleanup)
    defaultReferenceAudio = nil
  }

  // MARK: - Playback

  public func play(_ audio: AudioResult) async {
    guard case let .samples(samples, _, _) = audio else {
      Log.audio.warning("Cannot play AudioResult.file - use AudioFilePlayer instead")
      return
    }

    isPlaying = true
    await audioPlayer.play(samples: samples)
    isPlaying = false
  }

  // MARK: - Reference Audio Preparation

  /// Prepare reference audio from a URL (local file or remote)
  ///
  /// This performs the expensive conditioning computation once. The returned
  /// `ChatterboxReferenceAudio` can be reused across multiple `say()` or `generate()` calls.
  ///
  /// - Parameters:
  ///   - url: URL to audio file (local file path or remote URL)
  ///   - exaggeration: Emotion exaggeration factor (0-1, default: 0.5)
  /// - Returns: Prepared reference audio ready for generation
  public func prepareReferenceAudio(
    from url: URL,
    exaggeration: Float = 0.5,
  ) async throws -> ChatterboxReferenceAudio {
    if !isLoaded {
      try await load()
    }

    guard let chatterboxTTS else {
      throw TTSError.modelNotLoaded
    }

    let (samples, sampleRate) = try await loadAudioSamples(from: url)
    let description = url.lastPathComponent

    return await prepareReferenceAudioFromSamples(
      samples,
      sampleRate: sampleRate,
      exaggeration: exaggeration,
      description: description,
      tts: chatterboxTTS,
    )
  }

  /// Prepare reference audio from raw samples
  ///
  /// - Parameters:
  ///   - samples: Audio samples as Float array
  ///   - sampleRate: Sample rate of the audio
  ///   - exaggeration: Emotion exaggeration factor (0-1, default: 0.5)
  /// - Returns: Prepared reference audio ready for generation
  public func prepareReferenceAudio(
    fromSamples samples: [Float],
    sampleRate: Int,
    exaggeration: Float = 0.5,
  ) async throws -> ChatterboxReferenceAudio {
    if !isLoaded {
      try await load()
    }

    guard let chatterboxTTS else {
      throw TTSError.modelNotLoaded
    }

    let duration = Double(samples.count) / Double(sampleRate)
    let description = String(format: "Custom audio (%.1fs)", duration)

    return await prepareReferenceAudioFromSamples(
      samples,
      sampleRate: sampleRate,
      exaggeration: exaggeration,
      description: description,
      tts: chatterboxTTS,
    )
  }

  /// Prepare the default reference audio (LibriVox public domain sample)
  ///
  /// - Parameter exaggeration: Emotion exaggeration factor (0-1, default: 0.5)
  /// - Returns: Prepared reference audio ready for generation
  public func prepareDefaultReferenceAudio(
    exaggeration: Float = 0.5,
  ) async throws -> ChatterboxReferenceAudio {
    try await prepareReferenceAudio(from: defaultReferenceAudioURL, exaggeration: exaggeration)
  }

  // MARK: - Private Audio Loading

  private func prepareReferenceAudioFromSamples(
    _ samples: [Float],
    sampleRate: Int,
    exaggeration: Float,
    description: String,
    tts: ChatterboxTTS,
  ) async -> ChatterboxReferenceAudio {
    Log.tts.debug("Preparing reference audio: \(description)")

    let refWav = MLXArray(samples)
    let conditionals = await tts.prepareConditionals(
      refWav: refWav,
      refSr: sampleRate,
      exaggeration: exaggeration,
    )

    Log.tts.debug("Reference audio prepared: \(description)")

    return ChatterboxReferenceAudio(
      conditionals: conditionals,
      sampleRate: sampleRate,
      sampleCount: samples.count,
      description: description,
    )
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

  /// Generate audio from text using reference audio
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - referenceAudio: Prepared reference audio (if nil, uses default sample)
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    referenceAudio: ChatterboxReferenceAudio? = nil,
  ) async throws -> AudioResult {
    if !isLoaded {
      try await load()
    }

    guard let chatterboxTTS else {
      throw TTSError.modelNotLoaded
    }

    // Use provided reference audio, or prepare default
    let ref: ChatterboxReferenceAudio
    if let referenceAudio {
      ref = referenceAudio
    } else {
      // Lazy-load default reference audio
      if defaultReferenceAudio == nil {
        defaultReferenceAudio = try await prepareDefaultReferenceAudio(exaggeration: exaggeration)
      }
      ref = defaultReferenceAudio!
    }

    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedText.isEmpty else {
      throw TTSError.invalidArgument("Text cannot be empty")
    }

    generationTask?.cancel()
    isGenerating = true
    generationTime = 0

    do {
      // Generate audio using pre-computed conditionals
      let result = await chatterboxTTS.generate(
        text: trimmedText,
        conditionals: ref.conditionals,
        exaggeration: exaggeration,
        cfgWeight: cfgWeight,
        temperature: temperature,
        repetitionPenalty: repetitionPenalty,
        minP: minP,
        topP: topP,
        maxNewTokens: maxNewTokens,
      )

      generationTime = result.processingTime
      isGenerating = false

      Log.tts.timing("Chatterbox generation", duration: result.processingTime)
      Log.tts.rtf("Chatterbox", rtf: result.realTimeFactor)

      do {
        let fileURL = try AudioFileWriter.save(
          samples: result.audio,
          sampleRate: result.sampleRate,
          filename: TTSConstants.outputFilename,
        )
        lastGeneratedAudioURL = fileURL
      } catch {
        Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
      }

      return .samples(
        data: result.audio,
        sampleRate: result.sampleRate,
        processingTime: result.processingTime,
      )

    } catch {
      isGenerating = false
      Log.tts.error("Chatterbox generation failed: \(error.localizedDescription)")
      throw TTSError.generationFailed(underlying: error)
    }
  }

  /// Generate and immediately play audio
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - referenceAudio: Prepared reference audio (if nil, uses default sample)
  public func say(
    _ text: String,
    referenceAudio: ChatterboxReferenceAudio? = nil,
  ) async throws {
    let audio = try await generate(text, referenceAudio: referenceAudio)
    await play(audio)
  }
}
