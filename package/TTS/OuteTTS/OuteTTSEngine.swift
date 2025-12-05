import Foundation

/// OuteTTS engine - TTS with custom speaker profiles
///
/// Supports custom speaker profiles with reference audio.
@Observable
@MainActor
public final class OuteTTSEngine: TTSEngine {
  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .outetts
  public let streamingGranularity: StreamingGranularity = .sentence
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - OuteTTS-Specific Properties

  /// Temperature for sampling (higher = more variation)
  public var temperature: Float = 0.4

  /// Top-p (nucleus) sampling threshold
  public var topP: Float = 0.9

  // MARK: - Private Properties

  @ObservationIgnored private var outeTTS: OuteTTS?
  @ObservationIgnored private let audioPlayer = AudioSamplePlayer(sampleRate: TTSProvider.outetts.sampleRate)
  @ObservationIgnored private var generationTask: Task<Void, Never>?

  // MARK: - Initialization

  public init() {
    Log.tts.debug("OuteTTSEngine initialized")
  }

  deinit {
    generationTask?.cancel()
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.tts.debug("OuteTTSEngine already loaded")
      return
    }

    Log.model.info("Loading OuteTTS model...")

    do {
      outeTTS = try await OuteTTS.load(
        progressHandler: progressHandler ?? { _ in },
      )

      isLoaded = true
      Log.model.info("OuteTTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load OuteTTS model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  public func stop() async {
    generationTask?.cancel()
    generationTask = nil
    isGenerating = false

    await audioPlayer.stop()
    isPlaying = false

    Log.tts.debug("OuteTTSEngine stopped")
  }

  public func unload() async {
    await stop()

    outeTTS = nil
    isLoaded = false

    Log.tts.debug("OuteTTSEngine unloaded")
  }

  public func cleanup() async throws {
    await unload()
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

  // MARK: - Generation

  /// Generate audio from text
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Optional speaker profile (nil uses default voice)
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    speaker: OuteTTSSpeakerProfile? = nil,
  ) async throws -> AudioResult {
    if !isLoaded {
      try await load()
    }

    guard let outeTTS else {
      throw TTSError.modelNotLoaded
    }

    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedText.isEmpty else {
      throw TTSError.invalidArgument("Text cannot be empty")
    }

    generationTask?.cancel()
    isGenerating = true
    generationTime = 0

    do {
      let result = try await outeTTS.generate(
        text: trimmedText,
        speaker: speaker,
        temperature: temperature,
        topP: topP,
      )

      generationTime = result.processingTime
      isGenerating = false

      Log.tts.timing("OuteTTS generation", duration: result.processingTime)
      Log.tts.rtf("OuteTTS", rtf: result.realTimeFactor)

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
      Log.tts.error("OuteTTS generation failed: \(error.localizedDescription)")
      throw TTSError.generationFailed(underlying: error)
    }
  }

  /// Generate and immediately play audio
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Optional speaker profile (nil uses default voice)
  public func say(
    _ text: String,
    speaker: OuteTTSSpeakerProfile? = nil,
  ) async throws {
    let audio = try await generate(text, speaker: speaker)
    await play(audio)
  }

  // MARK: - Streaming

  /// Generate audio as a stream of chunks (one per sentence)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Optional speaker profile (nil uses default voice)
  /// - Returns: An async stream of audio chunks
  public func generateStreaming(
    _ text: String,
    speaker: OuteTTSSpeakerProfile? = nil,
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    sentenceStreamingGenerate(text: text, sampleRate: provider.sampleRate) { [self] sentence in
      guard let outeTTS else {
        throw TTSError.modelNotLoaded
      }
      let result = try await outeTTS.generate(
        text: sentence,
        speaker: speaker,
        temperature: temperature,
        topP: topP,
      )
      return result.audio
    }
  }

  /// Play audio with streaming (plays as chunks arrive)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Optional speaker profile (nil uses default voice)
  @discardableResult
  public func sayStreaming(
    _ text: String,
    speaker: OuteTTSSpeakerProfile? = nil,
  ) async throws -> AudioResult {
    if !isLoaded {
      try await load()
    }

    isPlaying = true
    isGenerating = true
    var allSamples: [Float] = []
    var totalProcessingTime: TimeInterval = 0

    do {
      for try await chunk in generateStreaming(text, speaker: speaker) {
        allSamples.append(contentsOf: chunk.samples)
        totalProcessingTime = chunk.processingTime
        audioPlayer.enqueue(samples: chunk.samples, prebufferSeconds: 0)
      }

      // Streaming complete - audio continues playing from queue
      isPlaying = false
      isGenerating = false

      if !allSamples.isEmpty {
        do {
          let fileURL = try AudioFileWriter.save(
            samples: allSamples,
            sampleRate: provider.sampleRate,
            filename: TTSConstants.outputFilename,
          )
          lastGeneratedAudioURL = fileURL
        } catch {
          Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
        }
      }

      return .samples(
        data: allSamples,
        sampleRate: provider.sampleRate,
        processingTime: totalProcessingTime,
      )
    } catch {
      isPlaying = false
      isGenerating = false
      throw error
    }
  }
}
