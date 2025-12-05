//
//  MarvisEngine.swift
//  MLXAudio
//
//  Marvis TTS engine conforming to TTSEngine protocol.
//

import Foundation
import MLX

/// Actor wrapper for MarvisOrchestrator that provides thread-safe generation
public actor MarvisTTS {
  private var orchestrator: MarvisOrchestrator?

  var isInitialized: Bool { orchestrator != nil }

  func initialize(
    modelRepoId: String,
    progressHandler: @escaping @Sendable (Progress) -> Void,
  ) async throws {
    orchestrator = try await MarvisOrchestrator.fromPretrained(
      repoId: modelRepoId,
      progressHandler: progressHandler,
    )
  }

  func generate(
    text: String,
    voice: MarvisEngine.Voice,
    quality: MarvisEngine.QualityLevel,
  ) throws -> MarvisOrchestrator.GenerationResult {
    guard let orchestrator else { throw TTSError.modelNotLoaded }
    return try orchestrator.generateAudio(text: text, voice: voice, quality: quality)
  }

  func generateStreaming(
    text: String,
    voice: MarvisEngine.Voice,
    quality: MarvisEngine.QualityLevel,
    interval: Double,
  ) throws -> AsyncThrowingStream<MarvisOrchestrator.GenerationResult, Error> {
    guard let orchestrator else { throw TTSError.modelNotLoaded }

    return AsyncThrowingStream { continuation in
      do {
        try orchestrator.generateAudioStream(
          text: text,
          voice: voice,
          quality: quality,
          interval: interval,
        ) { result in
          continuation.yield(result)
        }
        continuation.finish()
      } catch {
        continuation.finish(throwing: error)
      }
    }
  }

  func cleanUp() throws {
    try orchestrator?.cleanUpMemory()
    orchestrator = nil
  }
}

/// Marvis TTS engine - advanced conversational TTS with streaming support
@Observable
@MainActor
public final class MarvisEngine: TTSEngine, StreamingTTSEngine {
  // MARK: - Types

  /// Model variants for Marvis TTS
  public enum ModelVariant: String, CaseIterable, Sendable {
    case model100m_v0_2_6bit = "Marvis-AI/marvis-tts-100m-v0.2-MLX-6bit"
    case model250m_v0_2_6bit = "Marvis-AI/marvis-tts-250m-v0.2-MLX-6bit"

    public static let `default`: ModelVariant = .model100m_v0_2_6bit

    public var displayName: String {
      switch self {
        case .model100m_v0_2_6bit:
          "100M v0.2 (6-bit)"
        case .model250m_v0_2_6bit:
          "250M v0.2 (6-bit)"
      }
    }

    public var repoId: String {
      rawValue
    }
  }

  /// Available voices for Marvis TTS
  public enum Voice: String, CaseIterable, Sendable {
    case conversationalA = "conversational_a"
    case conversationalB = "conversational_b"
  }

  /// Quality levels for audio generation
  public enum QualityLevel: String, CaseIterable, Sendable {
    case low
    case medium
    case high
    case maximum

    public var codebookCount: Int {
      switch self {
        case .low: 8
        case .medium: 16
        case .high: 24
        case .maximum: 32
      }
    }
  }

  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .marvis
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - Marvis-Specific Properties

  /// Model variant to use
  public var modelVariant: ModelVariant = .default

  /// Quality level (affects codebook count)
  public var qualityLevel: QualityLevel = .maximum

  /// Streaming interval in seconds
  public var streamingInterval: Double = TTSConstants.Timing.defaultStreamingInterval

  // MARK: - Private Properties

  @ObservationIgnored private let marvisTTS = MarvisTTS()
  @ObservationIgnored private let audioPlayer = AudioSamplePlayer(sampleRate: TTSProvider.marvis.sampleRate)
  @ObservationIgnored private var generationTask: Task<Void, Never>?
  @ObservationIgnored private var lastModelVariant: ModelVariant?

  // MARK: - Initialization

  public init() {
    Log.tts.debug("MarvisEngine initialized")
  }

  deinit {
    generationTask?.cancel()
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    let isInitialized = await marvisTTS.isInitialized

    // Check if we need to reload
    if isInitialized, lastModelVariant == modelVariant {
      Log.tts.debug("MarvisEngine already loaded with same configuration")
      return
    }

    // Clean up existing model if configuration changed
    if isInitialized {
      Log.model.info("Configuration changed, reloading...")
      try await cleanup()
    }

    do {
      try await marvisTTS.initialize(
        modelRepoId: modelVariant.repoId,
        progressHandler: progressHandler ?? { _ in },
      )

      lastModelVariant = modelVariant
      isLoaded = true
      Log.model.info("Marvis TTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load Marvis model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  public func stop() async {
    generationTask?.cancel()
    generationTask = nil

    await audioPlayer.stop()
    isGenerating = false
    isPlaying = false

    Log.tts.debug("MarvisEngine stopped")
  }

  public func unload() async {
    await stop()

    // Clear model but preserve audio player
    try? await marvisTTS.cleanUp()
    lastModelVariant = nil
    isLoaded = false

    Log.tts.debug("MarvisEngine unloaded")
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
  ///   - voice: The voice to use
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    voice: Voice,
  ) async throws -> AudioResult {
    if !isLoaded {
      try await load()
    }

    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedText.isEmpty else {
      throw TTSError.invalidArgument("Text cannot be empty")
    }

    generationTask?.cancel()
    isGenerating = true
    generationTime = 0

    do {
      let result = try await marvisTTS.generate(
        text: trimmedText,
        voice: voice,
        quality: qualityLevel,
      )

      generationTime = result.processingTime
      isGenerating = false

      Log.tts.timing("Marvis generation", duration: result.processingTime)
      Log.tts.rtf("Marvis", rtf: result.realTimeFactor)

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
      Log.tts.error("Marvis generation failed: \(error.localizedDescription)")
      throw TTSError.generationFailed(underlying: error)
    }
  }

  /// Generate and immediately play audio
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  public func say(
    _ text: String,
    voice: Voice,
  ) async throws {
    let audio = try await generate(text, voice: voice)
    await play(audio)
  }

  // MARK: - Streaming

  /// Generate audio as a stream of chunks (no playback)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  /// - Returns: An async stream of audio chunks
  public func generateStreaming(
    _ text: String,
    voice: Voice,
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    let quality = qualityLevel
    let interval = streamingInterval
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Text cannot be empty")) }
    }

    guard isLoaded else {
      return AsyncThrowingStream { $0.finish(throwing: TTSError.modelNotLoaded) }
    }

    return AsyncThrowingStream { continuation in
      Task { @MainActor [weak self] in
        guard let self else {
          continuation.finish()
          return
        }

        isGenerating = true
        generationTime = 0

        var isFirst = true

        do {
          let stream = try await marvisTTS.generateStreaming(
            text: trimmedText,
            voice: voice,
            quality: quality,
            interval: interval,
          )

          for try await result in stream {
            if isFirst {
              generationTime = result.processingTime
              isFirst = false
            }

            let chunk = AudioChunk(
              samples: result.audio,
              sampleRate: result.sampleRate,
              isLast: false,
              processingTime: result.processingTime,
            )
            continuation.yield(chunk)
          }

          isGenerating = false
          continuation.finish()

        } catch {
          isGenerating = false
          Log.tts.error("Marvis streaming failed: \(error.localizedDescription)")
          continuation.finish(throwing: TTSError.generationFailed(underlying: error))
        }
      }
    }
  }

  /// Play audio with streaming (plays as chunks arrive)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  @discardableResult
  public func sayStreaming(
    _ text: String,
    voice: Voice,
  ) async throws -> AudioResult {
    isPlaying = true
    var allSamples: [Float] = []
    var totalProcessingTime: TimeInterval = 0

    do {
      for try await chunk in generateStreaming(text, voice: voice) {
        allSamples.append(contentsOf: chunk.samples)
        totalProcessingTime = chunk.processingTime
        audioPlayer.enqueue(samples: chunk.samples)
      }

      await audioPlayer.awaitCompletion()
      isPlaying = false

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
      throw error
    }
  }
}

// MARK: - Quality Level Helpers

extension MarvisEngine {
  /// Available quality levels
  static let qualityLevels = QualityLevel.allCases

  /// Description for each quality level
  func qualityDescription(for level: QualityLevel) -> String {
    switch level {
      case .low:
        "\(level.codebookCount) codebooks - Fastest, lower quality"
      case .medium:
        "\(level.codebookCount) codebooks - Balanced"
      case .high:
        "\(level.codebookCount) codebooks - Slower, better quality"
      case .maximum:
        "\(level.codebookCount) codebooks - Slowest, best quality"
    }
  }
}
