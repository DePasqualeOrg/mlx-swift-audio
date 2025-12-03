//
//  MarvisEngine.swift
//  MLXAudio
//
//  Marvis TTS engine conforming to TTSEngine protocol.
//  Wraps the existing MarvisTTS implementation.
//

import Foundation
import MLX

/// Actor wrapper that owns and serializes access to MarvisTTS
actor MarvisTTSSession {
  private var tts: MarvisTTS?

  var isInitialized: Bool { tts != nil }

  func initialize(
    modelRepoId: String,
    progressHandler: @escaping @Sendable (Progress) -> Void,
  ) async throws {
    tts = try await MarvisTTS.fromPretrained(
      repoId: modelRepoId,
      progressHandler: progressHandler,
    )
  }

  func generate(
    text: String,
    voice: MarvisTTS.Voice,
    quality: MarvisTTS.QualityLevel,
  ) throws -> MarvisTTS.GenerationResult {
    guard let tts else { throw TTSError.modelNotLoaded }
    return try tts.generateAudio(text: text, voice: voice, quality: quality)
  }

  func generateStreaming(
    text: String,
    voice: MarvisTTS.Voice,
    quality: MarvisTTS.QualityLevel,
    interval: Double,
  ) throws -> AsyncThrowingStream<MarvisTTS.GenerationResult, Error> {
    guard let tts else { throw TTSError.modelNotLoaded }

    return AsyncThrowingStream { continuation in
      do {
        try tts.generateAudioStream(
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

  func stopPlayback() {
    tts?.stopPlayback()
  }

  func cleanUp() throws {
    try tts?.cleanUpMemory()
    tts = nil
  }
}

/// Marvis TTS engine - advanced conversational TTS with streaming support
@Observable
@MainActor
public final class MarvisEngine: TTSEngine, StreamingTTSEngine {
  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .marvis
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - Marvis-Specific Properties

  /// Model variant to use
  public var modelVariant: MarvisTTS.ModelVariant = .default

  /// Quality level (affects codebook count)
  public var qualityLevel: MarvisTTS.QualityLevel = .maximum

  /// Streaming interval in seconds
  public var streamingInterval: Double = TTSConstants.Timing.defaultStreamingInterval

  // MARK: - Private Properties

  @ObservationIgnored private let session = MarvisTTSSession()
  @ObservationIgnored private var audioPlayer: AudioSamplePlayer?
  @ObservationIgnored private var generationTask: Task<Void, Never>?
  @ObservationIgnored private var lastModelVariant: MarvisTTS.ModelVariant?

  // MARK: - Initialization

  public init() {
    Log.tts.debug("MarvisEngine initialized")
  }

  deinit {
    generationTask?.cancel()
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    let sessionInitialized = await session.isInitialized

    // Check if we need to reload
    if sessionInitialized, lastModelVariant == modelVariant {
      Log.tts.debug("MarvisEngine already loaded with same configuration")
      return
    }

    // Clean up existing session if configuration changed
    if sessionInitialized {
      Log.model.info("Configuration changed, reloading...")
      try await cleanup()
    }

    do {
      try await session.initialize(
        modelRepoId: modelVariant.repoId,
        progressHandler: progressHandler ?? { _ in },
      )

      audioPlayer = AudioSamplePlayer(sampleRate: TTSConstants.Audio.marvisSampleRate)

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

    await session.stopPlayback()

    await audioPlayer?.stop()
    isGenerating = false
    isPlaying = false

    Log.tts.debug("MarvisEngine stopped")
  }

  public func cleanup() async throws {
    await stop()

    try await session.cleanUp()
    audioPlayer = nil
    lastModelVariant = nil
    isLoaded = false

    Log.tts.debug("MarvisEngine cleaned up")
  }

  // MARK: - Generation

  /// Generate audio from text
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    voice: MarvisTTS.Voice,
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
      let result = try await session.generate(
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
          filename: TTSConstants.FileNames.marvisOutput.replacingOccurrences(of: ".wav", with: ""),
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
    voice: MarvisTTS.Voice,
  ) async throws {
    let audio = try await generate(text, voice: voice)
    isPlaying = true
    await audio.play()
    isPlaying = false
  }

  // MARK: - Streaming

  /// Generate audio as a stream of chunks (no playback)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  /// - Returns: An async stream of audio chunks
  public func generateStreaming(
    _ text: String,
    voice: MarvisTTS.Voice,
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
          let stream = try await session.generateStreaming(
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
    voice: MarvisTTS.Voice,
  ) async throws -> AudioResult {
    guard let audioPlayer else {
      throw TTSError.modelNotLoaded
    }

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
            sampleRate: TTSConstants.Audio.marvisSampleRate,
            filename: TTSConstants.FileNames.marvisOutput.replacingOccurrences(of: ".wav", with: ""),
          )
          lastGeneratedAudioURL = fileURL
        } catch {
          Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
        }
      }

      return .samples(
        data: allSamples,
        sampleRate: TTSConstants.Audio.marvisSampleRate,
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
  static let qualityLevels = MarvisTTS.QualityLevel.allCases

  /// Description for each quality level
  func qualityDescription(for level: MarvisTTS.QualityLevel) -> String {
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
