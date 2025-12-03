//
//  KokoroEngine.swift
//  MLXAudio
//
//  Kokoro TTS engine conforming to TTSEngine protocol.
//  Wraps the existing KokoroTTS implementation.
//

import AVFoundation
import Foundation

/// Kokoro TTS engine - fast, lightweight TTS with many voice options
@Observable
@MainActor
public final class KokoroEngine: TTSEngine, StreamingTTSEngine {
  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .kokoro
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - Private Properties

  @ObservationIgnored private var kokoroTTS: KokoroTTS?
  @ObservationIgnored private var audioPlayer: AudioSamplePlayer?
  @ObservationIgnored private var generationTask: Task<Void, Never>?

  // MARK: - Initialization

  public init() {
    Log.tts.debug("KokoroEngine initialized")
  }

  deinit {
    generationTask?.cancel()
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.tts.debug("KokoroEngine already loaded")
      return
    }

    Log.model.info("Loading Kokoro TTS model...")

    kokoroTTS = KokoroTTS(
      repoId: KokoroWeightLoader.defaultRepoId,
      progressHandler: progressHandler ?? { _ in },
    )

    audioPlayer = AudioSamplePlayer(sampleRate: TTSConstants.Audio.kokoroSampleRate)

    isLoaded = true
    Log.model.info("Kokoro TTS model loaded successfully")
  }

  public func stop() async {
    generationTask?.cancel()
    generationTask = nil
    isGenerating = false

    await audioPlayer?.stop()
    isPlaying = false

    Log.tts.debug("KokoroEngine stopped")
  }

  public func cleanup() async throws {
    await stop()

    await kokoroTTS?.resetModel(preserveTextProcessing: false)
    kokoroTTS = nil
    audioPlayer = nil
    isLoaded = false

    Log.tts.debug("KokoroEngine cleaned up")
  }

  // MARK: - Generation

  /// Generate audio from text
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  ///   - speed: Playback speed multiplier (default: 1.0)
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    voice: KokoroTTS.Voice,
    speed: Float = 1.0,
  ) async throws -> AudioResult {
    if !isLoaded {
      try await load()
    }

    guard let kokoroTTS else {
      throw TTSError.modelNotLoaded
    }

    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedText.isEmpty else {
      throw TTSError.invalidArgument("Text cannot be empty")
    }

    generationTask?.cancel()
    isGenerating = true
    generationTime = 0

    let startTime = Date()
    var allSamples: [Float] = []
    var firstChunkTime: TimeInterval = 0

    do {
      for try await samples in try await kokoroTTS.generateAudioStream(
        voice: voice,
        text: trimmedText,
        speed: speed,
      ) {
        if firstChunkTime == 0 {
          firstChunkTime = Date().timeIntervalSince(startTime)
          generationTime = firstChunkTime
        }

        allSamples.append(contentsOf: samples)
      }

      isGenerating = false

      let totalTime = Date().timeIntervalSince(startTime)
      Log.tts.timing("Kokoro generation", duration: totalTime)

      do {
        let fileURL = try AudioFileWriter.save(
          samples: allSamples,
          sampleRate: TTSConstants.Audio.kokoroSampleRate,
          filename: TTSConstants.FileNames.kokoroOutput.replacingOccurrences(of: ".wav", with: ""),
        )
        lastGeneratedAudioURL = fileURL
      } catch {
        Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
      }

      return .samples(
        data: allSamples,
        sampleRate: TTSConstants.Audio.kokoroSampleRate,
        processingTime: generationTime,
      )

    } catch {
      isGenerating = false
      Log.tts.error("Kokoro generation failed: \(error.localizedDescription)")
      throw TTSError.generationFailed(underlying: error)
    }
  }

  /// Generate and immediately play audio
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  ///   - speed: Playback speed multiplier (default: 1.0)
  public func say(
    _ text: String,
    voice: KokoroTTS.Voice,
    speed: Float = 1.0,
  ) async throws {
    let audio = try await generate(text, voice: voice, speed: speed)
    isPlaying = true
    await audio.play()
    isPlaying = false
  }

  // MARK: - Streaming

  /// Generate audio as a stream of chunks (no playback)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  ///   - speed: Playback speed multiplier (default: 1.0)
  /// - Returns: An async stream of audio chunks
  public func generateStreaming(
    _ text: String,
    voice: KokoroTTS.Voice,
    speed: Float = 1.0,
  ) -> AsyncThrowingStream<AudioChunk, Error> {
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

        guard let kokoroTTS else {
          continuation.finish(throwing: TTSError.modelNotLoaded)
          return
        }

        isGenerating = true
        generationTime = 0

        let startTime = Date()
        var isFirst = true

        do {
          for try await samples in try await kokoroTTS.generateAudioStream(
            voice: voice,
            text: trimmedText,
            speed: speed,
          ) {
            if isFirst {
              generationTime = Date().timeIntervalSince(startTime)
              isFirst = false
            }

            let chunk = AudioChunk(
              samples: samples,
              sampleRate: TTSConstants.Audio.kokoroSampleRate,
              isLast: false,
              processingTime: Date().timeIntervalSince(startTime),
            )
            continuation.yield(chunk)
          }

          isGenerating = false
          continuation.finish()

        } catch {
          isGenerating = false
          Log.tts.error("Kokoro streaming failed: \(error.localizedDescription)")
          continuation.finish(throwing: TTSError.generationFailed(underlying: error))
        }
      }
    }
  }

  /// Play audio with streaming (plays as chunks arrive)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  ///   - speed: Playback speed multiplier (default: 1.0)
  @discardableResult
  public func sayStreaming(
    _ text: String,
    voice: KokoroTTS.Voice,
    speed: Float = 1.0,
  ) async throws -> AudioResult {
    guard let audioPlayer else {
      throw TTSError.modelNotLoaded
    }

    isPlaying = true
    var allSamples: [Float] = []
    var totalProcessingTime: TimeInterval = 0

    do {
      for try await chunk in generateStreaming(text, voice: voice, speed: speed) {
        allSamples.append(contentsOf: chunk.samples)
        totalProcessingTime = chunk.processingTime
        audioPlayer.enqueue(samples: chunk.samples, prebufferSeconds: 0)
      }

      await audioPlayer.awaitCompletion()
      isPlaying = false

      if !allSamples.isEmpty {
        do {
          let fileURL = try AudioFileWriter.save(
            samples: allSamples,
            sampleRate: TTSConstants.Audio.kokoroSampleRate,
            filename: TTSConstants.FileNames.kokoroOutput.replacingOccurrences(of: ".wav", with: ""),
          )
          lastGeneratedAudioURL = fileURL
        } catch {
          Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
        }
      }

      return .samples(
        data: allSamples,
        sampleRate: TTSConstants.Audio.kokoroSampleRate,
        processingTime: totalProcessingTime,
      )
    } catch {
      isPlaying = false
      throw error
    }
  }
}
