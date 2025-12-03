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
public final class KokoroEngine: TTSEngine {
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
  ///   - voice: The voice to use (default: .afHeart)
  ///   - speed: Playback speed multiplier (default: 1.0)
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    voice: KokoroTTS.Voice = .afHeart,
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
  ///   - voice: The voice to use (default: .afHeart)
  ///   - speed: Playback speed multiplier (default: 1.0)
  public func say(
    _ text: String,
    voice: KokoroTTS.Voice = .afHeart,
    speed: Float = 1.0,
  ) async throws {
    let audio = try await generate(text, voice: voice, speed: speed)
    isPlaying = true
    await audio.play()
    isPlaying = false
  }

  // MARK: - Streaming

  /// Generate audio with streaming playback (audio plays as it generates)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use (default: .afHeart)
  ///   - speed: Playback speed multiplier (default: 1.0)
  /// - Returns: The generated audio result
  public func generateWithStreaming(
    _ text: String,
    voice: KokoroTTS.Voice = .afHeart,
    speed: Float = 1.0,
  ) async throws -> AudioResult {
    if !isLoaded {
      try await load()
    }

    guard let kokoroTTS, let audioPlayer else {
      throw TTSError.modelNotLoaded
    }

    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedText.isEmpty else {
      throw TTSError.invalidArgument("Text cannot be empty")
    }

    generationTask?.cancel()
    isGenerating = true
    isPlaying = true
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
        audioPlayer.enqueue(samples: samples, prebufferSeconds: 0)
      }

      isGenerating = false

      await audioPlayer.awaitCompletion()
      isPlaying = false

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
      isPlaying = false
      throw TTSError.generationFailed(underlying: error)
    }
  }
}
