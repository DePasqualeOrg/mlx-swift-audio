//
//  OuteTTSEngine+Public.swift
//  MLXAudio
//
//  OuteTTS engine conforming to TTSEngine protocol.
//  Wraps the OuteTTSEngine actor implementation.
//

import Foundation

/// OuteTTS engine - TTS with custom speaker profiles
///
/// Supports custom speaker profiles with reference audio.
@Observable
@MainActor
public final class OuteTTSEngine: TTSEngine {
  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .outetts
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

  @ObservationIgnored private var outeTTS: OuteTTSSession?
  @ObservationIgnored private var generationTask: Task<Void, Never>?

  private static let sampleRate = 24000

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
      let engine = OuteTTSSession()
      try await engine.load(progressHandler: progressHandler ?? { _ in })
      outeTTS = engine

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
    isPlaying = false

    Log.tts.debug("OuteTTSEngine stopped")
  }

  public func cleanup() async throws {
    await stop()

    outeTTS = nil
    isLoaded = false

    Log.tts.debug("OuteTTSEngine cleaned up")
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

    let startTime = Date()

    do {
      let result = try await outeTTS.generate(
        text: trimmedText,
        speaker: speaker,
        temperature: temperature,
        topP: topP,
      )

      generationTime = Date().timeIntervalSince(startTime)
      Log.tts.timing("OuteTTS generation", duration: generationTime)

      isGenerating = false

      let audioDuration = result.duration
      let rtf = generationTime / audioDuration
      Log.tts.rtf("OuteTTS", rtf: rtf)

      do {
        let fileURL = try AudioFileWriter.save(
          samples: result.audio,
          sampleRate: Self.sampleRate,
          filename: "outetts_output",
        )
        lastGeneratedAudioURL = fileURL
      } catch {
        Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
      }

      return .samples(
        data: result.audio,
        sampleRate: Self.sampleRate,
        processingTime: generationTime,
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
    isPlaying = true
    await audio.play()
    isPlaying = false
  }

  // MARK: - Speaker Profile Helpers

  /// Load a speaker profile from a JSON file
  /// - Parameter path: Path to the speaker profile JSON file
  /// - Returns: The loaded speaker profile
  public static func loadSpeaker(from path: String) throws -> OuteTTSSpeakerProfile {
    try OuteTTSSpeakerProfile.load(from: path)
  }
}
