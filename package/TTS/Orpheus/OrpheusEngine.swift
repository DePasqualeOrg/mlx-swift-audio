//
//  OrpheusEngine.swift
//  MLXAudio
//
//  Orpheus TTS engine conforming to TTSEngine protocol.
//  Wraps the existing OrpheusTTS implementation.
//

import Foundation

/// Orpheus TTS engine - high quality with emotional expressions
///
/// Supports expressions: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`,
/// `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`
@Observable
@MainActor
public final class OrpheusEngine: TTSEngine {
  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .orpheus
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - Orpheus-Specific Properties

  /// Temperature for sampling (higher = more variation)
  public var temperature: Float = 0.6

  /// Top-p (nucleus) sampling threshold
  public var topP: Float = 0.8

  // MARK: - Private Properties

  @ObservationIgnored private var orpheusTTS: OrpheusTTS?
  @ObservationIgnored private var generationTask: Task<Void, Never>?

  // MARK: - Initialization

  public init() {
    Log.tts.debug("OrpheusEngine initialized")
  }

  deinit {
    generationTask?.cancel()
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.tts.debug("OrpheusEngine already loaded")
      return
    }

    Log.model.info("Loading Orpheus TTS model...")

    do {
      orpheusTTS = try await OrpheusTTS.load(
        progressHandler: progressHandler ?? { _ in },
      )

      isLoaded = true
      Log.model.info("Orpheus TTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load Orpheus model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  public func stop() async {
    generationTask?.cancel()
    generationTask = nil
    isGenerating = false
    isPlaying = false

    Log.tts.debug("OrpheusEngine stopped")
  }

  public func cleanup() async throws {
    await stop()

    orpheusTTS = nil
    isLoaded = false

    Log.tts.debug("OrpheusEngine cleaned up")
  }

  // MARK: - Generation

  /// Generate audio from text
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use (default: .tara)
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    voice: OrpheusTTS.Voice = .tara,
  ) async throws -> AudioResult {
    if !isLoaded {
      try await load()
    }

    guard let orpheusTTS else {
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
      let samples = try await orpheusTTS.generateAudio(
        voice: voice,
        text: trimmedText,
        temperature: temperature,
        topP: topP,
      )

      generationTime = Date().timeIntervalSince(startTime)
      Log.tts.timing("Orpheus generation", duration: generationTime)

      isGenerating = false

      let audioDuration = Double(samples.count) / Double(TTSConstants.Audio.orpheusSampleRate)
      let rtf = generationTime / audioDuration
      Log.tts.rtf("Orpheus", rtf: rtf)

      do {
        let fileURL = try AudioFileWriter.save(
          samples: samples,
          sampleRate: TTSConstants.Audio.orpheusSampleRate,
          filename: TTSConstants.FileNames.orpheusOutput.replacingOccurrences(of: ".wav", with: ""),
        )
        lastGeneratedAudioURL = fileURL
      } catch {
        Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
      }

      return .samples(
        data: samples,
        sampleRate: TTSConstants.Audio.orpheusSampleRate,
        processingTime: generationTime,
      )

    } catch {
      isGenerating = false
      Log.tts.error("Orpheus generation failed: \(error.localizedDescription)")
      throw TTSError.generationFailed(underlying: error)
    }
  }

  /// Generate and immediately play audio
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use (default: .tara)
  public func say(
    _ text: String,
    voice: OrpheusTTS.Voice = .tara,
  ) async throws {
    let audio = try await generate(text, voice: voice)
    isPlaying = true
    await audio.play()
    isPlaying = false
  }
}
