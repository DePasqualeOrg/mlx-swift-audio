import Foundation

/// Orpheus TTS engine - high quality with emotional expressions
///
/// Supports expressions: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`,
/// `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`
@Observable
@MainActor
public final class OrpheusEngine: TTSEngine {
  // MARK: - Voice

  /// Available voices for Orpheus TTS
  public enum Voice: String, CaseIterable, Sendable {
    case tara // Female, conversational, clear
    case leah // Female, warm, gentle
    case jess // Female, energetic, youthful
    case leo // Male, authoritative, deep
    case dan // Male, friendly, casual
    case mia // Female, professional, articulate
    case zac // Male, enthusiastic, dynamic
    case zoe // Female, calm, soothing

    /// Convert to generic Voice struct for UI display
    public func toVoice() -> MLXAudio.Voice {
      MLXAudio.Voice(id: rawValue, displayName: rawValue.capitalized, languageCode: "en-US")
    }

    /// All voices as generic Voice structs
    public static var allVoices: [MLXAudio.Voice] {
      allCases.map { $0.toVoice() }
    }
  }

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
  @ObservationIgnored private let audioPlayer = AudioSamplePlayer(sampleRate: TTSProvider.orpheus.sampleRate)
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

    await audioPlayer.stop()
    isPlaying = false

    Log.tts.debug("OrpheusEngine stopped")
  }

  public func unload() async {
    await stop()

    orpheusTTS = nil
    isLoaded = false

    Log.tts.debug("OrpheusEngine unloaded")
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

    do {
      let result = try await orpheusTTS.generate(
        text: trimmedText,
        voice: voice,
        temperature: temperature,
        topP: topP,
      )

      generationTime = result.processingTime
      isGenerating = false

      Log.tts.timing("Orpheus generation", duration: result.processingTime)
      Log.tts.rtf("Orpheus", rtf: result.realTimeFactor)

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
      Log.tts.error("Orpheus generation failed: \(error.localizedDescription)")
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
}
