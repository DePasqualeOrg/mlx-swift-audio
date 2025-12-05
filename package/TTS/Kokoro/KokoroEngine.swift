import AVFoundation
import Foundation

/// Kokoro TTS engine - fast, lightweight TTS with many voice options
@Observable
@MainActor
public final class KokoroEngine: TTSEngine {
  // MARK: - Voice

  /// Available voices for Kokoro TTS
  public enum Voice: String, CaseIterable, Sendable {
    // American Female
    case afAlloy
    case afAoede
    case afBella
    case afHeart
    case afJessica
    case afKore
    case afNicole
    case afNova
    case afRiver
    case afSarah
    case afSky
    // American Male
    case amAdam
    case amEcho
    case amEric
    case amFenrir
    case amLiam
    case amMichael
    case amOnyx
    case amPuck
    case amSanta
    // British Female
    case bfAlice
    case bfEmma
    case bfIsabella
    case bfLily
    // British Male
    case bmDaniel
    case bmFable
    case bmGeorge
    case bmLewis
    // Spanish
    case efDora
    case emAlex
    // French
    case ffSiwis
    // Hindi
    case hfAlpha
    case hfBeta
    case hfOmega
    case hmPsi
    // Italian
    case ifSara
    case imNicola
    // Japanese
    case jfAlpha
    case jfGongitsune
    case jfNezumi
    case jfTebukuro
    case jmKumo
    // Portuguese
    case pfDora
    case pmSanta
    // Chinese
    case zfXiaobei
    case zfXiaoni
    case zfXiaoxiao
    case zfXiaoyi
    case zmYunjian
    case zmYunxi
    case zmYunxia
    case zmYunyang

    /// Voice ID in snake_case format (e.g., "af_heart")
    public var voiceID: String {
      // Convert camelCase rawValue to snake_case (e.g., "afHeart" -> "af_heart")
      let raw = rawValue
      guard raw.count >= 2 else { return raw.lowercased() }
      // Insert underscore after the 2-letter prefix
      let prefix = raw.prefix(2).lowercased()
      let name = raw.dropFirst(2).lowercased()
      return "\(prefix)_\(name)"
    }

    /// Convert to generic Voice struct for UI display
    public func toVoice() -> MLXAudio.Voice {
      MLXAudio.Voice.fromKokoroID(voiceID)
    }

    /// All voices as generic Voice structs
    public static var allVoices: [MLXAudio.Voice] {
      allCases.map { $0.toVoice() }
    }
  }

  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .kokoro
  public let streamingGranularity: StreamingGranularity = .sentence
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - Private Properties

  @ObservationIgnored private var kokoroTTS: KokoroTTS?
  @ObservationIgnored private let audioPlayer = AudioSamplePlayer(sampleRate: TTSProvider.kokoro.sampleRate)
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

    do {
      kokoroTTS = try await KokoroTTS.load(
        repoId: KokoroWeightLoader.defaultRepoId,
        progressHandler: progressHandler ?? { _ in },
      )

      isLoaded = true
      Log.model.info("Kokoro TTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load Kokoro model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  public func stop() async {
    generationTask?.cancel()
    generationTask = nil
    isGenerating = false

    await audioPlayer.stop()
    isPlaying = false

    Log.tts.debug("KokoroEngine stopped")
  }

  public func unload() async {
    await stop()
    kokoroTTS = nil
    isLoaded = false

    Log.tts.debug("KokoroEngine unloaded")
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
  ///   - speed: Playback speed multiplier (default: 1.0)
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    voice: Voice,
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
      for try await samples in try await kokoroTTS.generateStream(
        text: trimmedText,
        voice: voice,
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
          sampleRate: provider.sampleRate,
          filename: TTSConstants.outputFilename,
        )
        lastGeneratedAudioURL = fileURL
      } catch {
        Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
      }

      return .samples(
        data: allSamples,
        sampleRate: provider.sampleRate,
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
    voice: Voice,
    speed: Float = 1.0,
  ) async throws {
    let audio = try await generate(text, voice: voice, speed: speed)
    await play(audio)
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
    voice: Voice,
    speed: Float = 1.0,
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Text cannot be empty")) }
    }

    return AsyncThrowingStream { continuation in
      Task { @MainActor [weak self] in
        guard let self else {
          continuation.finish()
          return
        }

        // Auto-load if needed
        if !isLoaded {
          do {
            try await load()
          } catch {
            continuation.finish(throwing: error)
            return
          }
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
          for try await samples in try await kokoroTTS.generateStream(
            text: trimmedText,
            voice: voice,
            speed: speed,
          ) {
            if isFirst {
              generationTime = Date().timeIntervalSince(startTime)
              isFirst = false
            }

            let chunk = AudioChunk(
              samples: samples,
              sampleRate: provider.sampleRate,
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
    voice: Voice,
    speed: Float = 1.0,
  ) async throws -> AudioResult {
    isPlaying = true
    var allSamples: [Float] = []
    var totalProcessingTime: TimeInterval = 0

    do {
      for try await chunk in generateStreaming(text, voice: voice, speed: speed) {
        allSamples.append(contentsOf: chunk.samples)
        totalProcessingTime = chunk.processingTime
        audioPlayer.enqueue(samples: chunk.samples, prebufferSeconds: 0)
      }

      // Streaming complete - audio continues playing from queue
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
