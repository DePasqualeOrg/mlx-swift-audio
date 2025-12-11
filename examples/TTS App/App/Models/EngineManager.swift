import Foundation
import Kokoro
import MLX
import MLXAudio

/// Manages TTS engine lifecycle and state
@MainActor
@Observable
final class EngineManager {
  // MARK: - Engines

  private(set) var kokoroEngine = KokoroEngine()
  private(set) var orpheusEngine = OrpheusEngine()
  private(set) var marvisEngine = MarvisEngine()
  private(set) var outeTTSEngine = OuteTTSEngine()
  private(set) var chatterboxEngine = ChatterboxEngine()
  private(set) var cosyVoice2Engine = CosyVoice2Engine()

  // MARK: - State

  /// Currently selected provider
  private(set) var selectedProvider: TTSProvider = .cosyVoice2

  /// Whether a model is currently being loaded
  private(set) var isLoading: Bool = false

  /// Model loading progress (0.0 to 1.0)
  private(set) var loadingProgress: Double = 0

  /// Last error that occurred
  private(set) var error: TTSError?

  // MARK: - Voice Selection (string-based for UI)

  var kokoroVoice: KokoroEngine.Voice = .afHeart
  var orpheusVoice: OrpheusEngine.Voice = .tara
  var marvisVoice: MarvisEngine.Voice = .conversationalA

  // MARK: - Chatterbox Reference Audio

  /// Prepared reference audio for Chatterbox (enables fast speaker switching)
  var chatterboxReferenceAudio: ChatterboxReferenceAudio?

  // MARK: - OuteTTS Speaker Profile

  /// Speaker profile for OuteTTS (nil uses bundled default)
  var outeTTSSpeaker: OuteTTSSpeakerProfile?

  // MARK: - CosyVoice2 Speaker

  /// Prepared speaker for CosyVoice2 (enables fast speaker switching)
  var cosyVoice2Speaker: CosyVoice2Speaker?

  // MARK: - Computed Properties

  var currentEngine: any TTSEngine {
    switch selectedProvider {
      case .kokoro: kokoroEngine
      case .orpheus: orpheusEngine
      case .marvis: marvisEngine
      case .outetts: outeTTSEngine
      case .chatterbox: chatterboxEngine
      case .cosyVoice2: cosyVoice2Engine
    }
  }

  var isLoaded: Bool { currentEngine.isLoaded }
  var isGenerating: Bool { currentEngine.isGenerating }
  var isPlaying: Bool { currentEngine.isPlaying }
  var generationTime: TimeInterval { currentEngine.generationTime }
  var lastGeneratedAudioURL: URL? { currentEngine.lastGeneratedAudioURL }

  // MARK: - Initialization

  init(initialProvider: TTSProvider = .cosyVoice2) {
    selectedProvider = initialProvider
  }

  // MARK: - Engine Lifecycle

  /// Switch to a different TTS provider
  func selectProvider(_ provider: TTSProvider) async {
    guard provider != selectedProvider else { return }

    // Stop playback and unload the previous engine to free GPU memory
    let previousEngine = currentEngine
    if previousEngine.isLoaded {
      await previousEngine.unload() // unload() calls stop() internally
    } else {
      await previousEngine.stop() // Just stop playback if not loaded
    }

    selectedProvider = provider
    error = nil
  }

  /// Load the current engine's model
  func loadEngine() async throws {
    guard !currentEngine.isLoaded else {
      Log.model.debug("Engine already loaded")
      return
    }

    isLoading = true
    loadingProgress = 0
    error = nil

    MLXMemory.configureForPlatform()

    do {
      try await currentEngine.load { [weak self] progress in
        Task { @MainActor in
          self?.loadingProgress = progress.fractionCompleted
        }
      }

      // Chatterbox-specific: load default reference audio if needed
      if selectedProvider == .chatterbox, chatterboxReferenceAudio == nil {
        chatterboxReferenceAudio = try await chatterboxEngine.prepareDefaultReferenceAudio()
      }

      // CosyVoice2-specific: load default speaker if needed
      if selectedProvider == .cosyVoice2, cosyVoice2Speaker == nil {
        cosyVoice2Speaker = try await cosyVoice2Engine.prepareDefaultSpeaker()
      }

      isLoading = false
      loadingProgress = 1.0
    } catch {
      isLoading = false
      loadingProgress = 0
      let ttsError = TTSError.modelLoadFailed(underlying: error)
      self.error = ttsError
      throw ttsError
    }
  }

  /// Generate audio from text
  func generate(text: String, speed: Float) async throws -> AudioResult {
    guard currentEngine.isLoaded else {
      throw TTSError.modelNotLoaded
    }

    error = nil

    do {
      switch selectedProvider {
        case .kokoro:
          return try await kokoroEngine.generate(text, voice: kokoroVoice, speed: speed)
        case .orpheus:
          return try await orpheusEngine.generate(text, voice: orpheusVoice)
        case .marvis:
          return try await marvisEngine.generate(text, voice: marvisVoice)
        case .outetts:
          return try await outeTTSEngine.generate(text, speaker: outeTTSSpeaker)
        case .chatterbox:
          return try await chatterboxEngine.generate(text, referenceAudio: chatterboxReferenceAudio)
        case .cosyVoice2:
          // Handle voice conversion mode specially
          if cosyVoice2Engine.generationMode == .voiceConversion {
            return try await cosyVoice2Engine.generateVoiceConversion(speaker: cosyVoice2Speaker)
          } else {
            return try await cosyVoice2Engine.generate(text, speaker: cosyVoice2Speaker)
          }
      }
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      let ttsError = (error as? TTSError) ?? TTSError.generationFailed(underlying: error)
      self.error = ttsError
      throw ttsError
    }
  }

  /// Generate with streaming
  func generateStreaming(text: String, speed: Float) -> AsyncThrowingStream<AudioChunk, Error> {
    switch selectedProvider {
      case .kokoro:
        kokoroEngine.generateStreaming(text, voice: kokoroVoice, speed: speed)
      case .marvis:
        marvisEngine.generateStreaming(text, voice: marvisVoice)
      case .orpheus:
        orpheusEngine.generateStreaming(text, voice: orpheusVoice)
      case .outetts:
        outeTTSEngine.generateStreaming(text, speaker: outeTTSSpeaker)
      case .chatterbox:
        chatterboxEngine.generateStreaming(text, referenceAudio: chatterboxReferenceAudio)
      case .cosyVoice2:
        cosyVoice2Engine.generateStreaming(text, speaker: cosyVoice2Speaker)
    }
  }

  /// Stream and play audio in real time
  func sayStreaming(text: String, speed: Float) async throws -> AudioResult {
    error = nil

    do {
      switch selectedProvider {
        case .kokoro:
          return try await kokoroEngine.sayStreaming(text, voice: kokoroVoice, speed: speed)
        case .marvis:
          return try await marvisEngine.sayStreaming(text, voice: marvisVoice)
        case .orpheus:
          return try await orpheusEngine.sayStreaming(text, voice: orpheusVoice)
        case .outetts:
          return try await outeTTSEngine.sayStreaming(text, speaker: outeTTSSpeaker)
        case .chatterbox:
          return try await chatterboxEngine.sayStreaming(text, referenceAudio: chatterboxReferenceAudio)
        case .cosyVoice2:
          return try await cosyVoice2Engine.sayStreaming(text, speaker: cosyVoice2Speaker)
      }
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      let ttsError = (error as? TTSError) ?? TTSError.generationFailed(underlying: error)
      self.error = ttsError
      throw ttsError
    }
  }

  /// Play audio result using the current engine's player
  func play(_ audio: AudioResult) async {
    await currentEngine.play(audio)
  }

  /// Stop generation and playback
  func stop() async {
    await currentEngine.stop()
  }
}
