import Foundation
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

  // MARK: - State

  /// Currently selected provider
  private(set) var selectedProvider: TTSProvider = .kokoro

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

  // MARK: - Computed Properties

  var currentEngine: any TTSEngine {
    switch selectedProvider {
      case .kokoro: kokoroEngine
      case .orpheus: orpheusEngine
      case .marvis: marvisEngine
      case .outetts: outeTTSEngine
      case .chatterbox: chatterboxEngine
    }
  }

  var isLoaded: Bool { currentEngine.isLoaded }
  var isGenerating: Bool { currentEngine.isGenerating }
  var isPlaying: Bool { currentEngine.isPlaying }
  var generationTime: TimeInterval { currentEngine.generationTime }
  var lastGeneratedAudioURL: URL? { currentEngine.lastGeneratedAudioURL }

  // MARK: - Initialization

  init(initialProvider: TTSProvider = .kokoro) {
    selectedProvider = initialProvider
  }

  // MARK: - Engine Lifecycle

  /// Switch to a different TTS provider
  func selectProvider(_ provider: TTSProvider) async {
    guard provider != selectedProvider else { return }

    // Unload the previous engine to free GPU memory
    // (preserves cached data like speaker profiles for faster reload)
    let previousEngine = currentEngine
    if previousEngine.isLoaded {
      await previousEngine.unload()
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

    MLX.GPU.set(cacheLimit: TTSConstants.Memory.gpuCacheLimit)

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
    MLX.GPU.set(cacheLimit: TTSConstants.Memory.gpuCacheLimit)

    do {
      switch selectedProvider {
        case .kokoro:
          return try await kokoroEngine.generate(text, voice: kokoroVoice, speed: speed)
        case .orpheus:
          return try await orpheusEngine.generate(text, voice: orpheusVoice)
        case .marvis:
          return try await marvisEngine.generate(text, voice: marvisVoice)
        case .outetts:
          return try await outeTTSEngine.generate(text)
        case .chatterbox:
          return try await chatterboxEngine.generate(text, referenceAudio: chatterboxReferenceAudio)
      }
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
        outeTTSEngine.generateStreaming(text)
      case .chatterbox:
        chatterboxEngine.generateStreaming(text, referenceAudio: chatterboxReferenceAudio)
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
          return try await outeTTSEngine.sayStreaming(text)
        case .chatterbox:
          return try await chatterboxEngine.sayStreaming(text, referenceAudio: chatterboxReferenceAudio)
      }
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
