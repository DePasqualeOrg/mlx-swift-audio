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

  var kokoroVoice: KokoroTTS.Voice = .afHeart
  var orpheusVoice: OrpheusTTS.Voice = .tara
  var marvisVoice: MarvisTTS.Voice = .conversationalA

  // MARK: - Computed Properties

  var currentEngine: any TTSEngine {
    switch selectedProvider {
      case .kokoro: kokoroEngine
      case .orpheus: orpheusEngine
      case .marvis: marvisEngine
      case .outetts: outeTTSEngine
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
      switch selectedProvider {
        case .kokoro:
          try await kokoroEngine.load { [weak self] progress in
            Task { @MainActor in
              self?.loadingProgress = progress.fractionCompleted
            }
          }
        case .orpheus:
          try await orpheusEngine.load { [weak self] progress in
            Task { @MainActor in
              self?.loadingProgress = progress.fractionCompleted
            }
          }
        case .marvis:
          try await marvisEngine.load { [weak self] progress in
            Task { @MainActor in
              self?.loadingProgress = progress.fractionCompleted
            }
          }
        case .outetts:
          try await outeTTSEngine.load { [weak self] progress in
            Task { @MainActor in
              self?.loadingProgress = progress.fractionCompleted
            }
          }
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
      }
    } catch let e as TTSError {
      error = e
      throw e
    } catch {
      let ttsError = TTSError.generationFailed(underlying: error)
      self.error = ttsError
      throw ttsError
    }
  }

  /// Generate with streaming (Kokoro and Marvis)
  func generateStreaming(text: String, speed: Float) -> AsyncThrowingStream<AudioChunk, Error> {
    switch selectedProvider {
      case .kokoro:
        kokoroEngine.generateStreaming(text, voice: kokoroVoice, speed: speed)
      case .marvis:
        marvisEngine.generateStreaming(text, voice: marvisVoice)
      default:
        AsyncThrowingStream { continuation in
          continuation.finish(throwing: TTSError.invalidArgument("Streaming not supported for \(selectedProvider.displayName)"))
        }
    }
  }

  /// Stream and play audio in real time (Kokoro and Marvis)
  func sayStreaming(text: String, speed: Float) async throws -> AudioResult {
    switch selectedProvider {
      case .kokoro:
        try await kokoroEngine.sayStreaming(text, voice: kokoroVoice, speed: speed)
      case .marvis:
        try await marvisEngine.sayStreaming(text, voice: marvisVoice)
      default:
        throw TTSError.invalidArgument("Streaming not supported for \(selectedProvider.displayName)")
    }
  }

  /// Play audio result
  func play(_ audio: AudioResult) async {
    await audio.play()
  }

  /// Stop generation and playback
  func stop() async {
    await currentEngine.stop()
  }
}
