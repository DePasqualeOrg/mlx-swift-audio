import Foundation
import MLX
import MLXAudio

/// Manages WhisperEngine lifecycle and state
@MainActor
@Observable
final class EngineManager {
  // MARK: - Engine

  /// Current Whisper engine instance (recreated when model/quantization changes)
  private(set) var whisperEngine: WhisperEngine?

  // MARK: - Configuration (what the loaded engine was created with)

  private(set) var loadedModelSize: WhisperModelSize?
  private(set) var loadedQuantization: WhisperQuantization?

  // MARK: - State

  /// Whether a model is currently being loaded
  private(set) var isLoading: Bool = false

  /// Model loading progress (0.0 to 1.0)
  private(set) var loadingProgress: Double = 0

  /// Last error that occurred
  private(set) var error: STTError?

  // MARK: - Computed Properties

  var isLoaded: Bool { whisperEngine?.isLoaded ?? false }
  var isTranscribing: Bool { whisperEngine?.isTranscribing ?? false }
  var transcriptionTime: TimeInterval { whisperEngine?.transcriptionTime ?? 0 }

  /// Check if engine needs to be recreated due to config change
  func needsReload(modelSize: WhisperModelSize, quantization: WhisperQuantization) -> Bool {
    loadedModelSize != modelSize || loadedQuantization != quantization
  }

  // MARK: - Engine Lifecycle

  /// Load or reload the engine with specified configuration
  func loadEngine(
    modelSize: WhisperModelSize,
    quantization: WhisperQuantization
  ) async throws {
    // If already loaded with same config, skip
    if isLoaded, loadedModelSize == modelSize, loadedQuantization == quantization {
      return
    }

    // Unload existing engine if config changed
    if whisperEngine != nil {
      await unload()
    }

    isLoading = true
    loadingProgress = 0
    error = nil

    MLXMemory.configureForPlatform()

    do {
      // Create new engine with requested config
      whisperEngine = WhisperEngine(modelSize: modelSize, quantization: quantization)

      try await whisperEngine?.load { [weak self] progress in
        Task { @MainActor in
          self?.loadingProgress = progress.fractionCompleted
        }
      }

      loadedModelSize = modelSize
      loadedQuantization = quantization
      isLoading = false
      loadingProgress = 1.0
    } catch {
      isLoading = false
      loadingProgress = 0
      whisperEngine = nil
      let sttError = STTError.modelLoadFailed(underlying: error)
      self.error = sttError
      throw sttError
    }
  }

  /// Transcribe audio file
  func transcribe(
    url: URL,
    language: Language?,
    timestamps: TimestampGranularity
  ) async throws -> TranscriptionResult {
    guard let engine = whisperEngine, engine.isLoaded else {
      throw STTError.modelNotLoaded
    }

    error = nil

    do {
      return try await engine.transcribe(
        url,
        language: language,
        temperature: 0.0,
        timestamps: timestamps
      )
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      let sttError = (error as? STTError) ?? STTError.transcriptionFailed(underlying: error)
      self.error = sttError
      throw sttError
    }
  }

  /// Translate audio file to English
  func translate(
    url: URL,
    language: Language?,
    timestamps: TimestampGranularity
  ) async throws -> TranscriptionResult {
    guard let engine = whisperEngine, engine.isLoaded else {
      throw STTError.modelNotLoaded
    }

    error = nil

    do {
      return try await engine.translate(
        url,
        language: language,
        timestamps: timestamps
      )
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      let sttError = (error as? STTError) ?? STTError.transcriptionFailed(underlying: error)
      self.error = sttError
      throw sttError
    }
  }

  /// Detect language of audio file
  func detectLanguage(url: URL) async throws -> (Language, Float) {
    guard let engine = whisperEngine, engine.isLoaded else {
      throw STTError.modelNotLoaded
    }

    error = nil

    do {
      return try await engine.detectLanguage(url)
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      let sttError = (error as? STTError) ?? STTError.transcriptionFailed(underlying: error)
      self.error = sttError
      throw sttError
    }
  }

  /// Unload current engine
  func unload() async {
    await whisperEngine?.unload()
    whisperEngine = nil
    loadedModelSize = nil
    loadedQuantization = nil
  }

  /// Stop current transcription
  func stop() async {
    await whisperEngine?.stop()
  }
}
