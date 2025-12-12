import Foundation

/// Core protocol that all STT engines must conform to.
///
/// Transcription configuration and output are engine-specific since each engine
/// may have different capabilities (multilingual, translation, timestamps, etc.)
@MainActor
public protocol STTEngine: Observable {
  /// The provider type for this engine
  var provider: STTProvider { get }

  // MARK: - State Properties

  /// Whether the model is loaded and ready for transcription
  var isLoaded: Bool { get }

  /// Whether transcription is currently in progress
  var isTranscribing: Bool { get }

  /// Time taken for the last transcription (seconds)
  var transcriptionTime: TimeInterval { get }

  // MARK: - Lifecycle Methods

  /// Load the model with optional progress reporting
  /// - Parameter progressHandler: Optional callback for download/load progress
  func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws

  /// Stop any ongoing transcription
  func stop() async

  /// Unload model weights to free GPU memory.
  ///
  /// Preserves cached data (tokenizer, audio buffers) for faster reload.
  /// Use this when switching between engines to free memory while keeping
  /// expensive pre-computed data.
  func unload() async

  /// Full cleanup - releases everything including cached data.
  ///
  /// Use before deallocating the engine or when you need to free all resources.
  func cleanup() async throws
}

// MARK: - Default Implementations

public extension STTEngine {
  /// Load the model without progress reporting
  func load() async throws {
    try await load(progressHandler: nil)
  }
}

// MARK: - Factory

/// Namespace for discovering and creating STT engines with full type safety.
///
/// Each method returns a concrete engine type, enabling autocomplete for
/// engine-specific features.
///
/// ```swift
/// let engine = STT.whisper(model: .base)
/// try await engine.load()
/// let result = try await engine.transcribe(audioURL: url)
/// print(result.text)
/// ```
@MainActor
public enum STT {
  /// Whisper: multilingual speech recognition
  public static func whisper(model: WhisperModelSize = .base) -> WhisperEngine {
    WhisperEngine(modelSize: model)
  }
}
