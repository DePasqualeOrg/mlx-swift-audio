//
//  TTSEngine.swift
//  MLXAudio
//
//  Protocol-oriented foundation for all TTS engines.
//

import Foundation

/// Core protocol that all TTS engines must conform to.
///
/// Voice selection and generation are engine-specific since each engine
/// has different voice types (enums, speaker profiles with reference audio, etc.)
@MainActor
public protocol TTSEngine: Observable {
  /// The provider type for this engine
  var provider: TTSProvider { get }

  // MARK: - State Properties

  /// Whether the model is loaded and ready for generation
  var isLoaded: Bool { get }

  /// Whether audio generation is currently in progress
  var isGenerating: Bool { get }

  /// Whether audio playback is currently in progress
  var isPlaying: Bool { get }

  /// URL of the last generated audio file (for sharing/export)
  var lastGeneratedAudioURL: URL? { get }

  /// Time taken for the last generation (seconds)
  var generationTime: TimeInterval { get }

  // MARK: - Lifecycle Methods

  /// Load the model with optional progress reporting
  /// - Parameter progressHandler: Optional callback for download/load progress
  func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws

  /// Stop any ongoing generation or playback
  func stop() async

  /// Clean up resources (model weights, audio buffers, etc.)
  func cleanup() async throws
}

// MARK: - Default Implementations

public extension TTSEngine {
  /// Load the model without progress reporting
  func load() async throws {
    try await load(progressHandler: nil)
  }
}

/// A chunk of audio data for streaming playback
public struct AudioChunk: Sendable {
  /// Raw audio samples
  public let samples: [Float]

  /// Sample rate in Hz (e.g., 24000)
  public let sampleRate: Int

  /// Whether this is the final chunk in the stream
  public let isLast: Bool

  /// Processing time for this chunk
  public let processingTime: TimeInterval

  public init(samples: [Float], sampleRate: Int, isLast: Bool, processingTime: TimeInterval) {
    self.samples = samples
    self.sampleRate = sampleRate
    self.isLast = isLast
    self.processingTime = processingTime
  }
}
