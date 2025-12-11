import Foundation

/// Core protocol that all TTS engines must conform to.
///
/// Voice selection and generation are engine-specific since each engine
/// has different voice types (enums, speaker profiles with reference audio, etc.)
///
/// TODO: Evaluate configuration approach when adding more engines.
/// Currently: Kokoro passes `speed` to methods, other engines use mutable properties
/// (temperature, topP, qualityLevel, etc.). Consider unifying if a clear pattern emerges.
@MainActor
public protocol TTSEngine: Observable {
  /// The provider type for this engine
  var provider: TTSProvider { get }

  /// The granularity at which this engine streams audio
  var streamingGranularity: StreamingGranularity { get }

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

  /// Unload model weights to free GPU memory.
  ///
  /// Preserves cached data (speaker profiles, reference audio, text processing)
  /// for faster reload. Use this when switching between engines to free memory
  /// while keeping expensive pre-computed data.
  func unload() async

  /// Full cleanup - releases everything including cached data.
  ///
  /// Use before deallocating the engine or when you need to free all resources.
  func cleanup() async throws

  // MARK: - Playback

  /// Play an audio result using the engine's internal player.
  ///
  /// Each engine maintains its own audio player, avoiding the overhead of
  /// creating a new player for each playback.
  func play(_ audio: AudioResult) async
}

// MARK: - Default Implementations

public extension TTSEngine {
  /// Load the model without progress reporting
  func load() async throws {
    try await load(progressHandler: nil)
  }
}

// MARK: - Factory

/// Namespace for discovering and creating TTS engines with full type safety.
///
/// Each method returns a concrete engine type, enabling autocomplete for
/// engine-specific features like typed voices.
///
/// ```swift
/// let engine = TTS.orpheus()
/// try await engine.load()
/// try await engine.say("Hello", voice: .dan)  // typed voice enum
/// ```
///
/// For Kokoro, import MLXAudioKokoro and use `KokoroEngine()` directly.
@MainActor
public enum TTS {
  /// Orpheus: 8 voices, emotional expressions
  public static func orpheus() -> OrpheusEngine { OrpheusEngine() }

  /// Marvis: streaming, quality levels
  public static func marvis() -> MarvisEngine { MarvisEngine() }

  /// OuteTTS: custom speaker profiles
  public static func outetts() -> OuteTTSEngine { OuteTTSEngine() }

  /// Chatterbox: reference audio, emotion control
  public static func chatterbox() -> ChatterboxEngine { ChatterboxEngine() }

  /// CosyVoice2: voice matching with zero-shot and cross-lingual modes
  public static func cosyVoice2() -> CosyVoice2Engine { CosyVoice2Engine() }
}

/// Describes how an engine streams audio output
public enum StreamingGranularity: Sendable {
  /// Audio is streamed sentence-by-sentence. Each chunk contains a complete sentence.
  /// Higher latency per chunk, but natural break points.
  case sentence

  /// Audio is streamed frame-by-frame at regular intervals.
  /// Lower latency, continuous output.
  case frame

  /// Human-readable description for UI display
  public var description: String {
    switch self {
      case .sentence:
        "Sentence-by-sentence"
      case .frame:
        "Frame-by-frame"
    }
  }

  /// Short description for compact UI
  public var shortDescription: String {
    switch self {
      case .sentence:
        "Per sentence"
      case .frame:
        "Continuous"
    }
  }
}

/// A chunk of audio data for streaming playback
public struct AudioChunk: Sendable {
  /// Raw audio samples
  public let samples: [Float]

  /// Sample rate in Hz (e.g., 24000)
  public let sampleRate: Int

  /// Processing time for this chunk
  public let processingTime: TimeInterval

  public init(samples: [Float], sampleRate: Int, processingTime: TimeInterval) {
    self.samples = samples
    self.sampleRate = sampleRate
    self.processingTime = processingTime
  }
}

/// Result from TTS audio generation
public struct TTSGenerationResult: Sendable {
  /// Raw audio samples
  public let audio: [Float]

  /// Sample rate in Hz (e.g., 24000)
  public let sampleRate: Int

  /// Duration of the generated audio in seconds
  public let duration: TimeInterval

  /// Time taken to generate the audio in seconds
  public let processingTime: TimeInterval

  /// Real-time factor (duration / processingTime)
  /// Values > 1 mean faster than real-time
  public var realTimeFactor: Double {
    processingTime > 0 ? duration / processingTime : 0
  }

  public init(audio: [Float], sampleRate: Int, duration: TimeInterval, processingTime: TimeInterval) {
    self.audio = audio
    self.sampleRate = sampleRate
    self.duration = duration
    self.processingTime = processingTime
  }

  /// Convenience initializer that computes duration from sample count
  public init(audio: [Float], sampleRate: Int, processingTime: TimeInterval) {
    self.audio = audio
    self.sampleRate = sampleRate
    duration = Double(audio.count) / Double(sampleRate)
    self.processingTime = processingTime
  }
}
