import Foundation

// MARK: - TranscriptionResult

/// Complete transcription result from STT engine
public struct TranscriptionResult: Sendable {
  /// Full transcription text
  public let text: String

  /// Detected or specified language code (e.g., "en", "zh", "es")
  public let language: String

  /// Individual segments with timestamps
  public let segments: [TranscriptionSegment]

  /// Processing time in seconds
  public let processingTime: TimeInterval

  /// Audio duration in seconds
  public let duration: TimeInterval

  /// Real-time factor (processingTime / duration)
  /// Values < 1.0 mean faster than real-time
  public var realTimeFactor: Double {
    duration > 0 ? processingTime / duration : 0
  }

  public init(
    text: String,
    language: String,
    segments: [TranscriptionSegment],
    processingTime: TimeInterval,
    duration: TimeInterval
  ) {
    self.text = text
    self.language = language
    self.segments = segments
    self.processingTime = processingTime
    self.duration = duration
  }
}

// MARK: - Segment

/// A transcription segment with timestamps
public struct TranscriptionSegment: Sendable {
  /// Segment text
  public let text: String

  /// Start time in seconds
  public let start: TimeInterval

  /// End time in seconds
  public let end: TimeInterval

  /// Token IDs for this segment
  public let tokens: [Int]

  /// Average log probability
  public let avgLogProb: Float

  /// No-speech probability (0-1)
  /// Higher values indicate the segment likely contains no speech
  public let noSpeechProb: Float

  /// Word-level timestamps (optional, not implemented yet)
  public let words: [Word]?

  public init(
    text: String,
    start: TimeInterval,
    end: TimeInterval,
    tokens: [Int],
    avgLogProb: Float,
    noSpeechProb: Float,
    words: [Word]? = nil
  ) {
    self.text = text
    self.start = start
    self.end = end
    self.tokens = tokens
    self.avgLogProb = avgLogProb
    self.noSpeechProb = noSpeechProb
    self.words = words
  }
}

// MARK: - Word

/// Word-level timestamp (future feature)
public struct Word: Sendable {
  /// The word text
  public let word: String

  /// Start time in seconds
  public let start: TimeInterval

  /// End time in seconds
  public let end: TimeInterval

  /// Confidence probability (0-1)
  public let probability: Float

  public init(
    word: String,
    start: TimeInterval,
    end: TimeInterval,
    probability: Float
  ) {
    self.word = word
    self.start = start
    self.end = end
    self.probability = probability
  }
}

// MARK: - TranscriptionTask

/// The task to perform during transcription
public enum TranscriptionTask: String, Sendable, CaseIterable {
  /// Transcribe in the original language
  case transcribe

  /// Translate to English
  case translate

  public var displayName: String {
    switch self {
      case .transcribe: "Transcribe"
      case .translate: "Translate to English"
    }
  }
}

// MARK: - TimestampGranularity

/// Granularity of timestamps in transcription
public enum TimestampGranularity: Sendable {
  /// No timestamps
  case none

  /// Segment-level timestamps only
  case segment

  /// Word-level timestamps (not implemented yet)
  case word

  public var displayName: String {
    switch self {
      case .none: "None"
      case .segment: "Segment-level"
      case .word: "Word-level"
    }
  }
}

// MARK: - WhisperModelSize

/// Available Whisper model sizes
public enum WhisperModelSize: String, Sendable, CaseIterable {
  // Multilingual models (use multilingual.tiktoken)
  case tiny = "whisper-tiny"
  case base = "whisper-base"
  case small = "whisper-small"
  case medium = "whisper-medium"
  case large = "whisper-large-v3"
  case largeTurbo = "whisper-large-v3-turbo"

  // English-only models (use gpt2.tiktoken)
  case tinyEn = "whisper-tiny.en"
  case baseEn = "whisper-base.en"
  case smallEn = "whisper-small.en"
  case mediumEn = "whisper-medium.en"

  /// Whether this model is currently available
  ///
  /// Models marked as unavailable either:
  /// - Use incompatible weight file structures (split encoder/decoder format)
  /// - Missing safetensors weight files (MLX Swift only supports .safetensors, not .npz)
  ///
  /// TODO: Find or create unified safetensors repos for all model sizes
  public var isAvailable: Bool {
    switch self {
      // Only largeTurbo has compatible safetensors format
      case .largeTurbo:
        true
      // Models with incompatible weight structure (split encoder/decoder)
      case .tiny, .base:
        false
      // Models without safetensors (use .npz)
      case .small, .medium, .large, .tinyEn, .baseEn, .smallEn, .mediumEn:
        false
    }
  }

  /// HuggingFace repository ID
  public var repoId: String {
    switch self {
      // Models with safetensors (split encoder/decoder format)
      case .tiny:
        "jkrukowski/whisper-tiny-mlx-safetensors"
      case .base:
        "jkrukowski/whisper-base-mlx-safetensors"
      // Models with safetensors (single file format)
      case .largeTurbo:
        "mlx-community/whisper-large-v3-turbo"
      // Models without safetensors (kept for future use)
      // TODO: Replace with safetensors repos when available
      case .small:
        "mlx-community/whisper-small-mlx" // Uses .npz
      case .medium:
        "mlx-community/whisper-medium-mlx" // Uses .npz
      case .large:
        "mlx-community/whisper-large-v3-mlx" // Uses .npz
      case .tinyEn, .baseEn, .smallEn, .mediumEn:
        "mlx-community/\(rawValue)-mlx" // Uses .npz
    }
  }

  /// Approximate parameter count
  public var parameters: String {
    switch self {
      case .tiny, .tinyEn: "39M"
      case .base, .baseEn: "74M"
      case .small, .smallEn: "244M"
      case .medium, .mediumEn: "769M"
      case .large: "1550M"
      case .largeTurbo: "809M"
    }
  }

  /// Display name
  public var displayName: String {
    switch self {
      case .tiny: "Tiny (39M)"
      case .tinyEn: "Tiny English-only (39M)"
      case .base: "Base (74M)"
      case .baseEn: "Base English-only (74M)"
      case .small: "Small (244M)"
      case .smallEn: "Small English-only (244M)"
      case .medium: "Medium (769M)"
      case .mediumEn: "Medium English-only (769M)"
      case .large: "Large v3 (1550M)"
      case .largeTurbo: "Large v3 Turbo (809M)"
    }
  }
}
