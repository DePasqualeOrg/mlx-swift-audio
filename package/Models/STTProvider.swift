import Foundation

/// Available STT (Speech-to-Text) providers
public enum STTProvider: String, CaseIterable, Identifiable, Sendable {
  case whisper

  public var id: String { rawValue }

  /// Canonical display name with proper casing/branding
  public var displayName: String {
    switch self {
      case .whisper: "Whisper"
    }
  }

  // MARK: - Audio Properties

  /// Sample rate for this provider's audio input (Hz)
  public var sampleRate: Int {
    switch self {
      case .whisper: 16000
    }
  }

  // MARK: - Feature Flags

  /// Whether this provider supports language detection
  public var supportsLanguageDetection: Bool {
    switch self {
      case .whisper: true
    }
  }

  /// Whether this provider supports translation to English
  public var supportsTranslation: Bool {
    switch self {
      case .whisper: true
    }
  }

  /// Whether this provider supports word-level timestamps
  public var supportsWordTimestamps: Bool {
    switch self {
      case .whisper: false // Will be added in future phase
    }
  }

  /// Whether this provider supports streaming transcription
  public var supportsStreaming: Bool {
    switch self {
      case .whisper: false // Will be added in future phase
    }
  }
}
