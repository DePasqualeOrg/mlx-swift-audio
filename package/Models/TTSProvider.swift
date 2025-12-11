import Foundation

// MARK: - ChatterboxQuantization

/// Quantization level for Chatterbox models
public enum ChatterboxQuantization: String, Sendable, CaseIterable {
  /// 16-bit floating point (best quality, larger size)
  case fp16

  /// 8-bit quantization (good balance of quality and size)
  case q8 = "8bit"

  /// 4-bit quantization (smallest size, some quality tradeoff)
  case q4 = "4bit"

  /// Display name
  public var displayName: String {
    switch self {
      case .fp16: "FP16 (Best Quality)"
      case .q8: "8-bit (Balanced)"
      case .q4: "4-bit (Smallest)"
    }
  }

  /// Approximate size multiplier relative to fp16
  public var sizeMultiplier: Float {
    switch self {
      case .fp16: 1.0
      case .q8: 0.5
      case .q4: 0.25
    }
  }
}

// MARK: - TTSProvider

/// Available TTS providers
public enum TTSProvider: String, CaseIterable, Identifiable, Sendable {
  case cosyVoice2
  case chatterbox
  case outetts
  case kokoro
  case orpheus
  case marvis

  public var id: String { rawValue }

  /// Canonical display name with proper casing/branding
  public var displayName: String {
    switch self {
      case .outetts: "OuteTTS"
      case .chatterbox: "Chatterbox"
      case .cosyVoice2: "CosyVoice2"
      default: rawValue.capitalized
    }
  }

  // MARK: - Audio Properties

  /// Sample rate for this provider's audio output (Hz)
  public var sampleRate: Int {
    switch self {
      default: 24000
    }
  }

  // MARK: - Feature Flags

  /// Whether this provider supports speed adjustment
  public var supportsSpeed: Bool {
    switch self {
      case .kokoro: true
      default: false
    }
  }

  /// Whether this provider supports emotional expressions
  public var supportsExpressions: Bool {
    switch self {
      case .orpheus: true
      default: false
    }
  }

  /// Whether this provider supports quality level selection
  public var supportsQualityLevels: Bool {
    switch self {
      case .marvis: true
      default: false
    }
  }

  /// Whether this provider supports reference audio
  public var supportsReferenceAudio: Bool {
    switch self {
      case .outetts, .chatterbox, .cosyVoice2: true
      default: false
    }
  }

  /// Whether this provider supports reference text transcription (for zero-shot mode)
  public var supportsReferenceText: Bool {
    switch self {
      case .cosyVoice2: true
      default: false
    }
  }
}
