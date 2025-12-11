import MLXAudio

extension TTSProvider {
  /// Description of the provider's capabilities
  var description: String {
    switch self {
      case .kokoro:
        "Fast, lightweight TTS with many voices"
      case .orpheus:
        "High quality with emotional expressions"
      case .marvis:
        "Advanced conversational TTS with streaming"
      case .outetts:
        "TTS with speaker profiles"
      case .chatterbox:
        "TTS with reference audio support"
      case .cosyVoice2:
        "Voice matching with zero-shot and cross-lingual modes"
    }
  }

  /// Status message shown in the UI (warnings, tips, etc.)
  var statusMessage: String {
    switch self {
      case .kokoro:
        ""
      case .orpheus:
        "Supports expressions: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
      case .marvis:
        "Marvis: Advanced conversational TTS with streaming support."
      case .outetts:
        "OuteTTS: Supports custom speaker profiles."
      case .chatterbox:
        "Chatterbox: TTS with reference audio support."
      case .cosyVoice2:
        "CosyVoice2: Voice matching with 4 inference modes."
    }
  }
}
