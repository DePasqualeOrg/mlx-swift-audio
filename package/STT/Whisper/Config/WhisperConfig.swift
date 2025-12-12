import Foundation

/// Model dimensions for Whisper
public struct ModelDimensions: Codable, Sendable {
  /// Number of mel frequency bins
  public let n_mels: Int

  /// Number of audio context tokens
  public let n_audio_ctx: Int

  /// Audio encoder hidden size
  public let n_audio_state: Int

  /// Number of audio encoder attention heads
  public let n_audio_head: Int

  /// Number of audio encoder layers
  public let n_audio_layer: Int

  /// Vocabulary size
  public let n_vocab: Int

  /// Text decoder context length
  public let n_text_ctx: Int

  /// Text decoder hidden size
  public let n_text_state: Int

  /// Number of text decoder attention heads
  public let n_text_head: Int

  /// Number of text decoder layers
  public let n_text_layer: Int

  enum CodingKeys: String, CodingKey {
    case n_mels
    case n_audio_ctx
    case n_audio_state
    case n_audio_head
    case n_audio_layer
    case n_vocab
    case n_text_ctx
    case n_text_state
    case n_text_head
    case n_text_layer
  }

  public init(
    n_mels: Int,
    n_audio_ctx: Int,
    n_audio_state: Int,
    n_audio_head: Int,
    n_audio_layer: Int,
    n_vocab: Int,
    n_text_ctx: Int,
    n_text_state: Int,
    n_text_head: Int,
    n_text_layer: Int
  ) {
    self.n_mels = n_mels
    self.n_audio_ctx = n_audio_ctx
    self.n_audio_state = n_audio_state
    self.n_audio_head = n_audio_head
    self.n_audio_layer = n_audio_layer
    self.n_vocab = n_vocab
    self.n_text_ctx = n_text_ctx
    self.n_text_state = n_text_state
    self.n_text_head = n_text_head
    self.n_text_layer = n_text_layer
  }

  /// Load model dimensions from config.json file
  ///
  /// This is the canonical way to get model dimensions. All dimensions are loaded
  /// dynamically from HuggingFace config.json to avoid hardcoded mismatches.
  public static func load(from url: URL) throws -> ModelDimensions {
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    return try decoder.decode(ModelDimensions.self, from: data)
  }
}
