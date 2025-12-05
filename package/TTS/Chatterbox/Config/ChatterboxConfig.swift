//
//  ChatterboxConfig.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/config.py
//

import Foundation

/// LLaMA 520M configuration for T3 backbone
struct T3LlamaConfig: Codable, Sendable {
  var modelType: String = "llama"
  var vocabSize: Int = 8 // Unused due to custom input layers
  var hiddenSize: Int = 1024
  var numHiddenLayers: Int = 30
  var intermediateSize: Int = 4096
  var numAttentionHeads: Int = 16
  var numKeyValueHeads: Int = 16
  var headDim: Int = 64
  var maxPositionEmbeddings: Int = 131_072
  var rmsNormEps: Float = 1e-05
  var ropeTheta: Float = 500_000.0
  var ropeScaling: RopeScaling = .init()
  var attentionBias: Bool = false
  var mlpBias: Bool = false
  var tieWordEmbeddings: Bool = false

  struct RopeScaling: Codable, Sendable {
    var factor: Float = 8.0
    var highFreqFactor: Float = 4.0
    var lowFreqFactor: Float = 1.0
    var originalMaxPositionEmbeddings: Int = 8192
    var ropeType: String = "llama3"

    enum CodingKeys: String, CodingKey {
      case factor
      case highFreqFactor = "high_freq_factor"
      case lowFreqFactor = "low_freq_factor"
      case originalMaxPositionEmbeddings = "original_max_position_embeddings"
      case ropeType = "rope_type"
    }

    init() {}
  }

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case vocabSize = "vocab_size"
    case hiddenSize = "hidden_size"
    case numHiddenLayers = "num_hidden_layers"
    case intermediateSize = "intermediate_size"
    case numAttentionHeads = "num_attention_heads"
    case numKeyValueHeads = "num_key_value_heads"
    case headDim = "head_dim"
    case maxPositionEmbeddings = "max_position_embeddings"
    case rmsNormEps = "rms_norm_eps"
    case ropeTheta = "rope_theta"
    case ropeScaling = "rope_scaling"
    case attentionBias = "attention_bias"
    case mlpBias = "mlp_bias"
    case tieWordEmbeddings = "tie_word_embeddings"
  }

  init() {}

  /// Default LLaMA 520M configuration
  static let llama520M = T3LlamaConfig()
}

/// Configuration for T3 (Token-to-Token) model
struct T3Config: Codable, Sendable {
  // Text token configuration
  var textTokensDictSize: Int = 704 // English: 704, Multilingual: 2454
  var startTextToken: Int = 255
  var stopTextToken: Int = 0
  var maxTextTokens: Int = 2048

  // Speech token configuration
  var speechTokensDictSize: Int = 8194
  var startSpeechToken: Int = 6561
  var stopSpeechToken: Int = 6562
  var maxSpeechTokens: Int = 4096

  // Model architecture
  var llamaConfigName: String = "Llama_520M"
  var inputPosEmb: String = "learned" // "learned" or "rope"
  var speechCondPromptLen: Int = 150

  // Conditioning
  var encoderType: String = "voice_encoder"
  var speakerEmbedSize: Int = 256
  var usePerceiverResampler: Bool = true
  var emotionAdv: Bool = true

  /// Get hidden size from LLaMA config
  var nChannels: Int {
    T3LlamaConfig.llama520M.hiddenSize
  }

  /// Check if model is multilingual
  var isMultilingual: Bool {
    textTokensDictSize == 2454
  }

  enum CodingKeys: String, CodingKey {
    case textTokensDictSize = "text_tokens_dict_size"
    case startTextToken = "start_text_token"
    case stopTextToken = "stop_text_token"
    case maxTextTokens = "max_text_tokens"
    case speechTokensDictSize = "speech_tokens_dict_size"
    case startSpeechToken = "start_speech_token"
    case stopSpeechToken = "stop_speech_token"
    case maxSpeechTokens = "max_speech_tokens"
    case llamaConfigName = "llama_config_name"
    case inputPosEmb = "input_pos_emb"
    case speechCondPromptLen = "speech_cond_prompt_len"
    case encoderType = "encoder_type"
    case speakerEmbedSize = "speaker_embed_size"
    case usePerceiverResampler = "use_perceiver_resampler"
    case emotionAdv = "emotion_adv"
  }

  init() {}

  /// Create configuration for English-only TTS model
  static func englishOnly() -> T3Config {
    var config = T3Config()
    config.textTokensDictSize = 704
    return config
  }

  /// Create configuration for multilingual TTS model
  static func multilingual() -> T3Config {
    var config = T3Config()
    config.textTokensDictSize = 2454
    return config
  }
}

/// Voice encoder configuration
struct VoiceEncConfig: Codable, Sendable {
  var numMels: Int = 40
  var sampleRate: Int = 16000
  var speakerEmbedSize: Int = 256
  var veHiddenSize: Int = 256
  var nFft: Int = 400
  var hopSize: Int = 160
  var winSize: Int = 400
  var fmax: Int = 8000
  var fmin: Int = 0
  var preemphasis: Float = 0.0
  var melPower: Float = 2.0
  var melType: String = "amp"
  var normalizedMels: Bool = false
  var vePartialFrames: Int = 160
  var veFinalRelu: Bool = true
  var stftMagnitudeMin: Float = 1e-4

  enum CodingKeys: String, CodingKey {
    case numMels = "num_mels"
    case sampleRate = "sample_rate"
    case speakerEmbedSize = "speaker_embed_size"
    case veHiddenSize = "ve_hidden_size"
    case nFft = "n_fft"
    case hopSize = "hop_size"
    case winSize = "win_size"
    case fmax
    case fmin
    case preemphasis
    case melPower = "mel_power"
    case melType = "mel_type"
    case normalizedMels = "normalized_mels"
    case vePartialFrames = "ve_partial_frames"
    case veFinalRelu = "ve_final_relu"
    case stftMagnitudeMin = "stft_magnitude_min"
  }

  init() {}
}

/// Main configuration for Chatterbox TTS model
struct ChatterboxModelConfig: Codable, Sendable {
  // Model type for auto-detection
  var modelType: String = "chatterbox"

  // Model components
  var t3Config: T3Config

  // Sample rates
  var s3Sr: Int = 16000 // S3 tokenizer sample rate
  var s3genSr: Int = 24000 // S3Gen output sample rate
  var sampleRate: Int = 24000 // Output sample rate (alias for s3genSr)

  // Conditioning lengths
  var encCondLen: Int = 6 * 16000 // 6 seconds at 16kHz
  var decCondLen: Int = 10 * 24000 // 10 seconds at 24kHz

  // Model path (set by load_model for tokenizer initialization)
  var modelPath: String?

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case t3Config = "t3_config"
    case s3Sr = "s3_sr"
    case s3genSr = "s3gen_sr"
    case sampleRate = "sample_rate"
    case encCondLen = "enc_cond_len"
    case decCondLen = "dec_cond_len"
    case modelPath = "model_path"
  }

  init() {
    t3Config = T3Config.englishOnly()
  }

  init(
    modelType: String = "chatterbox",
    t3Config: T3Config? = nil,
    s3Sr: Int = 16000,
    s3genSr: Int = 24000,
    sampleRate: Int? = nil,
    encCondLen: Int = 6 * 16000,
    decCondLen: Int = 10 * 24000,
    modelPath: String? = nil,
  ) {
    self.modelType = modelType
    self.t3Config = t3Config ?? T3Config.englishOnly()
    self.s3Sr = s3Sr
    self.s3genSr = s3genSr
    self.sampleRate = sampleRate ?? s3genSr
    self.encCondLen = encCondLen
    self.decCondLen = decCondLen
    self.modelPath = modelPath
  }
}

/// Constants used throughout the Chatterbox model
enum ChatterboxConstants {
  static let s3Sr: Int = 16000 // S3 tokenizer sample rate
  static let s3genSr: Int = 24000 // S3Gen output sample rate
  static let speechVocabSize: Int = 6561 // 3^8 vocabulary size
  static let encCondLen: Int = 6 * 16000 // 6 seconds at 16kHz (96000 samples)
  static let decCondLen: Int = 10 * 24000 // 10 seconds at 24kHz (240000 samples)
}
