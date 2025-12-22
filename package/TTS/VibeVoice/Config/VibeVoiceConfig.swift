// Copyright © Microsoft (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/microsoft/VibeVoice
// License: licenses/vibevoice.txt

import Foundation

/// Configuration for the acoustic tokenizer (VAE decoder)
public struct AcousticTokenizerConfig: Codable, Sendable {
  public var modelType: String = "vibevoice_acoustic_tokenizer"
  public var channels: Int = 1
  public var corpusNormalize: Float = 0.0
  public var causal: Bool = true
  public var vaeDim: Int = 64
  public var fixStd: Float = 0.5
  public var stdDistType: String = "gaussian"

  // Common parameters
  public var mixerLayer: String = "depthwise_conv"
  public var convNorm: String = "none"
  public var padMode: String = "constant"
  public var disableLastNorm: Bool = true
  public var layernorm: String = "RMSNorm"
  public var layernormEps: Float = 1e-5
  public var layernormElementwiseAffine: Bool = true
  public var convBias: Bool = true
  public var layerScaleInitValue: Float = 1e-6
  public var weightInitValue: Float = 0.01

  // Encoder specific
  public var encoderNFilters: Int = 32
  public var encoderRatios: [Int] = [8, 5, 5, 4, 2, 2]
  public var encoderDepths: String = "3-3-3-3-3-3-8"

  // Decoder specific
  public var decoderNFilters: Int?
  public var decoderRatios: [Int]?
  public var decoderDepths: String?

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case channels
    case corpusNormalize = "corpus_normalize"
    case causal
    case vaeDim = "vae_dim"
    case fixStd = "fix_std"
    case stdDistType = "std_dist_type"
    case mixerLayer = "mixer_layer"
    case convNorm = "conv_norm"
    case padMode = "pad_mode"
    case disableLastNorm = "disable_last_norm"
    case layernorm
    case layernormEps = "layernorm_eps"
    case layernormElementwiseAffine = "layernorm_elementwise_affine"
    case convBias = "conv_bias"
    case layerScaleInitValue = "layer_scale_init_value"
    case weightInitValue = "weight_init_value"
    case encoderNFilters = "encoder_n_filters"
    case encoderRatios = "encoder_ratios"
    case encoderDepths = "encoder_depths"
    case decoderNFilters = "decoder_n_filters"
    case decoderRatios = "decoder_ratios"
    case decoderDepths = "decoder_depths"
  }

  public init() {}

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "vibevoice_acoustic_tokenizer"
    channels = try container.decodeIfPresent(Int.self, forKey: .channels) ?? 1
    corpusNormalize = try container.decodeIfPresent(Float.self, forKey: .corpusNormalize) ?? 0.0
    causal = try container.decodeIfPresent(Bool.self, forKey: .causal) ?? true
    vaeDim = try container.decodeIfPresent(Int.self, forKey: .vaeDim) ?? 64
    fixStd = try container.decodeIfPresent(Float.self, forKey: .fixStd) ?? 0.5
    stdDistType = try container.decodeIfPresent(String.self, forKey: .stdDistType) ?? "gaussian"
    mixerLayer = try container.decodeIfPresent(String.self, forKey: .mixerLayer) ?? "depthwise_conv"
    convNorm = try container.decodeIfPresent(String.self, forKey: .convNorm) ?? "none"
    padMode = try container.decodeIfPresent(String.self, forKey: .padMode) ?? "constant"
    disableLastNorm = try container.decodeIfPresent(Bool.self, forKey: .disableLastNorm) ?? true
    layernorm = try container.decodeIfPresent(String.self, forKey: .layernorm) ?? "RMSNorm"
    layernormEps = try container.decodeIfPresent(Float.self, forKey: .layernormEps) ?? 1e-5
    layernormElementwiseAffine = try container.decodeIfPresent(Bool.self, forKey: .layernormElementwiseAffine) ?? true
    convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? true
    layerScaleInitValue = try container.decodeIfPresent(Float.self, forKey: .layerScaleInitValue) ?? 1e-6
    weightInitValue = try container.decodeIfPresent(Float.self, forKey: .weightInitValue) ?? 0.01
    encoderNFilters = try container.decodeIfPresent(Int.self, forKey: .encoderNFilters) ?? 32
    encoderRatios = try container.decodeIfPresent([Int].self, forKey: .encoderRatios) ?? [8, 5, 5, 4, 2, 2]
    encoderDepths = try container.decodeIfPresent(String.self, forKey: .encoderDepths) ?? "3-3-3-3-3-3-8"
    decoderNFilters = try container.decodeIfPresent(Int.self, forKey: .decoderNFilters)
    decoderRatios = try container.decodeIfPresent([Int].self, forKey: .decoderRatios)
    decoderDepths = try container.decodeIfPresent(String.self, forKey: .decoderDepths)
  }
}

/// Configuration for the diffusion prediction head
public struct DiffusionHeadConfig: Codable, Sendable {
  public var modelType: String = "vibevoice_diffusion_head"
  public var hiddenSize: Int = 896
  public var headLayers: Int = 4
  public var headFfnRatio: Float = 3.0
  public var rmsNormEps: Float = 1e-5
  public var latentSize: Int = 64
  public var speechVaeDim: Int? = 64
  public var predictionType: String = "v_prediction"
  public var diffusionType: String = "ddpm"
  public var ddpmNumSteps: Int = 1000
  public var ddpmNumInferenceSteps: Int = 20
  public var ddpmBetaSchedule: String = "cosine"
  public var ddpmBatchMul: Int = 4

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case hiddenSize = "hidden_size"
    case headLayers = "head_layers"
    case headFfnRatio = "head_ffn_ratio"
    case rmsNormEps = "rms_norm_eps"
    case latentSize = "latent_size"
    case speechVaeDim = "speech_vae_dim"
    case predictionType = "prediction_type"
    case diffusionType = "diffusion_type"
    case ddpmNumSteps = "ddpm_num_steps"
    case ddpmNumInferenceSteps = "ddpm_num_inference_steps"
    case ddpmBetaSchedule = "ddpm_beta_schedule"
    case ddpmBatchMul = "ddpm_batch_mul"
  }

  public init() {}

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "vibevoice_diffusion_head"
    hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 896
    headLayers = try container.decodeIfPresent(Int.self, forKey: .headLayers) ?? 4
    headFfnRatio = try container.decodeIfPresent(Float.self, forKey: .headFfnRatio) ?? 3.0
    rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
    latentSize = try container.decodeIfPresent(Int.self, forKey: .latentSize) ?? 64
    speechVaeDim = try container.decodeIfPresent(Int.self, forKey: .speechVaeDim)
    predictionType = try container.decodeIfPresent(String.self, forKey: .predictionType) ?? "v_prediction"
    diffusionType = try container.decodeIfPresent(String.self, forKey: .diffusionType) ?? "ddpm"
    ddpmNumSteps = try container.decodeIfPresent(Int.self, forKey: .ddpmNumSteps) ?? 1000
    ddpmNumInferenceSteps = try container.decodeIfPresent(Int.self, forKey: .ddpmNumInferenceSteps) ?? 20
    ddpmBetaSchedule = try container.decodeIfPresent(String.self, forKey: .ddpmBetaSchedule) ?? "cosine"
    ddpmBatchMul = try container.decodeIfPresent(Int.self, forKey: .ddpmBatchMul) ?? 4
  }
}

/// Configuration for the Qwen2 decoder backbone
public struct Qwen2DecoderConfig: Codable, Sendable {
  public var modelType: String = "qwen2"
  public var attentionDropout: Float = 0.0
  public var hiddenAct: String = "silu"
  public var hiddenSize: Int = 896
  public var initializerRange: Float = 0.02
  public var intermediateSize: Int = 4864
  public var maxPositionEmbeddings: Int = 8192
  public var maxWindowLayers: Int = 24
  public var numAttentionHeads: Int = 14
  public var numHiddenLayers: Int = 24
  public var numKeyValueHeads: Int = 2
  public var rmsNormEps: Float = 1e-6
  public var ropeTheta: Float = 1_000_000.0
  public var tieWordEmbeddings: Bool = false
  public var useCache: Bool = true
  public var useSlidingWindow: Bool = false
  public var vocabSize: Int = 151_936
  public var headDim: Int?

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case attentionDropout = "attention_dropout"
    case hiddenAct = "hidden_act"
    case hiddenSize = "hidden_size"
    case initializerRange = "initializer_range"
    case intermediateSize = "intermediate_size"
    case maxPositionEmbeddings = "max_position_embeddings"
    case maxWindowLayers = "max_window_layers"
    case numAttentionHeads = "num_attention_heads"
    case numHiddenLayers = "num_hidden_layers"
    case numKeyValueHeads = "num_key_value_heads"
    case rmsNormEps = "rms_norm_eps"
    case ropeTheta = "rope_theta"
    case tieWordEmbeddings = "tie_word_embeddings"
    case useCache = "use_cache"
    case useSlidingWindow = "use_sliding_window"
    case vocabSize = "vocab_size"
    case headDim = "head_dim"
  }

  public init() {}

  public init(
    modelType: String = "qwen2",
    hiddenSize: Int = 896,
    intermediateSize: Int = 4864,
    numAttentionHeads: Int = 14,
    numKeyValueHeads: Int = 2,
    numHiddenLayers: Int = 24,
    rmsNormEps: Float = 1e-6,
    vocabSize: Int = 151_936,
    maxPositionEmbeddings: Int = 8192,
    ropeTheta: Float = 1_000_000.0,
    headDim: Int? = nil
  ) {
    self.modelType = modelType
    self.hiddenSize = hiddenSize
    self.intermediateSize = intermediateSize
    self.numAttentionHeads = numAttentionHeads
    self.numKeyValueHeads = numKeyValueHeads
    self.numHiddenLayers = numHiddenLayers
    self.rmsNormEps = rmsNormEps
    self.vocabSize = vocabSize
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.ropeTheta = ropeTheta
    self.headDim = headDim
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen2"
    attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
    hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
    hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 896
    initializerRange = try container.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
    intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4864
    maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
    maxWindowLayers = try container.decodeIfPresent(Int.self, forKey: .maxWindowLayers) ?? 24
    numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 14
    numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
    numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
    rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
    ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
    tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    useCache = try container.decodeIfPresent(Bool.self, forKey: .useCache) ?? true
    useSlidingWindow = try container.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? false
    vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 151_936
    headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
  }
}

/// Main configuration for VibeVoice streaming model
public struct VibeVoiceConfig: Codable, Sendable {
  public var modelType: String = "vibevoice_streaming"
  public var modelPath: String?
  public var sampleRate: Int = 24000

  // Sub-configurations
  public var acousticTokenizerConfig: AcousticTokenizerConfig = .init()
  public var decoderConfig: Qwen2DecoderConfig = .init()
  public var diffusionHeadConfig: DiffusionHeadConfig = .init()

  // Model architecture parameters
  public var acousticVaeDim: Int = 64
  public var ttsBackboneNumHiddenLayers: Int = 20

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case modelPath = "model_path"
    case sampleRate = "sample_rate"
    case acousticTokenizerConfig = "acoustic_tokenizer_config"
    case decoderConfig = "decoder_config"
    case diffusionHeadConfig = "diffusion_head_config"
    case acousticVaeDim = "acoustic_vae_dim"
    case ttsBackboneNumHiddenLayers = "tts_backbone_num_hidden_layers"
  }

  public init() {}

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "vibevoice_streaming"
    modelPath = try container.decodeIfPresent(String.self, forKey: .modelPath)
    sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
    acousticTokenizerConfig = try container.decodeIfPresent(AcousticTokenizerConfig.self, forKey: .acousticTokenizerConfig) ?? .init()
    decoderConfig = try container.decodeIfPresent(Qwen2DecoderConfig.self, forKey: .decoderConfig) ?? .init()
    diffusionHeadConfig = try container.decodeIfPresent(DiffusionHeadConfig.self, forKey: .diffusionHeadConfig) ?? .init()
    acousticVaeDim = try container.decodeIfPresent(Int.self, forKey: .acousticVaeDim) ?? 64
    ttsBackboneNumHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .ttsBackboneNumHiddenLayers) ?? 20
  }

  /// Load configuration from a pretrained model directory
  public static func fromPretrained(modelPath: String) throws -> VibeVoiceConfig {
    let configPath = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
    let data = try Data(contentsOf: configPath)
    let decoder = JSONDecoder()
    return try decoder.decode(VibeVoiceConfig.self, from: data)
  }
}

/// Constants used throughout the VibeVoice model
public enum VibeVoiceConstants {
  /// Output sample rate (24kHz)
  public static let sampleRate: Int = 24000

  /// Text window size for streaming
  public static let ttsTextWindowSize: Int = 5

  /// Speech window size for streaming
  public static let ttsSpeechWindowSize: Int = 6

  /// Acoustic VAE latent dimension
  public static let acousticVaeDim: Int = 64

  /// Default HuggingFace repository ID
  public static let defaultRepoId = "mlx-community/VibeVoice-Realtime-0.5B-4bit"
}
