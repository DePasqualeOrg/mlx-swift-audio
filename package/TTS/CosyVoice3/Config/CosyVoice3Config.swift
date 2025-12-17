// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

// CosyVoice3 configuration structs
// Ported from mlx-audio-plus cosyvoice3/config.py

import Foundation

/// Configuration for DiT (Diffusion Transformer) module
public struct DiTConfig: Codable, Sendable {
  public var dim: Int = 1024
  public var depth: Int = 22
  public var heads: Int = 16
  public var dimHead: Int = 64
  public var ffMult: Int = 2
  public var dropout: Float = 0.1
  public var melDim: Int = 80
  public var muDim: Int = 80
  public var spkDim: Int = 80
  public var outChannels: Int = 80
  public var staticChunkSize: Int = 50 // chunk_size * token_mel_ratio
  public var numDecodingLeftChunks: Int = -1
  public var longSkipConnection: Bool = false

  enum CodingKeys: String, CodingKey {
    case dim
    case depth
    case heads
    case dimHead = "dim_head"
    case ffMult = "ff_mult"
    case dropout
    case melDim = "mel_dim"
    case muDim = "mu_dim"
    case spkDim = "spk_dim"
    case outChannels = "out_channels"
    case staticChunkSize = "static_chunk_size"
    case numDecodingLeftChunks = "num_decoding_left_chunks"
    case longSkipConnection = "long_skip_connection"
  }

  public init() {}

  public init(
    dim: Int = 1024,
    depth: Int = 22,
    heads: Int = 16,
    dimHead: Int = 64,
    ffMult: Int = 2,
    dropout: Float = 0.1,
    melDim: Int = 80,
    muDim: Int = 80,
    spkDim: Int = 80,
    outChannels: Int = 80,
    staticChunkSize: Int = 50,
    numDecodingLeftChunks: Int = -1,
    longSkipConnection: Bool = false
  ) {
    self.dim = dim
    self.depth = depth
    self.heads = heads
    self.dimHead = dimHead
    self.ffMult = ffMult
    self.dropout = dropout
    self.melDim = melDim
    self.muDim = muDim
    self.spkDim = spkDim
    self.outChannels = outChannels
    self.staticChunkSize = staticChunkSize
    self.numDecodingLeftChunks = numDecodingLeftChunks
    self.longSkipConnection = longSkipConnection
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    dim = try container.decodeIfPresent(Int.self, forKey: .dim) ?? 1024
    depth = try container.decodeIfPresent(Int.self, forKey: .depth) ?? 22
    heads = try container.decodeIfPresent(Int.self, forKey: .heads) ?? 16
    dimHead = try container.decodeIfPresent(Int.self, forKey: .dimHead) ?? 64
    ffMult = try container.decodeIfPresent(Int.self, forKey: .ffMult) ?? 2
    dropout = try container.decodeIfPresent(Float.self, forKey: .dropout) ?? 0.1
    melDim = try container.decodeIfPresent(Int.self, forKey: .melDim) ?? 80
    muDim = try container.decodeIfPresent(Int.self, forKey: .muDim) ?? 80
    spkDim = try container.decodeIfPresent(Int.self, forKey: .spkDim) ?? 80
    outChannels = try container.decodeIfPresent(Int.self, forKey: .outChannels) ?? 80
    staticChunkSize = try container.decodeIfPresent(Int.self, forKey: .staticChunkSize) ?? 50
    numDecodingLeftChunks = try container.decodeIfPresent(Int.self, forKey: .numDecodingLeftChunks) ?? -1
    longSkipConnection = try container.decodeIfPresent(Bool.self, forKey: .longSkipConnection) ?? false
  }
}

/// Configuration for Qwen2-based LLM in CosyVoice3
public struct CosyVoice3LLMConfig: Codable, Sendable {
  public var llmInputSize: Int = 896
  public var llmOutputSize: Int = 896
  public var speechTokenSize: Int = 6561
  /// CosyVoice3 uses +200 extended vocabulary
  public var extendedVocabSize: Int = 200
  public var mixRatio: [Int] = [5, 15]

  // Qwen2 model config
  public var hiddenSize: Int = 896
  public var numHiddenLayers: Int = 24
  public var intermediateSize: Int = 4864
  public var numAttentionHeads: Int = 14
  public var numKeyValueHeads: Int = 2
  public var rmsNormEps: Float = 1e-6
  public var vocabSize: Int = 151_936

  enum CodingKeys: String, CodingKey {
    case llmInputSize = "llm_input_size"
    case llmOutputSize = "llm_output_size"
    case speechTokenSize = "speech_token_size"
    case extendedVocabSize = "extended_vocab_size"
    case mixRatio = "mix_ratio"
    case hiddenSize = "hidden_size"
    case numHiddenLayers = "num_hidden_layers"
    case intermediateSize = "intermediate_size"
    case numAttentionHeads = "num_attention_heads"
    case numKeyValueHeads = "num_key_value_heads"
    case rmsNormEps = "rms_norm_eps"
    case vocabSize = "vocab_size"
  }

  public init() {}

  public init(
    llmInputSize: Int = 896,
    llmOutputSize: Int = 896,
    speechTokenSize: Int = 6561,
    extendedVocabSize: Int = 200,
    mixRatio: [Int] = [5, 15],
    hiddenSize: Int = 896,
    numHiddenLayers: Int = 24,
    intermediateSize: Int = 4864,
    numAttentionHeads: Int = 14,
    numKeyValueHeads: Int = 2,
    rmsNormEps: Float = 1e-6,
    vocabSize: Int = 151_936
  ) {
    self.llmInputSize = llmInputSize
    self.llmOutputSize = llmOutputSize
    self.speechTokenSize = speechTokenSize
    self.extendedVocabSize = extendedVocabSize
    self.mixRatio = mixRatio
    self.hiddenSize = hiddenSize
    self.numHiddenLayers = numHiddenLayers
    self.intermediateSize = intermediateSize
    self.numAttentionHeads = numAttentionHeads
    self.numKeyValueHeads = numKeyValueHeads
    self.rmsNormEps = rmsNormEps
    self.vocabSize = vocabSize
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    llmInputSize = try container.decodeIfPresent(Int.self, forKey: .llmInputSize) ?? 896
    llmOutputSize = try container.decodeIfPresent(Int.self, forKey: .llmOutputSize) ?? 896
    speechTokenSize = try container.decodeIfPresent(Int.self, forKey: .speechTokenSize) ?? 6561
    extendedVocabSize = try container.decodeIfPresent(Int.self, forKey: .extendedVocabSize) ?? 200
    mixRatio = try container.decodeIfPresent([Int].self, forKey: .mixRatio) ?? [5, 15]
    hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 896
    numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
    intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4864
    numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 14
    numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
    rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
    vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 151_936
  }
}

/// Configuration for Flow Matching module (CosyVoice3 with DiT)
public struct CosyVoice3FlowConfig: Codable, Sendable {
  public var inputSize: Int = 512
  public var outputSize: Int = 80
  public var spkEmbedDim: Int = 192
  public var outputType: String = "mel"
  public var vocabSize: Int = 6561
  public var inputFrameRate: Int = 25
  public var onlyMaskLoss: Bool = true
  public var tokenMelRatio: Int = 2
  public var preLookaheadLen: Int = 3
  public var nTimesteps: Int = 10

  // PreLookaheadLayer config
  public var preLookaheadChannels: Int = 512

  // DiT config (embedded)
  public var dit: DiTConfig = .init()

  // CFM params
  public var cfmSigmaMin: Float = 1e-6
  public var cfmTScheduler: String = "cosine"
  public var cfmInferenceCfgRate: Float = 0.7

  enum CodingKeys: String, CodingKey {
    case inputSize = "input_size"
    case outputSize = "output_size"
    case spkEmbedDim = "spk_embed_dim"
    case outputType = "output_type"
    case vocabSize = "vocab_size"
    case inputFrameRate = "input_frame_rate"
    case onlyMaskLoss = "only_mask_loss"
    case tokenMelRatio = "token_mel_ratio"
    case preLookaheadLen = "pre_lookahead_len"
    case nTimesteps = "n_timesteps"
    case preLookaheadChannels = "pre_lookahead_channels"
    case dit
    case cfmSigmaMin = "cfm_sigma_min"
    case cfmTScheduler = "cfm_t_scheduler"
    case cfmInferenceCfgRate = "cfm_inference_cfg_rate"
  }

  public init() {}

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    inputSize = try container.decodeIfPresent(Int.self, forKey: .inputSize) ?? 512
    outputSize = try container.decodeIfPresent(Int.self, forKey: .outputSize) ?? 80
    spkEmbedDim = try container.decodeIfPresent(Int.self, forKey: .spkEmbedDim) ?? 192
    outputType = try container.decodeIfPresent(String.self, forKey: .outputType) ?? "mel"
    vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 6561
    inputFrameRate = try container.decodeIfPresent(Int.self, forKey: .inputFrameRate) ?? 25
    onlyMaskLoss = try container.decodeIfPresent(Bool.self, forKey: .onlyMaskLoss) ?? true
    tokenMelRatio = try container.decodeIfPresent(Int.self, forKey: .tokenMelRatio) ?? 2
    preLookaheadLen = try container.decodeIfPresent(Int.self, forKey: .preLookaheadLen) ?? 3
    nTimesteps = try container.decodeIfPresent(Int.self, forKey: .nTimesteps) ?? 10
    preLookaheadChannels = try container.decodeIfPresent(Int.self, forKey: .preLookaheadChannels) ?? 512
    dit = try container.decodeIfPresent(DiTConfig.self, forKey: .dit) ?? DiTConfig()
    cfmSigmaMin = try container.decodeIfPresent(Float.self, forKey: .cfmSigmaMin) ?? 1e-6
    cfmTScheduler = try container.decodeIfPresent(String.self, forKey: .cfmTScheduler) ?? "cosine"
    cfmInferenceCfgRate = try container.decodeIfPresent(Float.self, forKey: .cfmInferenceCfgRate) ?? 0.7
  }
}

/// Configuration for Causal HiFi-GAN vocoder (CosyVoice3 24kHz)
public struct CosyVoice3HiFiGANConfig: Codable, Sendable {
  public var inChannels: Int = 80
  public var baseChannels: Int = 512
  public var nbHarmonics: Int = 8
  public var samplingRate: Int = 24000
  public var nsfAlpha: Float = 0.1
  public var nsfSigma: Float = 0.003
  public var nsfVoicedThreshold: Float = 10.0
  public var upsampleRates: [Int] = [8, 5, 3]
  public var upsampleKernelSizes: [Int] = [16, 11, 7]
  public var istftNFft: Int = 16
  public var istftHopLen: Int = 4
  public var resblockKernelSizes: [Int] = [3, 7, 11]
  public var resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
  // Note: Fun-CosyVoice3 uses 3 source resblocks, matching upsample stages
  public var sourceResblockKernelSizes: [Int] = [7, 7, 11]
  public var sourceResblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
  // Causal-specific config
  public var convPreLookRight: Int = 4
  public var causal: Bool = true

  enum CodingKeys: String, CodingKey {
    case inChannels = "in_channels"
    case baseChannels = "base_channels"
    case nbHarmonics = "nb_harmonics"
    case samplingRate = "sampling_rate"
    case nsfAlpha = "nsf_alpha"
    case nsfSigma = "nsf_sigma"
    case nsfVoicedThreshold = "nsf_voiced_threshold"
    case upsampleRates = "upsample_rates"
    case upsampleKernelSizes = "upsample_kernel_sizes"
    case istftNFft = "istft_n_fft"
    case istftHopLen = "istft_hop_len"
    case resblockKernelSizes = "resblock_kernel_sizes"
    case resblockDilationSizes = "resblock_dilation_sizes"
    case sourceResblockKernelSizes = "source_resblock_kernel_sizes"
    case sourceResblockDilationSizes = "source_resblock_dilation_sizes"
    case convPreLookRight = "conv_pre_look_right"
    case causal
  }

  public init() {}

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 80
    baseChannels = try container.decodeIfPresent(Int.self, forKey: .baseChannels) ?? 512
    nbHarmonics = try container.decodeIfPresent(Int.self, forKey: .nbHarmonics) ?? 8
    samplingRate = try container.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
    nsfAlpha = try container.decodeIfPresent(Float.self, forKey: .nsfAlpha) ?? 0.1
    nsfSigma = try container.decodeIfPresent(Float.self, forKey: .nsfSigma) ?? 0.003
    nsfVoicedThreshold = try container.decodeIfPresent(Float.self, forKey: .nsfVoicedThreshold) ?? 10.0
    upsampleRates = try container.decodeIfPresent([Int].self, forKey: .upsampleRates) ?? [8, 5, 3]
    upsampleKernelSizes = try container.decodeIfPresent([Int].self, forKey: .upsampleKernelSizes) ?? [16, 11, 7]
    istftNFft = try container.decodeIfPresent(Int.self, forKey: .istftNFft) ?? 16
    istftHopLen = try container.decodeIfPresent(Int.self, forKey: .istftHopLen) ?? 4
    resblockKernelSizes = try container.decodeIfPresent([Int].self, forKey: .resblockKernelSizes) ?? [3, 7, 11]
    resblockDilationSizes = try container.decodeIfPresent([[Int]].self, forKey: .resblockDilationSizes) ?? [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    sourceResblockKernelSizes = try container.decodeIfPresent([Int].self, forKey: .sourceResblockKernelSizes) ?? [7, 7, 11]
    sourceResblockDilationSizes = try container.decodeIfPresent([[Int]].self, forKey: .sourceResblockDilationSizes) ?? [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    convPreLookRight = try container.decodeIfPresent(Int.self, forKey: .convPreLookRight) ?? 4
    causal = try container.decodeIfPresent(Bool.self, forKey: .causal) ?? true
  }
}

/// Full configuration for CosyVoice3 model
public struct CosyVoice3Config: Codable, Sendable {
  public var llm: CosyVoice3LLMConfig = .init()
  public var flow: CosyVoice3FlowConfig = .init()
  public var hifigan: CosyVoice3HiFiGANConfig = .init()

  // Model paths
  public var llmPath: String?
  public var flowPath: String?
  public var hifiganPath: String?

  // Generation defaults
  public var defaultSampling: Int = 25
  public var maxTokenTextRatio: Float = 20.0
  public var minTokenTextRatio: Float = 2.0

  enum CodingKeys: String, CodingKey {
    case llm
    case flow
    case hifigan
    case llmPath = "llm_path"
    case flowPath = "flow_path"
    case hifiganPath = "hifigan_path"
    case defaultSampling = "default_sampling"
    case maxTokenTextRatio = "max_token_text_ratio"
    case minTokenTextRatio = "min_token_text_ratio"
  }

  public init() {}

  /// Load configuration from a pretrained model directory
  public static func fromPretrained(modelPath: String) throws -> CosyVoice3Config {
    let configPath = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
    let data = try Data(contentsOf: configPath)

    // Custom decoding to handle nested dit/estimator configs
    let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]

    var config = CosyVoice3Config()

    // Parse LLM config
    if let llmDict = json["llm"] as? [String: Any] {
      let llmData = try JSONSerialization.data(withJSONObject: llmDict)
      config.llm = try JSONDecoder().decode(CosyVoice3LLMConfig.self, from: llmData)
    }

    // Parse Flow config with nested DiT
    if let flowDict = json["flow"] as? [String: Any] {
      var flatFlow = flowDict

      // Extract DiT config from dit or estimator key
      if let ditDict = flowDict["dit"] as? [String: Any] ?? flowDict["estimator"] as? [String: Any] {
        let ditData = try JSONSerialization.data(withJSONObject: ditDict)
        config.flow.dit = try JSONDecoder().decode(DiTConfig.self, from: ditData)
        flatFlow.removeValue(forKey: "dit")
        flatFlow.removeValue(forKey: "estimator")
      }

      let flowData = try JSONSerialization.data(withJSONObject: flatFlow)
      let flowBase = try JSONDecoder().decode(CosyVoice3FlowConfig.self, from: flowData)
      // Preserve the DiT config we already parsed
      let ditConfig = config.flow.dit
      config.flow = flowBase
      config.flow.dit = ditConfig
    }

    // Parse HiFi-GAN config (may be named "hift" in config.json)
    if let hifiganDict = json["hifigan"] as? [String: Any] ?? json["hift"] as? [String: Any] {
      let hifiganData = try JSONSerialization.data(withJSONObject: hifiganDict)
      config.hifigan = try JSONDecoder().decode(CosyVoice3HiFiGANConfig.self, from: hifiganData)
    }

    return config
  }
}

/// Constants used throughout the CosyVoice3 model
public enum CosyVoice3Constants {
  /// Output sample rate (24kHz for CosyVoice3)
  public static let sampleRate: Int = 24000

  /// S3 tokenizer sample rate (16kHz)
  public static let s3TokenizerRate: Int = 16000

  /// Speech token vocabulary size (FSQ 3^8)
  public static let speechTokenSize: Int = 6561

  /// Extended vocabulary size for CosyVoice3
  public static let extendedVocabSize: Int = 200

  /// Total speech embedding size (speechTokenSize + extendedVocabSize)
  public static var totalSpeechVocabSize: Int { speechTokenSize + extendedVocabSize }

  /// Special token indices (CosyVoice3 unified in speech_embedding)
  public static var sosToken: Int { speechTokenSize + 0 } // 6561
  public static var eosToken: Int { speechTokenSize + 1 } // 6562
  public static var taskIdToken: Int { speechTokenSize + 2 } // 6563
  public static var fillToken: Int { speechTokenSize + 3 } // 6564

  /// Mel bins for flow matching
  public static let melBins: Int = 80

  /// Mel bins for S3 tokenizer
  public static let s3MelBins: Int = 128

  /// CAMPlus speaker embedding dimension
  public static let speakerEmbedDim: Int = 192

  /// Qwen2 hidden size
  public static let qwen2HiddenSize: Int = 896

  /// Number of Qwen2 transformer layers
  public static let qwen2NumLayers: Int = 24

  /// Token to mel ratio (mel_len = token_len * 2)
  public static let tokenMelRatio: Int = 2

  /// Default HuggingFace repository ID for CosyVoice3
  public static let defaultRepoId = "mlx-community/Fun-CosyVoice3-0.5B-2512-4bit"
}

/// Model configuration for CosyVoice3 (compatible with generate API)
public struct CosyVoice3ModelConfig: Codable, Sendable {
  public var modelType: String = "cosyvoice3"
  public var sampleRate: Int = 24000
  public var modelPath: String?

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case sampleRate = "sample_rate"
    case modelPath = "model_path"
  }

  public init() {}

  public init(modelType: String = "cosyvoice3", sampleRate: Int = 24000, modelPath: String? = nil) {
    self.modelType = modelType
    self.sampleRate = sampleRate
    self.modelPath = modelPath
  }

  public static func fromDict(_ config: [String: Any]) -> CosyVoice3ModelConfig {
    CosyVoice3ModelConfig(
      modelType: config["model_type"] as? String ?? "cosyvoice3",
      sampleRate: config["sample_rate"] as? Int ?? 24000,
      modelPath: config["model_path"] as? String
    )
  }
}
