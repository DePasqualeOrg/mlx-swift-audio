// CosyVoice2 configuration structs
// Ported from mlx-audio-plus cosyvoice2/config.py

import Foundation

/// Configuration for Qwen2-based LLM
struct LLMConfig: Codable, Sendable {
  var llmInputSize: Int = 896
  var llmOutputSize: Int = 896
  var speechTokenSize: Int = 6561
  var mixRatio: [Int] = [5, 15]

  // Qwen2 model config
  var hiddenSize: Int = 896
  var numHiddenLayers: Int = 24
  var intermediateSize: Int = 4864
  var numAttentionHeads: Int = 14
  var numKeyValueHeads: Int = 2
  var rmsNormEps: Float = 1e-6
  var vocabSize: Int = 151_936

  enum CodingKeys: String, CodingKey {
    case llmInputSize = "llm_input_size"
    case llmOutputSize = "llm_output_size"
    case speechTokenSize = "speech_token_size"
    case mixRatio = "mix_ratio"
    case hiddenSize = "hidden_size"
    case numHiddenLayers = "num_hidden_layers"
    case intermediateSize = "intermediate_size"
    case numAttentionHeads = "num_attention_heads"
    case numKeyValueHeads = "num_key_value_heads"
    case rmsNormEps = "rms_norm_eps"
    case vocabSize = "vocab_size"
  }

  init() {}

  init(
    llmInputSize: Int = 896,
    llmOutputSize: Int = 896,
    speechTokenSize: Int = 6561,
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
    self.mixRatio = mixRatio
    self.hiddenSize = hiddenSize
    self.numHiddenLayers = numHiddenLayers
    self.intermediateSize = intermediateSize
    self.numAttentionHeads = numAttentionHeads
    self.numKeyValueHeads = numKeyValueHeads
    self.rmsNormEps = rmsNormEps
    self.vocabSize = vocabSize
  }
}

/// Configuration for Flow Matching module
struct FlowConfig: Codable, Sendable {
  var inputSize: Int = 512
  var outputSize: Int = 80
  var spkEmbedDim: Int = 192
  var outputType: String = "mel"
  var vocabSize: Int = 6561
  var inputFrameRate: Int = 25
  var onlyMaskLoss: Bool = true
  var tokenMelRatio: Int = 2
  var preLookaheadLen: Int = 3
  var nTimesteps: Int = 10

  // Encoder config
  var encoderInputSize: Int = 512
  var encoderOutputSize: Int = 512
  var encoderAttentionHeads: Int = 8
  var encoderLinearUnits: Int = 2048
  var encoderNumBlocks: Int = 6
  var encoderNumUpBlocks: Int = 4
  var encoderDropoutRate: Float = 0.1
  var encoderPositionalDropoutRate: Float = 0.1
  var encoderAttentionDropoutRate: Float = 0.1
  var encoderNormalizeBefore: Bool = true
  var encoderMacaronStyle: Bool = false
  var encoderUseCnnModule: Bool = false
  var encoderCnnModuleKernel: Int = 15
  var encoderCausal: Bool = true
  var encoderUpsampleStride: Int = 2
  var encoderStaticChunkSize: Int = 25
  var encoderPosEncLayerType: String = "rel_pos_espnet"

  // Decoder config
  var decoderInChannels: Int = 320
  var decoderOutChannel: Int = 80
  var decoderChannels: [Int] = [256]
  var decoderDropout: Float = 0.0
  var decoderAttentionHeadDim: Int = 64
  var decoderNBlocks: Int = 4
  var decoderNumMidBlocks: Int = 12
  var decoderNumHeads: Int = 8
  var decoderActFn: String = "gelu"
  var decoderStaticChunkSize: Int = 50
  var decoderNumDecodingLeftChunks: Int = -1

  // CFM params
  var cfmInChannels: Int = 240
  var cfmSigmaMin: Float = 1e-6
  var cfmTScheduler: String = "cosine"
  var cfmInferenceCfgRate: Float = 0.7

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
    case encoderInputSize = "encoder_input_size"
    case encoderOutputSize = "encoder_output_size"
    case encoderAttentionHeads = "encoder_attention_heads"
    case encoderLinearUnits = "encoder_linear_units"
    case encoderNumBlocks = "encoder_num_blocks"
    case encoderNumUpBlocks = "encoder_num_up_blocks"
    case encoderDropoutRate = "encoder_dropout_rate"
    case encoderPositionalDropoutRate = "encoder_positional_dropout_rate"
    case encoderAttentionDropoutRate = "encoder_attention_dropout_rate"
    case encoderNormalizeBefore = "encoder_normalize_before"
    case encoderMacaronStyle = "encoder_macaron_style"
    case encoderUseCnnModule = "encoder_use_cnn_module"
    case encoderCnnModuleKernel = "encoder_cnn_module_kernel"
    case encoderCausal = "encoder_causal"
    case encoderUpsampleStride = "encoder_upsample_stride"
    case encoderStaticChunkSize = "encoder_static_chunk_size"
    case encoderPosEncLayerType = "encoder_pos_enc_layer_type"
    case decoderInChannels = "decoder_in_channels"
    case decoderOutChannel = "decoder_out_channel"
    case decoderChannels = "decoder_channels"
    case decoderDropout = "decoder_dropout"
    case decoderAttentionHeadDim = "decoder_attention_head_dim"
    case decoderNBlocks = "decoder_n_blocks"
    case decoderNumMidBlocks = "decoder_num_mid_blocks"
    case decoderNumHeads = "decoder_num_heads"
    case decoderActFn = "decoder_act_fn"
    case decoderStaticChunkSize = "decoder_static_chunk_size"
    case decoderNumDecodingLeftChunks = "decoder_num_decoding_left_chunks"
    case cfmInChannels = "cfm_in_channels"
    case cfmSigmaMin = "cfm_sigma_min"
    case cfmTScheduler = "cfm_t_scheduler"
    case cfmInferenceCfgRate = "cfm_inference_cfg_rate"
  }

  init() {}
}

/// Configuration for HiFi-GAN vocoder (CosyVoice2 24kHz)
struct HiFiGANConfig: Codable, Sendable {
  var inChannels: Int = 80
  var baseChannels: Int = 512
  var nbHarmonics: Int = 8
  var samplingRate: Int = 24000
  var nsfAlpha: Float = 0.1
  var nsfSigma: Float = 0.003
  var nsfVoicedThreshold: Float = 10.0
  var upsampleRates: [Int] = [8, 5, 3]
  var upsampleKernelSizes: [Int] = [16, 11, 7]
  var istftNFft: Int = 16
  var istftHopLen: Int = 4
  var resblockKernelSizes: [Int] = [3, 7, 11]
  var resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
  var sourceResblockKernelSizes: [Int] = [7, 7, 11]
  var sourceResblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
  var useInterpolation: Bool = true

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
    case useInterpolation = "use_interpolation"
  }

  init() {}
}

/// Full configuration for CosyVoice2 model
struct CosyVoice2Config: Codable, Sendable {
  var llm: LLMConfig = .init()
  var flow: FlowConfig = .init()
  var hifigan: HiFiGANConfig = .init()

  // Model paths
  var llmPath: String?
  var flowPath: String?
  var hifiganPath: String?

  // Generation defaults
  var defaultSampling: Int = 25
  var maxTokenTextRatio: Float = 20.0
  var minTokenTextRatio: Float = 2.0

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

  init() {}

  /// Load configuration from a pretrained model directory
  static func fromPretrained(modelPath: String) throws -> CosyVoice2Config {
    let configPath = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
    let data = try Data(contentsOf: configPath)

    // Custom decoding to handle nested encoder/decoder configs
    let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]

    var config = CosyVoice2Config()

    // Parse LLM config
    if let llmDict = json["llm"] as? [String: Any] {
      let llmData = try JSONSerialization.data(withJSONObject: llmDict)
      config.llm = try JSONDecoder().decode(LLMConfig.self, from: llmData)
    }

    // Parse Flow config with flattening of nested encoder/decoder
    if let flowDict = json["flow"] as? [String: Any] {
      var flatFlow: [String: Any] = [:]

      for (key, value) in flowDict {
        if key == "encoder", let encoderDict = value as? [String: Any] {
          for (ek, ev) in encoderDict {
            flatFlow["encoder_\(ek)"] = ev
          }
        } else if key == "decoder", let decoderDict = value as? [String: Any] {
          for (dk, dv) in decoderDict {
            if dk == "out_channels" {
              flatFlow["decoder_out_channel"] = dv
            } else {
              flatFlow["decoder_\(dk)"] = dv
            }
          }
        } else {
          flatFlow[key] = value
        }
      }

      let flowData = try JSONSerialization.data(withJSONObject: flatFlow)
      config.flow = try JSONDecoder().decode(FlowConfig.self, from: flowData)
    }

    // Parse HiFi-GAN config (may be named "hift" in config.json)
    if let hifiganDict = json["hifigan"] as? [String: Any] ?? json["hift"] as? [String: Any] {
      let hifiganData = try JSONSerialization.data(withJSONObject: hifiganDict)
      config.hifigan = try JSONDecoder().decode(HiFiGANConfig.self, from: hifiganData)
    }

    return config
  }
}

/// Model configuration for CosyVoice2 (compatible with generate API)
struct CosyVoice2ModelConfig: Codable, Sendable {
  var modelType: String = "cosyvoice2"
  var sampleRate: Int = 24000
  var modelPath: String?

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case sampleRate = "sample_rate"
    case modelPath = "model_path"
  }

  init() {}

  init(modelType: String = "cosyvoice2", sampleRate: Int = 24000, modelPath: String? = nil) {
    self.modelType = modelType
    self.sampleRate = sampleRate
    self.modelPath = modelPath
  }

  static func fromDict(_ config: [String: Any]) -> CosyVoice2ModelConfig {
    CosyVoice2ModelConfig(
      modelType: config["model_type"] as? String ?? "cosyvoice2",
      sampleRate: config["sample_rate"] as? Int ?? 24000,
      modelPath: config["model_path"] as? String
    )
  }
}

/// Constants used throughout the CosyVoice2 model
enum CosyVoice2Constants {
  /// Output sample rate (24kHz for CosyVoice2)
  static let sampleRate: Int = 24000

  /// S3 tokenizer sample rate (16kHz)
  static let s3TokenizerRate: Int = 16000

  /// Speech token vocabulary size (FSQ 3^8)
  static let speechTokenSize: Int = 6561

  /// Number of special tokens (sos/eos, task_id, fill_token)
  static let numSpecialTokens: Int = 3

  /// Mel bins for flow matching
  static let melBins: Int = 80

  /// Mel bins for S3 tokenizer
  static let s3MelBins: Int = 128

  /// CAMPlus speaker embedding dimension
  static let speakerEmbedDim: Int = 192

  /// Qwen2 hidden size
  static let qwen2HiddenSize: Int = 896

  /// Number of Qwen2 transformer layers
  static let qwen2NumLayers: Int = 24

  /// Token to mel ratio (mel_len = token_len * 2)
  static let tokenMelRatio: Int = 2
}
