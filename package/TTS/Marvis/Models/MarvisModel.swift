import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Configs

struct DepthDecoderConfig: Codable, Sendable {
  let attentionBias: Bool
  let attentionDropout: Double
  let backboneHiddenSize: Int
  let headDim: Int
  let hiddenAct: String
  let hiddenSize: Int
  let initializerRange: Double
  let intermediateSize: Int
  let maxPositionEmbeddings: Int
  let mlpBias: Bool
  let modelType: String
  let numAttentionHeads: Int
  let numCodebooks: Int
  let numHiddenLayers: Int
  let numKeyValueHeads: Int
  let rmsNormEps: Double
  let ropeScaling: [String: StringOrNumber]?
  let ropeTheta: Int
  let useCache: Bool
  let vocabSize: Int

  init(
    attentionBias: Bool,
    attentionDropout: Double,
    backboneHiddenSize: Int,
    headDim: Int,
    hiddenAct: String,
    hiddenSize: Int,
    initializerRange: Double,
    intermediateSize: Int,
    maxPositionEmbeddings: Int,
    mlpBias: Bool,
    modelType: String,
    numAttentionHeads: Int,
    numCodebooks: Int,
    numHiddenLayers: Int,
    numKeyValueHeads: Int,
    rmsNormEps: Double,
    ropeScaling: [String: StringOrNumber]?,
    ropeTheta: Int,
    useCache: Bool,
    vocabSize: Int,
  ) {
    self.attentionBias = attentionBias
    self.attentionDropout = attentionDropout
    self.backboneHiddenSize = backboneHiddenSize
    self.headDim = headDim
    self.hiddenAct = hiddenAct
    self.hiddenSize = hiddenSize
    self.initializerRange = initializerRange
    self.intermediateSize = intermediateSize
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.mlpBias = mlpBias
    self.modelType = modelType
    self.numAttentionHeads = numAttentionHeads
    self.numCodebooks = numCodebooks
    self.numHiddenLayers = numHiddenLayers
    self.numKeyValueHeads = numKeyValueHeads
    self.rmsNormEps = rmsNormEps
    self.ropeScaling = ropeScaling
    self.ropeTheta = ropeTheta
    self.useCache = useCache
    self.vocabSize = vocabSize
  }

  private enum CodingKeys: String, CodingKey {
    case attentionBias = "attention_bias"
    case attentionDropout = "attention_dropout"
    case backboneHiddenSize = "backbone_hidden_size"
    case headDim = "head_dim"
    case hiddenAct = "hidden_act"
    case hiddenSize = "hidden_size"
    case initializerRange = "initializer_range"
    case intermediateSize = "intermediate_size"
    case maxPositionEmbeddings = "max_position_embeddings"
    case mlpBias = "mlp_bias"
    case modelType = "model_type"
    case numAttentionHeads = "num_attention_heads"
    case numCodebooks = "num_codebooks"
    case numHiddenLayers = "num_hidden_layers"
    case numKeyValueHeads = "num_key_value_heads"
    case rmsNormEps = "rms_norm_eps"
    case ropeScaling = "rope_scaling"
    case ropeTheta = "rope_theta"
    case useCache = "use_cache"
    case vocabSize = "vocab_size"
  }
}

struct MarvisConfig: Codable, Sendable {
  let modelType: String
  let backboneFlavor: String?
  let decoderFlavor: String?
  let textVocabSize: Int
  let audioVocabSize: Int
  let audioNumCodebooks: Int
  let attentionBias: Bool
  let attentionDropout: Double
  let audioEosTokenId: Int
  let audioTokenId: Int
  let bosTokenId: Int
  let codebookEosTokenId: Int
  let codebookPadTokenId: Int
  let depthDecoderConfig: DepthDecoderConfig?
  let headDim: Int
  let hiddenAct: String
  let hiddenSize: Int
  let initializerRange: Double
  let intermediateSize: Int
  let maxPositionEmbeddings: Int
  let mlpBias: Bool
  let numAttentionHeads: Int
  let numCodebooks: Int
  let numHiddenLayers: Int
  let numKeyValueHeads: Int
  let padTokenId: Int
  let rmsNormEps: Double
  let ropeScaling: [String: StringOrNumber]?
  let ropeTheta: Int
  let tieCodebooksEmbeddings: Bool
  let tieWordEmbeddings: Bool
  let useCache: Bool
  let vocabSize: Int
  let quantization: [String: StringOrNumber]?

  init(
    modelType: String,
    backboneFlavor: String?,
    decoderFlavor: String?,
    textVocabSize: Int,
    audioVocabSize: Int,
    audioNumCodebooks: Int,
    attentionBias: Bool,
    attentionDropout: Double,
    audioEosTokenId: Int,
    audioTokenId: Int,
    bosTokenId: Int,
    codebookEosTokenId: Int,
    codebookPadTokenId: Int,
    depthDecoderConfig: DepthDecoderConfig?,
    headDim: Int,
    hiddenAct: String,
    hiddenSize: Int,
    initializerRange: Double,
    intermediateSize: Int,
    maxPositionEmbeddings: Int,
    mlpBias: Bool,
    numAttentionHeads: Int,
    numCodebooks: Int,
    numHiddenLayers: Int,
    numKeyValueHeads: Int,
    padTokenId: Int,
    rmsNormEps: Double,
    ropeScaling: [String: StringOrNumber]?,
    ropeTheta: Int,
    tieCodebooksEmbeddings: Bool,
    tieWordEmbeddings: Bool,
    useCache: Bool,
    vocabSize: Int,
    quantization: [String: StringOrNumber]?,
  ) {
    self.modelType = modelType
    self.backboneFlavor = backboneFlavor
    self.decoderFlavor = decoderFlavor
    self.textVocabSize = textVocabSize
    self.audioVocabSize = audioVocabSize
    self.audioNumCodebooks = audioNumCodebooks
    self.attentionBias = attentionBias
    self.attentionDropout = attentionDropout
    self.audioEosTokenId = audioEosTokenId
    self.audioTokenId = audioTokenId
    self.bosTokenId = bosTokenId
    self.codebookEosTokenId = codebookEosTokenId
    self.codebookPadTokenId = codebookPadTokenId
    self.depthDecoderConfig = depthDecoderConfig
    self.headDim = headDim
    self.hiddenAct = hiddenAct
    self.hiddenSize = hiddenSize
    self.initializerRange = initializerRange
    self.intermediateSize = intermediateSize
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.mlpBias = mlpBias
    self.numAttentionHeads = numAttentionHeads
    self.numCodebooks = numCodebooks
    self.numHiddenLayers = numHiddenLayers
    self.numKeyValueHeads = numKeyValueHeads
    self.padTokenId = padTokenId
    self.rmsNormEps = rmsNormEps
    self.ropeScaling = ropeScaling
    self.ropeTheta = ropeTheta
    self.tieCodebooksEmbeddings = tieCodebooksEmbeddings
    self.tieWordEmbeddings = tieWordEmbeddings
    self.useCache = useCache
    self.vocabSize = vocabSize
    self.quantization = quantization
  }

  private enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case backboneFlavor = "backbone_flavor"
    case decoderFlavor = "decoder_flavor"
    case textVocabSize = "text_vocab_size"
    case audioVocabSize = "audio_vocab_size"
    case audioNumCodebooks = "audio_num_codebooks"
    case attentionBias = "attention_bias"
    case attentionDropout = "attention_dropout"
    case audioEosTokenId = "audio_eos_token_id"
    case audioTokenId = "audio_token_id"
    case bosTokenId = "bos_token_id"
    case codebookEosTokenId = "codebook_eos_token_id"
    case codebookPadTokenId = "codebook_pad_token_id"
    case depthDecoderConfig = "depth_decoder_config"
    case headDim = "head_dim"
    case hiddenAct = "hidden_act"
    case hiddenSize = "hidden_size"
    case initializerRange = "initializer_range"
    case intermediateSize = "intermediate_size"
    case maxPositionEmbeddings = "max_position_embeddings"
    case mlpBias = "mlp_bias"
    case numAttentionHeads = "num_attention_heads"
    case numCodebooks = "num_codebooks"
    case numHiddenLayers = "num_hidden_layers"
    case numKeyValueHeads = "num_key_value_heads"
    case padTokenId = "pad_token_id"
    case rmsNormEps = "rms_norm_eps"
    case ropeScaling = "rope_scaling"
    case ropeTheta = "rope_theta"
    case tieCodebooksEmbeddings = "tie_codebooks_embeddings"
    case tieWordEmbeddings = "tie_word_embeddings"
    case useCache = "use_cache"
    case vocabSize = "vocab_size"
    case quantization
  }
}

extension MarvisConfig {
  static func load(from data: Data) throws -> MarvisConfig {
    let dec = JSONDecoder()
    dec.keyDecodingStrategy = .useDefaultKeys
    return try dec.decode(MarvisConfig.self, from: data)
  }

  static func load(from url: URL) throws -> MarvisConfig {
    try load(from: Data(contentsOf: url))
  }
}

func createMarvisBackboneConfigForBackbone(_ config: MarvisConfig) -> MarvisBackboneConfig {
  MarvisBackboneConfig(
    hiddenSize: config.hiddenSize,
    hiddenLayers: config.numHiddenLayers,
    intermediateSize: config.intermediateSize,
    attentionHeads: config.numAttentionHeads,
    headDimensions: config.headDim,
    rmsNormEps: Float(config.rmsNormEps),
    vocabularySize: config.textVocabSize,
    kvHeads: config.numKeyValueHeads,
    maxPositionEmbeddings: config.maxPositionEmbeddings,
    ropeTheta: Float(config.ropeTheta),
    ropeTraditional: false,
    ropeScaling: config.ropeScaling,
    tieWordEmbeddings: config.tieWordEmbeddings,
    attentionBias: config.attentionBias,
    mlpBias: config.mlpBias,
  )
}

func createMarvisBackboneConfigForDecoder(_ d: DepthDecoderConfig) -> MarvisBackboneConfig {
  MarvisBackboneConfig(
    hiddenSize: d.hiddenSize,
    hiddenLayers: d.numHiddenLayers,
    intermediateSize: d.intermediateSize,
    attentionHeads: d.numAttentionHeads,
    headDimensions: d.headDim,
    rmsNormEps: Float(d.rmsNormEps),
    vocabularySize: d.vocabSize,
    kvHeads: d.numKeyValueHeads,
    maxPositionEmbeddings: d.maxPositionEmbeddings,
    ropeTheta: Float(d.ropeTheta),
    ropeTraditional: false,
    ropeScaling: d.ropeScaling,
    tieWordEmbeddings: true,
    attentionBias: d.attentionBias,
    mlpBias: d.mlpBias,
  )
}

func createMarvisBackboneConfig(flavor: String) throws -> MarvisBackboneConfig {
  switch flavor {
    case "llama-1B":
      return MarvisBackboneConfig(
        hiddenSize: 2048,
        hiddenLayers: 16,
        intermediateSize: 8192,
        attentionHeads: 32,
        headDimensions: 64,
        rmsNormEps: 1e-5,
        vocabularySize: 128_256,
        kvHeads: 8,
        maxPositionEmbeddings: 2048,
        ropeTheta: 500_000,
        ropeTraditional: false,
        ropeScaling: [
          "factor": .float(32.0),
          "low_freq_factor": .float(1.0),
          "high_freq_factor": .float(4.0),
          "original_max_position_embeddings": .float(8192.0),
          "rope_type": .string("llama3"),
        ],
        tieWordEmbeddings: true,
        attentionBias: false,
        mlpBias: false,
      )

    case "llama-100M":
      return MarvisBackboneConfig(
        hiddenSize: 1024,
        hiddenLayers: 4,
        intermediateSize: 8192,
        attentionHeads: 8,
        headDimensions: 128,
        rmsNormEps: 1e-5,
        vocabularySize: 128_256,
        kvHeads: 2,
        maxPositionEmbeddings: 2048,
        ropeTheta: 500_000,
        ropeTraditional: false,
        ropeScaling: [
          "factor": .float(32.0),
          "low_freq_factor": .float(1.0),
          "high_freq_factor": .float(4.0),
          "original_max_position_embeddings": .float(8192.0),
          "rope_type": .string("llama3"),
        ],
        tieWordEmbeddings: true,
        attentionBias: false,
        mlpBias: false,
      )

    default:
      struct UnknownFlavor: Error {}
      throw UnknownFlavor()
  }
}

// MARK: - Model

final class MarvisModel: Module {
  let args: MarvisConfig

  @ModuleInfo var backbone: MarvisBackbone
  @ModuleInfo var decoder: MarvisBackbone

  @ModuleInfo(key: "text_embeddings") var textEmbeddings: Embedding
  @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: Embedding
  @ModuleInfo var projection: Linear // backbone_dim -> decoder_dim
  @ModuleInfo(key: "codebook0_head") var codebook0Head: Linear // logits for codebook 0
  @ModuleInfo(key: "audio_head") var audioHead: MLXArray // [nq-1, decoder_dim, audio_vocab]

  private var backboneCausalMask: MLXArray?
  private var decoderCausalMask: MLXArray?

  var backboneCache: [KVCache]?
  var decoderCache: [KVCache]?
  var cachesEnabled: Bool = false

  init(config: MarvisConfig) throws {
    args = config

    let backboneConfig: MarvisBackboneConfig
    let decoderConfig: MarvisBackboneConfig
    if let depth = config.depthDecoderConfig {
      backboneConfig = createMarvisBackboneConfigForBackbone(config)
      decoderConfig = createMarvisBackboneConfigForDecoder(depth)
    } else {
      guard let backboneFlavor = config.backboneFlavor, let decoderFlavor = config.decoderFlavor else {
        fatalError("Either depthDecoderConfig or both backboneFlavor and decoderFlavor must be provided")
      }
      do {
        backboneConfig = try createMarvisBackboneConfig(flavor: backboneFlavor)
        decoderConfig = try createMarvisBackboneConfig(flavor: decoderFlavor)
      } catch {
        fatalError("Failed to create MarvisBackboneConfig: \(error). Backbone flavor: \(backboneFlavor), Decoder flavor: \(decoderFlavor)")
      }
    }

    _backbone.wrappedValue = MarvisBackbone(backboneConfig)
    _decoder.wrappedValue = MarvisBackbone(decoderConfig)

    let backboneDim = backboneConfig.hiddenSize
    let decoderDim = decoderConfig.hiddenSize

    _textEmbeddings.wrappedValue = Embedding(embeddingCount: args.textVocabSize, dimensions: backboneDim)
    let audioVocabCombined = args.audioVocabSize * args.audioNumCodebooks
    _audioEmbeddings.wrappedValue = Embedding(embeddingCount: audioVocabCombined, dimensions: backboneDim)

    _projection.wrappedValue = Linear(backboneDim, decoderDim, bias: false)
    _codebook0Head.wrappedValue = Linear(backboneDim, args.audioVocabSize, bias: false)

    let restCodebooks = max(args.audioNumCodebooks - 1, 0)
    _audioHead.wrappedValue = MLXArray.zeros([restCodebooks, decoderDim, args.audioVocabSize])

    backboneCache = nil
    decoderCache = nil
    cachesEnabled = false
  }

  func cachesAreEnabled() -> Bool { cachesEnabled }

  func resetCaches() throws {
    let backboneConfig: MarvisBackboneConfig
    let decoderConfig: MarvisBackboneConfig

    if let depth = args.depthDecoderConfig {
      backboneConfig = createMarvisBackboneConfigForBackbone(args)
      decoderConfig = createMarvisBackboneConfigForDecoder(depth)
    } else {
      guard let backboneFlavor = args.backboneFlavor, let decoderFlavor = args.decoderFlavor else {
        fatalError("Either depthDecoderConfig or both backboneFlavor and decoderFlavor must be provided")
      }
      do {
        backboneConfig = try createMarvisBackboneConfig(flavor: backboneFlavor)
        decoderConfig = try createMarvisBackboneConfig(flavor: decoderFlavor)
      } catch {
        fatalError("Failed to create MarvisBackboneConfig: \(error). Backbone flavor: \(backboneFlavor), Decoder flavor: \(decoderFlavor)")
      }
    }

    backboneCache = (0 ..< backboneConfig.hiddenLayers).map { _ in KVCacheSimple() }
    decoderCache = (0 ..< decoderConfig.hiddenLayers).map { _ in KVCacheSimple() }
    cachesEnabled = true
  }

  func generateFrame(
    maxCodebooks: Int,
    tokens: MLXArray,
    tokensMask: MLXArray,
    sampler: (MLXArray) -> MLXArray,
  ) throws -> MLXArray {
    precondition(cachesEnabled, "backbone caches are not enabled")

    let embeds = _embedTokens(tokens) // [B, T, Cb+1, D]
    let masked = embeds * tokensMask.expandedDimensions(axis: -1) // [B, T, Cb+1, D]
    var h = sum(masked, axis: 2) // [B, T, D]

    h = backbone(h, cache: backboneCache) // [B, T, D]

    let B = h.shape[0]
    let dBackbone = h.shape[2]
    let lastT = h.shape[1] - 1
    let split1 = split(h, indices: [lastT], axis: 1)
    let lastSlice = split(split1[1], indices: [1], axis: 1)[0] // [B, 1, D]
    let lastH = lastSlice.reshaped([B, dBackbone]) // [B, D]

    let c0Logits = codebook0Head(lastH) // [B, vocab_audio]
    let c0SampleVec = sampler(c0Logits) // [B]
    let c0Sample = c0SampleVec.expandedDimensions(axis: -1) // [B, 1]
    let c0Embed = _embedAudio(codebook: 0, tokens: c0Sample) // [B, 1, dBackbone]

    let lastH3 = expandedDimensions(lastH, axis: 1) // [B, 1, dBackbone]
    var currH = concatenated([lastH3, c0Embed], axis: 1) // [B, 2, dBackbone]
    var currSample = c0Sample // [B, 1]

    // TODO: Use MLX.arange after next mlx-swift release (see MLXArray+Extensions.swift)
    let basePos = MLXArray.arange(2).reshaped([1, 2])
    var currPos = repeated(basePos, count: B, axis: 0) // [B, 2]

    let decoderConfig: MarvisBackboneConfig
    if let depth = args.depthDecoderConfig {
      decoderConfig = createMarvisBackboneConfigForDecoder(depth)
    } else {
      guard let decoderFlavor = args.decoderFlavor else {
        fatalError("Either depthDecoderConfig or decoderFlavor must be provided")
      }
      do {
        decoderConfig = try createMarvisBackboneConfig(flavor: decoderFlavor)
      } catch {
        fatalError("Failed to create MarvisBackboneConfig for decoder: \(error). Decoder flavor: \(decoderFlavor)")
      }
    }
    decoderCache = (0 ..< decoderConfig.hiddenLayers).map { _ in KVCacheSimple() }

    let codeBooks = min(args.audioNumCodebooks, maxCodebooks)
    if codeBooks > 1 {
      for i in 1 ..< codeBooks {
        let decH = decoder(projection(currH), cache: decoderCache) // [B, Tcur, dDec]

        let dDec = decH.shape[2]
        let lastSplit1 = split(decH, indices: [decH.shape[1] - 1], axis: 1)
        let lastDec = split(lastSplit1[1], indices: [1], axis: 1)[0].reshaped([B, dDec]) // [B, dDec]

        let Wi = take2DHead(audioHead, index: i - 1)
        let ciLogits = matmul(lastDec, Wi) // [B, vocab_audio]

        let ciSampleVec = sampler(ciLogits) // [B]
        let ciSample = expandedDimensions(ciSampleVec, axis: -1) // [B, 1]
        let ciEmbed = _embedAudio(codebook: i, tokens: ciSample) // [B, 1, dBackbone]

        currH = ciEmbed // [B, 1, dBackbone]
        currSample = concatenated([currSample, ciSample], axis: 1)
        currPos = split(currPos, indices: [1], axis: 1)[1] + MLXArray(1)
      }
    }

    return currSample // [B, codeBooks]
  }

  private func _embedAudio(codebook: Int, tokens: MLXArray) -> MLXArray {
    let offset = codebook * args.audioVocabSize
    let shifted = tokens + MLXArray(offset)
    return audioEmbeddings(shifted)
  }

  private func _embedTokens(_ tokens: MLXArray) -> MLXArray {
    let B = tokens.shape[0]
    let T = tokens.shape[1]
    let CbPlus = tokens.shape[2]
    let Cb = CbPlus - 1

    let split1 = split(tokens, indices: [Cb], axis: 2)
    let audioIds = split1[0] // [B, T, Cb]
    let textIds = split(split1[1], indices: [1], axis: 2)[0].reshaped([B, T]) // [B, T]

    var textEmb = textEmbeddings(textIds) // [B, T, D]
    textEmb = expandedDimensions(textEmb, axis: -2) // [B, T, 1, D]

    // TODO: Use MLX.arange after next mlx-swift release (see MLXArray+Extensions.swift)
    let cbIdx = MLXArray.arange(Cb) // [Cb]
    let cbOffsets = (cbIdx * MLXArray(Int32(args.audioVocabSize))).reshaped([1, 1, Cb])
    let shiftedAudioIds = audioIds + cbOffsets // [B, T, Cb]

    let flat = shiftedAudioIds.flattened() // [B*T*Cb]
    let audioFlatEmb = audioEmbeddings(flat) // [B*T*Cb, D]
    let D = audioFlatEmb.shape[1]
    let audioEmb = audioFlatEmb.reshaped([B, T, Cb, D]) // [B, T, Cb, D]

    return concatenated([audioEmb, textEmb], axis: 2) // [B, T, Cb+1, D]
  }

  private func take2DHead(_ W: MLXArray, index i: Int) -> MLXArray {
    if W.ndim == 3 {
      let left = split(W, indices: [i], axis: 0)
      let tail = split(left[1], indices: [1], axis: 0)
      return tail[0].reshaped([W.shape[1], W.shape[2]])
    }
    return W
  }
}
