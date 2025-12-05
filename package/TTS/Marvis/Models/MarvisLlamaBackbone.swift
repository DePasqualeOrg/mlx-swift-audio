import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Marvis-specific Llama3 Scaled RoPE

/// Llama3-style RoPE with manual cos/sin cache, specifically tuned for Marvis.
/// This implementation uses a different frequency convention than the standard MLXFast.RoPE.
private class Llama3ScaledRoPE {
  let dims: Int
  var maxSeqLen: Int
  let base: Float
  let scaleFactor: Float
  let lowFreqFactor: Float
  let highFreqFactor: Float
  let oldContextLen: Float

  private var theta: MLXArray?
  private var cache: MLXArray?
  private var isCacheBuilt = false

  init(
    dims: Int,
    maxSeqLen: Int = 2048,
    base: Float = 500_000.0,
    scaleFactor: Float = 32.0,
    lowFreqFactor: Float = 1.0,
    highFreqFactor: Float = 4.0,
    oldContextLen: Float = 8192.0,
  ) {
    precondition(dims % 2 == 0, "RoPE dims must be even")
    self.dims = dims
    self.maxSeqLen = maxSeqLen
    self.base = base
    self.scaleFactor = scaleFactor
    self.lowFreqFactor = lowFreqFactor
    self.highFreqFactor = highFreqFactor
    self.oldContextLen = oldContextLen

    ropeInit()
  }

  convenience init(dims: Int, config: MarvisBackboneConfig) {
    let base = config.ropeTheta
    let rs = config.ropeScaling
    func num(_ k: String, _ d: Float) -> Float {
      guard let v = rs?[k] else { return d }
      switch v {
        case let .float(x): return Float(x)
        case let .int(x): return Float(x)
        case let .string(s): return Float(s) ?? d
        default: return d
      }
    }
    self.init(
      dims: dims,
      maxSeqLen: config.maxPositionEmbeddings ?? 2048,
      base: base,
      scaleFactor: num("factor", 32.0),
      lowFreqFactor: num("low_freq_factor", 1.0),
      highFreqFactor: num("high_freq_factor", 4.0),
      oldContextLen: num("original_max_position_embeddings", 8192.0),
    )
  }

  func ropeInit() {
    let indices = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32)
    let exponents = indices / MLXArray(Float(dims))
    let freqs = MLX.pow(MLXArray(base), -exponents)

    let th = applyScaling(
      freqs: freqs,
      scaleFactor: scaleFactor,
      lowFreqFactor: lowFreqFactor,
      highFreqFactor: highFreqFactor,
      oldContextLen: oldContextLen,
    )
    theta = th
    buildRoPECache(maxSeqLen)
    isCacheBuilt = true
  }

  private func buildRoPECache(_ L: Int) {
    guard let th = theta else { return }
    let seqIdx = MLXArray(stride(from: 0, to: L, by: 1)).asType(th.dtype)
    let idxTheta = (seqIdx.reshaped([L, 1]) * th.reshaped([1, th.shape[0]])).asType(.float32)
    let cosT = cos(idxTheta)
    let sinT = sin(idxTheta)
    cache = stacked([cosT, sinT], axis: -1)
    maxSeqLen = L
  }

  private func applyScaling(
    freqs: MLXArray,
    scaleFactor: Float,
    lowFreqFactor: Float,
    highFreqFactor: Float,
    oldContextLen: Float,
  ) -> MLXArray {
    let twoPi = MLXArray(2.0 * Float.pi)
    let wavelens = twoPi / freqs

    let hiThr = MLXArray(oldContextLen / highFreqFactor)
    let loThr = MLXArray(oldContextLen / lowFreqFactor)

    let freqDiv = freqs / MLXArray(scaleFactor)

    let smoothRaw = (MLXArray(oldContextLen) / wavelens - MLXArray(lowFreqFactor))
      / MLXArray(highFreqFactor - lowFreqFactor)
    let smooth = clip(smoothRaw, min: 0.0, max: 1.0)
    let smoothBlend = (MLXArray(1.0) - smooth) * freqDiv + smooth * freqs

    let lessHi = wavelens .< hiThr
    let greaterLo = wavelens .> loThr
    let mid = MLX.where(greaterLo, freqDiv, smoothBlend)
    let out = MLX.where(lessHi, freqs, mid)
    return out.asType(freqs.dtype)
  }

  func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
    precondition(isCacheBuilt, "RoPE cache is not built. Call ropeInit() first.")
    guard var cache else { return x }

    let seqAxis = (x.ndim == 4) ? 2 : 1
    let seqLen = x.shape[seqAxis]
    let need = offset + seqLen
    if need > cache.shape[0] { buildRoPECache(need); cache = self.cache! }

    let start = max(offset, 0)
    let head = split(cache, indices: [start], axis: 0)[1]
    let seg = split(head, indices: [seqLen], axis: 0)[0]

    let pairs = dims / 2
    let xF = x.asType(.float32)
    let xShaped = xF.reshaped(Array(xF.shape.dropLast()) + [pairs, 2])

    var ropeShape = Array(repeating: 1, count: xShaped.ndim)
    ropeShape[seqAxis] = seqLen
    ropeShape[xShaped.ndim - 2] = pairs
    ropeShape[xShaped.ndim - 1] = 2
    let rope = seg.reshaped(ropeShape)

    func splitLast2(_ a: MLXArray) -> (MLXArray, MLXArray) {
      let p = split(a, indices: [1], axis: a.ndim - 1)
      return (p[0], p[1])
    }
    let (x0, x1) = splitLast2(xShaped)
    let (c, s) = splitLast2(rope)

    let y0 = x0 * c - x1 * s
    let y1 = x1 * c + x0 * s
    let y = stacked([y0, y1], axis: xShaped.ndim - 1)
    let out = y.reshaped(x.shape).asType(x.dtype)
    return out
  }
}

// MARK: - Attention

private class MarvisAttention: Module {
  let args: MarvisBackboneConfig
  let scale: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear

  let rope: Llama3ScaledRoPE

  init(_ args: MarvisBackboneConfig) {
    self.args = args

    let dim = args.hiddenSize
    let heads = args.attentionHeads
    let kvHeads = args.kvHeads

    let headDim = args.resolvedHeadDimensions
    scale = pow(Float(headDim), -0.5)

    _qProj.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
    _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
    _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
    _oProj.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

    rope = Llama3ScaledRoPE(dims: headDim, config: args)
  }

  func callAsFunction(
    _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
  ) -> MLXArray {
    let (B, L) = (x.dim(0), x.dim(1))

    var queries = qProj(x)
    var keys = kProj(x)
    var values = vProj(x)

    queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
    keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
    values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

    let offset = cache?.offset ?? 0
    queries = rope(queries, offset: offset)
    keys = rope(keys, offset: offset)

    if let cache {
      let (updatedKeys, updatedValues) = cache.update(keys: keys, values: values)
      let attnResult = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: updatedKeys,
        values: updatedValues,
        scale: scale,
        mask: mask,
      )
      let transposed = attnResult.transposed(0, 2, 1, 3)
      let output = transposed.reshaped([B, L, args.attentionHeads * args.resolvedHeadDimensions])
      return oProj(output)
    } else {
      let attnResult = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: scale,
        mask: mask,
      )
      let transposed = attnResult.transposed(0, 2, 1, 3)
      let output = transposed.reshaped([B, L, args.attentionHeads * args.resolvedHeadDimensions])
      return oProj(output)
    }
  }
}

// MARK: - Transformer Block

private class MarvisTransformerBlock: Module {
  @ModuleInfo(key: "self_attn") var attention: MarvisAttention
  @ModuleInfo(key: "mlp") var mlp: SwiGLUMLP

  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  init(_ args: MarvisBackboneConfig) {
    _attention.wrappedValue = MarvisAttention(args)
    _mlp.wrappedValue = SwiGLUMLP(
      hiddenSize: args.hiddenSize,
      intermediateSize: args.intermediateSize,
      bias: args.mlpBias,
    )
    _inputLayerNorm.wrappedValue = RMSNorm(
      dimensions: args.hiddenSize, eps: args.rmsNormEps,
    )
    _postAttentionLayerNorm.wrappedValue = RMSNorm(
      dimensions: args.hiddenSize, eps: args.rmsNormEps,
    )
  }

  func callAsFunction(
    _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
  ) -> MLXArray {
    var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
    let h = x + r
    r = mlp(postAttentionLayerNorm(h))
    let out = h + r
    return out
  }
}

// MARK: - Backbone

class MarvisBackbone: Module, LLMModel, KVCacheDimensionProvider {
  let vocabularySize: Int
  let kvHeads: [Int]

  fileprivate let layers: [MarvisTransformerBlock]
  let norm: RMSNorm

  init(_ args: MarvisBackboneConfig) {
    precondition(args.vocabularySize > 0)
    vocabularySize = args.vocabularySize
    kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
    layers = (0 ..< args.hiddenLayers).map { _ in MarvisTransformerBlock(args) }
    norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
  }

  func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
    var h = inputs

    let mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal

    for (i, layer) in layers.enumerated() {
      h = layer(h, mask: mask, cache: cache?[i])
    }

    return norm(h)
  }

  func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    weights.filter {
      !$0.key.contains("self_attn.rotary_emb.inv_freq")
    }
  }
}

// MARK: - Configuration

struct MarvisBackboneConfig: Codable, Sendable {
  var hiddenSize: Int
  var hiddenLayers: Int
  var intermediateSize: Int
  var attentionHeads: Int
  var headDimensions: Int?
  var rmsNormEps: Float
  var vocabularySize: Int
  var kvHeads: Int
  var maxPositionEmbeddings: Int?
  var ropeTheta: Float = 10000
  var ropeTraditional: Bool = false
  var ropeScaling: [String: StringOrNumber]?
  var tieWordEmbeddings: Bool = true
  var attentionBias: Bool = false
  var mlpBias: Bool = false

  init(
    hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int, attentionHeads: Int,
    headDimensions: Int? = nil, rmsNormEps: Float, vocabularySize: Int, kvHeads: Int,
    maxPositionEmbeddings: Int? = nil, ropeTheta: Float = 10000, ropeTraditional: Bool = false,
    ropeScaling: [String: StringOrNumber]? = nil, tieWordEmbeddings: Bool = true,
    attentionBias: Bool = false, mlpBias: Bool = false,
  ) {
    self.hiddenSize = hiddenSize
    self.hiddenLayers = hiddenLayers
    self.intermediateSize = intermediateSize
    self.attentionHeads = attentionHeads
    self.headDimensions = headDimensions
    self.rmsNormEps = rmsNormEps
    self.vocabularySize = vocabularySize
    self.kvHeads = kvHeads
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.ropeTheta = ropeTheta
    self.ropeTraditional = ropeTraditional
    self.ropeScaling = ropeScaling
    self.tieWordEmbeddings = tieWordEmbeddings
    self.attentionBias = attentionBias
    self.mlpBias = mlpBias
  }

  var resolvedHeadDimensions: Int {
    headDimensions ?? (hiddenSize / attentionHeads)
  }

  enum CodingKeys: String, CodingKey {
    case hiddenSize = "hidden_size"
    case hiddenLayers = "num_hidden_layers"
    case intermediateSize = "intermediate_size"
    case attentionHeads = "num_attention_heads"
    case headDimensions = "head_dim"
    case rmsNormEps = "rms_norm_eps"
    case vocabularySize = "vocab_size"
    case kvHeads = "num_key_value_heads"
    case maxPositionEmbeddings = "max_position_embeddings"
    case ropeTheta = "rope_theta"
    case ropeTraditional = "rope_traditional"
    case ropeScaling = "rope_scaling"
    case tieWordEmbeddings = "tie_word_embeddings"
    case attentionBias = "attention_bias"
    case mlpBias = "mlp_bias"
  }

  init(from decoder: Swift.Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
    hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
    intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
    attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
    headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
    rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
    vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
    kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
    maxPositionEmbeddings = try container.decodeIfPresent(
      Int.self, forKey: .maxPositionEmbeddings,
    )
    if let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) {
      self.ropeTheta = ropeTheta
    }
    if let ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) {
      self.ropeTraditional = ropeTraditional
    }
    ropeScaling = try container.decodeIfPresent(
      [String: StringOrNumber].self, forKey: .ropeScaling,
    )
    if let tieWordEmbeddings = try container.decodeIfPresent(
      Bool.self, forKey: .tieWordEmbeddings,
    ) {
      self.tieWordEmbeddings = tieWordEmbeddings
    }
    if let attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) {
      self.attentionBias = attentionBias
    }
    if let mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) {
      self.mlpBias = mlpBias
    }

    if let ropeScaling {
      if ropeScaling["factor"] == nil {
        throw DecodingError.dataCorruptedError(
          forKey: .ropeScaling, in: container,
          debugDescription: "rope_scaling must contain 'factor'",
        )
      }
      if let ropeType = ropeScaling["type"] ?? ropeScaling["rope_type"] {
        if case .string = ropeType {
          let options = [
            StringOrNumber.string("linear"), StringOrNumber.string("dynamic"),
            StringOrNumber.string("llama3"),
          ]
          if !options.contains(ropeType) {
            throw DecodingError.dataCorruptedError(
              forKey: .ropeScaling, in: container,
              debugDescription:
              "rope_scaling 'type' currently only supports 'linear', 'dynamic', or 'llama3'",
            )
          }
        }
      } else {
        throw DecodingError.dataCorruptedError(
          forKey: .ropeScaling, in: container,
          debugDescription: "rope_scaling must contain either 'type' or 'rope_type'",
        )
      }
    }
  }
}

// MARK: - LoRA

extension MarvisBackbone: LoRAModel {
  var loraLayers: [Module] {
    layers
  }
}
