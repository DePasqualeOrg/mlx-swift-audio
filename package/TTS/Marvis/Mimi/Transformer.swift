import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Config

struct MimiTransformerConfig {
  let dModel: Int
  let numHeads: Int
  let numLayers: Int
  let causal: Bool
  let normFirst: Bool
  let biasFF: Bool
  let biasAttn: Bool
  let layerScale: Float?
  let positionalEmbedding: String
  let useConvBlock: Bool
  let crossAttention: Bool
  let convKernelSize: Int
  let useConvBias: Bool
  let gating: Bool
  let norm: String
  let context: Int
  let maxPeriod: Int
  let maxSeqLen: Int
  let kvRepeat: Int
  let dimFeedforward: Int
  let convLayout: Bool

  init(
    dModel: Int,
    numHeads: Int,
    numLayers: Int,
    causal: Bool,
    normFirst: Bool,
    biasFF: Bool,
    biasAttn: Bool,
    layerScale: Float?,
    positionalEmbedding: String,
    useConvBlock: Bool,
    crossAttention: Bool,
    convKernelSize: Int,
    useConvBias: Bool,
    gating: Bool,
    norm: String,
    context: Int,
    maxPeriod: Int,
    maxSeqLen: Int,
    kvRepeat: Int,
    dimFeedforward: Int,
    convLayout: Bool,
  ) {
    self.dModel = dModel
    self.numHeads = numHeads
    self.numLayers = numLayers
    self.causal = causal
    self.normFirst = normFirst
    self.biasFF = biasFF
    self.biasAttn = biasAttn
    self.layerScale = layerScale
    self.positionalEmbedding = positionalEmbedding
    self.useConvBlock = useConvBlock
    self.crossAttention = crossAttention
    self.convKernelSize = convKernelSize
    self.useConvBias = useConvBias
    self.gating = gating
    self.norm = norm
    self.context = context
    self.maxPeriod = maxPeriod
    self.maxSeqLen = maxSeqLen
    self.kvRepeat = kvRepeat
    self.dimFeedforward = dimFeedforward
    self.convLayout = convLayout
  }

  var headDim: Int { dModel / numHeads }
}

// MARK: - Utilities

@inline(__always)
func geluApprox(_ x: MLXArray) -> MLXArray {
  // 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
  let c0 = MLXArray(0.7978845608028654) // sqrt(2/pi)
  let c1 = MLXArray(0.044715)
  let x3 = x * x * x
  return 0.5 * x * (1 + tanh(c0 * (x + c1 * x3)))
}

final class Id: Module {
  override init() {}
  func callAsFunction(_ xs: MLXArray) -> MLXArray { xs }
}

final class LayerScale: Module {
  var scale: MLXArray
  init(dim: Int) {
    scale = MLXArray.ones([dim])
  }

  func callAsFunction(_ xs: MLXArray) -> MLXArray {
    xs * scale
  }
}

// MARK: - Attention

final class Attention: Module {
  private let config: MimiTransformerConfig
  @ModuleInfo(key: "in_proj") var inProj: Linear
  @ModuleInfo(key: "out_proj") var outProj: Linear
  @ModuleInfo var rope: RoPE?

  private let scale: Float

  init(config: MimiTransformerConfig) {
    self.config = config
    // Only kv_repeat == 1 supported (parity with your python)
    precondition(config.kvRepeat == 1, "only kv_repeat == 1 is supported")

    let numKV = config.numHeads / config.kvRepeat
    let outDim = config.dModel + 2 * numKV * (config.dModel / config.numHeads) // => 3*dModel for kv_repeat=1
    _inProj.wrappedValue = Linear(config.dModel, outDim, bias: config.biasAttn)
    _outProj.wrappedValue = Linear(config.dModel, config.dModel, bias: config.biasAttn)
    scale = 1.0 / Float(Double(config.headDim).squareRoot())

    if config.positionalEmbedding == "rope" {
      _rope.wrappedValue = RoPE(dimensions: config.headDim, traditional: true, base: Float(config.maxPeriod))
    } else {
      _rope.wrappedValue = nil
    }
  }

  func callAsFunction(
    _ xs: MLXArray, // [B, T, D]
    cache: KVCache,
    mask: MLXArray? = nil,
  ) -> MLXArray {
    let b = xs.shape[0]
    let t = xs.shape[1]
    let hd = xs.shape[2] // d_model

    let qkv = inProj(xs).reshaped([b, t, 3, config.numHeads, config.headDim])

    var q = swappedAxes(qkv[0 ..< qkv.shape[0], 0 ..< qkv.shape[1], 0, 0 ..< qkv.shape[3], 0 ..< qkv.shape[4]], 1, 2)
    var k = swappedAxes(qkv[0 ..< qkv.shape[0], 0 ..< qkv.shape[1], 1, 0 ..< qkv.shape[3], 0 ..< qkv.shape[4]], 1, 2)
    var v = swappedAxes(qkv[0 ..< qkv.shape[0], 0 ..< qkv.shape[1], 2, 0 ..< qkv.shape[3], 0 ..< qkv.shape[4]], 1, 2)

    if let rope {
      q = rope(q, offset: cache.offset)
      k = rope(k, offset: cache.offset)
    }

    (k, v) = cache.update(keys: k, values: v)

    let kLen = k.shape[2]
    let kTargetLen = t + min(config.context, kLen - t)
    if kTargetLen < kLen {
      let start = kLen - kTargetLen
      k = split(k, indices: [start], axis: 2)[1]
      v = split(v, indices: [start], axis: 2)[1]
    }

    let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = if let mask { .array(mask) } else { .none }
    var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: maskMode)
    out = swappedAxes(out, 1, 2).reshaped([b, t, hd])
    return outProj(out)
  }
}

// MARK: - MLP

final class MlpGating: Module {
  @ModuleInfo(key: "linear_in") var linearIn: Linear
  @ModuleInfo(key: "linear_out") var linearOut: Linear

  init(config: MimiTransformerConfig) {
    var hidden = 2 * config.dimFeedforward / 3
    if config.dimFeedforward == 4 * config.dModel {
      hidden = 11 * config.dModel / 4
    }
    _linearIn.wrappedValue = Linear(config.dModel, 2 * hidden, bias: config.biasFF)
    _linearOut.wrappedValue = Linear(hidden, config.dModel, bias: config.biasFF)
  }

  func callAsFunction(_ xs: MLXArray) -> MLXArray {
    let b = xs.shape[0]
    let t = xs.shape[1]
    let doubled = linearIn(xs) // [B, T, 2*H]
    let hidden = doubled.shape[2] / 2
    let split2 = doubled.reshaped([b, t, 2, hidden])

    // split along axis=2 at 1 -> [B,T,1,H], [B,T,1,H]
    let parts = split(split2, indices: [1], axis: 2)
    let a = parts[0] // gate input
    let bpart = parts[1]

    // SiLU(a) * b -> [B,T,1,H] then reshape to [B,T,H]
    let gated = silu(a) * bpart
    let flat = gated.reshaped([b, t, hidden])

    return linearOut(flat)
  }
}

final class MlpNoGating: Module {
  @ModuleInfo var linear1: Linear
  @ModuleInfo var linear2: Linear

  init(config: MimiTransformerConfig) {
    _linear1.wrappedValue = Linear(config.dModel, config.dimFeedforward, bias: config.biasFF)
    _linear2.wrappedValue = Linear(config.dimFeedforward, config.dModel, bias: config.biasFF)
  }

  func callAsFunction(_ xs: MLXArray) -> MLXArray {
    linear2(geluApprox(linear1(xs)))
  }
}

// MARK: - Transformer layer

final class TransformerLayer: Module {
  @ModuleInfo var gating: Module
  @ModuleInfo var norm1: Module
  @ModuleInfo var norm2: Module
  @ModuleInfo(key: "layer_scale_1") var layerScale1: Module
  @ModuleInfo(key: "layer_scale_2") var layerScale2: Module
  @ModuleInfo(key: "self_attn") var selfAttn: Attention

  init(config: MimiTransformerConfig) {
    precondition(!config.useConvBlock, "conv-block is not supported")
    precondition(!config.crossAttention, "cross-attn is not supported")

    if config.gating {
      _gating.wrappedValue = MlpGating(config: config)
    } else {
      _gating.wrappedValue = MlpNoGating(config: config)
    }

    switch config.norm {
      case "layer_norm":
        _norm1.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        _norm2.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
      case "rms_norm":
        _norm1.wrappedValue = RMSNorm(dimensions: config.dModel, eps: 1e-8)
        _norm2.wrappedValue = RMSNorm(dimensions: config.dModel, eps: 1e-8)
      default:
        fatalError("unsupported norm type \(config.norm)")
    }

    if let _ = config.layerScale {
      _layerScale1.wrappedValue = LayerScale(dim: config.dModel)
      _layerScale2.wrappedValue = LayerScale(dim: config.dModel)
    } else {
      _layerScale1.wrappedValue = Id()
      _layerScale2.wrappedValue = Id()
    }

    _selfAttn.wrappedValue = Attention(config: config)
  }

  func callAsFunction(
    _ xs: MLXArray,
    cache: KVCache,
  ) -> MLXArray {
    var x = xs
    var n1 = (norm1 as! UnaryLayer)(x)
    n1 = selfAttn(n1, cache: cache)
    x = x + (layerScale1 as! LayerScale)(n1)
    x = x + (layerScale2 as! LayerScale)((gating as! MlpNoGating)((norm2 as! LayerNorm)(x)))
    return x
  }
}

// MARK: - Transformer

final class Transformer: Module {
  private let config: MimiTransformerConfig
  @ModuleInfo var layers: [TransformerLayer]

  init(config: MimiTransformerConfig) {
    self.config = config
    _layers.wrappedValue = (0 ..< config.numLayers).map { _ in TransformerLayer(config: config) }
  }

  func callAsFunction(
    _ xs: MLXArray,
    cache: [KVCache],
  ) -> MLXArray {
    var x = xs
    for (layer, c) in zip(layers, cache) {
      x = layer(x, cache: c)
    }
    return x
  }

  func makeCache() -> [KVCache] {
    (0 ..< config.numLayers).map { _ in KVCacheSimple() }
  }
}

// MARK: - ProjectedTransformer

final class ProjectedTransformer: Module {
  private let convLayout: Bool
  @ModuleInfo var transformer: Transformer
  @ModuleInfo(key: "input_proj") var inputProj: Linear?
  @ModuleInfo(key: "output_projs") var outputProjs: [Linear?]

  init(config: MimiTransformerConfig, inputDim: Int, outputDims: [Int]) {
    convLayout = config.convLayout
    _transformer.wrappedValue = Transformer(config: config)

    if inputDim == config.dModel {
      _inputProj.wrappedValue = nil
    } else {
      _inputProj.wrappedValue = Linear(inputDim, config.dModel, bias: false)
    }

    var outs: [Linear?] = []
    for od in outputDims {
      if od == config.dModel {
        outs.append(nil)
      } else {
        outs.append(Linear(config.dModel, od, bias: false))
      }
    }
    _outputProjs.wrappedValue = outs
  }

  func callAsFunction(
    _ xsIn: MLXArray,
    cache: [KVCache],
  ) -> [MLXArray] {
    var xs = xsIn
    if convLayout { xs = swappedAxes(xs, 1, 2) } // [B,C,T] -> [B,T,C]

    if let ip = inputProj { xs = ip(xs) }

    xs = transformer(xs, cache: cache)

    if outputProjs.compactMap({ $0 }).count == 0 {
      return [swappedAxes(xs, 1, 2)]
    } else {
      var outs: [MLXArray] = []
      for op in outputProjs {
        guard let op else { continue }
        var out = op(xs)
        if convLayout { out = swappedAxes(out, 1, 2) } // back to [B,C,T] if needed
        outs.append(out)
      }
      return outs
    }
  }

  func makeCache() -> [KVCache] { transformer.makeCache() }
}
