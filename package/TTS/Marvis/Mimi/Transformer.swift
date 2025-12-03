import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Config

struct TransformerConfig {
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

protocol AttentionCache {
  var offset: Int { get }
  func updateAndFetch(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray)
}

final class Attention: Module {
  private let cfg: TransformerConfig
  @ModuleInfo var in_proj: Linear
  @ModuleInfo var out_proj: Linear
  @ModuleInfo var rope: RoPE?

  private let scale: Float

  init(cfg: TransformerConfig) {
    self.cfg = cfg
    // Only kv_repeat == 1 supported (parity with your python)
    precondition(cfg.kvRepeat == 1, "only kv_repeat == 1 is supported")

    let numKV = cfg.numHeads / cfg.kvRepeat
    let outDim = cfg.dModel + 2 * numKV * (cfg.dModel / cfg.numHeads) // => 3*dModel for kv_repeat=1
    _in_proj = ModuleInfo(wrappedValue: Linear(cfg.dModel, outDim, bias: cfg.biasAttn))
    _out_proj = ModuleInfo(wrappedValue: Linear(cfg.dModel, cfg.dModel, bias: cfg.biasAttn))
    scale = 1.0 / Float(Double(cfg.headDim).squareRoot())

    if cfg.positionalEmbedding == "rope" {
      _rope = ModuleInfo(wrappedValue: RoPE(dimensions: cfg.headDim, traditional: true, base: Float(cfg.maxPeriod)))
    } else {
      _rope = ModuleInfo(wrappedValue: nil)
    }
  }

  func callAsFunction(
    _ xs: MLXArray, // [B, T, D]
    cache: any AttentionCache,
    mask: MLXArray? = nil,
  ) -> MLXArray {
    let b = xs.shape[0]
    let t = xs.shape[1]
    let hd = xs.shape[2] // d_model

    let qkv = in_proj(xs).reshaped([b, t, 3, cfg.numHeads, cfg.headDim])

    var q = swappedAxes(qkv[0 ..< qkv.shape[0], 0 ..< qkv.shape[1], 0, 0 ..< qkv.shape[3], 0 ..< qkv.shape[4]], 1, 2)
    var k = swappedAxes(qkv[0 ..< qkv.shape[0], 0 ..< qkv.shape[1], 1, 0 ..< qkv.shape[3], 0 ..< qkv.shape[4]], 1, 2)
    var v = swappedAxes(qkv[0 ..< qkv.shape[0], 0 ..< qkv.shape[1], 2, 0 ..< qkv.shape[3], 0 ..< qkv.shape[4]], 1, 2)

    if let rope {
      q = rope(q, offset: cache.offset)
      k = rope(k, offset: cache.offset)
    }

    (k, v) = cache.updateAndFetch(k, v)

    let kLen = k.shape[2]
    let kTargetLen = t + min(cfg.context, kLen - t)
    if kTargetLen < kLen {
      let start = kLen - kTargetLen
      k = split(k, indices: [start], axis: 2)[1]
      v = split(v, indices: [start], axis: 2)[1]
    }

    var out = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
    out = swappedAxes(out, 1, 2).reshaped([b, t, hd])
    return out_proj(out)
  }
}

// MARK: - MLP

final class MlpGating: Module {
  @ModuleInfo var linear_in: Linear
  @ModuleInfo var linear_out: Linear

  init(cfg: TransformerConfig) {
    var hidden = 2 * cfg.dimFeedforward / 3
    if cfg.dimFeedforward == 4 * cfg.dModel {
      hidden = 11 * cfg.dModel / 4
    }
    _linear_in = ModuleInfo(wrappedValue: Linear(cfg.dModel, 2 * hidden, bias: cfg.biasFF))
    _linear_out = ModuleInfo(wrappedValue: Linear(hidden, cfg.dModel, bias: cfg.biasFF))
  }

  func callAsFunction(_ xs: MLXArray) -> MLXArray {
    let b = xs.shape[0]
    let t = xs.shape[1]
    let doubled = linear_in(xs) // [B, T, 2*H]
    let hidden = doubled.shape[2] / 2
    let split2 = doubled.reshaped([b, t, 2, hidden])

    // split along axis=2 at 1 -> [B,T,1,H], [B,T,1,H]
    let parts = split(split2, indices: [1], axis: 2)
    let a = parts[0] // gate input
    let bpart = parts[1]

    // SiLU(a) * b -> [B,T,1,H] then reshape to [B,T,H]
    let gated = silu(a) * bpart
    let flat = gated.reshaped([b, t, hidden])

    return linear_out(flat)
  }
}

final class MlpNoGating: Module {
  @ModuleInfo var linear1: Linear
  @ModuleInfo var linear2: Linear

  init(cfg: TransformerConfig) {
    _linear1 = ModuleInfo(wrappedValue: Linear(cfg.dModel, cfg.dimFeedforward, bias: cfg.biasFF))
    _linear2 = ModuleInfo(wrappedValue: Linear(cfg.dimFeedforward, cfg.dModel, bias: cfg.biasFF))
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
  @ModuleInfo var layer_scale_1: Module
  @ModuleInfo var layer_scale_2: Module
  @ModuleInfo var self_attn: Attention

  init(cfg: TransformerConfig) {
    precondition(!cfg.useConvBlock, "conv-block is not supported")
    precondition(!cfg.crossAttention, "cross-attn is not supported")

    if cfg.gating {
      _gating = ModuleInfo(wrappedValue: MlpGating(cfg: cfg))
    } else {
      _gating = ModuleInfo(wrappedValue: MlpNoGating(cfg: cfg))
    }

    switch cfg.norm {
      case "layer_norm":
        _norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: cfg.dModel, eps: 1e-5))
        _norm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: cfg.dModel, eps: 1e-5))
      case "rms_norm":
        _norm1 = ModuleInfo(wrappedValue: RMSNorm(dimensions: cfg.dModel, eps: 1e-8))
        _norm2 = ModuleInfo(wrappedValue: RMSNorm(dimensions: cfg.dModel, eps: 1e-8))
      default:
        fatalError("unsupported norm type \(cfg.norm)")
    }

    if let _ = cfg.layerScale {
      _layer_scale_1 = ModuleInfo(wrappedValue: LayerScale(dim: cfg.dModel))
      _layer_scale_2 = ModuleInfo(wrappedValue: LayerScale(dim: cfg.dModel))
    } else {
      _layer_scale_1 = ModuleInfo(wrappedValue: Id())
      _layer_scale_2 = ModuleInfo(wrappedValue: Id())
    }

    _self_attn = ModuleInfo(wrappedValue: Attention(cfg: cfg))
  }

  func callAsFunction(
    _ xs: MLXArray,
    cache: any AttentionCache,
  ) -> MLXArray {
    var x = xs
    var n1 = (norm1 as! UnaryLayer)(x)
    n1 = self_attn(n1, cache: cache)
    x = x + (layer_scale_1 as! LayerScale)(n1)
    x = x + (layer_scale_2 as! LayerScale)((gating as! MlpNoGating)((norm2 as! LayerNorm)(x)))
    return x
  }
}

// MARK: - Transformer

final class Transformer: Module {
  private let cfg: TransformerConfig
  @ModuleInfo var layers: [TransformerLayer]

  init(cfg: TransformerConfig) {
    self.cfg = cfg
    _layers = ModuleInfo(wrappedValue: (0 ..< cfg.numLayers).map { _ in TransformerLayer(cfg: cfg) })
  }

  func callAsFunction(
    _ xs: MLXArray,
    cache: [AttentionCache],
  ) -> MLXArray {
    var x = xs
    for (layer, c) in zip(layers, cache) {
      x = layer(x, cache: c)
    }
    return x
  }

  func makeCache() -> [TTSKVCache] {
    // Assume your KVCache init matches the python: (head_dim, n_kv_heads)
    let numKVHeads = cfg.numHeads / cfg.kvRepeat
    return (0 ..< cfg.numLayers).map { _ in TTSKVCache(headDim: cfg.headDim, nKVHeads: numKVHeads) }
  }
}

// MARK: - ProjectedTransformer

final class ProjectedTransformer: Module {
  private let convLayout: Bool
  @ModuleInfo var transformer: Transformer
  @ModuleInfo var input_proj: Linear?
  @ModuleInfo var output_projs: [Linear?]

  init(cfg: TransformerConfig, inputDim: Int, outputDims: [Int]) {
    convLayout = cfg.convLayout
    _transformer = ModuleInfo(wrappedValue: Transformer(cfg: cfg))

    if inputDim == cfg.dModel {
      _input_proj = ModuleInfo(wrappedValue: nil)
    } else {
      _input_proj = ModuleInfo(wrappedValue: Linear(inputDim, cfg.dModel, bias: false))
    }

    var outs: [Linear?] = []
    for od in outputDims {
      if od == cfg.dModel {
        outs.append(nil)
      } else {
        outs.append(Linear(cfg.dModel, od, bias: false))
      }
    }
    _output_projs = ModuleInfo(wrappedValue: outs)
  }

  func callAsFunction(
    _ xsIn: MLXArray,
    cache: [AttentionCache],
  ) -> [MLXArray] {
    var xs = xsIn
    if convLayout { xs = swappedAxes(xs, 1, 2) } // [B,C,T] -> [B,T,C]

    if let ip = input_proj { xs = ip(xs) }

    xs = transformer(xs, cache: cache)

    if output_projs.compactMap({ $0 }).count == 0 {
      return [swappedAxes(xs, 1, 2)]
    } else {
      var outs: [MLXArray] = []
      for op in output_projs {
        guard let op else { continue }
        var out = op(xs)
        if convLayout { out = swappedAxes(out, 1, 2) } // back to [B,C,T] if needed
        outs.append(out)
      }
      return outs
    }
  }

  func makeCache() -> [TTSKVCache] { transformer.makeCache() }
}
