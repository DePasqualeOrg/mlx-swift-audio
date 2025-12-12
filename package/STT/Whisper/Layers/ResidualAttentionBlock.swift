import Foundation
import MLX
import MLXNN

/// Residual attention block for Whisper transformer
///
/// Implements pre-norm architecture with:
/// - Self-attention
/// - Optional cross-attention
/// - MLP with GELU activation
class ResidualAttentionBlock: Module {
  let attn: WhisperMultiHeadAttention
  @ModuleInfo(key: "attn_ln") var attnLn: LayerNorm

  @ModuleInfo(key: "cross_attn") var crossAttn: WhisperMultiHeadAttention?
  @ModuleInfo(key: "cross_attn_ln") var crossAttnLn: LayerNorm?

  let mlp1: Linear
  let mlp2: Linear
  @ModuleInfo(key: "mlp_ln") var mlpLn: LayerNorm

  init(nState: Int, nHead: Int, crossAttention: Bool = false) {
    attn = WhisperMultiHeadAttention(nState: nState, nHead: nHead)
    _attnLn.wrappedValue = LayerNorm(dimensions: nState)

    if crossAttention {
      _crossAttn.wrappedValue = WhisperMultiHeadAttention(nState: nState, nHead: nHead)
      _crossAttnLn.wrappedValue = LayerNorm(dimensions: nState)
    }

    let nMlp = nState * 4
    mlp1 = Linear(nState, nMlp)
    mlp2 = Linear(nMlp, nState)
    _mlpLn.wrappedValue = LayerNorm(dimensions: nState)
  }

  /// Forward pass
  ///
  /// - Parameters:
  ///   - x: Input tensor (batch, n_ctx, n_state)
  ///   - xa: Optional cross-attention input (batch, n_audio_ctx, n_audio_state)
  ///   - mask: Optional attention mask
  ///   - kvCache: Optional tuple of (self_kv_cache, cross_kv_cache)
  ///   - offset: Position offset for mask slicing when using KV cache
  /// - Returns: Tuple of (output, new_kv_cache, cross_attention_weights)
  func callAsFunction(
    _ x: MLXArray,
    xa: MLXArray? = nil,
    mask: MLXArray? = nil,
    kvCache: ((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)? = nil,
    offset: Int = 0
  ) -> (MLXArray, ((MLXArray, MLXArray)?, (MLXArray, MLXArray)?), MLXArray?) {
    var xVar = x

    // Extract KV caches
    let selfKvCache = kvCache?.0
    let crossKvCache = kvCache?.1

    // Self-attention
    let (y, newSelfKv, _) = attn(
      attnLn(xVar),
      mask: mask,
      kvCache: selfKvCache,
      offset: offset
    )
    xVar = xVar + y

    // Cross-attention (if enabled)
    var newCrossKv: (MLXArray, MLXArray)? = nil
    var crossQk: MLXArray? = nil

    if let crossAttn, let crossAttnLn, let xa {
      let (crossY, newCrossKvTmp, crossQkTmp) = crossAttn(
        crossAttnLn(xVar),
        xa: xa,
        kvCache: crossKvCache
      )
      xVar = xVar + crossY
      newCrossKv = newCrossKvTmp
      crossQk = crossQkTmp
    } else {
      newCrossKv = crossKvCache
    }

    // MLP
    let mlpOut = mlp2(GELU()(mlp1(mlpLn(xVar))))
    xVar = xVar + mlpOut

    return (xVar, (newSelfKv, newCrossKv), crossQk)
  }
}
