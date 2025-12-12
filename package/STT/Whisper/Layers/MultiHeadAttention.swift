import Foundation
import MLX
import MLXNN

/// Multi-head attention module for Whisper
///
/// Supports both self-attention and cross-attention with optional KV caching
class WhisperMultiHeadAttention: Module {
  let nHead: Int
  let query: Linear
  let key: Linear
  let value: Linear
  let out: Linear

  init(nState: Int, nHead: Int) {
    self.nHead = nHead
    query = Linear(nState, nState)
    key = Linear(nState, nState, bias: false)
    value = Linear(nState, nState)
    out = Linear(nState, nState)
  }

  /// Forward pass of multi-head attention
  ///
  /// - Parameters:
  ///   - x: Query input (batch, n_ctx, n_state)
  ///   - xa: Optional key/value input for cross-attention (batch, n_ctx, n_state)
  ///   - mask: Optional attention mask
  ///   - kvCache: Optional cached key/value tensors
  ///   - offset: Position offset for mask slicing when using KV cache
  /// - Returns: Tuple of (output, new_kv_cache, attention_weights)
  func callAsFunction(
    _ x: MLXArray,
    xa: MLXArray? = nil,
    mask: MLXArray? = nil,
    kvCache: (MLXArray, MLXArray)? = nil,
    offset: Int = 0
  ) -> (MLXArray, (MLXArray, MLXArray), MLXArray) {
    let q = query(x)

    var k: MLXArray
    var v: MLXArray

    if let xa {
      // Cross-attention: use xa for key/value
      if let kvCache {
        // Use cached cross-attention K,V
        k = kvCache.0
        v = kvCache.1
      } else {
        // Compute cross-attention K,V from audio features
        k = key(xa)
        v = value(xa)

        // Debug: verify audio features are being used
        #if DEBUG
        let xaMean = xa.mean().item(Float.self)
        let xaStd = xa.variance().sqrt().item(Float.self)
        if abs(xaMean) > 1.0 || xaStd < 0.01 {
          print("[CrossAttn] WARNING: xa stats unusual - mean=\(xaMean), std=\(xaStd)")
        }
        #endif
      }
    } else {
      // Self-attention: use x for key/value
      k = key(x)
      v = value(x)

      // Append to cache if provided
      if let kvCache {
        k = MLX.concatenated([kvCache.0, k], axis: 1)
        v = MLX.concatenated([kvCache.1, v], axis: 1)
      }
    }

    let (wv, qk) = qkvAttention(q: q, k: k, v: v, mask: mask, offset: offset)
    return (out(wv), (k, v), qk)
  }

  /// Scaled dot-product attention with multi-head support
  ///
  /// - Parameters:
  ///   - q: Query tensor (batch, n_ctx_q, n_state)
  ///   - k: Key tensor (batch, n_ctx_kv, n_state)
  ///   - v: Value tensor (batch, n_ctx_kv, n_state)
  ///   - mask: Optional attention mask
  ///   - offset: Position offset for mask slicing when using KV cache
  /// - Returns: Tuple of (attention_output, attention_weights)
  func qkvAttention(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    mask: MLXArray? = nil,
    offset: Int = 0
  ) -> (MLXArray, MLXArray) {
    let nBatch = q.shape[0]
    let nCtx = q.shape[1]
    let nState = q.shape[2]

    // Scale factor for attention (applied to both Q and K)
    let scale = pow(Float(nState / nHead), -0.25)

    // Reshape and transpose Q: (batch, n_ctx_q, n_head, head_dim) -> (batch, n_head, n_ctx_q, head_dim)
    var qReshaped = q.reshaped(nBatch, nCtx, nHead, nState / nHead)
    qReshaped = qReshaped.transposed(0, 2, 1, 3) * scale

    // Reshape and transpose K: (batch, n_ctx_kv, n_head, head_dim) -> (batch, n_head, head_dim, n_ctx_kv)
    let kCtx = k.shape[1]
    var kReshaped = k.reshaped(nBatch, kCtx, nHead, nState / nHead)
    kReshaped = kReshaped.transposed(0, 2, 3, 1) * scale

    // Reshape and transpose V: (batch, n_ctx_kv, n_head, head_dim) -> (batch, n_head, n_ctx_kv, head_dim)
    var vReshaped = v.reshaped(nBatch, kCtx, nHead, nState / nHead)
    vReshaped = vReshaped.transposed(0, 2, 1, 3)

    // Attention scores: (batch, n_head, n_ctx_q, n_ctx_kv)
    var qk = MLX.matmul(qReshaped, kReshaped)

    // Apply mask if provided
    if let mask {
      // Slice mask to match the actual Q and K dimensions
      // When using KV cache, offset indicates the current position in the sequence
      qk = qk + mask[offset ..< (offset + nCtx), 0 ..< kCtx]
    }

    // Softmax over key dimension
    let w = MLX.softmax(qk, axis: -1)

    // Apply attention to values: (batch, n_head, n_ctx_q, head_dim)
    let out = MLX.matmul(w, vReshaped)

    // Transpose back: (batch, n_ctx_q, n_head, head_dim)
    let outTransposed = out.transposed(0, 2, 1, 3)

    // Reshape to (batch, n_ctx_q, n_state)
    let outReshaped = outTransposed.reshaped(nBatch, nCtx, nState)

    return (outReshaped, qk)
  }
}
