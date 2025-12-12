import Foundation
import MLX
import MLXNN

/// Audio encoder for Whisper
///
/// Processes mel spectrogram input through:
/// - Two 1D convolution layers
/// - Sinusoidal positional embeddings
/// - Stack of transformer blocks
class AudioEncoder: Module {
  let conv1: Conv1d
  let conv2: Conv1d
  @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
  let blocks: [ResidualAttentionBlock]
  @ModuleInfo(key: "ln_post") var lnPost: LayerNorm

  init(nMels: Int, nCtx: Int, nState: Int, nHead: Int, nLayer: Int) {
    // Two conv layers: first with stride=1, second with stride=2
    conv1 = Conv1d(inputChannels: nMels, outputChannels: nState, kernelSize: 3, padding: 1)
    conv2 = Conv1d(inputChannels: nState, outputChannels: nState, kernelSize: 3, stride: 2, padding: 1)

    // Sinusoidal positional embeddings
    _positionalEmbedding.wrappedValue = sinusoids(length: nCtx, channels: nState)

    // Transformer blocks (no cross-attention)
    blocks = (0 ..< nLayer).map { _ in
      ResidualAttentionBlock(nState: nState, nHead: nHead, crossAttention: false)
    }

    _lnPost.wrappedValue = LayerNorm(dimensions: nState)
  }

  /// Forward pass
  ///
  /// - Parameter x: Mel spectrogram input (batch, n_frames, n_mels) - Conv1d expects (batch, length, channels)
  /// - Returns: Encoded audio features (batch, n_ctx, n_state)
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Apply convolutions with GELU activation
    // Conv1d input: (batch, length, channels) = (batch, n_frames, n_mels)
    // Conv1d output: (batch, length, channels) = (batch, n_ctx, n_state)
    var output = GELU()(conv1(x))
    output = GELU()(conv2(output))

    // Output from conv2 is already (batch, n_ctx, n_state), no transpose needed
    // Note: The Python version uses (batch, channels, length) and needs transpose,
    // but MLX Swift Conv1d uses (batch, length, channels) so no transpose needed here

    // Add positional embeddings
    let nCtx = output.shape[1]
    output = output + positionalEmbedding[0 ..< nCtx]

    // Apply transformer blocks
    for block in blocks {
      let (newOutput, _, _) = block(output)
      output = newOutput
    }

    // Final layer norm
    output = lnPost(output)

    return output
  }
}

/// Generate sinusoidal positional embeddings
///
/// - Parameters:
///   - length: Sequence length
///   - channels: Embedding dimension (must be even)
///   - maxTimescale: Maximum timescale for the sinusoids
/// - Returns: Positional embeddings (length, channels)
func sinusoids(length: Int, channels: Int, maxTimescale: Float = 10000.0) -> MLXArray {
  assert(channels % 2 == 0, "channels must be even")

  // Compute inverse timescales
  let logTimescaleIncrement = log(maxTimescale) / Float(channels / 2 - 1)
  let invTimescales = MLX.exp(
    -logTimescaleIncrement * MLXArray(0 ..< (channels / 2)).asType(.float32)
  )

  // Compute scaled time
  let positions = MLXArray(0 ..< length).asType(.float32)
  let scaledTime = positions.expandedDimensions(axis: 1) * invTimescales.expandedDimensions(axis: 0)

  // Concatenate sin and cos
  let sinPart = MLX.sin(scaledTime)
  let cosPart = MLX.cos(scaledTime)

  return MLX.concatenated([sinPart, cosPart], axis: 1)
}
