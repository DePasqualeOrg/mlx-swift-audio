//
//  SwiGLUMLP.swift
//  MLXAudio
//
//  Unified SwiGLU MLP implementation for Llama-style transformers.
//

import Foundation
import MLX
import MLXNN

/// SwiGLU MLP (Swish-Gated Linear Unit) feed-forward network.
///
/// Implements the SwiGLU variant: `down(silu(gate(x)) * up(x))`
/// Used in Llama-style transformer models.
class SwiGLUMLP: Module, UnaryLayer {
  @ModuleInfo(key: "gate_proj") var gate: Linear
  @ModuleInfo(key: "down_proj") var down: Linear
  @ModuleInfo(key: "up_proj") var up: Linear

  /// Initialize with explicit dimensions
  init(
    hiddenSize: Int,
    intermediateSize: Int,
    bias: Bool = false,
  ) {
    _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
    _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
    _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    down(silu(gate(x)) * up(x))
  }
}
