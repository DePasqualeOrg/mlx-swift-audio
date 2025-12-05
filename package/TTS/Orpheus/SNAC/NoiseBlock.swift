//
//  NoiseBlock.swift
//  MLXAudio
//
//  NoiseBlock for SNAC decoder - adds learned noise modulation
//
import Foundation
import MLX
import MLXNN

/// NoiseBlock for SNAC decoder - adds learned noise modulation
/// Weight keys: linear.weight_g, linear.weight_v
class SNACNoiseBlock: Module {
  @ModuleInfo var linear: WNConv1d

  init(dim: Int) {
    // WNConv1d for noise modulation - outputs 1 channel for noise scaling
    // Bias is false in Python implementation for NoiseBlock's WNConv1d
    _linear.wrappedValue = WNConv1d(
      inChannels: dim,
      outChannels: 1,
      kernelSize: 1,
      padding: 0,
      bias: false,
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Input shape is likely [N, C, T]
    let B = x.shape[0]
    let T = x.shape[2]

    // Generate noise [B, 1, T]
    let noise = MLXRandom.normal([B, 1, T])

    // Apply the linear transformation
    let h = linear(x)

    // Modulate noise by the linear output and add to input
    let n = noise * h
    return x + n
  }
}
