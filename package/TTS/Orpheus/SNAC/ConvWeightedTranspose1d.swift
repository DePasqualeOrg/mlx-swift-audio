//
//  ConvWeightedTranspose1d.swift
//  MLXAudio
//
//  ConvTranspose1d with weight normalization for SNAC decoder
//
import Foundation
import MLX
import MLXNN

/// ConvTranspose1d with weight normalization for SNAC decoder
/// Weight keys: weight_g, weight_v, bias
class WNConvTranspose1d: Module {
  @ModuleInfo(key: "weight_g") var weightG: MLXArray
  @ModuleInfo(key: "weight_v") var weightV: MLXArray
  @ModuleInfo var bias: MLXArray?

  let stride: Int
  let padding: Int
  let outputPadding: Int
  let dilation: Int
  let groups: Int

  // Helper for normalizing weight_v, equivalent to Python's normalize_weight(weight_v, except_dim=0)
  // Normalizes over axes 1 and 2 for a weight_v of shape [in_channels, kernel_size, out_channels_per_group]
  private static func normalizeWeightV(_ v: MLXArray) -> MLXArray {
    guard v.ndim == 3 else {
      fatalError("weight_v must have 3 dimensions for normalization")
    }
    // Python's normalize_weight(weight_v, except_dim=0) normalizes over axes 1 and 2
    let axesToSumOver = [1, 2]
    let vSquared = MLX.pow(v, 2)
    let sumSquared = MLX.sum(vSquared, axes: axesToSumOver, keepDims: true)
    return MLX.sqrt(sumSquared)
  }

  init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    padding: Int = 0,
    outputPadding: Int = 0,
    dilation: Int = 1,
    groups: Int = 1,
    bias: Bool = true,
  ) {
    self.stride = stride
    self.padding = padding
    self.outputPadding = outputPadding
    self.dilation = dilation
    self.groups = groups

    if dilation != 1 {
      Log.tts.warning("MLX.convTranspose1d might not support dilation != 1.")
    }

    // Initialize with placeholder values - will be replaced by model.update(parameters:)
    let scale = sqrt(1.0 / Double(inChannels * kernelSize))
    let weightInit = MLXRandom.uniform(
      low: -scale,
      high: scale,
      [inChannels, kernelSize, outChannels / groups],
    )
    let normWeight = Self.normalizeWeightV(weightInit)

    _weightG.wrappedValue = normWeight
    _weightV.wrappedValue = weightInit / (normWeight + 1e-12)
    _bias.wrappedValue = bias ? MLX.zeros([outChannels]) : nil
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // self.weightV has shape [in_channels, kernel_size, out_channels_per_group]
    // self.weightG has shape [in_channels, 1, 1]

    // Normalize self.weightV (like Python's normalize_weight(self.weight_v, except_dim=0))
    let normV = Self.normalizeWeightV(weightV) // Shape [in_channels, 1, 1]

    // Calculate effective weight before transposing for the MLX op
    // (weightG * weightV) / normV
    let effectiveWeightPreTranspose = (weightG * weightV) / (normV + 1e-12) // Shape [in_channels, kernel_size, out_channels_per_group]

    // MLX.convTransposed1d expects weight: [out_channels, kernel_width, in_channels / groups]
    // Our effectiveWeightPreTranspose is [in_channels, kernel_size, out_channels_per_group]
    // To match MLX Swift: transpose axes [2, 1, 0] (if groups=1, then out_channels_per_group = out_channels)
    // This matches Python's weight.swapaxes(0, 2)
    let weight = effectiveWeightPreTranspose.transposed(axes: [2, 1, 0]) // Shape [out_channels_per_group, kernel_size, in_channels]

    let output = MLX.convTransposed1d(
      x, // Expected input [batch, in_channels, length_in]
      weight, // Expected [out_channels, kernel_width, in_channels / groups]
      stride: stride,
      padding: padding,
      dilation: dilation,
      groups: groups,
    )

    if let bias {
      return output + bias.reshaped([1, 1, -1])
    } else {
      return output
    }
  }
}
