// Copyright © Microsoft (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/microsoft/VibeVoice
// License: licenses/vibevoice.txt

import Foundation
import MLX
import MLXNN

// MARK: - DiffusionRMSNorm

/// Root Mean Square Layer Normalization for diffusion head
class DiffusionRMSNorm: Module {
  let dim: Int
  let eps: Float
  let elementwiseAffine: Bool

  @ParameterInfo(key: "weight") var weight: MLXArray?

  init(dim: Int, eps: Float = 1e-6, elementwiseAffine: Bool = true) {
    self.dim = dim
    self.eps = eps
    self.elementwiseAffine = elementwiseAffine

    if elementwiseAffine {
      _weight.wrappedValue = MLXArray.ones([dim])
    } else {
      _weight.wrappedValue = nil
    }
  }

  private func norm(_ x: MLXArray) -> MLXArray {
    x * MLX.rsqrt(MLX.mean(x * x, axis: -1, keepDims: true) + eps)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = norm(x.asType(.float32)).asType(x.dtype)
    if let w = weight {
      out = out * w
    }
    return out
  }
}

// MARK: - Modulate

/// Apply adaptive layer normalization modulation
func modulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
  x * (1 + scale) + shift
}

// MARK: - TimestepEmbedder

/// Embeds scalar timesteps into vector representations
class TimestepEmbedder: Module {
  let frequencyEmbeddingSize: Int

  @ModuleInfo(key: "mlp") var mlp: Sequential

  init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256) {
    self.frequencyEmbeddingSize = frequencyEmbeddingSize

    _mlp.wrappedValue = Sequential {
      Linear(frequencyEmbeddingSize, hiddenSize, bias: false)
      SiLU()
      Linear(hiddenSize, hiddenSize, bias: false)
    }
  }

  /// Create sinusoidal timestep embeddings
  /// - Parameters:
  ///   - t: 1D tensor of timestep indices
  ///   - dim: Embedding dimension
  ///   - maxPeriod: Controls minimum frequency
  /// - Returns: Positional embeddings of shape (N, dim)
  static func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Int = 10000) -> MLXArray {
    let half = dim / 2
    let freqs = MLX.exp(
      -log(Float(maxPeriod)) * MLXArray(0 ..< half).asType(.float32) / Float(half)
    )
    let args = t[0..., .newAxis].asType(.float32) * freqs[.newAxis, 0...]
    var embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
    if dim % 2 != 0 {
      embedding = MLX.concatenated([embedding, MLXArray.zeros(like: embedding[0..., 0 ..< 1])], axis: -1)
    }
    return embedding
  }

  func callAsFunction(_ t: MLXArray) -> MLXArray {
    let tFreq = Self.timestepEmbedding(t, dim: frequencyEmbeddingSize)
    return mlp(tFreq)
  }
}

// MARK: - FeedForwardNetwork

/// Feed-forward network with SwiGLU activation for diffusion head
class DiffusionFeedForwardNetwork: Module {
  let embedDim: Int

  @ModuleInfo(key: "gate_proj") var gateProj: Linear
  @ModuleInfo(key: "up_proj") var upProj: Linear
  @ModuleInfo(key: "down_proj") var downProj: Linear

  init(embedDim: Int, ffnDim: Int) {
    self.embedDim = embedDim
    _gateProj.wrappedValue = Linear(embedDim, ffnDim, bias: false)
    _upProj.wrappedValue = Linear(embedDim, ffnDim, bias: false)
    _downProj.wrappedValue = Linear(ffnDim, embedDim, bias: false)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let gate = silu(gateProj(x))
    let up = upProj(x)
    return downProj(gate * up)
  }
}

// MARK: - HeadLayer

/// A layer in the diffusion head with adaptive layer norm
class HeadLayer: Module {
  let embedDim: Int
  let condDim: Int
  let ffnDim: Int

  @ModuleInfo(key: "ffn") var ffn: DiffusionFeedForwardNetwork
  @ModuleInfo(key: "norm") var norm: DiffusionRMSNorm
  @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: Sequential

  init(embedDim: Int, ffnDim: Int, condDim: Int, normEps: Float = 1e-5) {
    self.embedDim = embedDim
    self.condDim = condDim
    self.ffnDim = ffnDim

    _ffn.wrappedValue = DiffusionFeedForwardNetwork(embedDim: embedDim, ffnDim: ffnDim)
    _norm.wrappedValue = DiffusionRMSNorm(dim: embedDim, eps: normEps)

    // AdaLN modulation: outputs shift, scale, gate
    _adaLNModulation.wrappedValue = Sequential {
      SiLU()
      Linear(condDim, 3 * embedDim, bias: false)
    }
  }

  func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
    // Get modulation parameters
    let modulation = adaLNModulation(c)
    let parts = MLX.split(modulation, parts: 3, axis: -1)
    let shiftFfn = parts[0]
    let scaleFfn = parts[1]
    let gateFfn = parts[2]

    // Apply modulated FFN
    return x + gateFfn * ffn(modulate(norm(x), shift: shiftFfn, scale: scaleFfn))
  }
}

// MARK: - FinalLayer

/// Final layer in the diffusion head
class FinalLayer: Module {
  @ModuleInfo(key: "norm_final") var normFinal: DiffusionRMSNorm
  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: Sequential

  init(hiddenSize: Int, outputSize: Int, condSize: Int, normEps: Float = 1e-5) {
    _normFinal.wrappedValue = DiffusionRMSNorm(dim: hiddenSize, eps: normEps, elementwiseAffine: false)
    _linear.wrappedValue = Linear(hiddenSize, outputSize, bias: false)

    // AdaLN modulation
    _adaLNModulation.wrappedValue = Sequential {
      SiLU()
      Linear(condSize, 2 * hiddenSize, bias: false)
    }
  }

  func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
    let modulation = adaLNModulation(c)
    let parts = MLX.split(modulation, parts: 2, axis: -1)
    let shift = parts[0]
    let scale = parts[1]
    let modulated = modulate(normFinal(x), shift: shift, scale: scale)
    return linear(modulated)
  }
}

// MARK: - DiffusionHead

/// Diffusion prediction head for VibeVoice
/// This module predicts noise/velocity for the diffusion process
class DiffusionHead: Module {
  let config: DiffusionHeadConfig
  let condDim: Int
  let latentSize: Int

  @ModuleInfo(key: "noisy_images_proj") var noisyImagesProj: Linear
  @ModuleInfo(key: "cond_proj") var condProj: Linear
  @ModuleInfo(key: "t_embedder") var tEmbedder: TimestepEmbedder
  @ModuleInfo(key: "layers") var layers: [HeadLayer]
  @ModuleInfo(key: "final_layer") var finalLayer: FinalLayer

  init(config: DiffusionHeadConfig) {
    self.config = config
    condDim = config.hiddenSize
    latentSize = config.latentSize

    // Input projections
    _noisyImagesProj.wrappedValue = Linear(latentSize, config.hiddenSize, bias: false)
    _condProj.wrappedValue = Linear(config.hiddenSize, condDim, bias: false)

    // Timestep embedder
    _tEmbedder.wrappedValue = TimestepEmbedder(hiddenSize: condDim)

    // FFN dimension
    let ffnDim = Int(Float(config.hiddenSize) * config.headFfnRatio)

    // Intermediate layers
    _layers.wrappedValue = (0 ..< config.headLayers).map { _ in
      HeadLayer(
        embedDim: config.hiddenSize,
        ffnDim: ffnDim,
        condDim: condDim,
        normEps: config.rmsNormEps
      )
    }

    // Final layer
    _finalLayer.wrappedValue = FinalLayer(
      hiddenSize: config.hiddenSize,
      outputSize: latentSize,
      condSize: condDim,
      normEps: config.rmsNormEps
    )
  }

  /// Forward pass of the prediction head
  /// - Parameters:
  ///   - noisyImages: Noisy latents to denoise, shape (B, latentSize)
  ///   - timesteps: Diffusion timesteps, shape (B,)
  ///   - condition: Conditioning information, shape (B, hiddenSize)
  /// - Returns: Predicted noise/velocity, shape (B, latentSize)
  func callAsFunction(
    noisyImages: MLXArray,
    timesteps: MLXArray,
    condition: MLXArray
  ) -> MLXArray {
    var x = noisyImagesProj(noisyImages)
    let t = tEmbedder(timesteps)
    let cond = condProj(condition)
    let c = cond + t

    for layer in layers {
      x = layer(x, c: c)
    }

    x = finalLayer(x, c: c)
    return x
  }
}
