//
//  LearnedPositionEmbeddings.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/t3/learned_pos_emb.py
//

import Foundation
import MLX
import MLXNN
import MLXRandom

/// Learned position embeddings for T3 model
class LearnedPositionEmbeddings: Module {
  @ModuleInfo(key: "emb") var emb: Embedding

  init(seqLen: Int, modelDim: Int, initScale: Float = 0.02) {
    _emb.wrappedValue = Embedding(embeddingCount: seqLen, dimensions: modelDim)
    super.init()

    // Initialize with normal distribution (GPT-2 style)
    let newWeight = MLXRandom.normal([seqLen, modelDim]) * initScale
    try? emb.update(parameters: ModuleParameters.unflattened(["weight": newWeight]), verify: .noUnusedKeys)
  }

  /// Returns positional embeddings for index 0 up to the length of x.
  ///
  /// - Parameter x: Input tensor of shape (B, T, ...)
  /// - Returns: Positional embeddings of shape (T, model_dim)
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let sl = x.shape[1]
    return emb(MLXArray(0 ..< sl))
  }

  /// Get positional embeddings for specific indices.
  ///
  /// - Parameter idx: Scalar int for a specific position
  /// - Returns: Positional embeddings of shape (1, 1, dim)
  func getFixedEmbedding(_ idx: Int) -> MLXArray {
    let idxArray = MLXArray([Int32(idx)]).reshaped([1, 1])
    return emb(idxArray) // (1, 1, dim)
  }

  /// Get positional embeddings for an array of indices.
  ///
  /// - Parameter idx: Array of indices
  /// - Returns: Positional embeddings
  func getFixedEmbedding(_ idx: MLXArray) -> MLXArray {
    var idxReshaped = idx
    if idx.ndim == 1 {
      idxReshaped = idx.expandedDimensions(axis: 0)
    }
    precondition(idxReshaped.ndim == 2, "Expected 2D array, got shape \(idxReshaped.shape)")
    return emb(idxReshaped)
  }
}
