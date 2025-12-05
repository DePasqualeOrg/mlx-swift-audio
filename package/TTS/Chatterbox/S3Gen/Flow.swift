//
//  Flow.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/s3gen/flow.py
//  Causal masked diffusion model with speaker embeddings
//

import Foundation
import MLX
import MLXLinalg
import MLXNN

// MARK: - CausalMaskedDiffWithXvec

/// Causal masked diffusion model with speaker embeddings for streaming TTS
class CausalMaskedDiffWithXvec: Module {
  let inputSize: Int
  let outputSize: Int
  let vocabSize: Int
  let inputFrameRate: Int
  let onlyMaskLoss: Bool
  let tokenMelRatio: Int
  let preLookaheadLen: Int

  @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
  @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffineLayer: Linear
  @ModuleInfo(key: "encoder") var encoder: UpsampleConformerEncoder
  @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
  @ModuleInfo(key: "decoder") var decoder: CausalConditionalCFM

  init(
    inputSize: Int = 512,
    outputSize: Int = 80,
    spkEmbedDim: Int = 192,
    outputType _: String = "mel",
    vocabSize: Int = 6561,
    inputFrameRate: Int = 25,
    onlyMaskLoss: Bool = true,
    tokenMelRatio: Int = 2,
    preLookaheadLen: Int = 3,
    encoder: UpsampleConformerEncoder,
    decoder: CausalConditionalCFM,
  ) {
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.vocabSize = vocabSize
    self.inputFrameRate = inputFrameRate
    self.onlyMaskLoss = onlyMaskLoss
    self.tokenMelRatio = tokenMelRatio
    self.preLookaheadLen = preLookaheadLen

    _inputEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: inputSize)
    _spkEmbedAffineLayer.wrappedValue = Linear(spkEmbedDim, outputSize)
    _encoder.wrappedValue = encoder
    _encoderProj.wrappedValue = Linear(encoder.outputSize(), outputSize)
    _decoder.wrappedValue = decoder
  }

  /// Inference for streaming TTS
  func inference(
    token: MLXArray,
    tokenLen: MLXArray,
    promptToken: MLXArray,
    promptTokenLen: MLXArray,
    promptFeat: MLXArray,
    promptFeatLen _: MLXArray,
    embedding: MLXArray,
    finalize: Bool,
  ) -> (MLXArray, MLXArray?) {
    precondition(token.shape[0] == 1, "Batch size must be 1")

    // Speaker embedding projection (normalize then project)
    var embeddingNorm = embedding / MLXLinalg.norm(embedding, axis: 1, keepDims: true)
    embeddingNorm = spkEmbedAffineLayer(embeddingNorm)

    // Concatenate prompt and new tokens
    var combinedToken = MLX.concatenated([promptToken, token], axis: 1)
    var combinedTokenLen = promptTokenLen + tokenLen

    // Create mask
    let batchSize = combinedTokenLen.shape[0]
    let maxLen = Int(combinedTokenLen.max().item(Int32.self))
    let seqRange = MLXArray(0 ..< maxLen)
    let seqRangeExpand = MLX.broadcast(
      seqRange.expandedDimensions(axis: 0),
      to: [batchSize, maxLen],
    )
    let seqLengthExpand = combinedTokenLen.expandedDimensions(axis: -1)
    let mask = (seqRangeExpand .< seqLengthExpand).expandedDimensions(axis: -1).asType(embeddingNorm.dtype)

    // Embed tokens
    let numEmbeddings = inputEmbedding.weight.shape[0]
    combinedToken = MLX.clip(combinedToken, min: 0, max: numEmbeddings - 1)
    var tokenEmbed = inputEmbedding(combinedToken) * mask

    // Encode
    var (h, hLengths) = encoder(tokenEmbed, xsLens: combinedTokenLen.asType(.int32))

    // Trim lookahead for streaming (unless finalizing)
    if !finalize {
      let trimLen = preLookaheadLen * tokenMelRatio
      h = h[0..., 0 ..< h.shape[1] - trimLen, 0...]
    }

    let melLen1 = promptFeat.shape[1]
    let melLen2 = h.shape[1] - promptFeat.shape[1]
    h = encoderProj(h)

    // Get conditions (prompt mel features)
    var conds = MLXArray.zeros([1, melLen1 + melLen2, outputSize]).asType(h.dtype)
    // Copy prompt features to conds (first melLen1 frames)
    conds[0..., 0 ..< melLen1, 0...] = promptFeat
    conds = conds.transposed(0, 2, 1) // (1, D, T)

    // Create mask for decoder
    let totalLen = melLen1 + melLen2
    let decoderMask = MLXArray.ones([1, 1, totalLen]).asType(h.dtype)

    // Generate mel features
    let (feat, _) = decoder(
      mu: h.transposed(0, 2, 1),
      mask: decoderMask,
      nTimesteps: 10,
      temperature: 1.0,
      spks: embeddingNorm,
      cond: conds,
    )

    // Extract only the new portion (after prompt)
    let outputFeat = feat[0..., 0..., melLen1...]
    precondition(outputFeat.shape[2] == melLen2, "Output length mismatch")

    return (outputFeat, nil)
  }
}
