//
//  VoiceEncoder.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/voice_encoder/voice_encoder.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Utility Functions

/// Calculate number of windows and target length for partial utterance splitting
func getNumWins(
  nFrames: Int,
  step: Int,
  minCoverage: Float,
  config: VoiceEncConfig,
) -> (Int, Int) {
  precondition(nFrames > 0, "nFrames must be positive")
  let winSize = config.vePartialFrames
  var nWins = max(nFrames - winSize + step, 0) / step
  let remainder = max(nFrames - winSize + step, 0) % step

  if nWins == 0 || Float(remainder + (winSize - step)) / Float(winSize) >= minCoverage {
    nWins += 1
  }
  let targetN = winSize + step * (nWins - 1)
  return (nWins, targetN)
}

/// Compute how many frames separate two partial utterances
func getFrameStep(
  overlap: Float,
  rate: Float?,
  config: VoiceEncConfig,
) -> Int {
  precondition(overlap >= 0 && overlap < 1, "Overlap must be in [0, 1)")

  var frameStep = if let rate {
    Int(round(Float(config.sampleRate) / rate / Float(config.vePartialFrames)))
  } else {
    Int(round(Float(config.vePartialFrames) * (1 - overlap)))
  }

  precondition(frameStep > 0 && frameStep <= config.vePartialFrames, "Invalid frame step")
  return frameStep
}

// MARK: - Voice Encoder

/// LSTM-based voice encoder for speaker embeddings
class VoiceEncoder: Module {
  let config: VoiceEncConfig

  @ModuleInfo(key: "lstm") var lstm: ChatterboxLSTM
  @ModuleInfo(key: "proj") var proj: Linear

  // Cosine similarity scaling (learnable parameters)
  @ParameterInfo(key: "similarity_weight") var similarityWeight: MLXArray
  @ParameterInfo(key: "similarity_bias") var similarityBias: MLXArray

  init(config: VoiceEncConfig = VoiceEncConfig()) {
    self.config = config

    _lstm.wrappedValue = ChatterboxLSTM(
      inputSize: config.numMels,
      hiddenSize: config.veHiddenSize,
      numLayers: 3,
    )
    _proj.wrappedValue = Linear(config.veHiddenSize, config.speakerEmbedSize)

    _similarityWeight.wrappedValue = MLXArray([10.0])
    _similarityBias.wrappedValue = MLXArray([-5.0])
  }

  /// Computes the embeddings of a batch of partial utterances.
  ///
  /// - Parameter mels: Batch of unscaled mel spectrograms (B, T, M) where T is vePartialFrames
  /// - Returns: Embeddings as (B, E) where E is speakerEmbedSize. Embeddings are L2-normed.
  func callAsFunction(_ mels: MLXArray) -> MLXArray {
    if config.normalizedMels {
      let minVal = mels.min()
      let maxVal = mels.max()
      if minVal.item(Float.self) < 0 || maxVal.item(Float.self) > 1 {
        fatalError("Mels outside [0, 1]. Min=\(minVal), Max=\(maxVal)")
      }
    }

    // Pass full sequence through LSTM layers
    let (_, (hN, _)) = lstm(mels)

    // Get final hidden state from last layer
    let finalHidden = hN[-1] // (B, H)

    // Project
    var rawEmbeds = proj(finalHidden)

    // Apply ReLU if configured
    if config.veFinalRelu {
      rawEmbeds = relu(rawEmbeds)
    }

    // L2 normalize
    let norm = MLX.sqrt(MLX.sum(rawEmbeds * rawEmbeds, axis: 1, keepDims: true))
    let embeds = rawEmbeds / norm

    return embeds
  }

  /// Computes embeddings of a batch of full utterances.
  ///
  /// - Parameters:
  ///   - mels: (B, T, M) unscaled mels
  ///   - melLens: List of mel lengths for each batch item
  ///   - overlap: Overlap between partial windows (default 0.5)
  ///   - rate: Rate for frame step calculation
  ///   - minCoverage: Minimum coverage for partial windows (default 0.8)
  ///   - batchSize: Batch size for processing partials
  /// - Returns: (B, E) embeddings
  func inference(
    mels: MLXArray,
    melLens: [Int],
    overlap: Float = 0.5,
    rate: Float? = nil,
    minCoverage: Float = 0.8,
    batchSize: Int? = nil,
  ) -> MLXArray {
    // Compute where to split the utterances into partials
    let frameStep = getFrameStep(overlap: overlap, rate: rate, config: config)

    var nPartialsList: [Int] = []
    var targetLens: [Int] = []

    for l in melLens {
      let (nP, tL) = getNumWins(nFrames: l, step: frameStep, minCoverage: minCoverage, config: config)
      nPartialsList.append(nP)
      targetLens.append(tL)
    }

    // Possibly pad the mels to reach the target lengths
    var paddedMels = mels
    let lenDiff = (targetLens.max() ?? 0) - mels.shape[1]
    if lenDiff > 0 {
      let pad = MLXArray.zeros([mels.shape[0], lenDiff, config.numMels])
      paddedMels = MLX.concatenated([mels, pad], axis: 1)
    }

    // Group all partials together
    var partialList: [MLXArray] = []

    for (idx, nPartial) in nPartialsList.enumerated() {
      if nPartial > 0 {
        let mel = paddedMels[idx]
        // Extract partials
        for p in 0 ..< nPartial {
          let start = p * frameStep
          let end = start + config.vePartialFrames
          let partial = mel[start ..< end, 0...]
          partialList.append(partial)
        }
      }
    }

    guard !partialList.isEmpty else {
      return MLXArray.zeros([melLens.count, config.speakerEmbedSize])
    }

    let partials = MLX.stacked(partialList, axis: 0) // (total_partials, T, M)

    // Forward the partials (in batches if needed)
    var partialEmbeds: MLXArray
    if batchSize == nil || batchSize! >= partials.shape[0] {
      partialEmbeds = callAsFunction(partials)
    } else {
      var embedChunks: [MLXArray] = []
      let bs = batchSize!
      for i in stride(from: 0, to: partials.shape[0], by: bs) {
        let end = min(i + bs, partials.shape[0])
        let chunk = partials[i ..< end]
        embedChunks.append(callAsFunction(chunk))
      }
      partialEmbeds = MLX.concatenated(embedChunks, axis: 0)
    }

    // Reduce the partial embeds into full embeds
    var slices = [0]
    for n in nPartialsList {
      slices.append(slices.last! + n)
    }

    // Reduce partials to embeddings (independent - can be parallelized)
    let rawEmbeds = (0 ..< (slices.count - 1)).map { i -> MLXArray in
      MLX.mean(partialEmbeds[slices[i] ..< slices[i + 1]], axis: 0)
    }
    let rawEmbedsStacked = MLX.stacked(rawEmbeds, axis: 0)

    // L2-normalize the final embeds
    let norm = MLX.sqrt(MLX.sum(rawEmbedsStacked * rawEmbedsStacked, axis: 1, keepDims: true))
    let embeds = rawEmbedsStacked / norm

    return embeds
  }

  /// Takes L2-normalized utterance embeddings, computes mean and L2-normalizes to get a speaker embedding.
  static func uttToSpkEmbed(_ uttEmbeds: MLXArray) -> MLXArray {
    precondition(uttEmbeds.ndim == 2, "Expected 2D tensor for utterance embeddings")
    let mean = MLX.mean(uttEmbeds, axis: 0)
    let norm = MLX.sqrt(MLX.sum(mean * mean))
    return mean / norm
  }

  /// Cosine similarity for L2-normalized embeddings.
  static func voiceSimilarity(_ embedsX: MLXArray, _ embedsY: MLXArray) -> Float {
    var x = embedsX
    var y = embedsY

    if x.ndim != 1 {
      x = uttToSpkEmbed(x)
    }
    if y.ndim != 1 {
      y = uttToSpkEmbed(y)
    }

    return MLX.sum(x * y).item(Float.self)
  }

  /// Convenience function for deriving utterance or speaker embeddings from mel spectrograms.
  ///
  /// - Parameters:
  ///   - mels: List of (Ti, M) arrays
  ///   - melLens: Individual mel lengths if mels is stacked
  ///   - asSpk: Whether to return speaker embedding (single) or utterance embeddings (per-utt)
  ///   - batchSize: Batch size for processing
  ///   - overlap: Overlap between partial windows
  ///   - rate: Rate for frame step calculation
  ///   - minCoverage: Minimum coverage for partial windows
  /// - Returns: (B, E) embeddings if asSpk is false, else (E,) speaker embedding
  func embedsFromMels(
    mels: [MLXArray],
    melLens: [Int]? = nil,
    asSpk: Bool = false,
    batchSize: Int = 32,
    overlap: Float = 0.5,
    rate: Float? = nil,
    minCoverage: Float = 0.8,
  ) -> MLXArray {
    let actualMelLens = melLens ?? mels.map { $0.shape[0] }
    let maxLen = actualMelLens.max() ?? 0

    // Pad and stack
    var padded: [MLXArray] = []
    for mel in mels {
      if mel.shape[0] < maxLen {
        let pad = MLXArray.zeros([maxLen - mel.shape[0], mel.shape[1]])
        padded.append(MLX.concatenated([mel, pad], axis: 0))
      } else {
        padded.append(mel)
      }
    }
    let melStacked = MLX.stacked(padded, axis: 0)

    // Embed them
    let uttEmbeds = inference(
      mels: melStacked,
      melLens: actualMelLens,
      overlap: overlap,
      rate: rate,
      minCoverage: minCoverage,
      batchSize: batchSize,
    )

    return asSpk ? VoiceEncoder.uttToSpkEmbed(uttEmbeds) : uttEmbeds
  }

  /// Wrapper around embedsFromMels that first converts waveforms to mel spectrograms.
  ///
  /// - Parameters:
  ///   - wavs: List of waveform arrays (at 16kHz)
  ///   - asSpk: Whether to return speaker embedding
  ///   - batchSize: Batch size for processing
  ///   - rate: Rate for frame step calculation (default 1.3)
  /// - Returns: Embeddings
  func embedsFromWavs(
    wavs: [MLXArray],
    asSpk: Bool = false,
    batchSize: Int = 32,
    rate: Float = 1.3,
  ) -> MLXArray {
    // Convert waveforms to mel spectrograms
    var mels: [MLXArray] = []
    for wav in wavs {
      let mel = voiceEncoderMelspectrogram(wav: wav, config: config)
      // Transpose from (M, T) to (T, M)
      mels.append(mel.transposed())
    }

    return embedsFromMels(mels: mels, asSpk: asSpk, batchSize: batchSize, rate: rate)
  }
}
