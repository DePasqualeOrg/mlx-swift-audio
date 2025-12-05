//
//  T3CondEnc.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/t3/cond_enc.py
//

import Foundation
import MLX
import MLXNN

// MARK: - T3Cond

/// Container for T3 conditioning information
struct T3Cond: @unchecked Sendable {
  /// Speaker embedding from voice encoder (B, speaker_dim)
  var speakerEmb: MLXArray

  /// Optional CLAP embedding for semantic conditioning
  var clapEmb: MLXArray?

  /// Optional speech token prompt (B, T)
  var condPromptSpeechTokens: MLXArray?

  /// Optional embedded speech prompt (B, T, D)
  var condPromptSpeechEmb: MLXArray?

  /// Emotion exaggeration factor, typically 0.3-0.7 (scalar or (B, 1))
  var emotionAdv: MLXArray

  init(
    speakerEmb: MLXArray,
    clapEmb: MLXArray? = nil,
    condPromptSpeechTokens: MLXArray? = nil,
    condPromptSpeechEmb: MLXArray? = nil,
    emotionAdv: MLXArray? = nil,
  ) {
    self.speakerEmb = speakerEmb
    self.clapEmb = clapEmb
    self.condPromptSpeechTokens = condPromptSpeechTokens
    self.condPromptSpeechEmb = condPromptSpeechEmb
    self.emotionAdv = emotionAdv ?? MLXArray(0.5)
  }
}

// MARK: - T3CondEnc

/// Conditioning encoder for T3 model.
/// Handles speaker embeddings, emotion control, and prompt speech tokens.
class T3CondEnc: Module {
  let config: T3Config

  @ModuleInfo(key: "spkr_enc") var spkrEnc: Linear
  @ModuleInfo(key: "emotion_adv_fc") var emotionAdvFc: Linear
  @ModuleInfo(key: "perceiver") var perceiver: Perceiver

  init(config: T3Config) {
    self.config = config

    // Speaker embedding projection
    if config.encoderType == "voice_encoder" {
      _spkrEnc.wrappedValue = Linear(config.speakerEmbedSize, config.nChannels)
    } else {
      fatalError("encoder_type '\(config.encoderType)' not supported")
    }

    // Emotion control - always create since model weights include it
    _emotionAdvFc.wrappedValue = Linear(1, config.nChannels, bias: false)

    // Perceiver resampler - always create since model weights include it
    _perceiver.wrappedValue = Perceiver()

    super.init()
  }

  /// Process conditioning inputs into a single conditioning tensor.
  ///
  /// - Parameter cond: T3Cond struct with conditioning information
  /// - Returns: Conditioning embeddings (B, cond_len, D)
  func callAsFunction(_ cond: T3Cond) -> MLXArray {
    // Validate
    let hasTokens = cond.condPromptSpeechTokens != nil
    let hasEmb = cond.condPromptSpeechEmb != nil
    precondition(
      hasTokens == hasEmb,
      "condPromptSpeechTokens and condPromptSpeechEmb must both be provided or both be nil",
    )

    // Speaker embedding projection (B, speaker_dim) -> (B, 1, D)
    let B = cond.speakerEmb.shape[0]
    var condSpkr = spkrEnc(cond.speakerEmb.reshaped([B, config.speakerEmbedSize]))
    condSpkr = condSpkr.expandedDimensions(axis: 1) // (B, 1, D)

    // Empty placeholder for concatenation
    let empty = condSpkr[0..., 0 ..< 0, 0...] // (B, 0, D)

    // CLAP embedding (not implemented yet)
    if cond.clapEmb != nil {
      fatalError("clapEmb not yet implemented")
    }
    let condClap = empty // (B, 0, D)

    // Conditional prompt speech embeddings
    var condPromptSpeechEmb = cond.condPromptSpeechEmb
    if condPromptSpeechEmb == nil {
      condPromptSpeechEmb = empty // (B, 0, D)
    } else if config.usePerceiverResampler {
      // Resample to fixed length using Perceiver
      condPromptSpeechEmb = perceiver(condPromptSpeechEmb!)
    }

    // Emotion exaggeration
    var condEmotionAdv = empty // (B, 0, D)
    if config.emotionAdv {
      // Reshape to (B, 1, 1)
      var emotionVal = cond.emotionAdv
      if emotionVal.ndim == 0 {
        emotionVal = emotionVal.reshaped([1, 1, 1])
      } else if emotionVal.ndim == 1 {
        emotionVal = emotionVal.reshaped([-1, 1, 1])
      } else if emotionVal.ndim == 2 {
        emotionVal = emotionVal.expandedDimensions(axis: -1)
      }

      condEmotionAdv = emotionAdvFc(emotionVal)
    }

    // Concatenate all conditioning signals
    let condEmbeds = MLX.concatenated([
      condSpkr,
      condClap,
      condPromptSpeechEmb!,
      condEmotionAdv,
    ], axis: 1)

    return condEmbeds
  }
}
