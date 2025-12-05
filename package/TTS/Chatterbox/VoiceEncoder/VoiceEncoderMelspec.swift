//
//  VoiceEncoderMelspec.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/voice_encoder/melspec.py
//

import Foundation
import MLX
import MLXNN

/// Compute mel spectrogram from waveform for voice encoder
///
/// - Parameters:
///   - wav: Waveform (T,) in 16kHz
///   - config: Voice encoder configuration
///   - pad: Whether to pad the STFT
/// - Returns: Mel spectrogram (M, T')
func voiceEncoderMelspectrogram(
  wav: MLXArray,
  config: VoiceEncConfig,
  pad _: Bool = true,
) -> MLXArray {
  // Create Hann window
  let window = hanningWindow(length: config.winSize + 1)[0 ..< config.winSize]

  // Compute STFT
  let spec = stft(
    wav,
    window: window,
    nFft: config.nFft,
    hopLength: config.hopSize,
    winLength: config.winSize,
  )

  // Get magnitudes
  var specMagnitudes = MLX.abs(spec) // (T', F)

  // Apply power
  if config.melPower != 1.0 {
    specMagnitudes = MLX.pow(specMagnitudes, config.melPower)
  }

  // Create mel filterbank
  let filters = melFilters(
    sampleRate: config.sampleRate,
    nFft: config.nFft,
    nMels: config.numMels,
    fMin: Float(config.fmin),
    fMax: Float(config.fmax),
  )

  // Apply mel filterbank: (T', F) @ (F, M) -> (T', M)
  var mel = MLX.matmul(specMagnitudes, filters.T)
  mel = mel.transposed(1, 0) // (M, T')

  // Convert to dB if needed
  if config.melType == "db" {
    mel = 20 * MLX.log10(MLX.maximum(mel, MLXArray(config.stftMagnitudeMin)))
  }

  // Normalize if needed
  if config.normalizedMels {
    let minLevelDb = 20 * log10(config.stftMagnitudeMin)
    let headroomDb: Float = 15
    mel = (mel - minLevelDb) / (-minLevelDb + headroomDb)
  }

  return mel // (M, T')
}
