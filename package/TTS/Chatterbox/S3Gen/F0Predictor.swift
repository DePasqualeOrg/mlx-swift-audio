//
//  F0Predictor.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/s3gen/f0_predictor.py
//  Convolutional F0 (fundamental frequency / pitch) predictor
//

import Foundation
import MLX
import MLXNN

/// Convolutional F0 (fundamental frequency / pitch) predictor.
///
/// Predicts F0 from mel-spectrogram features using stacked convolutions.
class ConvRNNF0Predictor: Module {
  let numClass: Int

  @ModuleInfo(key: "condnet") var condnet: [Conv1d]
  @ModuleInfo(key: "classifier") var classifier: Linear

  init(
    numClass: Int = 1,
    inChannels: Int = 80,
    condChannels: Int = 512,
  ) {
    self.numClass = numClass

    // Stack of convolutional layers with ELU activation
    _condnet.wrappedValue = [
      Conv1d(inputChannels: inChannels, outputChannels: condChannels, kernelSize: 3, padding: 1),
      Conv1d(inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, padding: 1),
      Conv1d(inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, padding: 1),
      Conv1d(inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, padding: 1),
      Conv1d(inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, padding: 1),
    ]

    // Final classifier
    _classifier.wrappedValue = Linear(condChannels, numClass)
  }

  /// Predict F0 from mel-spectrogram.
  ///
  /// - Parameter x: Mel-spectrogram (B, C, T)
  /// - Returns: F0 predictions (B, T)
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x is (B, C, T), but MLX Conv1d expects (B, T, C)
    var result = x.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)

    // Convolutional stack with ELU activations
    for conv in condnet {
      result = elu(conv(result))
    }

    // result is now (B, T, C) which is correct for linear layer

    // Classify and take absolute value
    result = classifier(result)
    result = result.squeezed(axis: -1) // (B, T, 1) -> (B, T)

    return abs(result)
  }
}
