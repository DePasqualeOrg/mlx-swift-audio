// CAMPlus Speaker Encoder for CosyVoice2
// Wrapper around the existing CAMPPlus implementation from Chatterbox

import Foundation
import MLX
import MLXNN

/// CAMPlus speaker encoder that extracts 192-dim speaker embeddings.
/// Uses the pure MLX implementation from Chatterbox (no ONNX dependency).
class CAMPlusSpeakerEncoder {
  let embeddingDim: Int = 192
  let model: CAMPPlus
  private(set) var isLoaded: Bool = false

  /// Initialize CAMPlus speaker encoder.
  /// - Parameter modelPath: Path to CAMPlus weights file (.safetensors). If nil, model will be initialized but not loaded.
  init(modelPath: String? = nil) {
    model = CAMPPlus(
      featDim: 80,
      embeddingSize: embeddingDim,
      growthRate: 32,
      bnSize: 4,
      initChannels: 128,
      configStr: "batchnorm-relu",
      outputLevel: "segment"
    )

    if let path = modelPath {
      loadWeights(path)
    }
  }

  /// Load model weights from a pre-loaded dictionary.
  /// This is the preferred method when loading from a combined model.safetensors file.
  /// - Parameter weights: Dictionary of weight arrays (already stripped of 'campplus.' prefix)
  func loadWeights(from weights: [String: MLXArray]) {
    guard !weights.isEmpty else {
      print("Warning: Empty weights dictionary provided. Speaker embeddings will be zeros.")
      return
    }

    do {
      let weightsList = weights.map { (key: $0.key, value: $0.value) }
      try model.update(parameters: ModuleParameters.unflattened(weightsList), verify: [.noUnusedKeys])

      // Set eval mode for inference (BatchNorm uses running stats)
      model.train(false)
      isLoaded = true
    } catch {
      print("Warning: Failed to load CAMPlus weights: \(error). Speaker embeddings will be zeros.")
    }
  }

  /// Load model weights from file.
  /// - Parameter modelPath: Path to weights file (.safetensors or directory containing campplus.safetensors or model.safetensors)
  func loadWeights(_ modelPath: String) {
    let path = URL(fileURLWithPath: modelPath)
    var weightsPath = path

    if path.hasDirectoryPath {
      // First check for standalone campplus.safetensors
      let campplusSafetensors = path.appendingPathComponent("campplus.safetensors")
      let modelSafetensors = path.appendingPathComponent("model.safetensors")

      if FileManager.default.fileExists(atPath: campplusSafetensors.path) {
        weightsPath = campplusSafetensors
      } else if FileManager.default.fileExists(atPath: modelSafetensors.path) {
        // Load from main model.safetensors and extract campplus weights
        weightsPath = modelSafetensors
      } else {
        print(
          "Warning: No campplus weights found in \(modelPath). Speaker embeddings will be zeros.")
        return
      }
    }

    do {
      let weights = try MLX.loadArrays(url: weightsPath)

      // Sanitize weights (handles key remapping if needed)
      let sanitizedWeights = sanitizeWeights(weights)

      // Load into model
      let weightsList = sanitizedWeights.map { (key: $0.key, value: $0.value) }
      try model.update(parameters: ModuleParameters.unflattened(weightsList), verify: [.noUnusedKeys])

      // Set eval mode for inference (BatchNorm uses running stats)
      model.train(false)
      isLoaded = true
    } catch {
      print(
        "Warning: Failed to load CAMPlus weights from \(modelPath): \(error). Speaker embeddings will be zeros."
      )
    }
  }

  /// Sanitize weights from Python format to Swift format
  private func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]

    for (key, value) in weights {
      // Remove 'campplus.' prefix if present
      var newKey = key
      if newKey.hasPrefix("campplus.") {
        newKey = String(newKey.dropFirst("campplus.".count))
      }

      // Handle BatchNorm running_mean/running_var -> running_mean/running_var
      // (MLX BatchNorm uses same names)

      sanitized[newKey] = value
    }

    return sanitized
  }

  /// Extract speaker embedding from audio.
  /// - Parameters:
  ///   - audio: Audio waveform at 16kHz. Can be (T,) or (B, T)
  ///   - sampleRate: Sample rate (should be 16000)
  /// - Returns: Speaker embedding (1, 192) or (B, 192)
  func callAsFunction(_ audio: MLXArray, sampleRate _: Int = 16000) -> MLXArray {
    if !isLoaded {
      // Return zeros if model not loaded
      if audio.ndim == 1 {
        return MLXArray.zeros([1, embeddingDim])
      } else {
        return MLXArray.zeros([audio.shape[0], embeddingDim])
      }
    }

    // Ensure 1D or 2D
    var audioData = audio
    if audioData.ndim > 2 {
      audioData = audioData.squeezed()
    }

    // Use the model's inference method which handles fbank extraction
    var embedding = model.inference(audioData)

    // Ensure output is (B, embeddingDim)
    if embedding.ndim == 1 {
      embedding = embedding.expandedDimensions(axis: 0)
    }

    return embedding
  }

  /// Extract 80-dim filter bank features from audio.
  /// Uses the Kaldi-compatible fbank implementation from Chatterbox.
  /// - Parameters:
  ///   - audio: Audio waveform (samples,) at 16kHz
  ///   - sampleRate: Sample rate (should be 16000)
  /// - Returns: Filter bank features (T, 80)
  func extractFbank(_ audio: MLXArray, sampleRate: Int = 16000) -> MLXArray {
    // Ensure 1D
    var audioData = audio
    if audioData.ndim > 1 {
      audioData = audioData.squeezed()
    }

    return kaldiFbankCAMPPlus(audio: audioData, sampleRate: sampleRate, numMelBins: 80)
  }
}
