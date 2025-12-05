//
//  Kokoro-tts-lib
//
import Foundation
import Hub
import MLX
import MLXNN

class KokoroWeightLoader {
  private init() {}

  static let defaultRepoId = "mlx-community/Kokoro-82M-bf16"
  static let defaultWeightsFilename = "kokoro-v1_0.safetensors"

  static func loadWeights(
    repoId: String = defaultRepoId,
    filename: String = defaultWeightsFilename,
    progressHandler: @escaping (Progress) -> Void = { _ in },
  ) async throws -> [String: MLXArray] {
    let modelDirectoryURL = try await Hub.snapshot(
      from: repoId,
      matching: [filename],
      progressHandler: progressHandler,
    )
    let weightFileURL = modelDirectoryURL.appending(path: filename)
    return try loadWeights(from: weightFileURL)
  }

  static func loadWeights(from url: URL) throws -> [String: MLXArray] {
    let weights = try MLX.loadArrays(url: url)
    var sanitizedWeights: [String: MLXArray] = [:]

    for (key, value) in weights {
      if key.hasPrefix("bert") {
        if key.contains("position_ids") {
          continue
        }
        sanitizedWeights[key] = value
      } else if key.hasPrefix("predictor") {
        // Remap duration_proj.linear_layer.* to duration_proj.*
        var remappedKey = key
        if key.contains("duration_proj.linear_layer.") {
          remappedKey = key.replacingOccurrences(of: "duration_proj.linear_layer.", with: "duration_proj.")
        }

        // Remap text_encoder.lstms.X.* to text_encoder.lstmY.* or text_encoder.normY.*
        if remappedKey.contains("text_encoder.lstms.") {
          remappedKey = remapDurationEncoderKey(remappedKey)
        }

        if remappedKey.contains("F0_proj.weight") {
          sanitizedWeights[remappedKey] = value.transposed(0, 2, 1)
        } else if remappedKey.contains("N_proj.weight") {
          sanitizedWeights[remappedKey] = value.transposed(0, 2, 1)
        } else if remappedKey.contains("weight_v") {
          if checkArrayShape(arr: value) {
            sanitizedWeights[remappedKey] = value
          } else {
            sanitizedWeights[remappedKey] = value.transposed(0, 2, 1)
          }
        } else {
          sanitizedWeights[remappedKey] = value
        }
      } else if key.hasPrefix("text_encoder") {
        // Remap text_encoder.cnn.X.0.* to text_encoder.cnn.X.conv.*
        // Remap text_encoder.cnn.X.1.* to text_encoder.cnn.X.norm.*
        var remappedKey = key
        if key.contains(".cnn.") {
          remappedKey = remapTextEncoderCNNKey(key)
        }

        if remappedKey.contains("weight_v") {
          if checkArrayShape(arr: value) {
            sanitizedWeights[remappedKey] = value
          } else {
            sanitizedWeights[remappedKey] = value.transposed(0, 2, 1)
          }
        } else {
          sanitizedWeights[remappedKey] = value
        }
      } else if key.hasPrefix("decoder") {
        if key.contains("noise_convs"), key.hasSuffix(".weight") {
          sanitizedWeights[key] = value.transposed(0, 2, 1)
        } else if key.contains("weight_v") {
          if checkArrayShape(arr: value) {
            sanitizedWeights[key] = value
          } else {
            sanitizedWeights[key] = value.transposed(0, 2, 1)
          }
        } else {
          sanitizedWeights[key] = value
        }
      }
    }

    return sanitizedWeights
  }

  private static func checkArrayShape(arr: MLXArray) -> Bool {
    guard arr.shape.count != 3 else { return false }

    let outChannels = arr.shape[0]
    let kH = arr.shape[1]
    let kW = arr.shape[2]

    return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
  }

  /// Remap predictor.text_encoder.lstms.X.* to predictor.text_encoder.lstmY.* or normY.*
  /// lstms.0 → lstm0, lstms.1 → norm0, lstms.2 → lstm1, lstms.3 → norm1, etc.
  private static func remapDurationEncoderKey(_ key: String) -> String {
    // Pattern: predictor.text_encoder.lstms.{idx}.{rest}
    let parts = key.split(separator: ".")
    guard parts.count >= 5,
          parts[0] == "predictor",
          parts[1] == "text_encoder",
          parts[2] == "lstms",
          let idx = Int(parts[3])
    else {
      return key
    }

    // Even indices (0, 2, 4) are LSTMs → lstm0, lstm1, lstm2
    // Odd indices (1, 3, 5) are norms → norm0, norm1, norm2
    let newName = if idx % 2 == 0 {
      "lstm\(idx / 2)"
    } else {
      "norm\(idx / 2)"
    }

    var newParts = Array(parts[0 ..< 2]) // predictor.text_encoder
    newParts.append(Substring(newName))
    newParts.append(contentsOf: parts[4...])
    return newParts.joined(separator: ".")
  }

  /// Remap text_encoder.cnn.X.0.* to text_encoder.cnn.X.conv.*
  /// Remap text_encoder.cnn.X.1.* to text_encoder.cnn.X.norm.*
  private static func remapTextEncoderCNNKey(_ key: String) -> String {
    // Pattern: text_encoder.cnn.{blockIdx}.{layerIdx}.{rest}
    // where layerIdx is 0 (conv) or 1 (norm)
    let parts = key.split(separator: ".")
    guard parts.count >= 5,
          parts[0] == "text_encoder",
          parts[1] == "cnn"
    else {
      return key
    }

    // parts[2] is block index (0, 1, 2, ...)
    // parts[3] is layer index (0 or 1)
    // parts[4...] is the rest (weight_g, weight_v, bias, gamma, beta)
    let layerIndex = parts[3]
    let layerName: String
    if layerIndex == "0" {
      layerName = "conv"
    } else if layerIndex == "1" {
      layerName = "norm"
    } else {
      return key
    }

    var newParts = Array(parts[0 ..< 3])
    newParts.append(Substring(layerName))
    newParts.append(contentsOf: parts[4...])
    return newParts.joined(separator: ".")
  }
}
