//
//  OuteTTSTokens.swift
//  MLXAudio
//
//  Special tokens for OuteTTS text and audio processing
//  Ported from: mlx_audio/tts/models/outetts/tokens.py
//

import Foundation

/// Special tokens used for OuteTTS text and audio processing
struct OuteTTSSpecialTokens {
  // Sequence markers
  let bos = "<|im_start|>"
  let eos = "<|im_end|>"

  // Audio code tokens (format strings)
  let c1 = "<|c1_%d|>" // Codebook 1 token
  let c2 = "<|c2_%d|>" // Codebook 2 token

  // Text markers
  let textStart = "<|text_start|>"
  let textEnd = "<|text_end|>"

  // Voice characteristic markers
  let voiceCharacteristicStart = "<|voice_characteristic_start|>"
  let voiceCharacteristicEnd = "<|voice_characteristic_end|>"

  // Emotion markers
  let emotionStart = "<|emotion_start|>"
  let emotionEnd = "<|emotion_end|>"

  // Audio markers
  let audioStart = "<|audio_start|>"
  let audioEnd = "<|audio_end|>"

  // Time token (format string with 2 decimal places)
  let time = "<|t_%.2f|>"

  // Code marker
  let code = "<|code|>"

  // Audio feature tokens (format strings)
  let energy = "<|energy_%d|>"
  let spectralCentroid = "<|spectral_centroid_%d|>"
  let pitch = "<|pitch_%d|>"

  // Word markers
  let wordStart = "<|word_start|>"
  let wordEnd = "<|word_end|>"

  // Feature markers
  let features = "<|features|>"
  let globalFeaturesStart = "<|global_features_start|>"
  let globalFeaturesEnd = "<|global_features_end|>"

  init() {}

  /// Format a codebook 1 token
  func formatC1(_ value: Int) -> String {
    String(format: c1, value)
  }

  /// Format a codebook 2 token
  func formatC2(_ value: Int) -> String {
    String(format: c2, value)
  }

  /// Format a time token
  func formatTime(_ duration: Double) -> String {
    String(format: time, duration)
  }

  /// Format an energy token
  func formatEnergy(_ value: Int) -> String {
    String(format: energy, value)
  }

  /// Format a spectral centroid token
  func formatSpectralCentroid(_ value: Int) -> String {
    String(format: spectralCentroid, value)
  }

  /// Format a pitch token
  func formatPitch(_ value: Int) -> String {
    String(format: pitch, value)
  }
}

/// Audio features for a word or segment
struct OuteTTSAudioFeatures: Codable, Sendable {
  var energy: Int
  var spectralCentroid: Int
  var pitch: Int

  enum CodingKeys: String, CodingKey {
    case energy
    case spectralCentroid = "spectral_centroid"
    case pitch
  }

  init(energy: Int = 0, spectralCentroid: Int = 0, pitch: Int = 0) {
    self.energy = energy
    self.spectralCentroid = spectralCentroid
    self.pitch = pitch
  }
}

/// Word data with audio codes and features
struct OuteTTSWordData: Codable, Sendable {
  var word: String
  var duration: Double
  var c1: [Int]
  var c2: [Int]
  var features: OuteTTSAudioFeatures

  init(word: String, duration: Double, c1: [Int], c2: [Int], features: OuteTTSAudioFeatures) {
    self.word = word
    self.duration = duration
    self.c1 = c1
    self.c2 = c2
    self.features = features
  }
}

/// Speaker profile
struct OuteTTSSpeakerProfile: Codable, Sendable {
  var text: String
  var words: [OuteTTSWordData]
  var globalFeatures: OuteTTSAudioFeatures

  enum CodingKeys: String, CodingKey {
    case text
    case words
    case globalFeatures = "global_features"
  }

  init(text: String, words: [OuteTTSWordData], globalFeatures: OuteTTSAudioFeatures) {
    self.text = text
    self.words = words
    self.globalFeatures = globalFeatures
  }

  /// Load speaker profile from JSON file
  static func load(from path: String) throws -> OuteTTSSpeakerProfile {
    let expandedPath = NSString(string: path).expandingTildeInPath
    let url = URL(fileURLWithPath: expandedPath)
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(OuteTTSSpeakerProfile.self, from: data)
  }

  /// Save speaker profile to JSON file
  func save(to path: String) throws {
    let expandedPath = NSString(string: path).expandingTildeInPath
    let url = URL(fileURLWithPath: expandedPath)

    // Create directory if needed
    let directory = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

    let encoder = JSONEncoder()
    encoder.outputFormatting = .prettyPrinted
    let data = try encoder.encode(self)
    try data.write(to: url)
  }
}
