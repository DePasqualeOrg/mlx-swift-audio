//
//  AudioResult.swift
//  MLXAudio
//
//  Represents the result of TTS audio generation.
//

import Foundation

/// Audio generation results are either in-memory samples (for streaming/playback)
/// or a saved file (for sharing/export).
///
/// Using an enum makes the intent explicit and provides type safety.
public enum AudioResult: Sendable {
  /// In-memory audio samples ready for playback or further processing
  case samples(data: [Float], sampleRate: Int, processingTime: TimeInterval)

  /// Audio saved to a file URL (for sharing/export)
  case file(url: URL, processingTime: TimeInterval)

  // MARK: - Computed Properties

  /// Time taken to generate this audio (seconds)
  public var processingTime: TimeInterval {
    switch self {
      case let .samples(_, _, time), let .file(_, time):
        time
    }
  }

  /// Sample rate if available (nil for file-only results)
  public var sampleRate: Int? {
    switch self {
      case let .samples(_, rate, _):
        rate
      case .file:
        nil
    }
  }

  /// Number of samples if available
  public var sampleCount: Int? {
    switch self {
      case let .samples(data, _, _):
        data.count
      case .file:
        nil
    }
  }

  /// Audio duration in seconds (if calculable)
  public var duration: TimeInterval? {
    switch self {
      case let .samples(data, rate, _):
        Double(data.count) / Double(rate)
      case .file:
        nil
    }
  }

  /// Real-time factor (RTF) - how fast the generation was relative to audio length
  /// Values < 1.0 mean faster than real-time
  public var realTimeFactor: Double? {
    guard let duration, duration > 0 else { return nil }
    return processingTime / duration
  }

  /// File URL if this result has been saved to disk
  public var fileURL: URL? {
    switch self {
      case .samples:
        nil
      case let .file(url, _):
        url
    }
  }

  /// Raw audio samples if available
  public var samples: [Float]? {
    switch self {
      case let .samples(data, _, _):
        data
      case .file:
        nil
    }
  }
}
