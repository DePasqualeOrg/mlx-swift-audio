//
//  TTSError.swift
//  MLXAudio
//
//  Unified error handling across all TTS engines.
//

import Foundation

/// Unified error type for all TTS operations
public enum TTSError: LocalizedError {
  /// The model hasn't been loaded yet
  case modelNotLoaded

  /// Audio generation failed
  case generationFailed(underlying: Error)

  /// Audio playback failed
  case audioPlaybackFailed(underlying: Error)

  /// The requested voice is not valid for this engine
  case invalidVoice(String)

  /// Not enough memory to load or run the model
  case insufficientMemory

  /// The operation was cancelled by the user
  case cancelled

  /// Model download or loading failed
  case modelLoadFailed(underlying: Error)

  /// Invalid reference audio provided
  case invalidReferenceAudio(String)

  /// Voice not found in available voices
  case voiceNotFound(String)

  /// File I/O error
  case fileIOError(underlying: Error)

  /// Invalid configuration or arguments
  case invalidArgument(String)

  // MARK: - LocalizedError

  public var errorDescription: String? {
    switch self {
      case .modelNotLoaded:
        "Model not loaded. Call load() first."
      case let .generationFailed(error):
        "Generation failed: \(error.localizedDescription)"
      case let .audioPlaybackFailed(error):
        "Playback failed: \(error.localizedDescription)"
      case let .invalidVoice(id):
        "Invalid voice: \(id)"
      case .insufficientMemory:
        "Insufficient memory for model."
      case .cancelled:
        "Operation was cancelled."
      case let .modelLoadFailed(error):
        "Failed to load model: \(error.localizedDescription)"
      case let .invalidReferenceAudio(message):
        "Invalid reference audio: \(message)"
      case let .voiceNotFound(name):
        "Voice not found: \(name)"
      case let .fileIOError(error):
        "File I/O error: \(error.localizedDescription)"
      case let .invalidArgument(message):
        "Invalid argument: \(message)"
    }
  }

  public var failureReason: String? {
    switch self {
      case .modelNotLoaded:
        "The TTS model must be loaded before generating audio."
      case .generationFailed:
        "An error occurred during audio synthesis."
      case .audioPlaybackFailed:
        "The audio system encountered an error during playback."
      case .invalidVoice:
        "The specified voice is not available for this TTS engine."
      case .insufficientMemory:
        "The device does not have enough memory to run this model."
      case .cancelled:
        "The user cancelled the operation."
      case .modelLoadFailed:
        "The model weights could not be downloaded or loaded."
      case .invalidReferenceAudio:
        "The reference audio file is invalid or in an unsupported format."
      case .voiceNotFound:
        "The requested voice preset could not be found."
      case .fileIOError:
        "A file system operation failed."
      case .invalidArgument:
        "An invalid argument was provided."
    }
  }

  public var recoverySuggestion: String? {
    switch self {
      case .modelNotLoaded:
        "Call the load() method before attempting to generate audio."
      case .generationFailed:
        "Try again with different text or check the error details."
      case .audioPlaybackFailed:
        "Check that the audio session is configured correctly."
      case .invalidVoice:
        "Use availableVoices to see the list of valid voices."
      case .insufficientMemory:
        "Close other applications to free up memory, or use a smaller model."
      case .cancelled:
        nil
      case .modelLoadFailed:
        "Check your internet connection and try again."
      case .invalidReferenceAudio:
        "Provide a mono WAV file at 24kHz sample rate."
      case .voiceNotFound:
        "Check that the voice preset file exists in the model directory."
      case .fileIOError:
        "Check file permissions and available disk space."
      case .invalidArgument:
        "Review the method documentation for valid argument values."
    }
  }
}
