import Foundation

/// Unified error type for all STT operations
public enum STTError: LocalizedError {
  /// The model hasn't been loaded yet
  case modelNotLoaded

  /// Transcription failed
  case transcriptionFailed(underlying: Error)

  /// Audio loading failed
  case audioLoadFailed(underlying: Error)

  /// Not enough memory to load or run the model
  case insufficientMemory

  /// The operation was cancelled by the user
  case cancelled

  /// Model download or loading failed
  case modelLoadFailed(underlying: Error)

  /// Model is not available (missing required files)
  case modelUnavailable(String)

  /// Invalid audio file or format
  case invalidAudio(String)

  /// File I/O error
  case fileIOError(underlying: Error)

  /// Invalid configuration or arguments
  case invalidArgument(String)

  /// Language detection failed
  case languageDetectionFailed(String)

  /// Tokenization failed
  case tokenizationFailed(String)

  // MARK: - LocalizedError

  public var errorDescription: String? {
    switch self {
      case .modelNotLoaded:
        "Model not loaded. Call load() first."
      case let .transcriptionFailed(error):
        "Transcription failed: \(error.localizedDescription)"
      case let .audioLoadFailed(error):
        "Audio loading failed: \(error.localizedDescription)"
      case .insufficientMemory:
        "Insufficient memory for model."
      case .cancelled:
        "Operation was cancelled."
      case let .modelLoadFailed(error):
        "Failed to load model: \(error.localizedDescription)"
      case let .modelUnavailable(message):
        "Model unavailable: \(message)"
      case let .invalidAudio(message):
        "Invalid audio: \(message)"
      case let .fileIOError(error):
        "File I/O error: \(error.localizedDescription)"
      case let .invalidArgument(message):
        "Invalid argument: \(message)"
      case let .languageDetectionFailed(message):
        "Language detection failed: \(message)"
      case let .tokenizationFailed(message):
        "Tokenization failed: \(message)"
    }
  }

  public var failureReason: String? {
    switch self {
      case .modelNotLoaded:
        "The STT model must be loaded before transcribing audio."
      case .transcriptionFailed:
        "An error occurred during speech recognition."
      case .audioLoadFailed:
        "The audio file could not be loaded or decoded."
      case .insufficientMemory:
        "The device does not have enough memory to run this model."
      case .cancelled:
        "The user cancelled the operation."
      case .modelLoadFailed:
        "The model weights could not be downloaded or loaded."
      case .modelUnavailable:
        "The model is temporarily unavailable due to missing weight files."
      case .invalidAudio:
        "The audio file is invalid or in an unsupported format."
      case .fileIOError:
        "A file system operation failed."
      case .invalidArgument:
        "An invalid argument was provided."
      case .languageDetectionFailed:
        "Could not detect the language of the audio."
      case .tokenizationFailed:
        "Could not tokenize the text."
    }
  }

  public var recoverySuggestion: String? {
    switch self {
      case .modelNotLoaded:
        "Call the load() method before attempting to transcribe audio."
      case .transcriptionFailed:
        "Try again with different audio or check the error details."
      case .audioLoadFailed:
        "Ensure the audio file exists and is in a supported format (WAV, MP3, M4A)."
      case .insufficientMemory:
        "Close other applications to free up memory, or use a smaller model."
      case .cancelled:
        nil
      case .modelLoadFailed:
        "Check your internet connection and try again."
      case .modelUnavailable:
        "Use an available model (tiny, base, or large-v3-turbo) instead."
      case .invalidAudio:
        "Provide a valid audio file at 16kHz sample rate."
      case .fileIOError:
        "Check file permissions and available disk space."
      case .invalidArgument:
        "Review the method documentation for valid argument values."
      case .languageDetectionFailed:
        "Specify the language explicitly instead of using auto-detection."
      case .tokenizationFailed:
        "Ensure the tokenizer is properly initialized."
    }
  }
}
