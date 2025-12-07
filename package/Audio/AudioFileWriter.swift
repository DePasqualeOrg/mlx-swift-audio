import AVFoundation
import Foundation

/// Errors that can occur during audio file writing
enum AudioFileWriterError: LocalizedError {
  case formatCreationFailed
  case bufferCreationFailed
  case channelDataAccessFailed

  var errorDescription: String? {
    switch self {
      case .formatCreationFailed:
        "Failed to create audio format"
      case .bufferCreationFailed:
        "Failed to create audio buffer"
      case .channelDataAccessFailed:
        "Failed to get channel data"
    }
  }
}

/// Audio file format options
public enum AudioFileFormat {
  case wav
  case caf

  public var fileExtension: String {
    switch self {
      case .wav: "wav"
      case .caf: "caf"
    }
  }

  public var commonFormat: AVAudioCommonFormat {
    switch self {
      case .wav, .caf: .pcmFormatFloat32
    }
  }
}

/// Utility for saving audio samples to files
public enum AudioFileWriter {
  /// Save audio samples to a file
  /// - Parameters:
  ///   - samples: Audio samples as Float array
  ///   - sampleRate: Sample rate (e.g., 24000 for Kokoro)
  ///   - directory: Target directory URL (defaults to documents directory)
  ///   - filename: Base filename without extension
  ///   - format: Output format (default: .wav for compatibility)
  /// - Returns: URL of the saved file
  /// - Throws: TTSError.fileIOError on failure
  public static func save(
    samples: [Float],
    sampleRate: Int,
    to directory: URL? = nil,
    filename: String,
    format: AudioFileFormat = .wav,
  ) throws -> URL {
    guard !samples.isEmpty else {
      throw TTSError.invalidArgument("Cannot save empty audio samples")
    }

    // Determine output directory
    let outputDirectory = directory ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]

    // Create full file URL
    let fileURL = outputDirectory.appendingPathComponent("\(filename).\(format.fileExtension)")

    // Create audio format
    guard let audioFormat = AVAudioFormat(
      standardFormatWithSampleRate: Double(sampleRate),
      channels: 1,
    ) else {
      throw TTSError.fileIOError(underlying: AudioFileWriterError.formatCreationFailed)
    }

    // Create buffer
    guard let buffer = AVAudioPCMBuffer(
      pcmFormat: audioFormat,
      frameCapacity: AVAudioFrameCount(samples.count),
    ) else {
      throw TTSError.fileIOError(underlying: AudioFileWriterError.bufferCreationFailed)
    }

    buffer.frameLength = AVAudioFrameCount(samples.count)

    // Copy samples to buffer
    guard let channelData = buffer.floatChannelData else {
      throw TTSError.fileIOError(underlying: AudioFileWriterError.channelDataAccessFailed)
    }

    for i in 0 ..< samples.count {
      channelData[0][i] = samples[i]
    }

    // Write to file
    do {
      let audioFile = try AVAudioFile(
        forWriting: fileURL,
        settings: audioFormat.settings,
        commonFormat: format.commonFormat,
        interleaved: false,
      )
      try audioFile.write(from: buffer)
      Log.audio.debug("Audio saved to: \(fileURL.path)")
      return fileURL
    } catch {
      throw TTSError.fileIOError(underlying: error)
    }
  }
}
