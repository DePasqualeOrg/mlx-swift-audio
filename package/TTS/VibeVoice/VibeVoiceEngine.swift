// Copyright © Microsoft (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/microsoft/VibeVoice
// License: licenses/vibevoice.txt

import AVFoundation
import Foundation
import Hub
import MLX
import Observation

/// VibeVoice TTS Engine
///
/// A high-quality streaming TTS engine using diffusion-based speech synthesis.
///
/// Example usage:
/// ```swift
/// let engine = VibeVoiceEngine()
/// try await engine.load()
/// try await engine.loadVoice("default")
/// let audio = try await engine.generate(text: "Hello, world!")
/// await engine.play(audio)
/// ```
@MainActor
@Observable
public final class VibeVoiceEngine: TTSEngine {
  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .vibeVoice
  public let streamingGranularity: StreamingGranularity = .sentence

  public private(set) var isLoaded = false
  public private(set) var isGenerating = false
  public private(set) var isPlaying = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - Configuration

  /// HuggingFace repository ID for the model
  public var repoId: String = VibeVoiceConstants.defaultRepoId

  /// Maximum tokens to generate
  public var maxTokens: Int = 512

  /// Classifier-free guidance scale
  public var cfgScale: Float = 1.5

  /// Number of diffusion steps (higher = better quality, slower)
  public var ddpmSteps: Int = 20

  // MARK: - Private Properties

  private var tts: VibeVoiceTTS?
  private var audioPlayer: AVAudioPlayer?
  private var currentVoice: String?

  // MARK: - Initialization

  public init(repoId: String = VibeVoiceConstants.defaultRepoId) {
    self.repoId = repoId
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)? = nil) async throws {
    guard !isLoaded else { return }

    tts = try await VibeVoiceTTS.load(
      repoId: repoId,
      progressHandler: progressHandler ?? { _ in }
    )

    isLoaded = true
  }

  public func stop() async {
    audioPlayer?.stop()
    isPlaying = false
    isGenerating = false
  }

  public func unload() async {
    tts = nil
    currentVoice = nil
    isLoaded = false
  }

  public func cleanup() async throws {
    await stop()
    await unload()
    lastGeneratedAudioURL = nil
    generationTime = 0
  }

  public func play(_ audio: AudioResult) async {
    guard !audio.samples.isEmpty else { return }

    isPlaying = true
    defer { isPlaying = false }

    do {
      // Create audio file
      let tempURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("wav")

      try writeWAV(samples: audio.samples, sampleRate: audio.sampleRate, to: tempURL)
      lastGeneratedAudioURL = tempURL

      // Play audio
      audioPlayer = try AVAudioPlayer(contentsOf: tempURL)
      audioPlayer?.play()

      // Wait for playback to complete
      while audioPlayer?.isPlaying == true {
        try await Task.sleep(for: .milliseconds(100))
      }
    } catch {
      print("Error playing audio: \(error)")
    }
  }

  // MARK: - Voice Loading

  /// Load a voice for conditioning
  /// - Parameter voiceName: Name of the voice cache file (without .safetensors)
  public func loadVoice(_ voiceName: String) async throws {
    guard let tts else {
      throw VibeVoiceError.modelNotLoaded
    }

    try await tts.loadVoice(voiceName)
    currentVoice = voiceName
  }

  /// Currently loaded voice name
  public var voice: String? {
    currentVoice
  }

  // MARK: - Generation

  /// Generate audio from text
  /// - Parameter text: Text to synthesize
  /// - Returns: Audio result containing samples and metadata
  public func generate(text: String) async throws -> AudioResult {
    guard let tts else {
      throw VibeVoiceError.modelNotLoaded
    }

    isGenerating = true
    defer { isGenerating = false }

    let startTime = CFAbsoluteTimeGetCurrent()

    let result = await tts.generate(
      text: text,
      maxTokens: maxTokens,
      cfgScale: cfgScale,
      ddpmSteps: ddpmSteps
    )

    generationTime = CFAbsoluteTimeGetCurrent() - startTime

    return AudioResult(
      samples: result.audio,
      sampleRate: result.sampleRate,
      duration: result.duration
    )
  }

  /// Generate and immediately play audio
  /// - Parameter text: Text to synthesize
  public func say(_ text: String) async throws {
    let audio = try await generate(text: text)
    await play(audio)
  }

  // MARK: - Streaming Generation

  /// Generate audio with streaming output
  /// - Parameters:
  ///   - text: Text to synthesize
  ///   - onChunk: Callback for each audio chunk
  public func generateStreaming(
    text: String,
    onChunk: @escaping (AudioChunk) -> Void
  ) async throws {
    // For now, generate full audio and return as single chunk
    // Future: implement true streaming with sentence segmentation
    let result = try await generate(text: text)

    let chunk = AudioChunk(
      samples: result.samples,
      sampleRate: result.sampleRate,
      processingTime: generationTime
    )
    onChunk(chunk)
  }

  // MARK: - Private Helpers

  private func writeWAV(samples: [Float], sampleRate: Int, to url: URL) throws {
    let format = AVAudioFormat(
      commonFormat: .pcmFormatFloat32,
      sampleRate: Double(sampleRate),
      channels: 1,
      interleaved: false
    )!

    let buffer = AVAudioPCMBuffer(
      pcmFormat: format,
      frameCapacity: AVAudioFrameCount(samples.count)
    )!

    buffer.frameLength = AVAudioFrameCount(samples.count)

    if let channelData = buffer.floatChannelData?[0] {
      for (i, sample) in samples.enumerated() {
        channelData[i] = sample
      }
    }

    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    try file.write(from: buffer)
  }
}

// MARK: - AudioResult

/// Audio generation result
public struct AudioResult: Sendable {
  public let samples: [Float]
  public let sampleRate: Int
  public let duration: TimeInterval

  public init(samples: [Float], sampleRate: Int, duration: TimeInterval) {
    self.samples = samples
    self.sampleRate = sampleRate
    self.duration = duration
  }

  public init(samples: [Float], sampleRate: Int) {
    self.samples = samples
    self.sampleRate = sampleRate
    duration = Double(samples.count) / Double(sampleRate)
  }
}

// MARK: - TTS Extension

public extension TTS {
  /// VibeVoice: Diffusion-based high-quality TTS
  @MainActor
  static func vibeVoice() -> VibeVoiceEngine {
    VibeVoiceEngine()
  }
}
