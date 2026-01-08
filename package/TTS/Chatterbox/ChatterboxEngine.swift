// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

import AVFoundation
import Foundation
import MLX

// MARK: - Reference Audio

/// Prepared reference audio for Chatterbox TTS
///
/// Create using `ChatterboxEngine.prepareReferenceAudio(from:)` methods.
/// Can be reused across multiple `say()` or `generate()` calls for efficient multi-speaker scenarios.
///
/// ```swift
/// let speakerA = try await engine.prepareReferenceAudio(from: urlA)
/// let speakerB = try await engine.prepareReferenceAudio(from: urlB)
///
/// try await engine.say("Hello from A", referenceAudio: speakerA)
/// try await engine.say("Hello from B", referenceAudio: speakerB)  // instant switch
/// ```
public struct ChatterboxReferenceAudio: Sendable {
  /// The pre-computed conditionals for generation
  let conditionals: ChatterboxConditionals

  /// Sample rate of the original reference audio
  public let sampleRate: Int

  /// Duration of the reference audio in seconds
  public let duration: TimeInterval

  /// Description for display purposes
  public let description: String

  init(conditionals: ChatterboxConditionals, sampleRate: Int, sampleCount: Int, description: String) {
    self.conditionals = conditionals
    self.sampleRate = sampleRate
    duration = Double(sampleCount) / Double(sampleRate)
    self.description = description
  }
}

/// Default reference audio URL - LJ Speech Dataset sample
/// ~7 seconds (public domain)
public let defaultReferenceAudioURL = URL(
  string:
  "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav",
)!

/// Chatterbox TTS engine - TTS with reference audio
///
/// Supports generating speech using reference audio clips.
@Observable
@MainActor
public final class ChatterboxEngine: TTSEngine {
  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .chatterbox
  public let supportedStreamingGranularities: Set<StreamingGranularity> = [.sentence]
  public let defaultStreamingGranularity: StreamingGranularity = .sentence
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - Chatterbox-Specific Properties

  /// Quantization level
  public let quantization: ChatterboxQuantization

  /// Temperature for sampling (higher = more variation)
  public var temperature: Float = 0.8

  /// Top-p (nucleus) sampling threshold
  public var topP: Float = 1.0

  /// Minimum probability threshold for sampling
  public var minP: Float = 0.05

  /// Repetition penalty
  public var repetitionPenalty: Float = 1.2

  /// Classifier-free guidance weight
  public var cfgWeight: Float = 0.5

  /// Emotion exaggeration factor (0-1)
  public var exaggeration: Float = 0.1

  /// Maximum number of tokens to generate
  public var maxNewTokens: Int = 1000

  // MARK: - Private Properties

  @ObservationIgnored private var chatterboxTTS: ChatterboxTTS?
  @ObservationIgnored private let playback = TTSPlaybackController(sampleRate: TTSProvider.chatterbox.sampleRate)
  @ObservationIgnored private var defaultReferenceAudio: ChatterboxReferenceAudio?

  // MARK: - Initialization

  public init(quantization: ChatterboxQuantization = .q4) {
    self.quantization = quantization
    Log.tts.debug("ChatterboxEngine initialized with quantization: \(quantization.rawValue)")
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.tts.debug("ChatterboxEngine already loaded")
      return
    }

    let quantization = quantization
    Log.model.info("Loading Chatterbox TTS model (\(quantization.rawValue))...")

    do {
      chatterboxTTS = try await ChatterboxTTS.load(
        quantization: quantization,
        progressHandler: progressHandler ?? { _ in },
      )

      isLoaded = true
      Log.model.info("Chatterbox TTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load Chatterbox model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  public func stop() async {
    await playback.stop(
      setGenerating: { self.isGenerating = $0 },
      setPlaying: { self.isPlaying = $0 },
    )
    Log.tts.debug("ChatterboxEngine stopped")
  }

  public func unload() async {
    await stop()

    // Clear model but preserve prepared reference audio (expensive to recompute)
    chatterboxTTS = nil
    isLoaded = false

    Log.tts.debug("ChatterboxEngine unloaded (reference audio preserved)")
  }

  public func cleanup() async throws {
    await unload()

    // Also clear prepared reference audio (expensive to recompute, but full cleanup)
    defaultReferenceAudio = nil
  }

  // MARK: - Playback

  public func play(_ audio: AudioResult) async {
    await playback.play(audio, setPlaying: { self.isPlaying = $0 })
  }

  // MARK: - Reference Audio Preparation

  /// Prepare reference audio from a URL (local file or remote)
  ///
  /// This performs the expensive conditioning computation once. The returned
  /// `ChatterboxReferenceAudio` can be reused across multiple `say()` or `generate()` calls.
  ///
  /// - Parameters:
  ///   - url: URL to audio file (local file path or remote URL)
  ///   - exaggeration: Emotion exaggeration factor (0-1, default: 0.5)
  /// - Returns: Prepared reference audio ready for generation
  public func prepareReferenceAudio(
    from url: URL,
    exaggeration: Float = 0.5,
  ) async throws -> ChatterboxReferenceAudio {
    if !isLoaded {
      try await load()
    }

    guard let chatterboxTTS else {
      throw TTSError.modelNotLoaded
    }

    let (samples, sampleRate) = try await loadAudioSamples(from: url)
    let description = url.lastPathComponent

    return await prepareReferenceAudioFromSamples(
      samples,
      sampleRate: sampleRate,
      exaggeration: exaggeration,
      description: description,
      tts: chatterboxTTS,
    )
  }

  /// Prepare reference audio from raw samples
  ///
  /// - Parameters:
  ///   - samples: Audio samples as Float array
  ///   - sampleRate: Sample rate of the audio
  ///   - exaggeration: Emotion exaggeration factor (0-1, default: 0.5)
  /// - Returns: Prepared reference audio ready for generation
  public func prepareReferenceAudio(
    fromSamples samples: [Float],
    sampleRate: Int,
    exaggeration: Float = 0.5,
  ) async throws -> ChatterboxReferenceAudio {
    if !isLoaded {
      try await load()
    }

    guard let chatterboxTTS else {
      throw TTSError.modelNotLoaded
    }

    let duration = Double(samples.count) / Double(sampleRate)
    let description = String(format: "Custom audio (%.1f sec.)", duration)

    return await prepareReferenceAudioFromSamples(
      samples,
      sampleRate: sampleRate,
      exaggeration: exaggeration,
      description: description,
      tts: chatterboxTTS,
    )
  }

  /// Prepare the default reference audio (LibriVox public domain sample)
  ///
  /// - Parameter exaggeration: Emotion exaggeration factor (0-1, default: 0.5)
  /// - Returns: Prepared reference audio ready for generation
  public func prepareDefaultReferenceAudio(
    exaggeration: Float = 0.5,
  ) async throws -> ChatterboxReferenceAudio {
    try await prepareReferenceAudio(from: defaultReferenceAudioURL, exaggeration: exaggeration)
  }

  // MARK: - Private Audio Loading

  private func prepareReferenceAudioFromSamples(
    _ samples: [Float],
    sampleRate: Int,
    exaggeration: Float,
    description: String,
    tts: ChatterboxTTS,
  ) async -> ChatterboxReferenceAudio {
    Log.tts.debug("Preparing reference audio: \(description)")

    let originalDuration = Float(samples.count) / Float(sampleRate)
    Log.tts.debug("Original reference audio duration: \(originalDuration)s")

    // Trim silence from beginning and end (matching Python implementation)
    // Chatterbox Python uses trim_top_db=20 which is more aggressive than CosyVoice2
    let trimmedSamples = AudioTrimmer.trimSilence(
      samples,
      sampleRate: sampleRate,
      config: .chatterbox
    )

    let trimmedDuration = Float(trimmedSamples.count) / Float(sampleRate)
    Log.tts.debug("After silence trimming: \(trimmedDuration)s")

    let refWav = MLXArray(trimmedSamples)
    let conditionals = await tts.prepareConditionals(
      refWav: refWav,
      refSr: sampleRate,
      exaggeration: exaggeration,
    )

    Log.tts.debug("Reference audio prepared: \(description)")

    return ChatterboxReferenceAudio(
      conditionals: conditionals,
      sampleRate: sampleRate,
      sampleCount: trimmedSamples.count,
      description: description,
    )
  }

  private func loadAudioSamples(from url: URL) async throws -> (samples: [Float], sampleRate: Int) {
    if url.isFileURL {
      try await loadAudioFromFile(url)
    } else {
      try await loadAudioFromRemoteURL(url)
    }
  }

  private func loadAudioFromRemoteURL(_ url: URL) async throws -> (samples: [Float], sampleRate: Int) {
    Log.tts.debug("Downloading reference audio from URL: \(url)")

    let (data, response) = try await URLSession.shared.data(from: url)

    guard let httpResponse = response as? HTTPURLResponse,
          (200 ... 299).contains(httpResponse.statusCode)
    else {
      throw TTSError.invalidArgument("Failed to download reference audio from URL")
    }

    // Save to temporary file and load
    let tempURL = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString)
      .appendingPathExtension(url.pathExtension.isEmpty ? "mp3" : url.pathExtension)

    try data.write(to: tempURL)
    defer { try? FileManager.default.removeItem(at: tempURL) }

    return try await loadAudioFromFile(tempURL)
  }

  private func loadAudioFromFile(_ url: URL) async throws -> (samples: [Float], sampleRate: Int) {
    Log.tts.debug("Loading reference audio from file: \(url.path)")

    let audioFile = try AVAudioFile(forReading: url)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
      throw TTSError.invalidArgument("Failed to create audio buffer")
    }

    try audioFile.read(into: buffer)

    guard let floatData = buffer.floatChannelData else {
      throw TTSError.invalidArgument("Failed to read audio data")
    }

    // Convert to mono if stereo
    let samples: [Float]
    if format.channelCount == 1 {
      samples = Array(UnsafeBufferPointer(start: floatData[0], count: Int(buffer.frameLength)))
    } else {
      // Mix stereo to mono
      let left = UnsafeBufferPointer(start: floatData[0], count: Int(buffer.frameLength))
      let right = UnsafeBufferPointer(start: floatData[1], count: Int(buffer.frameLength))
      samples = zip(left, right).map { ($0 + $1) / 2.0 }
    }

    return (samples, Int(format.sampleRate))
  }

  // MARK: - Generation

  /// Generate audio from text using reference audio
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - referenceAudio: Prepared reference audio (if nil, uses default sample)
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    referenceAudio: ChatterboxReferenceAudio? = nil,
  ) async throws -> AudioResult {
    let (samples, processingTime) = try await playback.collectStream(
      generateStreaming(text, referenceAudio: referenceAudio),
    )

    Log.tts.timing("Chatterbox generation", duration: processingTime)
    lastGeneratedAudioURL = playback.saveAudioFile(samples: samples, sampleRate: provider.sampleRate)

    return .samples(
      data: samples,
      sampleRate: provider.sampleRate,
      processingTime: processingTime,
    )
  }

  /// Generate and immediately play audio
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - referenceAudio: Prepared reference audio (if nil, uses default sample)
  public func say(
    _ text: String,
    referenceAudio: ChatterboxReferenceAudio? = nil,
  ) async throws {
    let audio = try await generate(text, referenceAudio: referenceAudio)
    await play(audio)
  }

  // MARK: - Streaming

  /// Generate audio as a stream of chunks
  ///
  /// Text splitting (sentences and overflow) is handled by ChatterboxTTS.
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - referenceAudio: Prepared reference audio (if nil, uses default sample)
  /// - Returns: An async stream of audio chunks
  public func generateStreaming(
    _ text: String,
    referenceAudio: ChatterboxReferenceAudio? = nil,
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Text cannot be empty")) }
    }

    let sampleRate = provider.sampleRate

    // Capture current parameter values
    let currentExaggeration = exaggeration
    let currentCfgWeight = cfgWeight
    let currentTemperature = temperature
    let currentRepetitionPenalty = repetitionPenalty
    let currentMinP = minP
    let currentTopP = topP
    let currentMaxNewTokens = maxNewTokens

    return playback.createGenerationStream(
      setGenerating: { self.isGenerating = $0 },
      setGenerationTime: { self.generationTime = $0 },
    ) { [weak self] in
      guard let self else {
        return AsyncThrowingStream { $0.finish() }
      }

      // Auto-load if needed
      if !isLoaded {
        try await load()
      }

      guard let chatterboxTTS else {
        throw TTSError.modelNotLoaded
      }

      // Prepare reference audio if needed
      if referenceAudio == nil, defaultReferenceAudio == nil {
        defaultReferenceAudio = try await prepareDefaultReferenceAudio(exaggeration: currentExaggeration)
      }
      guard let referenceAudio = referenceAudio ?? defaultReferenceAudio else {
        throw TTSError.modelNotLoaded
      }

      // Wrap model stream to convert [Float] -> AudioChunk
      let modelStream = await chatterboxTTS.generateStreaming(
        text: trimmedText,
        conditionals: referenceAudio.conditionals,
        exaggeration: currentExaggeration,
        cfgWeight: currentCfgWeight,
        temperature: currentTemperature,
        repetitionPenalty: currentRepetitionPenalty,
        minP: currentMinP,
        topP: currentTopP,
        maxNewTokens: currentMaxNewTokens,
      )

      let startTime = Date()
      return AsyncThrowingStream { continuation in
        Task {
          do {
            for try await samples in modelStream {
              guard !Task.isCancelled else { break }
              let chunk = AudioChunk(
                samples: samples,
                sampleRate: sampleRate,
                processingTime: Date().timeIntervalSince(startTime),
              )
              continuation.yield(chunk)
            }
            continuation.finish()
          } catch is CancellationError {
            continuation.finish()
          } catch {
            continuation.finish(throwing: error)
          }
        }
      }
    }
  }

  /// Play audio with streaming (plays as chunks arrive)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - referenceAudio: Prepared reference audio (if nil, uses default sample)
  @discardableResult
  public func sayStreaming(
    _ text: String,
    referenceAudio: ChatterboxReferenceAudio? = nil,
  ) async throws -> AudioResult {
    let (samples, processingTime) = try await playback.playStream(
      generateStreaming(text, referenceAudio: referenceAudio),
      setPlaying: { self.isPlaying = $0 },
    )

    lastGeneratedAudioURL = playback.saveAudioFile(samples: samples, sampleRate: provider.sampleRate)

    return .samples(
      data: samples,
      sampleRate: provider.sampleRate,
      processingTime: processingTime,
    )
  }
}
