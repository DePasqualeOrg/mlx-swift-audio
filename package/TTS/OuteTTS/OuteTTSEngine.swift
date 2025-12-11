import AVFoundation
import Foundation
import MLX

/// OuteTTS engine - TTS with custom speaker profiles
///
/// Supports custom speaker profiles with reference audio.
@Observable
@MainActor
public final class OuteTTSEngine: TTSEngine {
  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .outetts
  public let streamingGranularity: StreamingGranularity = .sentence
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - OuteTTS-Specific Properties

  /// Temperature for sampling (higher = more variation)
  public var temperature: Float = 0.4

  /// Top-p (nucleus) sampling threshold
  public var topP: Float = 0.9

  // MARK: - Private Properties

  @ObservationIgnored private var outeTTS: OuteTTS?
  @ObservationIgnored private let playback = TTSPlaybackController(sampleRate: TTSProvider.outetts.sampleRate)
  @ObservationIgnored private var whisperEngine: WhisperEngine?

  // MARK: - Initialization

  public init() {
    Log.tts.debug("OuteTTSEngine initialized")
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.tts.debug("OuteTTSEngine already loaded")
      return
    }

    Log.model.info("Loading OuteTTS model...")

    do {
      outeTTS = try await OuteTTS.load(
        progressHandler: progressHandler ?? { _ in },
      )

      isLoaded = true
      Log.model.info("OuteTTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load OuteTTS model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  public func stop() async {
    await playback.stop(
      setGenerating: { self.isGenerating = $0 },
      setPlaying: { self.isPlaying = $0 },
    )
    Log.tts.debug("OuteTTSEngine stopped")
  }

  public func unload() async {
    await stop()
    outeTTS = nil
    isLoaded = false
    Log.tts.debug("OuteTTSEngine unloaded")
  }

  public func cleanup() async throws {
    await unload()
  }

  // MARK: - Playback

  public func play(_ audio: AudioResult) async {
    await playback.play(audio, setPlaying: { self.isPlaying = $0 })
  }

  // MARK: - Generation

  /// Generate audio from text
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Optional speaker profile (nil uses default voice)
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    speaker: OuteTTSSpeakerProfile? = nil,
  ) async throws -> AudioResult {
    let (samples, processingTime) = try await playback.collectStream(
      generateStreaming(text, speaker: speaker),
    )

    Log.tts.timing("OuteTTS generation", duration: processingTime)
    lastGeneratedAudioURL = playback.saveAudioFile(samples: samples, sampleRate: provider.sampleRate)

    return .samples(
      data: samples,
      sampleRate: provider.sampleRate,
      processingTime: processingTime,
    )
  }

  /// Generate and immediately play audio
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Optional speaker profile (nil uses default voice)
  public func say(
    _ text: String,
    speaker: OuteTTSSpeakerProfile? = nil,
  ) async throws {
    let audio = try await generate(text, speaker: speaker)
    await play(audio)
  }

  // MARK: - Streaming

  /// Generate audio as a stream of chunks
  ///
  /// Text splitting is handled by OuteTTS.
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - speaker: Optional speaker profile (nil uses default voice)
  /// - Returns: An async stream of audio chunks
  public func generateStreaming(
    _ text: String,
    speaker: OuteTTSSpeakerProfile? = nil,
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Text cannot be empty")) }
    }

    let sampleRate = provider.sampleRate
    let temp = temperature
    let top = topP

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

      guard let outeTTS else {
        throw TTSError.modelNotLoaded
      }

      // Wrap model stream to convert [Float] -> AudioChunk
      let modelStream = await outeTTS.generateStreaming(
        text: trimmedText,
        speaker: speaker,
        temperature: temp,
        topP: top,
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
  ///   - speaker: Optional speaker profile (nil uses default voice)
  @discardableResult
  public func sayStreaming(
    _ text: String,
    speaker: OuteTTSSpeakerProfile? = nil,
  ) async throws -> AudioResult {
    let (samples, processingTime) = try await playback.playStream(
      generateStreaming(text, speaker: speaker),
      setPlaying: { self.isPlaying = $0 },
    )

    lastGeneratedAudioURL = playback.saveAudioFile(samples: samples, sampleRate: provider.sampleRate)

    return .samples(
      data: samples,
      sampleRate: provider.sampleRate,
      processingTime: processingTime,
    )
  }

  // MARK: - Speaker Profile Creation

  /// Create a speaker profile from an audio file with automatic transcription
  /// - Parameter url: URL to the audio file (local or remote)
  /// - Returns: A speaker profile that can be used for voice synthesis
  public func createSpeakerProfile(from url: URL) async throws -> OuteTTSSpeakerProfile {
    if !isLoaded {
      try await load()
    }

    guard let outeTTS else {
      throw TTSError.modelNotLoaded
    }

    // Load audio samples
    let (samples, sampleRate) = try await loadAudioSamples(from: url)
    let audioArray = MLXArray(samples)

    // Transcribe audio with word-level timestamps
    let (text, words) = try await transcribeAudio(url: url)

    guard !words.isEmpty else {
      throw TTSError.invalidArgument("Could not transcribe audio - no words detected")
    }

    // Resample to 24kHz if needed (OuteTTS native sample rate)
    let targetSampleRate = provider.sampleRate
    let resampledAudio: MLXArray = if sampleRate != targetSampleRate {
      AudioResampler.resample(audioArray, from: sampleRate, to: targetSampleRate)
    } else {
      audioArray
    }

    // Create speaker profile using the audio processor
    let profile = try await outeTTS.getSpeaker(
      referenceAudio: resampledAudio,
      referenceText: text,
      referenceWords: words,
    )

    guard let profile else {
      throw TTSError.invalidArgument("Failed to create speaker profile from audio")
    }

    return profile
  }

  // MARK: - Audio Loading

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

  // MARK: - Speech Transcription

  private func transcribeAudio(url: URL) async throws -> (text: String, words: [(word: String, start: Double, end: Double)]) {
    // Load Whisper engine if needed
    if whisperEngine == nil {
      whisperEngine = WhisperEngine(modelSize: .base, quantization: .q4)
    }

    guard let whisper = whisperEngine else {
      throw TTSError.invalidArgument("Failed to create Whisper engine")
    }

    if !whisper.isLoaded {
      try await whisper.load(progressHandler: nil)
    }

    // Transcribe with word-level timestamps using DTW alignment
    let result = try await whisper.transcribe(url, timestamps: .word)

    // Extract word-level timestamps from segments (from DTW alignment)
    // Strip words and filter empty ones (matches Python's audio_processor.py line 267)
    var words: [(word: String, start: Double, end: Double)] = []

    for segment in result.segments {
      guard let segmentWords = segment.words else { continue }
      for word in segmentWords {
        let strippedWord = word.word.trimmingCharacters(in: .whitespaces)
        // Skip empty words (shouldn't happen after Whisper filtering, but be safe)
        guard !strippedWord.isEmpty else { continue }
        words.append((word: strippedWord, start: word.start, end: word.end))
      }
    }

    // Debug: Print word timestamps for verification
    print("[OuteTTS] Transcription: \"\(result.text)\"")
    print("[OuteTTS] Word timestamps (\(words.count) words):")
    for (i, word) in words.enumerated() {
      let duration = word.end - word.start
      print("  [\(i)] \"\(word.word)\" start=\(String(format: "%.3f", word.start))s end=\(String(format: "%.3f", word.end))s duration=\(String(format: "%.3f", duration))s")
    }

    return (result.text, words)
  }
}
