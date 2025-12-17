// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import AVFoundation
import Foundation
import MLX

/// Fun-ASR STT engine - LLM-based multilingual speech recognition
///
/// Combines SenseVoice encoder with Qwen3 decoder for high-quality transcription
/// and translation capabilities.
@Observable
@MainActor
public final class FunASREngine: STTEngine {
  // MARK: - STTEngine Protocol Properties

  public let provider: STTProvider = .funASR
  public private(set) var isLoaded: Bool = false
  public private(set) var isTranscribing: Bool = false
  public private(set) var transcriptionTime: TimeInterval = 0

  // MARK: - FunASR-Specific Properties

  /// Model variant (quantization level)
  public let variant: FunASRModelVariant

  // MARK: - Private Properties

  @ObservationIgnored private var funASRSTT: FunASRSTT?

  // MARK: - Initialization

  public init(variant: FunASRModelVariant = .nano4bit) {
    self.variant = variant
    Log.model.debug("FunASREngine initialized with variant: \(variant.repoId)")
  }

  public init(modelType: FunASRModelType = .nano, quantization: FunASRQuantization = .q4) {
    variant = FunASRModelVariant(modelType: modelType, quantization: quantization)
    let repoId = variant.repoId
    Log.model.debug("FunASREngine initialized with variant: \(repoId)")
  }

  // MARK: - STTEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.model.info("Fun-ASR model already loaded")
      return
    }

    let variant = variant
    Log.model.info("Loading Fun-ASR \(variant.repoId) model...")

    funASRSTT = try await FunASRSTT.load(
      variant: variant,
      progressHandler: progressHandler ?? { _ in }
    )

    isLoaded = true
    Log.model.info("Fun-ASR model loaded successfully")
  }

  public func stop() async {
    // TODO: Implement cancellation mechanism
    Log.model.info("Stop requested (not fully implemented yet)")
  }

  public func unload() async {
    funASRSTT = nil
    isLoaded = false
    Log.model.info("Fun-ASR model unloaded")
  }

  public func cleanup() async throws {
    await unload()
    Log.model.info("Fun-ASR cleanup complete")
  }

  // MARK: - Transcription Methods

  /// Transcribe audio in original language from a URL
  ///
  /// - Parameters:
  ///   - url: URL to audio file (supports WAV, MP3, M4A, etc.)
  ///   - language: Language of the audio (nil = auto-detect)
  ///   - temperature: Sampling temperature (0.0 = greedy, higher = more random)
  ///   - topP: Top-p (nucleus) sampling threshold
  ///   - topK: Top-k sampling (0 to disable)
  ///   - maxTokens: Maximum tokens to generate (nil = model default)
  ///   - initialPrompt: Custom instructions or context
  /// - Returns: Transcription result
  public nonisolated func transcribe(
    _ url: URL,
    language: FunASRLanguage = .auto,
    temperature: Float = 0.0,
    topP: Float = 0.95,
    topK: Int = 0,
    maxTokens: Int? = nil,
    initialPrompt: String? = nil
  ) async throws -> TranscriptionResult {
    guard await isLoaded, let funASRSTT = await funASRSTT else {
      throw STTError.modelNotLoaded
    }

    guard await !isTranscribing else {
      throw STTError.invalidArgument("Transcription already in progress")
    }

    await MainActor.run { isTranscribing = true }
    defer { Task { await MainActor.run { isTranscribing = false } } }

    // Load and preprocess audio
    let audio16k = try loadAndPreprocessAudio(from: url)

    // Transcribe
    let result = await funASRSTT.transcribe(
      audio: audio16k,
      language: language,
      task: .transcribe,
      targetLanguage: .english,
      temperature: temperature,
      topP: topP,
      topK: topK,
      maxTokens: maxTokens,
      initialPrompt: initialPrompt
    )

    await MainActor.run { transcriptionTime = result.processingTime }

    return result
  }

  /// Transcribe audio in original language from an MLXArray
  ///
  /// - Parameters:
  ///   - audio: Audio waveform (T,) at 16kHz
  ///   - language: Language of the audio (nil = auto-detect)
  ///   - temperature: Sampling temperature (0.0 = greedy, higher = more random)
  ///   - topP: Top-p (nucleus) sampling threshold
  ///   - topK: Top-k sampling (0 to disable)
  ///   - maxTokens: Maximum tokens to generate (nil = model default)
  ///   - initialPrompt: Custom instructions or context
  /// - Returns: Transcription result
  public nonisolated func transcribe(
    _ audio: sending MLXArray,
    language: FunASRLanguage = .auto,
    temperature: Float = 0.0,
    topP: Float = 0.95,
    topK: Int = 0,
    maxTokens: Int? = nil,
    initialPrompt: String? = nil
  ) async throws -> TranscriptionResult {
    guard await isLoaded, let funASRSTT = await funASRSTT else {
      throw STTError.modelNotLoaded
    }

    guard await !isTranscribing else {
      throw STTError.invalidArgument("Transcription already in progress")
    }

    await MainActor.run { isTranscribing = true }
    defer { Task { await MainActor.run { isTranscribing = false } } }

    // Transcribe
    let result = await funASRSTT.transcribe(
      audio: audio,
      language: language,
      task: .transcribe,
      targetLanguage: .english,
      temperature: temperature,
      topP: topP,
      topK: topK,
      maxTokens: maxTokens,
      initialPrompt: initialPrompt
    )

    await MainActor.run { transcriptionTime = result.processingTime }

    return result
  }

  /// Translate audio to target language from a URL
  ///
  /// - Note: Translation works best with MLT (multilingual) model variants.
  ///   Standard variants are trained primarily for transcription and may not
  ///   reliably follow translation instructions.
  ///
  /// - Parameters:
  ///   - url: URL to audio file (supports WAV, MP3, M4A, etc.)
  ///   - sourceLanguage: Source language hint (nil = auto-detect)
  ///   - targetLanguage: Target language for translation (default: English)
  ///   - temperature: Sampling temperature (0.0 = greedy, higher = more random)
  ///   - topP: Top-p (nucleus) sampling threshold
  ///   - maxTokens: Maximum tokens to generate (nil = model default)
  /// - Returns: Translation result
  public nonisolated func translate(
    _ url: URL,
    sourceLanguage: FunASRLanguage = .auto,
    targetLanguage: FunASRLanguage = .english,
    temperature: Float = 0.0,
    topP: Float = 0.95,
    maxTokens: Int? = nil
  ) async throws -> TranscriptionResult {
    guard await isLoaded, let funASRSTT = await funASRSTT else {
      throw STTError.modelNotLoaded
    }

    guard await !isTranscribing else {
      throw STTError.invalidArgument("Transcription already in progress")
    }

    await MainActor.run { isTranscribing = true }
    defer { Task { await MainActor.run { isTranscribing = false } } }

    // Load and preprocess audio
    let audio16k = try loadAndPreprocessAudio(from: url)

    // Translate
    let result = await funASRSTT.transcribe(
      audio: audio16k,
      language: sourceLanguage,
      task: .translate,
      targetLanguage: targetLanguage,
      temperature: temperature,
      topP: topP,
      topK: 0,
      maxTokens: maxTokens,
      initialPrompt: nil
    )

    await MainActor.run { transcriptionTime = result.processingTime }

    return result
  }

  /// Translate audio to target language from an MLXArray
  ///
  /// - Note: Translation works best with MLT (multilingual) model variants.
  ///   Standard variants are trained primarily for transcription and may not
  ///   reliably follow translation instructions.
  ///
  /// - Parameters:
  ///   - audio: Audio waveform (T,) at 16kHz
  ///   - sourceLanguage: Source language hint (nil = auto-detect)
  ///   - targetLanguage: Target language for translation (default: English)
  ///   - temperature: Sampling temperature (0.0 = greedy, higher = more random)
  ///   - topP: Top-p (nucleus) sampling threshold
  ///   - maxTokens: Maximum tokens to generate (nil = model default)
  /// - Returns: Translation result
  public nonisolated func translate(
    _ audio: sending MLXArray,
    sourceLanguage: FunASRLanguage = .auto,
    targetLanguage: FunASRLanguage = .english,
    temperature: Float = 0.0,
    topP: Float = 0.95,
    maxTokens: Int? = nil
  ) async throws -> TranscriptionResult {
    guard await isLoaded, let funASRSTT = await funASRSTT else {
      throw STTError.modelNotLoaded
    }

    guard await !isTranscribing else {
      throw STTError.invalidArgument("Transcription already in progress")
    }

    await MainActor.run { isTranscribing = true }
    defer { Task { await MainActor.run { isTranscribing = false } } }

    // Translate
    let result = await funASRSTT.transcribe(
      audio: audio,
      language: sourceLanguage,
      task: .translate,
      targetLanguage: targetLanguage,
      temperature: temperature,
      topP: topP,
      topK: 0,
      maxTokens: maxTokens,
      initialPrompt: nil
    )

    await MainActor.run { transcriptionTime = result.processingTime }

    return result
  }

  // MARK: - Streaming Methods

  /// Stream tokens during transcription from a URL
  ///
  /// - Parameters:
  ///   - url: URL to audio file
  ///   - language: Source language
  ///   - task: Task type (transcribe or translate)
  ///   - targetLanguage: Target language for translation
  ///   - temperature: Sampling temperature
  ///   - topP: Top-p sampling threshold
  ///   - topK: Top-k sampling
  ///   - maxTokens: Maximum tokens to generate
  ///   - initialPrompt: Custom instructions
  /// - Returns: AsyncThrowingStream yielding token strings
  public nonisolated func transcribeStreaming(
    _ url: URL,
    language: FunASRLanguage = .auto,
    task: FunASRTask = .transcribe,
    targetLanguage: FunASRLanguage = .english,
    temperature: Float = 0.0,
    topP: Float = 0.95,
    topK: Int = 0,
    maxTokens: Int? = nil,
    initialPrompt: String? = nil
  ) async throws -> AsyncThrowingStream<String, Error> {
    guard await isLoaded, let funASRSTT = await funASRSTT else {
      throw STTError.modelNotLoaded
    }

    // Load and preprocess audio
    let audio16k = try loadAndPreprocessAudio(from: url)

    let tokenStream = await funASRSTT.generateStreaming(
      audio: audio16k,
      language: language,
      task: task,
      targetLanguage: targetLanguage,
      temperature: temperature,
      topP: topP,
      topK: topK,
      maxTokens: maxTokens,
      initialPrompt: initialPrompt
    )

    // Convert token IDs to strings
    return AsyncThrowingStream { continuation in
      Task {
        do {
          for try await tokenId in tokenStream {
            let text = await funASRSTT.decodeToken(tokenId)
            continuation.yield(text)
          }
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
    }
  }

  // MARK: - Audio Loading

  /// Load audio file, resample to 16kHz, and return as MLXArray
  ///
  /// - Parameter url: URL to audio file
  /// - Returns: Audio array at 16kHz
  private nonisolated func loadAndPreprocessAudio(from url: URL) throws -> MLXArray {
    let (audioArray, sampleRate) = try loadAudioFile(from: url)

    // Resample to 16kHz if needed
    if sampleRate != FunASRAudio.sampleRate {
      Log.model.debug("Resampling audio from \(sampleRate)Hz to \(FunASRAudio.sampleRate)Hz")
      return AudioResampler.resample(
        audioArray,
        from: sampleRate,
        to: FunASRAudio.sampleRate
      )
    } else {
      return audioArray
    }
  }

  /// Load audio file and convert to MLXArray
  ///
  /// - Parameter url: URL to audio file
  /// - Returns: Tuple of (audio_array, sample_rate)
  private nonisolated func loadAudioFile(from url: URL) throws -> (MLXArray, Int) {
    // Load audio using AVFoundation
    let audioFile = try AVAudioFile(forReading: url)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
      throw STTError.invalidAudio("Failed to create audio buffer")
    }

    try audioFile.read(into: buffer)

    // Convert to mono Float32 array
    let channelCount = Int(format.channelCount)
    let sampleRate = Int(format.sampleRate)
    let length = Int(buffer.frameLength)

    var audioSamples: [Float] = []
    audioSamples.reserveCapacity(length)

    if let floatData = buffer.floatChannelData {
      if channelCount == 1 {
        // Mono audio
        for i in 0 ..< length {
          audioSamples.append(floatData[0][i])
        }
      } else {
        // Stereo or multi-channel: average all channels to mono
        for i in 0 ..< length {
          var sum: Float = 0.0
          for channel in 0 ..< channelCount {
            sum += floatData[channel][i]
          }
          audioSamples.append(sum / Float(channelCount))
        }
      }
    } else {
      throw STTError.invalidAudio("Audio buffer has no float data")
    }

    return (MLXArray(audioSamples), sampleRate)
  }
}
