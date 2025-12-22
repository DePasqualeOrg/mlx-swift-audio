import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import Testing

@testable import MLXAudio

// MARK: - Unit Tests

@Suite
struct VibeVoiceUnitTests {
  /// Test configuration loading
  @Test func testConfigDefaults() async throws {
    // Test AcousticTokenizerConfig defaults
    let acousticConfig = AcousticTokenizerConfig()
    #expect(acousticConfig.vaeDim == 64)
    #expect(acousticConfig.channels == 1)
    #expect(acousticConfig.encoderNFilters == 32)
    #expect(acousticConfig.encoderRatios == [8, 5, 5, 4, 2, 2])

    // Test DiffusionHeadConfig defaults
    let diffusionConfig = DiffusionHeadConfig()
    #expect(diffusionConfig.hiddenSize == 896)
    #expect(diffusionConfig.headLayers == 4)
    #expect(diffusionConfig.headFfnRatio == 3.0)
    #expect(diffusionConfig.latentSize == 64)
    #expect(diffusionConfig.ddpmNumSteps == 1000)
    #expect(diffusionConfig.ddpmNumInferenceSteps == 20)

    // Test Qwen2DecoderConfig defaults
    let qwenConfig = Qwen2DecoderConfig()
    #expect(qwenConfig.hiddenSize == 896)
    #expect(qwenConfig.numHiddenLayers == 24)
    #expect(qwenConfig.numAttentionHeads == 14)
    #expect(qwenConfig.numKeyValueHeads == 2)
    #expect(qwenConfig.intermediateSize == 4864)
    #expect(qwenConfig.vocabSize == 151_936)
  }

  /// Test DPM-Solver scheduler step calculation
  @Test func testDPMSolverScheduler() async throws {
    let scheduler = DPMSolverMultistepScheduler(
      numTrainTimesteps: 1000,
      betaSchedule: "cosine",
      predictionType: "v_prediction"
    )

    // Set timesteps for inference
    scheduler.setTimesteps(20)

    #expect(scheduler.numInferenceSteps == 20)
    #expect(scheduler.timesteps != nil)
    #expect(scheduler.timesteps!.shape[0] == 20)

    // Verify cached values are populated
    #expect(scheduler.cachedAlphaT.count == 21) // 20 + 1 for final
    #expect(scheduler.cachedSigmaT.count == 21)
    #expect(scheduler.cachedLambda.count == 21)

    // Test a scheduler step with dummy data
    let batchSize = 1
    let latentDim = 64
    let sample = MLXArray.zeros([batchSize, latentDim])
    let modelOutput = MLXArray.zeros([batchSize, latentDim])

    let output = scheduler.step(
      modelOutput: modelOutput,
      timestep: 999,
      sample: sample,
      prevX0: nil
    )

    #expect(output.prevSample.shape == [batchSize, latentDim])
    #expect(output.x0Pred != nil)
    #expect(output.x0Pred!.shape == [batchSize, latentDim])
  }

  /// Test rotary embedding computation
  @Test func testVibeVoiceRotaryEmbedding() async throws {
    let dim = 64
    let seqLen = 100
    let base: Float = 1_000_000.0

    let rotaryEmb = VibeVoiceRotaryEmbedding(dim: dim, base: base)

    let positionIds = MLXArray(0 ..< seqLen)
    let (cos, sin) = rotaryEmb(positionIds)

    #expect(cos.shape == [seqLen, dim])
    #expect(sin.shape == [seqLen, dim])

    // Verify values are in expected range
    let cosValues = cos.asArray(Float.self)
    let sinValues = sin.asArray(Float.self)
    for i in 0 ..< min(10, cosValues.count) {
      #expect(cosValues[i] >= -1.0 && cosValues[i] <= 1.0)
      #expect(sinValues[i] >= -1.0 && sinValues[i] <= 1.0)
    }
  }

  /// Test diffusion head components
  @Test func testTimestepEmbedding() async throws {
    let hiddenSize = 896
    let embedder = TimestepEmbedder(hiddenSize: hiddenSize)

    let timesteps = MLXArray([0.5, 0.8])
    let embedding = embedder(timesteps)

    #expect(embedding.shape == [2, hiddenSize])
  }

  /// Test model loading
  @Test func testModelLoading() async throws {
    let modelRepoId = VibeVoiceConstants.defaultRepoId

    print("Loading VibeVoice model...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    _ = try await VibeVoiceTTS.load(repoId: modelRepoId)
    let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
    print("Model loaded in \(String(format: "%.2f", loadTime))s")
  }
}

// MARK: - End-to-End Integration Tests

@Suite(.serialized)
struct VibeVoiceIntegrationTests {
  /// HuggingFace repo ID for VibeVoice model
  static let modelRepoId = VibeVoiceConstants.defaultRepoId

  /// Reference audio from LJ Speech dataset (public domain)
  /// This is a clear female voice reading: "The examination and testimony of the experts
  /// enabled the commission to conclude that five shots may have been fired"
  static let referenceAudioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
  static let referenceTranscription =
    "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired"

  /// Output directory for generated audio
  static let outputDir = URL(fileURLWithPath: "/tmp/vibevoice-test")

  /// Download audio from URL and return as MLXArray at 24kHz
  static func downloadAudio(from url: URL) async throws -> MLXArray {
    let cacheURL = try await TestAudioCache.downloadToFile(from: url)
    return try loadAudioFile(at: cacheURL)
  }

  /// Load audio file and resample to 24kHz mono
  static func loadAudioFile(at url: URL) throws -> MLXArray {
    let file = try AVAudioFile(forReading: url)

    guard let buffer = AVAudioPCMBuffer(
      pcmFormat: file.processingFormat,
      frameCapacity: AVAudioFrameCount(file.length)
    ) else {
      throw TestError(message: "Failed to create buffer")
    }

    try file.read(into: buffer)

    guard let floatData = buffer.floatChannelData else {
      throw TestError(message: "No float data in buffer")
    }

    let frameCount = Int(buffer.frameLength)
    let channelCount = Int(buffer.format.channelCount)

    // Convert to mono if stereo
    var samples = [Float](repeating: 0, count: frameCount)
    if channelCount == 1 {
      for i in 0 ..< frameCount {
        samples[i] = floatData[0][i]
      }
    } else {
      // Average channels for mono
      for i in 0 ..< frameCount {
        var sum: Float = 0
        for ch in 0 ..< channelCount {
          sum += floatData[ch][i]
        }
        samples[i] = sum / Float(channelCount)
      }
    }

    // Resample to 24kHz if needed
    let sourceSR = Int(file.fileFormat.sampleRate)
    if sourceSR != 24000 {
      let ratio = Float(24000) / Float(sourceSR)
      let newLength = Int(Float(frameCount) * ratio)
      var resampled = [Float](repeating: 0, count: newLength)
      for i in 0 ..< newLength {
        let srcIdx = Float(i) / ratio
        let idx0 = Int(srcIdx)
        let idx1 = min(idx0 + 1, frameCount - 1)
        let frac = srcIdx - Float(idx0)
        resampled[i] = samples[idx0] * (1 - frac) + samples[idx1] * frac
      }
      samples = resampled
    }

    return MLXArray(samples)
  }

  /// Save audio to WAV file
  static func saveAudio(_ audio: [Float], to url: URL, sampleRate: Int = 24000) throws {
    // Create output directory if needed
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )

    let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audio.count))!
    buffer.frameLength = AVAudioFrameCount(audio.count)

    for (i, sample) in audio.enumerated() {
      buffer.floatChannelData![0][i] = sample
    }

    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    try file.write(from: buffer)
  }

  /// Compute word accuracy between expected and transcribed text
  static func computeWordAccuracy(expected: String, transcribed: String) -> Float {
    let punctuation = CharacterSet.punctuationCharacters
    let expectedWords = Set(
      expected.lowercased()
        .components(separatedBy: punctuation).joined()
        .split(separator: " ").map(String.init)
    )
    let transcribedWords = Set(
      transcribed.lowercased()
        .components(separatedBy: punctuation).joined()
        .split(separator: " ").map(String.init)
    )
    let matchedWords = transcribedWords.intersection(expectedWords)
    return Float(matchedWords.count) / Float(expectedWords.count)
  }

  /// List available voices in the model directory
  static func listAvailableVoices(modelDirectory: URL) -> [String] {
    let voicesDir = modelDirectory.appendingPathComponent("voices")
    guard let contents = try? FileManager.default.contentsOfDirectory(atPath: voicesDir.path) else {
      return []
    }
    return contents
      .filter { $0.hasSuffix(".safetensors") }
      .map { $0.replacingOccurrences(of: ".safetensors", with: "") }
  }

  /// Audio generation test using VibeVoiceTTS
  /// Listen to output manually at /tmp/vibevoice-test/
  @Test func testAudioGeneration() async throws {
    print("=== VibeVoice Audio Generation Test ===\n")

    // Create output directory
    try FileManager.default.createDirectory(at: Self.outputDir, withIntermediateDirectories: true)

    // Load model
    print("Loading VibeVoice model from \(Self.modelRepoId)...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let tts = try await VibeVoiceTTS.load(repoId: Self.modelRepoId)
    let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
    print("  Model loaded in \(String(format: "%.2f", loadTime))s")

    // Download model directory to check for voices
    let modelDirectory = try await HubConfiguration.shared.snapshot(from: Self.modelRepoId)
    let availableVoices = Self.listAvailableVoices(modelDirectory: modelDirectory)
    print("  Available voices: \(availableVoices)")

    // Load a voice if available
    if let voiceName = availableVoices.first {
      print("\nLoading voice: \(voiceName)...")
      try await tts.loadVoice(voiceName)
      print("  Voice loaded successfully")
    } else {
      print("\nWarning: No voice caches found, generation may fail")
    }

    // Generate audio
    let text = "Hello, this is a test of the VibeVoice text to speech system."
    print("\nGenerating: \"\(text)\"")

    let start = CFAbsoluteTimeGetCurrent()
    let result = await tts.generate(text: text, maxTokens: 256, cfgScale: 1.5, ddpmSteps: 20)
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    let duration = result.duration

    // Save audio
    let outputURL = Self.outputDir.appendingPathComponent("vibevoice_test.wav")
    try Self.saveAudio(result.audio, to: outputURL, sampleRate: result.sampleRate)

    print("\n✓ Generated \(String(format: "%.2f", duration))s audio in \(String(format: "%.2f", elapsed))s")
    print("  RTF: \(String(format: "%.2fx", Float(elapsed) / Float(duration)))")
    print("  Output: \(outputURL.path)")
    print("\nOpen with: open \"\(outputURL.path)\"")

    // Basic assertions
    #expect(result.audio.count > 0, "Audio should not be empty")
    #expect(result.sampleRate == 24000, "Sample rate should be 24kHz")
  }

  /// End-to-end test with Whisper verification
  @Test func testVoiceGenerationWithWhisperVerification() async throws {
    print("=== VibeVoice Generation Test with Whisper Verification ===\n")

    // Create output directory
    try FileManager.default.createDirectory(at: Self.outputDir, withIntermediateDirectories: true)

    // === Step 1: Load models ===
    print("Step 1: Loading models...")

    let ttsStart = CFAbsoluteTimeGetCurrent()
    let tts = try await VibeVoiceTTS.load(repoId: Self.modelRepoId)
    print("  VibeVoice loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - ttsStart))s")

    let whisperStart = CFAbsoluteTimeGetCurrent()
    let whisper = await STT.whisper(model: .largeTurbo, quantization: .q4)
    try await whisper.load()
    print("  Whisper loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - whisperStart))s")

    // === Step 2: Load voice ===
    print("\nStep 2: Loading voice...")
    let modelDirectory = try await HubConfiguration.shared.snapshot(from: Self.modelRepoId)
    let availableVoices = Self.listAvailableVoices(modelDirectory: modelDirectory)

    guard let voiceName = availableVoices.first else {
      print("  No voice caches available, skipping test")
      await whisper.unload()
      return
    }

    try await tts.loadVoice(voiceName)
    print("  Voice '\(voiceName)' loaded")

    // === Step 3: Generate audio and verify ===
    print("\nStep 3: Generating audio...")

    let testCases = [
      "Hello, this is a test of voice synthesis.",
      "The quick brown fox jumps over the lazy dog.",
      "Artificial intelligence is transforming how we interact with technology.",
    ]

    for (index, text) in testCases.enumerated() {
      print("\n--- Test \(index + 1)/\(testCases.count) ---")
      print("  Input: \"\(text)\"")

      let genStart = CFAbsoluteTimeGetCurrent()
      let result = await tts.generate(text: text, maxTokens: 256, cfgScale: 1.5, ddpmSteps: 20)
      let genTime = CFAbsoluteTimeGetCurrent() - genStart
      let audioDuration = result.duration
      let rtf = Float(genTime) / Float(audioDuration)

      // Save audio
      let outputURL = Self.outputDir.appendingPathComponent("vibevoice_test_\(index + 1).wav")
      try Self.saveAudio(result.audio, to: outputURL, sampleRate: result.sampleRate)

      // Transcribe with Whisper
      let transcription = try await whisper.transcribe(outputURL, language: .english)
      let accuracy = Self.computeWordAccuracy(expected: text, transcribed: transcription.text)

      print("  Output: \"\(transcription.text)\"")
      print("  Accuracy: \(String(format: "%.0f%%", accuracy * 100)), RTF: \(String(format: "%.2fx", rtf))")
      print("  Duration: \(String(format: "%.2f", audioDuration))s")
      print("  Saved: \(outputURL.path)")

      // Assertions
      #expect(accuracy >= 0.50, "Expected >=50% word accuracy, got \(String(format: "%.0f%%", accuracy * 100))")
      #expect(rtf < 10.0, "RTF \(String(format: "%.1f", rtf)) exceeds 10x threshold")
    }

    // Cleanup
    await whisper.unload()

    print("\n✓ All tests completed. Audio files saved to: \(Self.outputDir.path)")
    print("Open folder with: open \"\(Self.outputDir.path)\"")
  }

  /// Test generation with different CFG scales
  @Test func testDifferentCFGScales() async throws {
    print("=== VibeVoice CFG Scale Comparison Test ===\n")

    // Create output directory
    try FileManager.default.createDirectory(at: Self.outputDir, withIntermediateDirectories: true)

    // Load model
    let tts = try await VibeVoiceTTS.load(repoId: Self.modelRepoId)

    // Load voice
    let modelDirectory = try await HubConfiguration.shared.snapshot(from: Self.modelRepoId)
    let availableVoices = Self.listAvailableVoices(modelDirectory: modelDirectory)

    guard let voiceName = availableVoices.first else {
      print("No voice caches available, skipping test")
      return
    }

    try await tts.loadVoice(voiceName)

    let text = "This is a comparison of different guidance scales."
    let cfgScales: [Float] = [1.0, 1.5, 2.0, 3.0]

    print("Generating with different CFG scales...")
    for cfg in cfgScales {
      let start = CFAbsoluteTimeGetCurrent()
      let result = await tts.generate(text: text, maxTokens: 256, cfgScale: cfg, ddpmSteps: 20)
      let elapsed = CFAbsoluteTimeGetCurrent() - start

      let outputURL = Self.outputDir.appendingPathComponent("vibevoice_cfg_\(String(format: "%.1f", cfg)).wav")
      try Self.saveAudio(result.audio, to: outputURL, sampleRate: result.sampleRate)

      print("  CFG \(String(format: "%.1f", cfg)): \(String(format: "%.2f", result.duration))s audio in \(String(format: "%.2f", elapsed))s")
      print("    Saved: \(outputURL.path)")
    }

    print("\n✓ CFG comparison complete. Listen to files to compare quality.")
  }
}
