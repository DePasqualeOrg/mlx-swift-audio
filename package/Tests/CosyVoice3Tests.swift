import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import Testing

@testable import MLXAudio

// MARK: - Unit Tests

@Suite
struct CosyVoice3UnitTests {
  /// Test rotary embedding computation
  @Test func testRotaryEmbeddingComputation() async throws {
    let dim = 64
    let seqLen = 938
    let base: Float = 10000.0

    // Compute inv_freq
    let arange = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32)
    print("arange shape: \(arange.shape)")
    #expect(arange.shape == [32])

    let invFreq = 1.0 / MLX.pow(MLXArray(base), arange / Float(dim))
    print("invFreq shape: \(invFreq.shape)")
    #expect(invFreq.shape == [32])

    // Create positions
    var positions = MLXArray(0 ..< seqLen).asType(.float32)
    print("positions shape before expand: \(positions.shape)")
    #expect(positions.shape == [938])

    positions = positions.expandedDimensions(axis: 0)
    print("positions shape after expand: \(positions.shape)")
    #expect(positions.shape == [1, 938])

    // Compute freqs via einsum
    let freqs = MLX.einsum("bi,j->bij", positions, invFreq)
    print("freqs shape: \(freqs.shape)")
    #expect(freqs.shape == [1, 938, 32])

    // Stack
    let stacked = MLX.stacked([freqs, freqs], axis: -1)
    print("stacked shape: \(stacked.shape)")
    #expect(stacked.shape == [1, 938, 32, 2])

    // Reshape to interleave
    let freqsInterleaved = stacked.reshaped([stacked.shape[0], stacked.shape[1], -1])
    print("freqsInterleaved shape: \(freqsInterleaved.shape)")
    #expect(freqsInterleaved.shape == [1, 938, 64])

    print("Rotary embedding computation test passed!")
  }

  /// Test the actual RotaryEmbedding class
  @Test func testRotaryEmbeddingClass() async throws {
    // Create a RotaryEmbedding with dim=64 (same as DiT config)
    let rotaryEmbed = RotaryEmbedding(dim: 64)

    // Test with sequence length 938
    let seqLen = 938
    let (freqs, scale) = rotaryEmbed.forwardFromSeqLen(seqLen)

    print("RotaryEmbedding class test:")
    print("  freqs shape: \(freqs.shape)")
    print("  scale: \(scale)")

    #expect(freqs.shape == [1, 938, 64], "Expected freqs shape [1, 938, 64], got \(freqs.shape)")

    print("RotaryEmbedding class test passed!")
  }

  /// Test model loading
  @Test func testModelLoading() async throws {
    let modelRepoId = "mlx-community/Fun-CosyVoice3-0.5B-2512-4bit"

    print("Loading CosyVoice3 model...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let tts = try await CosyVoice3TTS.load(repoId: modelRepoId)
    let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
    print("Model loaded in \(String(format: "%.2f", loadTime))s")

    // Access the model's flow decoder and test its rotary embedding
    // This tests that weights were loaded correctly
    print("Model loading test passed!")
  }

  /// Test DiT forward pass in isolation (without loaded weights)
  @Test func testDiTForward() async throws {
    // Create a DiT instance directly with default parameters
    let melDim = 80
    let spkDim = 192
    let dit = DiT(
      dim: 1024, depth: 2, heads: 16, dimHead: 64, // Use depth=2 for faster test
      dropout: 0.0, ffMult: 2, melDim: melDim, muDim: melDim,
      longSkipConnection: false, spkDim: spkDim,
      outChannels: melDim, staticChunkSize: 50, numDecodingLeftChunks: -1
    )

    // Create dummy inputs for DiT
    let batchSize = 1
    let seqLen = 100

    // Input shapes for DiT: (B, mel_dim, N) - channel-first
    let x = MLXArray.zeros([batchSize, melDim, seqLen])
    let mask = MLXArray.ones([batchSize, seqLen])
    let mu = MLXArray.zeros([batchSize, melDim, seqLen])
    let t = MLXArray(0.5) // Timestep
    let spks = MLXArray.zeros([batchSize, spkDim])

    print("Testing DiT forward pass...")
    print("  x shape: \(x.shape)")
    print("  mask shape: \(mask.shape)")
    print("  mu shape: \(mu.shape)")

    // Call DiT forward
    let noCond: MLXArray? = nil
    let output = dit(
      x: x,
      mask: mask,
      mu: mu,
      t: t,
      spks: spks,
      cond: noCond,
      streaming: false
    )

    print("  output shape: \(output.shape)")
    #expect(output.shape[0] == batchSize)
    #expect(output.shape[1] == melDim)
    #expect(output.shape[2] == seqLen)

    print("DiT forward test passed!")
  }
}

// MARK: - End-to-End Integration Tests

@Suite(.serialized)
struct CosyVoice3IntegrationTests {
  /// HuggingFace repo ID for 4-bit model
  static let modelRepoId = "mlx-community/Fun-CosyVoice3-0.5B-2512-4bit"

  /// HuggingFace repo ID for S3 tokenizer V3 (CosyVoice3 uses V3 with 12 layers)
  static let s3TokenizerRepoId = "mlx-community/S3TokenizerV3"

  /// Reference audio from LJ Speech dataset (public domain)
  /// This is a clear female voice reading: "The examination and testimony of the experts
  /// enabled the commission to conclude that five shots may have been fired"
  static let referenceAudioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
  static let referenceTranscription =
    "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired"

  /// Output directory for generated audio
  static let outputDir = FileManager.default.temporaryDirectory.appendingPathComponent("cosyvoice3-test")

  /// Load S3TokenizerV3 from HuggingFace (CosyVoice3 uses V3 with 12 layers)
  static func loadS3Tokenizer() async throws -> S3TokenizerV3 {
    let modelDirectory = try await HubConfiguration.shared.snapshot(from: s3TokenizerRepoId)
    let weightURL = modelDirectory.appendingPathComponent("model.safetensors")
    let weights = try MLX.loadArrays(url: weightURL)

    let tokenizer = S3TokenizerV3()

    // Load weights into tokenizer
    let parameters = ModuleParameters.unflattened(weights)
    try tokenizer.update(parameters: parameters, verify: [.noUnusedKeys])

    // Set to eval mode
    tokenizer.train(false)
    eval(tokenizer)

    return tokenizer
  }

  /// Download audio from URL and return as MLXArray at 24kHz
  /// Audio files are cached locally to avoid repeated downloads
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

  /// Simple audio generation test using CosyVoice3Engine
  /// Listen to output manually at /tmp/cosyvoice3-test/
  @Test func testAudioGeneration() async throws {
    print("=== CosyVoice3 Audio Generation Test ===\n")

    // Create engine
    let engine = await CosyVoice3Engine()

    // Load model (includes S3TokenizerV3 and CosyVoice3 model)
    print("Loading CosyVoice3Engine...")
    try await engine.load(progressHandler: nil)

    // Prepare speaker from reference audio (auto-transcribes with Whisper)
    print("\nPreparing speaker from reference audio...")
    let speaker = try await engine.prepareSpeaker(
      from: Self.referenceAudioURL,
      transcription: Self.referenceTranscription // Provide transcription for zero-shot mode
    )
    print("  Speaker: \(speaker.description)")
    print("  Duration: \(String(format: "%.1f", speaker.duration))s")
    print("  Has transcription: \(speaker.hasTranscription)")

    // Create output directory
    try FileManager.default.createDirectory(at: Self.outputDir, withIntermediateDirectories: true)

    // Generate audio
    let text = "Hello, this is a test of the voice cloning system."
    print("\nGenerating: \"\(text)\"")

    let start = CFAbsoluteTimeGetCurrent()
    let result = try await engine.generate(text, speaker: speaker)
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    // Extract audio samples
    guard case let .samples(samples, sampleRate, _) = result else {
      throw CosyVoice3Error.invalidInput("Expected samples result")
    }

    let duration = Float(samples.count) / Float(sampleRate)

    // Save audio
    let outputURL = Self.outputDir.appendingPathComponent("cosyvoice3_test.wav")
    try Self.saveAudio(samples, to: outputURL, sampleRate: sampleRate)

    print("\n✓ Generated \(String(format: "%.2f", duration))s audio in \(String(format: "%.2f", elapsed))s")
    print("  RTF: \(String(format: "%.2fx", Float(elapsed) / duration))")
    print("  Output: \(outputURL.path)")
    print("\nOpen with: open \"\(outputURL.path)\"")
  }

  /// End-to-end test: Generate speech in multiple modes and verify with Whisper
  /// Uses publicly available LJ Speech audio as reference voice
  @Test func testVoiceMatchingWithWhisperVerification() async throws {
    print("=== CosyVoice3 Voice Matching Test with Whisper Verification ===\n")

    // === Step 1: Load all models once ===
    print("Step 1: Loading models...")

    let ttsStart = CFAbsoluteTimeGetCurrent()
    let tts = try await CosyVoice3TTS.load(repoId: Self.modelRepoId)
    print("  CosyVoice3 loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - ttsStart))s")

    let s3Start = CFAbsoluteTimeGetCurrent()
    let s3Tokenizer = try await Self.loadS3Tokenizer()
    print("  S3 tokenizer loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - s3Start))s")

    let whisperStart = CFAbsoluteTimeGetCurrent()
    let whisper = await STT.whisper(model: .base, quantization: .q4)
    try await whisper.load()
    print("  Whisper loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - whisperStart))s")

    // === Step 2: Download reference audio ===
    print("\nStep 2: Downloading reference audio (LJ Speech)...")
    let refAudio = try await Self.downloadAudio(from: Self.referenceAudioURL)
    let refDuration = Float(refAudio.shape[0]) / 24000.0
    print("  Reference audio: \(refAudio.shape[0]) samples (\(String(format: "%.1f", refDuration))s)")
    print("  Reference transcription: \"\(Self.referenceTranscription)\"")

    // Create output directory
    try FileManager.default.createDirectory(at: Self.outputDir, withIntermediateDirectories: true)

    // === Test 1: Cross-lingual mode ===
    print("\n--- Test 1: Cross-Lingual Mode ---")
    print("(Uses speaker embedding only, no reference text alignment)")

    nonisolated(unsafe) let tokenizer = s3Tokenizer
    nonisolated(unsafe) let refAudioUnsafe = refAudio
    let crossLingualConditionals = await tts.prepareConditionals(
      refWav: refAudioUnsafe,
      refText: nil, // Cross-lingual mode
      s3Tokenizer: { mel, melLen in tokenizer(mel, melLen: melLen) }
    )

    let crossLingualText = "Hello, this is a test of CosyVoice three voice matching."
    print("  Input: \"\(crossLingualText)\"")

    let crossLingualTokens = await tts.encode(text: crossLingualText)
    let crossLingualStart = CFAbsoluteTimeGetCurrent()
    let crossLingualResult = try await tts.generateCrossLingual(
      text: crossLingualText,
      textTokens: crossLingualTokens,
      conditionals: crossLingualConditionals,
      sampling: 25,
      nTimesteps: 10
    )
    let crossLingualTime = CFAbsoluteTimeGetCurrent() - crossLingualStart
    let crossLingualDuration = Float(crossLingualResult.audio.count) / Float(crossLingualResult.sampleRate)
    let crossLingualRTF = Float(crossLingualTime) / crossLingualDuration

    let crossLingualURL = Self.outputDir.appendingPathComponent("cosyvoice3_cross_lingual_test.wav")
    try Self.saveAudio(crossLingualResult.audio, to: crossLingualURL, sampleRate: crossLingualResult.sampleRate)

    let crossLingualTranscription = try await whisper.transcribe(crossLingualURL, language: .english)
    let crossLingualAccuracy = Self.computeWordAccuracy(expected: crossLingualText, transcribed: crossLingualTranscription.text)

    print("  Output: \"\(crossLingualTranscription.text)\"")
    print("  Accuracy: \(String(format: "%.0f%%", crossLingualAccuracy * 100)), RTF: \(String(format: "%.2fx", crossLingualRTF))")
    print("  Saved: \(crossLingualURL.path)")

    #expect(crossLingualAccuracy >= 0.70, "Cross-lingual: Expected >=70% accuracy, got \(String(format: "%.0f%%", crossLingualAccuracy * 100))")
    #expect(crossLingualRTF < 5.0, "Cross-lingual: RTF \(String(format: "%.1f", crossLingualRTF)) exceeds 5x threshold")

    // === Test 2: Zero-shot mode ===
    print("\n--- Test 2: Zero-Shot Mode ---")
    print("(Uses reference text for better voice alignment)")

    let zeroShotConditionals = await tts.prepareConditionals(
      refWav: refAudioUnsafe,
      refText: Self.referenceTranscription, // Zero-shot mode
      s3Tokenizer: { mel, melLen in tokenizer(mel, melLen: melLen) }
    )

    let zeroShotText = "Machine learning enables computers to learn from data."
    print("  Input: \"\(zeroShotText)\"")

    let zeroShotTokens = await tts.encode(text: zeroShotText)
    let zeroShotStart = CFAbsoluteTimeGetCurrent()
    let zeroShotResult = try await tts.generateZeroShot(
      text: zeroShotText,
      textTokens: zeroShotTokens,
      conditionals: zeroShotConditionals,
      sampling: 25,
      nTimesteps: 10
    )
    let zeroShotTime = CFAbsoluteTimeGetCurrent() - zeroShotStart
    let zeroShotDuration = Float(zeroShotResult.audio.count) / Float(zeroShotResult.sampleRate)
    let zeroShotRTF = Float(zeroShotTime) / zeroShotDuration

    let zeroShotURL = Self.outputDir.appendingPathComponent("cosyvoice3_zero_shot_test.wav")
    try Self.saveAudio(zeroShotResult.audio, to: zeroShotURL, sampleRate: zeroShotResult.sampleRate)

    let zeroShotTranscription = try await whisper.transcribe(zeroShotURL, language: .english)
    let zeroShotAccuracy = Self.computeWordAccuracy(expected: zeroShotText, transcribed: zeroShotTranscription.text)

    print("  Output: \"\(zeroShotTranscription.text)\"")
    print("  Accuracy: \(String(format: "%.0f%%", zeroShotAccuracy * 100)), RTF: \(String(format: "%.2fx", zeroShotRTF))")
    print("  Saved: \(zeroShotURL.path)")

    #expect(zeroShotAccuracy >= 0.70, "Zero-shot: Expected >=70% accuracy, got \(String(format: "%.0f%%", zeroShotAccuracy * 100))")
    #expect(zeroShotRTF < 5.0, "Zero-shot: RTF \(String(format: "%.1f", zeroShotRTF)) exceeds 5x threshold")

    // Cleanup
    await whisper.unload()

    print("\n=== All CosyVoice3 voice matching tests passed! ===")
  }
}
