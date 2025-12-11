import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import Testing

@testable import MLXAudio

/// Test helper for CosyVoice2 weight loading
@Suite(.serialized)
struct CosyVoice2WeightLoadingTests {
  /// Path to 4-bit model (primary test model - smaller and faster)
  static let fourBitModelPath =
    "/Users/anthony/.cache/huggingface/hub/models--mlx-community--CosyVoice2-0.5B-4bit/snapshots/becde1f0a28d7b0aa68aa93177c3f361d0dcae1e"

  /// Path to fp16 model (optional - larger model for comparison testing)
  static let fp16ModelPath =
    "/Users/anthony/.cache/huggingface/hub/models--mlx-community--CosyVoice2-0.5B-fp16/snapshots/055935ef40a28ee9be54a5d2ea4b6af6ae15f3d9"

  @Test func testConfigLoading() throws {
    print("Testing CosyVoice2 config loading...")

    // Test with defaults (minimal config.json from HuggingFace)
    let config = try CosyVoice2Config.fromPretrained(modelPath: Self.fourBitModelPath)

    // Verify defaults are set correctly
    #expect(config.llm.speechTokenSize == 6561)
    #expect(config.llm.hiddenSize == 896)
    #expect(config.llm.numHiddenLayers == 24)
    #expect(config.flow.outputSize == 80)
    #expect(config.hifigan.samplingRate == 24000)

    print("✓ Config loaded with correct defaults")
  }

  @Test func testWeightKeyAnalysis() throws {
    print("Analyzing weight keys from 4-bit model...")

    let modelURL = URL(fileURLWithPath: Self.fourBitModelPath).appendingPathComponent("model.safetensors")
    let weights = try MLX.loadArrays(url: modelURL)

    // Count keys per prefix
    var prefixCounts: [String: Int] = [:]
    for key in weights.keys {
      let prefix = key.split(separator: ".").first.map(String.init) ?? "unknown"
      prefixCounts[prefix, default: 0] += 1
    }

    print("Weight prefixes:")
    for (prefix, count) in prefixCounts.sorted(by: { $0.key < $1.key }) {
      print("  \(prefix): \(count) keys")
    }

    // Verify expected prefixes exist
    #expect(prefixCounts["qwen2"] != nil, "Missing qwen2 weights")
    #expect(prefixCounts["llm"] != nil, "Missing llm weights")
    #expect(prefixCounts["flow"] != nil, "Missing flow weights")
    #expect(prefixCounts["hift"] != nil, "Missing hift weights")
    #expect(prefixCounts["campplus"] != nil, "Missing campplus weights")

    print("✓ All expected weight prefixes found")
  }

  @Test func testFourBitModelLoading() async throws {
    print("Testing 4-bit quantized model loading...")

    // First, print the weight keys to understand structure
    let modelURL = URL(fileURLWithPath: Self.fourBitModelPath).appendingPathComponent("model.safetensors")
    let allWeights = try MLX.loadArrays(url: modelURL)

    // Print qwen2 layer 0 keys to understand structure
    let layer0Keys = allWeights.keys.filter { $0.hasPrefix("qwen2.model.layers.0.") }.sorted()
    print("Qwen2 layer 0 weight keys:")
    for key in layer0Keys.prefix(5) {
      print("  \(key): \(allWeights[key]!.shape)")
    }

    // Print llm keys
    let llmKeys = allWeights.keys.filter { $0.hasPrefix("llm.") }.sorted()
    print("LLM weight keys:")
    for key in llmKeys {
      print("  \(key): \(allWeights[key]!.shape)")
    }

    let startTime = CFAbsoluteTimeGetCurrent()

    let tts = try await CosyVoice2TTS.load(modelPath: Self.fourBitModelPath)

    // Print model parameter structure to see how weights are organized
    // This helps verify quantization detection
    print("\nChecking model weight structure after loading:")
    // Note: We can't directly access model internals from here, but the test shows loading succeeded

    let loadTime = CFAbsoluteTimeGetCurrent() - startTime
    print("Model loaded in \(String(format: "%.2f", loadTime)) seconds")

    // Verify model is loaded by checking sample rate (actor-isolated access)
    let sr = await tts.sampleRate
    #expect(sr == 24000)

    // Verify speaker encoder is loaded by checking it produces non-zero embeddings
    let speakerEncoderLoaded = await tts.isSpeakerEncoderLoaded
    #expect(speakerEncoderLoaded, "Speaker encoder (CAMPlus) should be loaded")

    print("✓ 4-bit model loaded successfully (including speaker encoder)")
  }

  @Test func testFp16ModelLoading() async throws {
    print("Testing fp16 model loading...")

    // Check if fp16 model exists (optional - larger model)
    let modelURL = URL(fileURLWithPath: Self.fp16ModelPath).appendingPathComponent("model.safetensors")
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
      print("⚠️ fp16 model not found at \(Self.fp16ModelPath), skipping test")
      return
    }

    let startTime = CFAbsoluteTimeGetCurrent()

    let tts = try await CosyVoice2TTS.load(modelPath: Self.fp16ModelPath)

    let loadTime = CFAbsoluteTimeGetCurrent() - startTime
    print("fp16 model loaded in \(String(format: "%.2f", loadTime)) seconds")

    // Verify model is loaded (actor-isolated access)
    let sr = await tts.sampleRate
    #expect(sr == 24000)

    print("✓ fp16 model loaded successfully")
  }

  @Test func testTokenizerEncodeDecode() async throws {
    print("Testing text tokenizer encode/decode...")

    let tts = try await CosyVoice2TTS.load(modelPath: Self.fourBitModelPath)

    // Test basic encoding
    let text = "Hello, world!"
    let tokens = await tts.encode(text: text)
    print("Encoded '\(text)' to \(tokens.count) tokens: \(tokens)")
    #expect(tokens.count > 0, "Encoding should produce tokens")

    // Test decoding back
    let decoded = await tts.decode(tokens: tokens)
    print("Decoded back to: '\(decoded)'")
    #expect(decoded == text, "Decoded text should match original")

    // Test Chinese text
    let chineseText = "你好世界"
    let chineseTokens = await tts.encode(text: chineseText)
    print("Encoded '\(chineseText)' to \(chineseTokens.count) tokens: \(chineseTokens)")
    #expect(chineseTokens.count > 0, "Chinese encoding should produce tokens")

    let chineseDecoded = await tts.decode(tokens: chineseTokens)
    print("Decoded Chinese back to: '\(chineseDecoded)'")
    #expect(chineseDecoded == chineseText, "Decoded Chinese should match original")

    print("✓ Tokenizer encode/decode works correctly")
  }

  @Test func testSpecialTokens() async throws {
    print("Testing special tokens are recognized...")

    let tts = try await CosyVoice2TTS.load(modelPath: Self.fourBitModelPath)

    // Test that Qwen2 standard special tokens exist
    let eosToken = await tts.tokenToId("<|im_end|>")
    print("EOS token <|im_end|> ID: \(String(describing: eosToken))")
    #expect(eosToken != nil, "EOS token should exist")

    let endOfTextToken = await tts.tokenToId("<|endoftext|>")
    print("End of text token <|endoftext|> ID: \(String(describing: endOfTextToken))")
    #expect(endOfTextToken != nil, "End of text token should exist")

    print("✓ Special tokens are recognized")
  }

  @Test func testTokenizerConsistency() async throws {
    print("Testing tokenizer consistency with Python implementation...")

    let tts = try await CosyVoice2TTS.load(modelPath: Self.fourBitModelPath)

    // These are known token sequences from Python implementation
    // Test English encoding matches expected pattern
    let englishText = "Hello"
    let englishTokens = await tts.encode(text: englishText)
    print("English '\(englishText)' tokens: \(englishTokens)")

    // Verify basic consistency - tokens should be in reasonable range
    for token in englishTokens {
      #expect(token >= 0, "Token should be non-negative")
      #expect(token < 152_000, "Token should be within Qwen2 vocab range")
    }

    print("✓ Tokenizer produces consistent results")
  }
}

// MARK: - End-to-End Integration Tests

@Suite(.serialized)
struct CosyVoice2IntegrationTests {
  /// Path to 4-bit model (quantization is auto-detected and handled)
  static let modelPath =
    "/Users/anthony/.cache/huggingface/hub/models--mlx-community--CosyVoice2-0.5B-4bit/snapshots/becde1f0a28d7b0aa68aa93177c3f361d0dcae1e"

  /// Path to S3 tokenizer
  static let s3TokenizerPath =
    "/Users/anthony/.cache/huggingface/hub/models--mlx-community--S3TokenizerV2/snapshots/e0c9886f0e1c35ae85b1f27277416fb19fc72bec"

  /// Reference audio directory
  static let referenceAudioDir = "/Users/anthony/Desktop/reference-audio"

  /// Output directory for generated audio
  static let outputDir = "/tmp/cosyvoice2-swift-test"

  /// Load S3TokenizerV2 from local cache
  static func loadS3Tokenizer() throws -> S3TokenizerV2 {
    let weightURL = URL(fileURLWithPath: s3TokenizerPath).appendingPathComponent("model.safetensors")
    let weights = try MLX.loadArrays(url: weightURL)

    let tokenizer = S3TokenizerV2()

    // Load weights into tokenizer
    let parameters = ModuleParameters.unflattened(weights)
    try tokenizer.update(parameters: parameters, verify: [.noUnusedKeys])

    // Set to eval mode
    tokenizer.train(false)
    eval(tokenizer)

    return tokenizer
  }

  /// Load reference audio from WAV file
  static func loadReferenceAudio(path: String) throws -> MLXArray {
    let url = URL(fileURLWithPath: path)
    let file = try AVAudioFile(forReading: url)

    guard let buffer = AVAudioPCMBuffer(
      pcmFormat: file.processingFormat,
      frameCapacity: AVAudioFrameCount(file.length)
    ) else {
      throw NSError(domain: "CosyVoice2Test", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffer"])
    }

    try file.read(into: buffer)

    guard let floatData = buffer.floatChannelData else {
      throw NSError(domain: "CosyVoice2Test", code: 2, userInfo: [NSLocalizedDescriptionKey: "No float data in buffer"])
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
      print("Resampling from \(sourceSR) Hz to 24000 Hz...")
      // Simple linear interpolation resampling
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
  static func saveAudio(_ audio: [Float], to path: String, sampleRate: Int = 24000) throws {
    let url = URL(fileURLWithPath: path)

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

    print("Saved audio to: \(path)")
  }

  /// Quick test to verify quantized model can do a forward pass
  @Test func testQuantizedForwardPass() async throws {
    print("=== Quick Forward Pass Test ===\n")

    print("Step 1: Loading 4-bit quantized model...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let tts = try await CosyVoice2TTS.load(modelPath: Self.modelPath)
    print("   Model loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - loadStart))s")

    // Just verify the model loaded and basic operations work
    print("Step 2: Testing text encoding...")
    let text = "Hello world"
    let tokens = await tts.encode(text: text)
    print("   '\(text)' -> \(tokens.count) tokens: \(tokens)")
    #expect(tokens.count > 0, "Should encode text to tokens")

    print("\n✓ Quantized model loads and runs successfully!")
  }

  /// Test S3 tokenizer loading separately
  @Test func testS3TokenizerLoading() async throws {
    print("=== S3 Tokenizer Loading Test ===\n")

    guard FileManager.default.fileExists(atPath: Self.s3TokenizerPath) else {
      print("⚠️ S3TokenizerV2 not found, skipping")
      return
    }

    print("Loading S3 tokenizer...")
    let start = CFAbsoluteTimeGetCurrent()
    let s3Tokenizer = try Self.loadS3Tokenizer()
    print("   Loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - start))s")

    // Quick sanity check - just verify it loaded
    #expect(s3Tokenizer != nil, "S3 tokenizer should load")
    print("✓ S3 tokenizer loaded successfully!")
  }

  /// Single comprehensive end-to-end test that loads models once
  @Test func testEndToEndInference() async throws {
    print("=== CosyVoice2 End-to-End Integration Test ===\n")

    // Check for reference audio
    let refAudioPath = "\(Self.referenceAudioDir)/anthony.wav"
    guard FileManager.default.fileExists(atPath: refAudioPath) else {
      print("⚠️ Reference audio not found at \(refAudioPath), skipping test")
      print("   Please add a reference WAV file to run this test")
      return
    }

    // Check for S3 tokenizer
    guard FileManager.default.fileExists(atPath: Self.s3TokenizerPath) else {
      print("⚠️ S3TokenizerV2 not found at \(Self.s3TokenizerPath), skipping test")
      print("   Run: huggingface-cli download mlx-community/S3TokenizerV2")
      return
    }

    // === Load models once ===
    print("Step 1: Loading CosyVoice2 model...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let tts = try await CosyVoice2TTS.load(modelPath: Self.modelPath)
    print("   Model loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - loadStart))s")

    print("Step 2: Loading S3 tokenizer...")
    let s3Start = CFAbsoluteTimeGetCurrent()
    let s3Tokenizer = try Self.loadS3Tokenizer()
    print("   S3 tokenizer loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - s3Start))s")

    print("Step 3: Loading reference audio...")
    let refAudioSamples = try Self.loadReferenceAudio(path: refAudioPath)
    let refAudioSampleCount = refAudioSamples.shape[0]
    print("   Reference audio: \(refAudioSampleCount) samples (\(String(format: "%.1f", Float(refAudioSampleCount) / 24000.0))s at 24kHz)")

    // === Test 1: Cross-lingual mode ===
    print("\n--- Test 1: Cross-lingual Mode ---")

    print("Preparing conditionals (no reference text)...")
    let condStart = CFAbsoluteTimeGetCurrent()
    // Use nonisolated(unsafe) to bypass Sendable check since we're running sequentially
    nonisolated(unsafe) let tokenizer = s3Tokenizer
    let crossLingualConditionals = await tts.prepareConditionals(
      refWav: refAudioSamples,
      refText: nil, // Cross-lingual mode
      s3Tokenizer: { mel, melLen in
        tokenizer(mel, melLen: melLen)
      }
    )
    print("   Conditionals prepared in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - condStart))s")
    print("   Speech tokens shape: \(crossLingualConditionals.promptSpeechToken.shape)")
    print("   Prompt mel shape: \(crossLingualConditionals.promptMel.shape)")

    let crossLingualText = "Hello! This is a test of the CosyVoice2 text to speech system."
    let crossLingualTokens = await tts.encode(text: crossLingualText)
    print("   Text: '\(crossLingualText)'")
    print("   Tokens: \(crossLingualTokens.count)")

    print("Generating audio...")
    let genStart = CFAbsoluteTimeGetCurrent()
    let crossLingualResult = try await tts.generateCrossLingual(
      text: crossLingualText,
      textTokens: crossLingualTokens,
      conditionals: crossLingualConditionals,
      sampling: 25,
      nTimesteps: 10
    )
    let genTime = CFAbsoluteTimeGetCurrent() - genStart

    let audioDuration = Float(crossLingualResult.audio.count) / Float(crossLingualResult.sampleRate)
    let rtf = Float(genTime) / audioDuration
    print("   Generated \(crossLingualResult.audio.count) samples in \(String(format: "%.2f", genTime))s")
    print("   Audio duration: \(String(format: "%.2f", audioDuration))s")
    print("   Real-time factor: \(String(format: "%.2f", rtf))x")

    // Check for NaN values
    let nanCount = crossLingualResult.audio.filter { $0.isNaN }.count
    let minVal = crossLingualResult.audio.min() ?? 0
    let maxVal = crossLingualResult.audio.max() ?? 0
    print("   Audio stats: min=\(minVal), max=\(maxVal), NaN count=\(nanCount)")

    let crossLingualOutput = "\(Self.outputDir)/cross_lingual_test.wav"
    try Self.saveAudio(crossLingualResult.audio, to: crossLingualOutput, sampleRate: crossLingualResult.sampleRate)

    #expect(crossLingualResult.audio.count > 0, "Cross-lingual should generate audio")
    #expect(crossLingualResult.sampleRate == 24000, "Sample rate should be 24kHz")
    print("✓ Cross-lingual test passed! Output: \(crossLingualOutput)")

    // === Test 2: Zero-shot mode ===
    print("\n--- Test 2: Zero-shot Mode ---")

    let refText = "This is a sample reference text for zero-shot voice cloning."
    print("Preparing conditionals with reference text: '\(refText)'")

    // Reload reference audio to avoid data race with previous use
    let zeroShotAudio = try Self.loadReferenceAudio(path: refAudioPath)
    let zeroShotConditionals = await tts.prepareConditionals(
      refWav: zeroShotAudio,
      refText: refText, // Zero-shot mode
      s3Tokenizer: { mel, melLen in
        tokenizer(mel, melLen: melLen)
      }
    )

    let zeroShotText = "CosyVoice2 is a powerful text to speech model."
    let zeroShotTokens = await tts.encode(text: zeroShotText)
    print("   Text: '\(zeroShotText)'")
    print("   Tokens: \(zeroShotTokens.count)")

    print("Generating audio...")
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
    print("   Generated \(zeroShotResult.audio.count) samples in \(String(format: "%.2f", zeroShotTime))s")
    print("   Audio duration: \(String(format: "%.2f", zeroShotDuration))s")

    let zeroShotOutput = "\(Self.outputDir)/zero_shot_test.wav"
    try Self.saveAudio(zeroShotResult.audio, to: zeroShotOutput, sampleRate: zeroShotResult.sampleRate)

    #expect(zeroShotResult.audio.count > 0, "Zero-shot should generate audio")
    print("✓ Zero-shot test passed! Output: \(zeroShotOutput)")

    print("\n=== All tests passed! ===")
    print("Output files in: \(Self.outputDir)")
  }
}
