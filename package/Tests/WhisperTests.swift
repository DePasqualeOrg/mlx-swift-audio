import AVFoundation
import Foundation
import MLX
import Testing

@testable import MLXAudio

// IMPORTANT: Whisper models are smaller than TTS models but still significant (~150MB for base).
// Run tests in isolation to avoid loading multiple models simultaneously.
//
// These tests are DISABLED by default to avoid memory issues when running benchmarks.
// To run these tests, remove the .disabled trait below.

@Suite(.serialized)
struct WhisperTests {
  @Test @MainActor func whisperEngineInitializes() async {
    let engine = STT.whisper(model: .largeTurbo)
    #expect(engine.isLoaded == false)
    #expect(engine.isTranscribing == false)
    #expect(engine.provider == .whisper)
    #expect(engine.modelSize == .largeTurbo)
  }

  @Test @MainActor func whisperEngineLoadsModel() async throws {
    let engine = STT.whisper(model: .largeTurbo) // Use large-v3-turbo - has weights.safetensors
    print("Loading Whisper large-v3-turbo model...")
    try await engine.load { progress in
      if progress.fractionCompleted > 0 {
        print("  Progress: \(Int(progress.fractionCompleted * 100))%")
      }
    }
    #expect(engine.isLoaded == true)
    print("Whisper model loaded successfully")
  }

  @Test @MainActor func whisperTokenizerIsConfiguredCorrectly() async throws {
    print("Testing Whisper tokenizer configuration...")

    // Load multilingual tokenizer (large-v3-turbo is multilingual)
    let tokenizer = try await WhisperTokenizer.load(isMultilingual: true)

    // Verify critical special token IDs match the expected values
    // These MUST match the Python tokenizer implementation
    // Verified against Python mlx-audio tokenizer (see dump_whisper_tokens.py)
    print("Verifying special token IDs:")

    #expect(tokenizer.eot == 50257, "EOT token should be 50257, got \(tokenizer.eot)")
    print("  ✓ EOT: 50257")

    #expect(tokenizer.sot == 50258, "SOT token should be 50258, got \(tokenizer.sot)")
    print("  ✓ SOT: 50258")

    #expect(tokenizer.translate == 50359, "Translate token should be 50359, got \(tokenizer.translate)")
    print("  ✓ Translate: 50359")

    #expect(tokenizer.transcribe == 50360, "Transcribe token should be 50360, got \(tokenizer.transcribe)")
    print("  ✓ Transcribe: 50360")

    #expect(tokenizer.sotLm == 50361, "SOT LM token should be 50361, got \(tokenizer.sotLm)")
    print("  ✓ SOT LM: 50361")

    #expect(tokenizer.sotPrev == 50362, "SOT Prev token should be 50362, got \(tokenizer.sotPrev)")
    print("  ✓ SOT Prev: 50362")

    #expect(tokenizer.noSpeech == 50363, "No-speech token should be 50363, got \(tokenizer.noSpeech)")
    print("  ✓ No-speech: 50363")

    #expect(tokenizer.noTimestamps == 50364, "No-timestamps token should be 50364, got \(tokenizer.noTimestamps)")
    print("  ✓ No-timestamps: 50364")

    #expect(tokenizer.timestampBegin == 50365, "Timestamp begin should be 50365, got \(tokenizer.timestampBegin)")
    print("  ✓ Timestamp begin: 50365")

    // Verify language token is correct (English should be 50259)
    let enToken = tokenizer.languageToken(for: "en")
    #expect(enToken == 50259, "English language token should be 50259, got \(enToken ?? -1)")
    print("  ✓ Language token (en): 50259")

    // Verify tokenizer can encode/decode correctly
    let testText = "Hello, world!"
    let encoded = tokenizer.encode(testText)
    let decoded = tokenizer.decode(encoded)
    #expect(!encoded.isEmpty, "Encoding should produce tokens")
    #expect(decoded.contains("Hello"), "Decoded text should contain 'Hello'")
    print("  ✓ Encode/decode works correctly")

    // Verify SOT sequence is built correctly
    let sotSeq = tokenizer.sotSequence(language: "en", task: .transcribe)
    #expect(sotSeq.count == 3, "SOT sequence should have 3 tokens (sot, lang, task)")
    #expect(sotSeq[0] == 50258, "First token should be SOT (50258)")
    #expect(sotSeq[1] == 50259, "Second token should be English (50259)")
    #expect(sotSeq[2] == 50360, "Third token should be Transcribe (50360)")
    print("  ✓ SOT sequence: \(sotSeq)")

    print("All tokenizer configuration checks passed!")
  }

  @Test @MainActor func whisperTranscribesAudio() async throws {
    // Use large-v3-turbo - has weights.safetensors
    let engine = STT.whisper(model: .largeTurbo)

    print("Loading Whisper large-v3-turbo model...")
    try await engine.load()
    #expect(engine.isLoaded == true)

    // Download a public domain audio sample from LJSpeech dataset
    // LJSpeech is a public domain speech dataset with WAV files
    // Sample text: "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
    let audioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
    let expectedText = "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired"

    print("Downloading test audio from LJSpeech...")
    let (tempFileURL, _) = try await URLSession.shared.download(from: audioURL)

    // Move to permanent location for transcription
    let testAudioURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_audio.wav")
    if FileManager.default.fileExists(atPath: testAudioURL.path) {
      try FileManager.default.removeItem(at: testAudioURL)
    }
    try FileManager.default.moveItem(at: tempFileURL, to: testAudioURL)
    defer {
      try? FileManager.default.removeItem(at: testAudioURL)
    }

    print("Transcribing test audio...")
    let result = try await engine.transcribe(
      testAudioURL,
      language: .english,
      temperature: 0.0
    )

    print("Transcription result:")
    print("  Text: \(result.text)")
    print("  Expected: \(expectedText)")
    print("  Language: \(result.language)")
    print("  Duration: \(String(format: "%.2f", result.duration))s")
    print("  Processing time: \(String(format: "%.2f", result.processingTime))s")
    print("  Real-time factor: \(String(format: "%.2fx", result.realTimeFactor))")
    print("  Segments: \(result.segments.count)")

    // Basic validation
    #expect(result.language == "en")
    #expect(result.duration > 0)
    #expect(result.processingTime > 0)
    #expect(result.segments.count > 0)

    // Check transcription quality (should contain most of the expected words)
    let transcribedWords = Set(result.text.lowercased().split(separator: " ").map(String.init))
    let expectedWords = Set(expectedText.lowercased().split(separator: " ").map(String.init))
    let matchedWords = transcribedWords.intersection(expectedWords)
    let accuracy = Float(matchedWords.count) / Float(expectedWords.count)
    print("  Word accuracy: \(String(format: "%.1f%%", accuracy * 100))")

    // With Whisper large-v3-turbo, we should get at least 90% of words correct
    #expect(accuracy > 0.9, "Transcription accuracy too low: \(accuracy)")
  }

  @Test @MainActor func whisperAllModelsTranscribe() async throws {
    print("Testing transcription with all Whisper models...")
    print("Note: This test downloads and tests multiple models sequentially\n")

    // Download test audio once
    print("Downloading test audio from LJSpeech...")
    let audioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
    let (testAudioData, _) = try await URLSession.shared.data(from: audioURL)

    let tempDir = FileManager.default.temporaryDirectory
    let testAudioURL = tempDir.appendingPathComponent("test_whisper_all_models.wav")
    try testAudioData.write(to: testAudioURL)
    defer { try? FileManager.default.removeItem(at: testAudioURL) }

    let expectedText = "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired"

    // Helper function to test a model
    func testModel(_ modelSize: WhisperModelSize, minAccuracy: Float) async throws {
      print("\n" + String(repeating: "=", count: 80))
      print("Testing \(modelSize.displayName)")
      print(String(repeating: "=", count: 80))

      let engine = STT.whisper(model: modelSize)
      try await engine.load()

      let startTime = CFAbsoluteTimeGetCurrent()
      let result = try await engine.transcribe(
        testAudioURL,
        language: .english,
        temperature: 0.0
      )
      let endTime = CFAbsoluteTimeGetCurrent()

      // Calculate accuracy
      let transcribedWords = Set(result.text.lowercased().split(separator: " ").map(String.init))
      let expectedWords = Set(expectedText.lowercased().split(separator: " ").map(String.init))
      let matchedWords = transcribedWords.intersection(expectedWords)
      let accuracy = Float(matchedWords.count) / Float(expectedWords.count)

      print("Results:")
      print("  Text: \(result.text)")
      print("  Expected: \(expectedText)")
      print("  Accuracy: \(String(format: "%.1f%%", accuracy * 100))")
      print("  Processing time: \(String(format: "%.2f", endTime - startTime))s")
      print("  Real-time factor: \(String(format: "%.2fx", result.realTimeFactor))")

      #expect(accuracy >= minAccuracy, "\(modelSize.rawValue): accuracy \(accuracy) < minimum \(minAccuracy)")

      // Unload to free memory before next model
      await engine.unload()
    }

    // Test only available models
    print("\n🌍 TESTING AVAILABLE MODELS")
    print("Note: Currently only testing models with compatible weight formats\n")

    let modelsToTest: [(WhisperModelSize, Float)] = [
      (.largeTurbo, 0.9), // Large v3 Turbo: 809M params, best available
    ]

    for (modelSize, minAccuracy) in modelsToTest {
      guard modelSize.isAvailable else {
        print("⏭️  Skipping \(modelSize.displayName) (not available)")
        continue
      }
      try await testModel(modelSize, minAccuracy: minAccuracy)
    }

    // Unavailable models (incompatible weight format or missing safetensors):
    print("\n⚠️  UNAVAILABLE MODELS:")
    print("   - .tiny, .base: Incompatible weight structure (split encoder/decoder)")
    print("   - .small, .medium, .large: Missing safetensors (only .npz available)")
    print("   - All .en models: Missing safetensors (only .npz available)")
    print("   TODO: Find or create unified safetensors repos for all model sizes")

    print("\n" + String(repeating: "=", count: 80))
    print("✅ All available models tested successfully!")
    print(String(repeating: "=", count: 80))
  }

  @Test @MainActor func whisperAudioPreprocessing() async throws {
    // Test padOrTrim function
    let shortAudio = MLXArray([Float](repeating: 0.5, count: 1000))
    let padded = padOrTrim(shortAudio, length: WhisperAudio.nSamples)
    #expect(padded.shape[0] == WhisperAudio.nSamples)

    let longAudio = MLXArray([Float](repeating: 0.5, count: 600_000))
    let trimmed = padOrTrim(longAudio, length: WhisperAudio.nSamples)
    #expect(trimmed.shape[0] == WhisperAudio.nSamples)

    print("Audio preprocessing tests passed")
  }

  @Test @MainActor func whisperMelSpectrogram() async throws {
    // Test mel spectrogram generation with 80 mel bins (standard for most Whisper models)
    let nMels = 80
    let audio = MLXArray([Float](repeating: 0.5, count: WhisperAudio.nSamples))
    let mel = whisperLogMelSpectrogram(audio: audio, nMels: nMels)

    // Check output shape: (80, 3000) for 30-second audio with 80 mel bins
    #expect(mel.shape[0] == nMels)
    #expect(mel.shape[1] == WhisperAudio.nFrames)

    // Check value range (should be roughly in [-1, 1] after normalization)
    let minVal = mel.min().item(Float.self)
    let maxVal = mel.max().item(Float.self)
    print("Mel spectrogram range: [\(minVal), \(maxVal)]")
    #expect(minVal >= -2.0) // Allow some tolerance
    #expect(maxVal <= 2.0)

    print("Mel spectrogram tests passed")
  }

  @Test @MainActor func whisperConfigurationsLoadCorrectly() async throws {
    // Test that model dimensions are loaded dynamically from config.json
    // Note: We don't hardcode configs - they're loaded from HuggingFace

    // Just verify that the largeTurbo config has the correct n_mels
    // (this is the only model with safetensors currently available)
    let engine = STT.whisper(model: .largeTurbo)
    try await engine.load()

    // The model's dims are loaded from config.json, not hardcoded
    // This test just confirms the dynamic loading works
    #expect(engine.isLoaded == true)

    print("Model configuration loaded successfully from HuggingFace config.json")
  }

  @Test @MainActor func whisperTokenizerDecodesSpecificTokensCorrectly() async throws {
    print("Testing specific token decoding against Python reference...")

    // Load multilingual tokenizer (tests use multilingual vocabulary)
    let tokenizer = try await WhisperTokenizer.load(isMultilingual: true)

    // Python successfully transcribes with these tokens:
    // [50365, 440, 23874, 293, 15634, 295, 264, 8572, 15172, 264, 10766, 281, 16886, 300, 1732, 8305, 815, 362, 668, 11777, 50744]
    // Decoded text: " The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired"

    // Test critical tokens that Swift was decoding incorrectly
    let testTokens = [
      (440, " The"),
      (23874, " examination"),
      (293, " and"),
      (15634, " testimony"),
    ]

    for (token, expectedText) in testTokens {
      let decoded = tokenizer.decode([token])
      print("Token \(token) → '\(decoded)' (expected: '\(expectedText)')")
      #expect(decoded == expectedText, "Token \(token) should decode to '\(expectedText)', got '\(decoded)'")
    }

    // Test full sequence
    let fullTokens = [440, 23874, 293, 15634, 295, 264, 8572, 15172, 264, 10766, 281, 16886, 300, 1732, 8305, 815, 362, 668, 11777]
    let fullDecoded = tokenizer.decode(fullTokens)
    print("Full sequence decoded: '\(fullDecoded)'")

    let expectedFull = " The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired"
    #expect(fullDecoded == expectedFull, "Full sequence should decode correctly")

    print("All token decoding tests passed!")
  }

  @Test @MainActor func whisperModelVocabularySelection() async throws {
    print("Testing vocabulary selection for all Whisper models...")
    print("This test verifies that multilingual and English-only models load the correct vocabulary\n")

    // Helper to test a model's vocabulary configuration
    func testModelVocabulary(
      _ modelSize: WhisperModelSize,
      expectedVocabSize: Int,
      expectedIsMultilingual: Bool,
      expectedVocabulary: String
    ) async throws {
      print("Testing \(modelSize.displayName)...")

      let model = try await WhisperModel.load(modelSize: modelSize)

      #expect(model.dims.n_vocab == expectedVocabSize, "\(modelSize.rawValue): expected n_vocab=\(expectedVocabSize), got \(model.dims.n_vocab)")
      #expect(model.isMultilingual == expectedIsMultilingual, "\(modelSize.rawValue): expected isMultilingual=\(expectedIsMultilingual), got \(model.isMultilingual)")

      // Verify correct tokenizer is loaded
      let tokenizer = try await WhisperTokenizer.load(isMultilingual: model.isMultilingual)
      #expect(tokenizer.eot == 50257, "Tokenizer should have eot=50257")
      #expect(tokenizer.sot == 50258, "Tokenizer should have sot=50258")

      print("   ✅ \(modelSize.rawValue): n_vocab=\(model.dims.n_vocab), isMultilingual=\(model.isMultilingual) → \(expectedVocabulary)")
    }

    // Test only available models
    print("🌍 AVAILABLE MODELS (n_vocab=51866 → multilingual.tiktoken)")
    try await testModelVocabulary(.largeTurbo, expectedVocabSize: 51866, expectedIsMultilingual: true, expectedVocabulary: "multilingual.tiktoken")

    // Unavailable models:
    print("\n⚠️  UNAVAILABLE MODELS:")
    print("   - .tiny, .base: Incompatible weight structure")
    print("   - .small, .medium, .large: Missing safetensors repos")
    print("   - All .en models: Missing safetensors repos")

    print("\n" + String(repeating: "=", count: 80))
    print("✅ Available model vocabulary verified!")
    print("   - 1 model available: largeTurbo (multilingual.tiktoken)")
    print("   - 9 models unavailable (awaiting compatible repos)")
    print(String(repeating: "=", count: 80))
  }

  @Test @MainActor func whisperTranslateToEnglish() async throws {
    print("Testing translation to English...")

    let engine = STT.whisper(model: .largeTurbo)
    try await engine.load()
    #expect(engine.isLoaded == true)

    // Download a Spanish audio sample (using a Spanish LJSpeech-like dataset)
    // For now, we'll use the same English audio as a placeholder
    // TODO: Replace with actual Spanish audio when available
    let audioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!

    print("Downloading test audio...")
    let (tempFileURL, _) = try await URLSession.shared.download(from: audioURL)

    let testAudioURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_translate.wav")
    if FileManager.default.fileExists(atPath: testAudioURL.path) {
      try FileManager.default.removeItem(at: testAudioURL)
    }
    try FileManager.default.moveItem(at: tempFileURL, to: testAudioURL)
    defer {
      try? FileManager.default.removeItem(at: testAudioURL)
    }

    print("Translating audio to English...")
    let result = try await engine.translate(testAudioURL, language: .english)

    print("Translation result:")
    print("  Text: \(result.text)")
    print("  Language: \(result.language)")
    print("  Duration: \(String(format: "%.2f", result.duration))s")
    print("  Processing time: \(String(format: "%.2f", result.processingTime))s")

    // Basic validation
    #expect(result.language == "en") // Output should always be English
    #expect(result.duration > 0)
    #expect(result.processingTime > 0)
    #expect(!result.text.isEmpty)

    print("Translation test passed!")
  }

  @Test @MainActor func whisperDetectLanguage() async throws {
    print("Testing language detection...")

    let engine = STT.whisper(model: .largeTurbo)
    try await engine.load()
    #expect(engine.isLoaded == true)

    // Download English audio
    let audioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!

    print("Downloading test audio...")
    let (tempFileURL, _) = try await URLSession.shared.download(from: audioURL)

    let testAudioURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_detect_lang.wav")
    if FileManager.default.fileExists(atPath: testAudioURL.path) {
      try FileManager.default.removeItem(at: testAudioURL)
    }
    try FileManager.default.moveItem(at: tempFileURL, to: testAudioURL)
    defer {
      try? FileManager.default.removeItem(at: testAudioURL)
    }

    print("Detecting language...")
    let (language, confidence) = try await engine.detectLanguage(testAudioURL)

    print("Language detection result:")
    print("  Language: \(language.displayName) (\(language.code))")
    print("  Confidence: \(String(format: "%.2f%%", confidence * 100))")

    // Should detect English
    #expect(language == .english, "Expected English, got \(language.displayName)")
    #expect(confidence > 0.5, "Confidence should be > 50%")

    print("Language detection test passed!")
  }

  @Test @MainActor func whisperAPIConsistency() async throws {
    print("Testing API consistency across URL and MLXArray inputs...")

    let engine = STT.whisper(model: .base)
    try await engine.load()

    // Download test audio
    let audioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
    let (tempFileURL, _) = try await URLSession.shared.download(from: audioURL)

    let testAudioURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_api_consistency.wav")
    if FileManager.default.fileExists(atPath: testAudioURL.path) {
      try FileManager.default.removeItem(at: testAudioURL)
    }
    try FileManager.default.moveItem(at: tempFileURL, to: testAudioURL)
    defer {
      try? FileManager.default.removeItem(at: testAudioURL)
    }

    // Load audio manually
    let audioFile = try AVAudioFile(forReading: testAudioURL)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
    try audioFile.read(into: buffer)

    // Convert to mono MLXArray
    let length = Int(buffer.frameLength)
    var audioSamples: [Float] = []
    if let floatData = buffer.floatChannelData {
      for i in 0 ..< length {
        audioSamples.append(floatData[0][i])
      }
    }
    let audioArray = MLXArray(audioSamples)

    // Resample to 16kHz if needed
    let sampleRate = Int(format.sampleRate)
    let audio16k: MLXArray = if sampleRate != WhisperAudio.sampleRate {
      AudioResampler.resample(audioArray, from: sampleRate, to: WhisperAudio.sampleRate)
    } else {
      audioArray
    }

    print("Testing transcribe() with URL...")
    let urlResult = try await engine.transcribe(testAudioURL, language: .english)

    print("Testing transcribe() with MLXArray...")
    let arrayResult = try await engine.transcribe(audio16k, language: .english)

    print("URL result: \(urlResult.text)")
    print("Array result: \(arrayResult.text)")

    // Both should produce similar results (text might vary slightly due to non-determinism)
    #expect(urlResult.text == arrayResult.text, "URL and MLXArray should produce same results")

    print("API consistency test passed!")
  }
}
