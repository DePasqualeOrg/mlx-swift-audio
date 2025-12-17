// Copyright © Anthony DePasquale

import Foundation
import MLX
import Testing

@testable import MLXAudio

@Suite(.serialized)
struct FunASRTests {
  @Test @MainActor func funASRBasicTranscribe() async throws {
    print("Testing Fun-ASR basic transcription...")

    // Download test audio (cached) - same as Whisper tests for comparison
    let audioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
    let testAudioURL = try await TestAudioCache.downloadToFile(from: audioURL)

    let expectedText = "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired"

    // Pre-compute expected words (strip punctuation, lowercase, split)
    let punctuation = CharacterSet.punctuationCharacters
    let expectedWords = Set(
      expectedText.lowercased()
        .components(separatedBy: punctuation).joined()
        .split(separator: " ").map(String.init)
    )

    print("Loading Fun-ASR nano (4-bit)...")
    let engine = STT.funASR(variant: .nano4bit)
    try await engine.load()
    #expect(engine.isLoaded == true)

    print("Transcribing...")
    let result = try await engine.transcribe(testAudioURL, language: .english)

    // Strip punctuation for word comparison
    let transcribedWords = Set(
      result.text.lowercased()
        .components(separatedBy: punctuation).joined()
        .split(separator: " ").map(String.init)
    )
    let accuracy = Float(transcribedWords.intersection(expectedWords).count) / Float(expectedWords.count)

    print("  Transcription: \(result.text)")
    print("  Expected: \(expectedText)")
    print("  Accuracy: \(String(format: "%.0f%%", accuracy * 100))")
    print("  Duration: \(String(format: "%.2f", result.duration))s")
    print("  Processing time: \(String(format: "%.2f", result.processingTime))s")
    print("  RTF: \(String(format: "%.2f", result.realTimeFactor))x")

    // Fun-ASR should achieve reasonable accuracy
    #expect(accuracy >= 0.7, "Expected at least 70% accuracy, got \(String(format: "%.0f%%", accuracy * 100))")
    #expect(result.duration > 0)
    #expect(result.processingTime > 0)

    await engine.unload()
    #expect(engine.isLoaded == false)
  }

  @Test @MainActor func funASRTranslateToEnglish() async throws {
    print("Testing Fun-ASR MLT translation from Spanish to English...")

    // Use MLT (multilingual) variant for translation tasks
    let engine = STT.funASR(variant: .mltNano4bit)
    try await engine.load()
    #expect(engine.isLoaded == true)

    // Spanish audio: counting 1-10 in Spanish (CC BY-SA 3.0, Wikimedia Commons)
    let audioURL = URL(string: "https://upload.wikimedia.org/wikipedia/commons/3/36/1-10-sp.ogg")!

    print("Downloading Spanish test audio...")
    let (tempFileURL, _) = try await URLSession.shared.download(from: audioURL)

    let testAudioURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_funasr_translate.ogg")
    if FileManager.default.fileExists(atPath: testAudioURL.path) {
      try FileManager.default.removeItem(at: testAudioURL)
    }
    try FileManager.default.moveItem(at: tempFileURL, to: testAudioURL)
    defer { try? FileManager.default.removeItem(at: testAudioURL) }

    print("Translating Spanish audio to English...")
    let result = try await engine.translate(testAudioURL, sourceLanguage: .auto, targetLanguage: .english)

    print("Translation result:")
    print("  Text: \(result.text)")
    print("  Language: \(result.language)")
    print("  Duration: \(String(format: "%.2f", result.duration))s")
    print("  Processing time: \(String(format: "%.2f", result.processingTime))s")
    print("  RTF: \(String(format: "%.2f", result.realTimeFactor))x")

    // Model may output words ("one two three..."), digits ("1 2 3..."), or concatenated ("12345...")
    let expectedWords = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    let expectedDigits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    let lowercasedResult = result.text.lowercased()
    let resultTokens = Set(lowercasedResult.split(separator: " ").map(String.init))

    // Check for word matches (space-separated)
    let matchedWords = expectedWords.filter { resultTokens.contains($0) }

    // Check for digit matches (space-separated)
    let matchedDigits = expectedDigits.filter { resultTokens.contains($0) }

    // Check for digits contained in output (handles concatenated output like "12345678910")
    let containedDigits = expectedDigits.filter { lowercasedResult.contains($0) }

    let totalMatched = max(matchedWords.count, matchedDigits.count, containedDigits.count)

    print("  Matched: \(totalMatched) (words: \(matchedWords.count), digits: \(matchedDigits.count), contained: \(containedDigits.count))")

    // Fun-ASR translation should capture at least some numbers
    #expect(totalMatched >= 3, "Expected at least 3 numbers in translation, got \(totalMatched)")
    #expect(result.duration > 0)
    #expect(result.processingTime > 0)

    await engine.unload()
  }

  @Test @MainActor func funASRModelVariants() async throws {
    print("Testing Fun-ASR model variants load correctly...")

    // Test that each variant can be loaded
    // Only test 4-bit for CI speed, others can be tested manually

    let engine = STT.funASR(variant: .nano4bit)
    try await engine.load()
    #expect(engine.isLoaded == true)
    #expect(engine.variant == .nano4bit)

    await engine.unload()
    #expect(engine.isLoaded == false)

    print("Model variant test passed")
  }

  @Test func funASRAudioPreprocessing() async throws {
    print("Testing Fun-ASR audio preprocessing...")

    // Test log mel spectrogram computation
    let sampleRate = FunASRAudio.sampleRate
    let duration: Float = 1.0
    let numSamples = Int(Float(sampleRate) * duration)

    // Create a simple test signal (sine wave)
    let frequency: Float = 440.0
    var samples: [Float] = []
    for i in 0 ..< numSamples {
      let t = Float(i) / Float(sampleRate)
      samples.append(sin(2 * .pi * frequency * t))
    }
    let audio = MLXArray(samples)

    // Test mel spectrogram computation
    let melSpec = funASRLogMelSpectrogram(
      audio: audio,
      nMels: FunASRAudio.nMels,
      nFft: FunASRAudio.nFft,
      hopLength: FunASRAudio.hopLength
    )

    print("  Mel spectrogram shape: \(melSpec.shape)")
    #expect(melSpec.ndim == 2)
    #expect(melSpec.shape[1] == FunASRAudio.nMels)

    // Test LFR processing
    let lfrFeatures = applyLFR(
      melSpec,
      lfrM: FunASRAudio.lfrM,
      lfrN: FunASRAudio.lfrN
    )

    print("  LFR features shape: \(lfrFeatures.shape)")
    #expect(lfrFeatures.ndim == 2)
    #expect(lfrFeatures.shape[1] == FunASRAudio.nMels * FunASRAudio.lfrM)

    // LFR should reduce sequence length by factor of lfrN
    let expectedLen = (melSpec.shape[0] + FunASRAudio.lfrN - 1) / FunASRAudio.lfrN
    #expect(lfrFeatures.shape[0] == expectedLen)

    print("Audio preprocessing tests passed")
  }

  @Test @MainActor func funASRAutoLanguageDetection() async throws {
    print("Testing Fun-ASR auto language detection...")

    // Download test audio (cached)
    let audioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
    let testAudioURL = try await TestAudioCache.downloadToFile(from: audioURL)

    let engine = STT.funASR(variant: .nano4bit)
    try await engine.load()

    // Transcribe with auto language detection
    let result = try await engine.transcribe(testAudioURL, language: .auto)

    print("  Detected language: \(result.language)")
    print("  Transcription: \(result.text.prefix(100))...")

    // Should detect English for English audio
    #expect(result.language == "en" || result.language == "english", "Expected English, got \(result.language)")
    #expect(!result.text.isEmpty)

    await engine.unload()
  }

  @Test @MainActor func funASRStreamingTranscription() async throws {
    print("Testing Fun-ASR streaming transcription...")

    // Download test audio (cached)
    let audioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
    let testAudioURL = try await TestAudioCache.downloadToFile(from: audioURL)

    let engine = STT.funASR(variant: .nano4bit)
    try await engine.load()

    let startTime = CFAbsoluteTimeGetCurrent()
    var tokens: [String] = []
    var firstTokenTime: Double?

    let stream = try await engine.transcribeStreaming(testAudioURL, language: .english)
    for try await token in stream {
      if firstTokenTime == nil {
        firstTokenTime = CFAbsoluteTimeGetCurrent() - startTime
      }
      tokens.append(token)
    }

    let totalTime = CFAbsoluteTimeGetCurrent() - startTime
    let fullText = tokens.joined()

    print("Streaming results:")
    print("  First token latency: \(String(format: "%.2f", firstTokenTime ?? 0))s")
    print("  Total time: \(String(format: "%.2f", totalTime))s")
    print("  Token count: \(tokens.count)")
    print("  Full text: \(fullText.prefix(100))...")

    #expect(tokens.count > 0, "Expected at least some tokens")
    #expect(firstTokenTime != nil, "Should have received first token")
    #expect(!fullText.isEmpty, "Full text should not be empty")

    await engine.unload()
  }
}
