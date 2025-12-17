// Copyright © Anthony DePasquale

import Foundation
import MLX
import Testing

@testable import MLXAudio

@Suite(.serialized)
struct WhisperTests {
  @Test @MainActor func whisperModelVariantsTranscribe() async throws {
    print("Testing representative Whisper model variants...\n")

    // Download test audio (cached)
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

    // Helper to test a model variant
    func testVariant(
      _ size: WhisperModelSize,
      _ quantization: WhisperQuantization,
      minAccuracy: Float
    ) async throws {
      print("Testing \(size.rawValue) [\(quantization.rawValue)]...")

      let engine = STT.whisper(model: size, quantization: quantization)
      try await engine.load()

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
      print("  Accuracy: \(String(format: "%.0f%%", accuracy * 100)), RTF: \(String(format: "%.2fx", result.realTimeFactor))")
      #expect(accuracy >= minAccuracy)

      await engine.unload()
    }

    // Test matrix covering key dimensions:
    // - Model sizes: tiny, base, small, large-turbo
    // - Languages: multilingual vs English-only
    // - Quantization: fp16 vs 4bit

    // All models achieve 100% accuracy on this clear English audio clip.
    // Any deviation indicates a regression.

    // Tiny models - fastest
    try await testVariant(.tiny, .fp16, minAccuracy: 1.0)
    try await testVariant(.tiny, .q4, minAccuracy: 1.0)
    try await testVariant(.tinyEn, .fp16, minAccuracy: 1.0)

    // Base models - good balance of speed and quality
    try await testVariant(.base, .fp16, minAccuracy: 1.0)
    try await testVariant(.base, .q4, minAccuracy: 1.0)
    try await testVariant(.baseEn, .fp16, minAccuracy: 1.0)

    // Small model - mid-size
    try await testVariant(.small, .q4, minAccuracy: 1.0)

    // Large turbo - best quality
    try await testVariant(.largeTurbo, .q4, minAccuracy: 1.0)
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

  /// Test translation from Spanish to English using a freely-licensed Spanish audio sample.
  @Test @MainActor func whisperTranslateToEnglish() async throws {
    print("Testing translation from Spanish to English...")

    let engine = STT.whisper(model: .largeTurbo, quantization: .q4)
    try await engine.load()
    #expect(engine.isLoaded == true)

    // Spanish audio: counting 1-10 in Spanish (CC BY-SA 3.0, Wikimedia Commons)
    let audioURL = URL(string: "https://upload.wikimedia.org/wikipedia/commons/3/36/1-10-sp.ogg")!

    print("Downloading Spanish test audio...")
    let (tempFileURL, _) = try await URLSession.shared.download(from: audioURL)

    let testAudioURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_translate.ogg")
    if FileManager.default.fileExists(atPath: testAudioURL.path) {
      try FileManager.default.removeItem(at: testAudioURL)
    }
    try FileManager.default.moveItem(at: tempFileURL, to: testAudioURL)
    defer {
      try? FileManager.default.removeItem(at: testAudioURL)
    }

    print("Translating Spanish audio to English...")
    let result = try await engine.translate(testAudioURL)

    print("Translation result:")
    print("  Text: \(result.text)")
    print("  Language: \(result.language)")
    print("  Duration: \(String(format: "%.2f", result.duration))s")
    print("  Processing time: \(String(format: "%.2f", result.processingTime))s")

    // Model may output words ("one two three...") or digits ("1 2 3...")
    let expectedWords = Set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])
    let expectedDigits = Set(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    let resultTokens = Set(result.text.lowercased().split(separator: " ").map(String.init))
    let matchedWords = resultTokens.intersection(expectedWords)
    let matchedDigits = resultTokens.intersection(expectedDigits)
    let totalMatched = max(matchedWords.count, matchedDigits.count)

    print("  Matched: \(totalMatched) (words: \(matchedWords.count), digits: \(matchedDigits.count))")
    #expect(totalMatched >= 8, "Expected at least 8 numbers in translation, got \(totalMatched)")
    #expect(result.duration > 0)
    #expect(result.processingTime > 0)
  }

  /// Test word-level timestamps using DTW alignment
  @Test @MainActor func whisperWordTimestamps() async throws {
    print("Testing word-level timestamps...")

    // Download test audio (cached)
    let audioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
    let testAudioURL = try await TestAudioCache.downloadToFile(from: audioURL)

    // Use base model for reliable word timestamps
    let engine = STT.whisper(model: .base, quantization: .q4)
    try await engine.load()

    // Transcribe with word-level timestamps
    let result = try await engine.transcribe(testAudioURL, language: .english, timestamps: .word)

    print("Transcription: \(result.text)")
    print("Segments: \(result.segments.count)")

    // Collect all words from all segments
    var allWords: [Word] = []
    for (idx, segment) in result.segments.enumerated() {
      print("Segment \(idx): \(segment.start)-\(segment.end)s")
      guard let words = segment.words else {
        Issue.record("Segment \(idx) has no word timestamps")
        continue
      }
      allWords.append(contentsOf: words)

      for word in words {
        let duration = word.end - word.start
        print("  \"\(word.word)\" \(String(format: "%.2f", word.start))-\(String(format: "%.2f", word.end))s (duration: \(String(format: "%.2f", duration))s, prob: \(String(format: "%.2f", word.probability)))")
      }
    }

    // Verify we got word timestamps
    #expect(!allWords.isEmpty, "Should have word-level timestamps")
    print("Total words: \(allWords.count)")

    // Verify timestamp validity
    for word in allWords {
      // Word text should not be empty (empty words are filtered)
      #expect(!word.word.trimmingCharacters(in: .whitespaces).isEmpty, "Word text should not be empty")

      // End time should be >= start time
      #expect(word.end >= word.start, "Word '\(word.word)' has invalid timestamps: end (\(word.end)) < start (\(word.start))")

      // Timestamps should be within audio duration (with small tolerance)
      #expect(word.start >= 0, "Word '\(word.word)' has negative start time")
      #expect(word.end <= result.duration + 1.0, "Word '\(word.word)' end time (\(word.end)) exceeds audio duration (\(result.duration))")

      // Probability should be valid
      #expect(word.probability >= 0 && word.probability <= 1, "Word '\(word.word)' has invalid probability: \(word.probability)")
    }

    // Verify timestamps are generally monotonic (each word starts after or at previous word's start)
    var previousStart: TimeInterval = 0
    for word in allWords {
      #expect(word.start >= previousStart - 0.1, "Word '\(word.word)' starts before previous word (non-monotonic)")
      previousStart = word.start
    }

    // Verify expected words are present (case-insensitive)
    // Note: Whisper words may have leading spaces, so trim whitespace and punctuation
    let expectedWords = ["examination", "testimony", "experts", "commission", "conclude", "five", "shots", "fired"]
    let transcribedWordsLower = Set(allWords.map {
      $0.word.lowercased()
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .trimmingCharacters(in: .punctuationCharacters)
    })

    var foundCount = 0
    for expected in expectedWords {
      if transcribedWordsLower.contains(expected) {
        foundCount += 1
      } else {
        print("  Missing expected word: \(expected)")
      }
    }

    let accuracy = Float(foundCount) / Float(expectedWords.count)
    print("Word accuracy: \(String(format: "%.0f%%", accuracy * 100)) (\(foundCount)/\(expectedWords.count) expected words found)")
    #expect(accuracy >= 0.75, "Expected at least 75% of key words to be present with timestamps")

    await engine.unload()
  }

  /// Test language detection using Spanish audio.
  @Test @MainActor func whisperDetectLanguage() async throws {
    print("Testing language detection...")

    let engine = STT.whisper(model: .largeTurbo, quantization: .q4)
    try await engine.load()
    #expect(engine.isLoaded == true)

    // Spanish audio: counting 1-10 in Spanish (CC BY-SA 3.0, Wikimedia Commons)
    let audioURL = URL(string: "https://upload.wikimedia.org/wikipedia/commons/3/36/1-10-sp.ogg")!

    print("Downloading Spanish test audio...")
    let (tempFileURL, _) = try await URLSession.shared.download(from: audioURL)

    let testAudioURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_detect_lang.ogg")
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

    // Should detect Spanish
    #expect(language == .spanish, "Expected Spanish, got \(language.displayName)")
    #expect(confidence > 0.5, "Confidence should be > 50%")
  }

  /// Test longer audio transcription with the Gettysburg Address (~2.5 minutes)
  /// This tests seek-based processing across multiple 30-second segments
  @Test @MainActor func whisperLongAudioTranscription() async throws {
    print("Testing longer audio transcription (Gettysburg Address)...")

    // LibriVox public domain recording of the Gettysburg Address
    let audioURL = URL(string: "https://archive.org/download/gettysburg_johng_librivox/gettysburg_address.mp3")!

    print("Downloading Gettysburg Address audio...")
    let (tempFileURL, _) = try await URLSession.shared.download(from: audioURL)

    let testAudioURL = FileManager.default.temporaryDirectory.appendingPathComponent("gettysburg_address.mp3")
    if FileManager.default.fileExists(atPath: testAudioURL.path) {
      try FileManager.default.removeItem(at: testAudioURL)
    }
    try FileManager.default.moveItem(at: tempFileURL, to: testAudioURL)
    defer { try? FileManager.default.removeItem(at: testAudioURL) }

    // Known text of the Gettysburg Address (key phrases to check)
    let expectedPhrases = [
      "four score and seven years ago",
      "brought forth on this continent",
      "new nation",
      "conceived in liberty",
      "dedicated to the proposition",
      "all men are created equal",
      "civil war",
      "testing whether that nation",
      "great battle field",
      "final resting place",
      "brave men living and dead",
      "little note nor long remember",
      "never forget what they did here",
      "unfinished work",
      "increased devotion",
      "these dead shall not have died in vain",
      "new birth of freedom",
      "government of the people",
      "by the people",
      "for the people",
      "shall not perish from the earth",
    ]

    // Test with whisper-base (multilingual) to verify seek-based processing
    // Use word timestamps for better hallucination detection
    let engine = STT.whisper(model: .base, quantization: .q4)
    try await engine.load()

    print("Transcribing Gettysburg Address with word timestamps...")
    let result = try await engine.transcribe(testAudioURL, language: .english, timestamps: .word)

    // Show full transcription
    print("\n--- FULL TRANSCRIPTION ---")
    print(result.text)
    print("--- END TRANSCRIPTION ---\n")

    // Show all segments with full text
    print("SEGMENTS:")
    for (i, segment) in result.segments.enumerated() {
      print("  [\(i)] \(String(format: "%.1f", segment.start))-\(String(format: "%.1f", segment.end))s: \(segment.text)")
    }

    print("\nSTATISTICS:")
    print("  Duration: \(String(format: "%.2f", result.duration))s")
    print("  Processing time: \(String(format: "%.2f", result.processingTime))s")
    print("  Segments: \(result.segments.count)")
    print("  Text length: \(result.text.count) characters")

    // Verify duration is around 2.5 minutes
    #expect(result.duration > 120, "Audio should be > 2 minutes")
    #expect(result.duration < 200, "Audio should be < 3.5 minutes")

    // Check how many expected phrases are found
    let transcriptLower = result.text.lowercased()
    var foundPhrases = 0
    for phrase in expectedPhrases {
      if transcriptLower.contains(phrase) {
        foundPhrases += 1
      } else {
        print("  Missing phrase: \"\(phrase)\"")
      }
    }

    let phraseAccuracy = Float(foundPhrases) / Float(expectedPhrases.count)
    print("  Phrase accuracy: \(String(format: "%.0f%%", phraseAccuracy * 100)) (\(foundPhrases)/\(expectedPhrases.count) phrases found)")

    // Expect at least 70% of key phrases to be present
    #expect(phraseAccuracy >= 0.70, "Expected at least 70% of key phrases, got \(String(format: "%.0f%%", phraseAccuracy * 100))")

    await engine.unload()
  }

  /// Benchmark transcription speed with and without word timestamps
  @Test @MainActor func whisperWordTimestampsBenchmark() async throws {
    print("Benchmarking word timestamps overhead...")

    // LibriVox public domain recording of the Gettysburg Address
    let audioURL = URL(string: "https://archive.org/download/gettysburg_johng_librivox/gettysburg_address.mp3")!

    print("Downloading Gettysburg Address audio...")
    let (tempFileURL, _) = try await URLSession.shared.download(from: audioURL)

    let testAudioURL = FileManager.default.temporaryDirectory.appendingPathComponent("gettysburg_benchmark.mp3")
    if FileManager.default.fileExists(atPath: testAudioURL.path) {
      try FileManager.default.removeItem(at: testAudioURL)
    }
    try FileManager.default.moveItem(at: tempFileURL, to: testAudioURL)
    defer { try? FileManager.default.removeItem(at: testAudioURL) }

    let engine = STT.whisper(model: .base, quantization: .q4)
    try await engine.load()

    // Without word timestamps
    print("\nTranscribing WITHOUT word timestamps...")
    let resultSegment = try await engine.transcribe(testAudioURL, language: .english, timestamps: .segment)
    print("  Duration: \(String(format: "%.2f", resultSegment.duration))s")
    print("  Processing: \(String(format: "%.2f", resultSegment.processingTime))s")
    print("  RTF: \(String(format: "%.2f", resultSegment.realTimeFactor))x")

    // With word timestamps
    print("\nTranscribing WITH word timestamps...")
    let resultWord = try await engine.transcribe(testAudioURL, language: .english, timestamps: .word)
    print("  Duration: \(String(format: "%.2f", resultWord.duration))s")
    print("  Processing: \(String(format: "%.2f", resultWord.processingTime))s")
    print("  RTF: \(String(format: "%.2f", resultWord.realTimeFactor))x")

    let overhead = resultWord.processingTime / resultSegment.processingTime
    print("\nWord timestamps overhead: \(String(format: "%.1f", overhead))x slower")

    // Word timestamps should add no more than 2.5x overhead (typically ~1.8x)
    // This catches performance regressions while allowing for variability
    #expect(overhead < 2.5, "Word timestamps overhead (\(String(format: "%.1f", overhead))x) exceeds 2.5x threshold - performance regression?")

    await engine.unload()
  }
}
