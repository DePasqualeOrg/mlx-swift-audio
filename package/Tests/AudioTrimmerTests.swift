// Copyright © Anthony DePasquale

import AVFoundation
import Foundation
import MLX
import XCTest

@testable import MLXAudio

final class AudioTrimmerTests: XCTestCase {
  // MARK: - Test Constants

  let testSampleRate = 16000

  // MARK: - Real Audio Test Data

  /// LJ Speech sample - clear female voice reading a sentence
  static let ljSpeechURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
  static let ljSpeechExpectedText =
    "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired"

  // MARK: - Silence Trimming Tests

  func testTrimSilenceRemovesLeadingSilence() {
    // Create audio: [1 second silence] + [1 second tone]
    let silenceSamples = Int(1.0 * Float(testSampleRate))
    let toneSamples = Int(1.0 * Float(testSampleRate))

    var audio = [Float](repeating: 0.0, count: silenceSamples)
    // Add a simple tone (0.5 amplitude)
    for i in 0 ..< toneSamples {
      audio.append(0.5 * sin(Float(i) * 440.0 * 2.0 * .pi / Float(testSampleRate)))
    }

    let trimmed = AudioTrimmer.trimSilence(audio, sampleRate: testSampleRate)

    // Trimmed audio should be significantly shorter (mostly the tone part)
    let originalDuration = Float(audio.count) / Float(testSampleRate)
    let trimmedDuration = Float(trimmed.count) / Float(testSampleRate)

    XCTAssertLessThan(trimmedDuration, originalDuration, "Trimmed audio should be shorter")
    XCTAssertGreaterThan(trimmedDuration, 0.5, "Should preserve most of the tone")
    XCTAssertLessThan(trimmedDuration, 1.5, "Should remove most of the silence")
  }

  func testTrimSilenceRemovesTrailingSilence() {
    // Create audio: [1 second tone] + [1 second silence]
    let toneSamples = Int(1.0 * Float(testSampleRate))
    let silenceSamples = Int(1.0 * Float(testSampleRate))

    var audio = [Float]()
    // Add a simple tone
    for i in 0 ..< toneSamples {
      audio.append(0.5 * sin(Float(i) * 440.0 * 2.0 * .pi / Float(testSampleRate)))
    }
    // Add silence
    audio.append(contentsOf: [Float](repeating: 0.0, count: silenceSamples))

    let trimmed = AudioTrimmer.trimSilence(audio, sampleRate: testSampleRate)

    let trimmedDuration = Float(trimmed.count) / Float(testSampleRate)

    XCTAssertGreaterThan(trimmedDuration, 0.5, "Should preserve most of the tone")
    XCTAssertLessThan(trimmedDuration, 1.5, "Should remove trailing silence")
  }

  func testTrimSilencePreservesAudioWithNoSilence() {
    // Create audio with just a tone (no silence)
    let toneSamples = Int(1.0 * Float(testSampleRate))

    var audio = [Float]()
    for i in 0 ..< toneSamples {
      audio.append(0.5 * sin(Float(i) * 440.0 * 2.0 * .pi / Float(testSampleRate)))
    }

    let trimmed = AudioTrimmer.trimSilence(audio, sampleRate: testSampleRate)

    // Should be approximately the same length
    let originalDuration = Float(audio.count) / Float(testSampleRate)
    let trimmedDuration = Float(trimmed.count) / Float(testSampleRate)

    XCTAssertEqual(trimmedDuration, originalDuration, accuracy: 0.1, "Audio without silence should be preserved")
  }

  func testTrimSilenceHandlesEmptyAudio() {
    let audio: [Float] = []
    let trimmed = AudioTrimmer.trimSilence(audio, sampleRate: testSampleRate)
    XCTAssertTrue(trimmed.isEmpty, "Empty audio should remain empty")
  }

  func testTrimSilenceHandlesPureSilence() {
    // Create pure silence
    let silenceSamples = Int(1.0 * Float(testSampleRate))
    let audio = [Float](repeating: 0.0, count: silenceSamples)

    let trimmed = AudioTrimmer.trimSilence(audio, sampleRate: testSampleRate)

    // Pure silence should return the original (can't trim everything)
    XCTAssertFalse(trimmed.isEmpty, "Pure silence should return original audio")
  }

  // MARK: - Speech Bounds Tests

  func testFindSpeechBoundsWithLeadingSilence() {
    // Create audio: [0.5s silence] + [1s tone] + [0.5s silence]
    let leadingSilence = Int(0.5 * Float(testSampleRate))
    let toneSamples = Int(1.0 * Float(testSampleRate))
    let trailingSilence = Int(0.5 * Float(testSampleRate))

    var audio = [Float](repeating: 0.0, count: leadingSilence)
    for i in 0 ..< toneSamples {
      audio.append(0.5 * sin(Float(i) * 440.0 * 2.0 * .pi / Float(testSampleRate)))
    }
    audio.append(contentsOf: [Float](repeating: 0.0, count: trailingSilence))

    guard let bounds = AudioTrimmer.findSpeechBounds(audio, sampleRate: testSampleRate) else {
      XCTFail("Should find speech bounds")
      return
    }

    // Start should be around the beginning of the tone
    let startTime = Float(bounds.start) / Float(testSampleRate)
    XCTAssertGreaterThan(startTime, 0.3, "Start should be after leading silence")
    XCTAssertLessThan(startTime, 0.7, "Start should be near beginning of tone")

    // End should be around the end of the tone
    let endTime = Float(bounds.end) / Float(testSampleRate)
    XCTAssertGreaterThan(endTime, 1.3, "End should be after tone")
    XCTAssertLessThan(endTime, 1.7, "End should be before trailing silence ends")
  }

  func testFindSpeechBoundsReturnsNilForEmptyAudio() {
    let audio: [Float] = []
    let bounds = AudioTrimmer.findSpeechBounds(audio, sampleRate: testSampleRate)
    XCTAssertNil(bounds, "Empty audio should return nil")
  }

  // MARK: - Word Anomaly Score Tests

  func testWordAnomalyScoreNormalWord() {
    let word = Word(word: "hello", start: 0.0, end: 0.5, probability: 0.9)
    let score = AudioTrimmer.wordAnomalyScore(word)
    XCTAssertEqual(score, 0.0, accuracy: 0.01, "Normal word should have zero anomaly score")
  }

  func testWordAnomalyScoreLowProbability() {
    let word = Word(word: "uhm", start: 0.0, end: 0.5, probability: 0.1)
    let score = AudioTrimmer.wordAnomalyScore(word)
    XCTAssertGreaterThan(score, 0.5, "Low probability word should have positive anomaly score")
  }

  func testWordAnomalyScoreVeryShortDuration() {
    let word = Word(word: "a", start: 0.0, end: 0.05, probability: 0.9)
    let score = AudioTrimmer.wordAnomalyScore(word)
    XCTAssertGreaterThan(score, 0.5, "Very short word should have positive anomaly score")
  }

  func testWordAnomalyScoreVeryLongDuration() {
    let word = Word(word: "silence", start: 0.0, end: 3.0, probability: 0.9)
    let score = AudioTrimmer.wordAnomalyScore(word)
    XCTAssertGreaterThan(score, 0.5, "Very long word should have positive anomaly score")
  }

  // MARK: - Drop Unreliable Trailing Words Tests

  func testDropUnreliableTrailingWordsRemovesLowProbability() {
    let words = [
      Word(word: "hello", start: 0.0, end: 0.5, probability: 0.9),
      Word(word: "world", start: 0.5, end: 1.0, probability: 0.9),
      Word(word: "uhm", start: 1.0, end: 1.5, probability: 0.1), // Low probability
    ]

    let filtered = AudioTrimmer.dropUnreliableTrailingWords(words, audioDuration: 2.0)

    // Should drop the low probability word plus safety margin word
    XCTAssertLessThan(filtered.count, words.count, "Should drop unreliable trailing words")
  }

  func testDropUnreliableTrailingWordsRemovesWordsAfterAudioEnd() {
    let words = [
      Word(word: "hello", start: 0.0, end: 0.5, probability: 0.9),
      Word(word: "world", start: 0.5, end: 1.0, probability: 0.9),
      Word(word: "hallucinated", start: 1.0, end: 2.0, probability: 0.8), // Ends after audio
    ]

    let filtered = AudioTrimmer.dropUnreliableTrailingWords(words, audioDuration: 1.5)

    // Should drop word that ends after audio duration
    XCTAssertLessThan(filtered.count, words.count, "Should drop words ending after audio")
    for word in filtered {
      XCTAssertLessThanOrEqual(Float(word.end), 1.55, "All remaining words should end within audio")
    }
  }

  func testDropUnreliableTrailingWordsPreservesReliableWords() {
    let words = [
      Word(word: "hello", start: 0.0, end: 0.5, probability: 0.9),
      Word(word: "world", start: 0.5, end: 1.0, probability: 0.9),
      Word(word: "good", start: 1.0, end: 1.5, probability: 0.9),
    ]

    // Use config with trailingWordsToDrop = 0 to test only anomaly filtering
    var config = AudioTrimConfig.default
    config.trailingWordsToDrop = 0

    let filtered = AudioTrimmer.dropUnreliableTrailingWords(words, audioDuration: 2.0, config: config)

    XCTAssertEqual(filtered.count, words.count, "Should preserve all reliable words")
  }

  // MARK: - Word Boundary Clip Point Tests

  func testFindWordBoundaryClipPointBasic() {
    let words = [
      Word(word: "hello", start: 0.0, end: 0.5, probability: 0.9),
      Word(word: "world", start: 0.5, end: 1.0, probability: 0.9),
      Word(word: "how", start: 1.0, end: 1.3, probability: 0.9),
      Word(word: "are", start: 1.3, end: 1.6, probability: 0.9),
      Word(word: "you", start: 1.6, end: 2.0, probability: 0.9),
    ]

    guard let (clipSample, validWords) = AudioTrimmer.findWordBoundaryClipPoint(
      words: words,
      maxDuration: 1.5,
      sampleRate: testSampleRate
    ) else {
      XCTFail("Should find clip point")
      return
    }

    // Should clip after "how" (ends at 1.3s, before maxDuration - safetyMargin)
    XCTAssertEqual(validWords.count, 3, "Should include hello, world, how")
    XCTAssertEqual(validWords.last?.word, "how", "Last word should be 'how'")

    let clipTime = Float(clipSample) / Float(testSampleRate)
    XCTAssertEqual(clipTime, 1.3, accuracy: 0.01, "Clip time should be at end of 'how'")
  }

  func testFindWordBoundaryClipPointReturnsNilWhenNoWordsInRange() {
    let words = [
      Word(word: "supercalifragilistic", start: 0.0, end: 3.0, probability: 0.9),
    ]

    let result = AudioTrimmer.findWordBoundaryClipPoint(
      words: words,
      maxDuration: 1.0,
      sampleRate: testSampleRate
    )

    // Should return nil since no word ends before maxDuration - safetyMargin
    XCTAssertNil(result, "Should return nil when no words fit within max duration")
  }

  // MARK: - Configuration Tests

  func testCosyVoice2ConfigValues() {
    let config = AudioTrimConfig.cosyVoice2
    XCTAssertEqual(config.topDb, 60.0, "CosyVoice2 should use topDb=60")
    XCTAssertEqual(config.frameLength, 0.025, "Frame length should be 25ms")
    XCTAssertEqual(config.hopLength, 0.0125, "Hop length should be 12.5ms")
  }

  func testChatterboxConfigValues() {
    let config = AudioTrimConfig.chatterbox
    XCTAssertEqual(config.topDb, 20.0, "Chatterbox should use topDb=20")
    XCTAssertEqual(config.frameLength, 0.025, "Frame length should be 25ms")
    XCTAssertEqual(config.hopLength, 0.0125, "Hop length should be 12.5ms")
  }

  // MARK: - Integration Tests

  func testTrimResultStructure() {
    let audio = [Float](repeating: 0.5, count: testSampleRate)
    let result = AudioTrimmer.trim(
      audio: audio,
      sampleRate: testSampleRate,
      maxDuration: nil
    )

    XCTAssertFalse(result.audio.isEmpty, "Result should have audio")
    XCTAssertEqual(result.sampleRate, testSampleRate, "Sample rate should be preserved")
    XCTAssertNil(result.transcription, "Transcription should be nil for simple trim")
    XCTAssertNil(result.words, "Words should be nil for simple trim")
    XCTAssertGreaterThan(result.originalDuration, 0, "Original duration should be positive")
    XCTAssertGreaterThan(result.trimmedDuration, 0, "Trimmed duration should be positive")
    XCTAssertFalse(result.clippedAtWordBoundary, "Should not be clipped at word boundary")
  }

  func testTrimWithMaxDuration() {
    // Create 3 seconds of audio
    let audio = [Float](repeating: 0.5, count: testSampleRate * 3)

    let result = AudioTrimmer.trim(
      audio: audio,
      sampleRate: testSampleRate,
      maxDuration: 1.5
    )

    XCTAssertLessThanOrEqual(result.trimmedDuration, 1.5, "Trimmed duration should not exceed max")
  }

  // MARK: - Real Audio Integration Tests

  // These tests require network access and MLX/Metal
  // Run with: xcodebuild test -scheme mlx-audio-Package -destination 'platform=macOS' -only-testing:'MLXAudioTests/AudioTrimmerTests/testSilenceTrimmingWithRealAudio'

  /// Helper to download audio file (cached)
  private static func downloadAudio(from url: URL) async throws -> (samples: [Float], sampleRate: Int) {
    let cacheURL = try await TestAudioCache.downloadToFile(from: url)
    return try loadAudioFile(at: cacheURL)
  }

  /// Helper to load audio file
  private static func loadAudioFile(at url: URL) throws -> (samples: [Float], sampleRate: Int) {
    let audioFile = try AVAudioFile(forReading: url)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
      throw TestError(message: "Failed to create buffer")
    }

    try audioFile.read(into: buffer)

    guard let floatData = buffer.floatChannelData else {
      throw TestError(message: "Failed to read audio data")
    }

    // Convert to mono if stereo
    let samples: [Float]
    if format.channelCount == 1 {
      samples = Array(UnsafeBufferPointer(start: floatData[0], count: Int(buffer.frameLength)))
    } else {
      let left = UnsafeBufferPointer(start: floatData[0], count: Int(buffer.frameLength))
      let right = UnsafeBufferPointer(start: floatData[1], count: Int(buffer.frameLength))
      samples = zip(left, right).map { ($0 + $1) / 2.0 }
    }

    return (samples, Int(format.sampleRate))
  }

  /// Test silence trimming with real audio (LJ Speech sample)
  @MainActor
  func testSilenceTrimmingWithRealAudio() async throws {
    print("\n=== Testing Silence Trimming with Real Audio ===")

    let (samples, sampleRate) = try await Self.downloadAudio(from: Self.ljSpeechURL)

    let originalDuration = Float(samples.count) / Float(sampleRate)
    print("Original audio: \(originalDuration)s at \(sampleRate)Hz")

    // Test with CosyVoice2 config (topDb=60)
    let trimmedCosyVoice2 = AudioTrimmer.trimSilence(samples, sampleRate: sampleRate, config: .cosyVoice2)
    let cosyVoice2Duration = Float(trimmedCosyVoice2.count) / Float(sampleRate)
    print("After CosyVoice2 trimming (topDb=60): \(cosyVoice2Duration)s")

    // Test with Chatterbox config (topDb=20)
    let trimmedChatterbox = AudioTrimmer.trimSilence(samples, sampleRate: sampleRate, config: .chatterbox)
    let chatterboxDuration = Float(trimmedChatterbox.count) / Float(sampleRate)
    print("After Chatterbox trimming (topDb=20): \(chatterboxDuration)s")

    // Find speech bounds
    if let bounds = AudioTrimmer.findSpeechBounds(samples, sampleRate: sampleRate) {
      let startTime = Float(bounds.start) / Float(sampleRate)
      let endTime = Float(bounds.end) / Float(sampleRate)
      print("Speech bounds: \(startTime)s - \(endTime)s")
    }

    // Verify trimming doesn't remove too much
    XCTAssertGreaterThan(cosyVoice2Duration, originalDuration * 0.5, "Should preserve most of the audio")
    XCTAssertLessThanOrEqual(cosyVoice2Duration, originalDuration, "Trimmed should not be longer than original")
  }

  /// Test Whisper transcription with word timestamps on real audio
  @MainActor
  func testWhisperWordTimestampsWithRealAudio() async throws {
    print("\n=== Testing Whisper Word Timestamps with Real Audio ===")

    let (samples, sampleRate) = try await Self.downloadAudio(from: Self.ljSpeechURL)

    let originalDuration = Float(samples.count) / Float(sampleRate)
    print("Original audio: \(originalDuration)s at \(sampleRate)Hz")

    // Load Whisper
    print("Loading Whisper...")
    let whisper = WhisperEngine(modelSize: .base, quantization: .q4)
    try await whisper.load(progressHandler: nil)

    // Resample to 16kHz for Whisper
    let audio16k: MLXArray = if sampleRate != 16000 {
      AudioResampler.resample(MLXArray(samples), from: sampleRate, to: 16000)
    } else {
      MLXArray(samples)
    }

    // Get audio duration before transcription (to avoid data race)
    let audioDuration = Float(audio16k.shape[0]) / 16000.0

    // Transcribe with word timestamps
    print("Transcribing with word timestamps...")
    let result = try await whisper.transcribe(audio16k, timestamps: .word)

    print("\n--- TRANSCRIPTION ---")
    print("Full text: \"\(result.text)\"")
    print("Expected:  \"\(Self.ljSpeechExpectedText)\"")
    print("\n--- WORD TIMESTAMPS ---")

    var allWords: [Word] = []
    for segment in result.segments {
      if let words = segment.words {
        for word in words {
          print("  [\(String(format: "%.2f", word.start))s - \(String(format: "%.2f", word.end))s] \"\(word.word)\" (prob: \(String(format: "%.2f", word.probability)))")
          allWords.append(word)
        }
      }
    }
    print("--- END WORD TIMESTAMPS ---\n")

    print("Total words: \(allWords.count)")

    // Test dropping unreliable trailing words
    let filteredWords = AudioTrimmer.dropUnreliableTrailingWords(allWords, audioDuration: audioDuration)
    print("After dropping unreliable trailing words: \(filteredWords.count) words")

    if filteredWords.count < allWords.count {
      print("Dropped words:")
      for word in allWords.suffix(allWords.count - filteredWords.count) {
        let anomaly = AudioTrimmer.wordAnomalyScore(word)
        print("  - \"\(word.word)\" (prob: \(String(format: "%.2f", word.probability)), anomaly: \(String(format: "%.2f", anomaly)))")
      }
    }

    XCTAssertFalse(allWords.isEmpty, "Should have transcribed words")
  }

  /// Test word-boundary clipping with real audio
  @MainActor
  func testWordBoundaryClippingWithRealAudio() async throws {
    print("\n=== Testing Word Boundary Clipping with Real Audio ===")

    let (samples, sampleRate) = try await Self.downloadAudio(from: Self.ljSpeechURL)

    let originalDuration = Float(samples.count) / Float(sampleRate)
    print("Original audio: \(originalDuration)s at \(sampleRate)Hz")

    // Load Whisper
    print("Loading Whisper...")
    let whisper = WhisperEngine(modelSize: .base, quantization: .q4)
    try await whisper.load(progressHandler: nil)

    // Test clipping at different max durations
    let maxDurations: [Float] = [3.0, 5.0, 7.0]

    for maxDuration in maxDurations {
      print("\n--- Testing maxDuration = \(maxDuration)s ---")

      // Resample to 16kHz for Whisper
      let audio16k: [Float]
      if sampleRate != 16000 {
        let resampled = AudioResampler.resample(MLXArray(samples), from: sampleRate, to: 16000)
        audio16k = resampled.asArray(Float.self)
      } else {
        audio16k = samples
      }

      let result = try await AudioTrimmer.trimAtWordBoundary(
        audio: audio16k,
        sampleRate: 16000,
        maxDuration: maxDuration,
        whisperEngine: whisper,
        config: .cosyVoice2
      )

      print("Original duration: \(result.originalDuration)s")
      print("Trimmed duration:  \(result.trimmedDuration)s")
      print("Clipped at word boundary: \(result.clippedAtWordBoundary)")

      if let transcription = result.transcription {
        print("Transcription: \"\(transcription)\"")
      }

      if let words = result.words {
        print("Words retained: \(words.count)")
        if let lastWord = words.last {
          print("Last word: \"\(lastWord.word)\" (ends at \(String(format: "%.2f", lastWord.end))s)")
        }
      }

      // Verify constraints
      XCTAssertLessThanOrEqual(result.trimmedDuration, maxDuration, "Trimmed duration should not exceed max")
      if result.clippedAtWordBoundary {
        XCTAssertNotNil(result.transcription, "Should have transcription when clipped at word boundary")
        XCTAssertNotNil(result.words, "Should have words when clipped at word boundary")
      }
    }
  }
}
