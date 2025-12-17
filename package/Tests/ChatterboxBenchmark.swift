// Copyright © Anthony DePasquale

import AVFoundation
import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXAudio

// MARK: - Memory Management

/// Configure GPU memory limits to prevent runaway memory growth during benchmarks.
private func configureMemoryLimits() {
  // Use library's memory configuration utility
  MLXMemory.configure(cacheLimit: 512 * 1024 * 1024)
  MLXMemory.logStats(prefix: "Initial")
}

/// Clear GPU cache between benchmark runs to prevent memory accumulation
private func clearMemoryBetweenRuns() {
  MLXMemory.clearCache()
}

/// Chatterbox Pipeline Benchmark
///
/// Measures timing for each stage of the Chatterbox TTS pipeline.
/// Run with: swift test --filter ChatterboxBenchmark
@Suite(.serialized)
struct ChatterboxBenchmark {
  // MARK: - Configuration

  /// Test text (should match Python benchmark)
  static let testText = "Hello, this is a test of the Chatterbox text to speech system."

  /// Random seed for reproducibility (should match Python benchmark)
  static let seed: UInt64 = 42

  /// Number of benchmark runs
  static let numRuns = 3

  /// Number of warmup runs
  static let warmupRuns = 1

  /// Default reference audio URL
  static let defaultReferenceAudioURL = URL(
    string: "https://archive.org/download/short_poetry_001_librivox/dead_boche_graves_sm.mp3",
  )!

  // MARK: - Detailed Pipeline Benchmark

  @Test @MainActor func pipelineBenchmark() async throws {
    print("=" * 60)
    print("Chatterbox Pipeline Benchmark (Swift MLX)")
    print("=" * 60)

    // Configure memory limits FIRST to prevent runaway growth
    configureMemoryLimits()

    print("\nRandom seed: \(Self.seed)")
    print("Test text: \"\(Self.testText)\"")

    // Stage 0: Model Loading (uses shared model)
    print("\n" + "-" * 40)
    print("Stage 0: Model Loading")
    print("-" * 40)

    let loadStart = CFAbsoluteTimeGetCurrent()
    let model = try await ChatterboxTestHelper.getOrLoadModel()
    let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
    print("  Model load time: \(String(format: "%.3f", loadTime))s (global shared instance)")

    // Load reference audio (default sample)
    print("\nLoading reference audio...")
    let (samples, sampleRate) = try await downloadAndLoadAudio(from: Self.defaultReferenceAudioURL)
    print("  Sample rate: \(sampleRate) Hz")
    print("  Duration: \(String(format: "%.2f", Float(samples.count) / Float(sampleRate)))s")

    let refWav = MLXArray(samples)
    let refSr = sampleRate

    // Warmup runs
    print("\nPerforming \(Self.warmupRuns) warmup run(s)...")
    MLXRandom.seed(Self.seed)
    for _ in 0 ..< Self.warmupRuns {
      let conds = model.prepareConditionals(refWav: refWav, refSr: refSr, exaggeration: 0.1)
      let wav = model.generate(
        text: Self.testText,
        conds: conds,
        exaggeration: 0.1,
        cfgWeight: 0.5,
        temperature: 0.8,
        repetitionPenalty: 1.2,
        minP: 0.05,
        topP: 1.0,
        maxNewTokens: 1000,
      )
      wav.eval()
    }

    // Clear cache after warmup to start fresh
    clearMemoryBetweenRuns()

    // Benchmark runs with detailed timing
    print("\nPerforming \(Self.numRuns) benchmark run(s)...")

    var stageTimes: [String: [Double]] = [
      "prepare_conditionals": [],
      "text_tokenization": [],
      "t3_inference": [],
      "s3gen_waveform": [],
      "total": [],
    ]

    var lastAudioDuration: Double = 0

    for run in 0 ..< Self.numRuns {
      print("\n--- Run \(run + 1)/\(Self.numRuns) ---")

      // Clear cache at start of each run to prevent accumulation
      if run > 0 {
        clearMemoryBetweenRuns()
      }

      // Reset seed for each run for reproducibility
      MLXRandom.seed(Self.seed + UInt64(run))

      let totalStart = CFAbsoluteTimeGetCurrent()

      // Stage 1: Prepare conditionals
      var stageStart = CFAbsoluteTimeGetCurrent()
      var conds = model.prepareConditionals(refWav: refWav, refSr: refSr, exaggeration: 0.1)
      conds.t3.speakerEmb.eval()
      conds.gen.embedding.eval()
      stageTimes["prepare_conditionals"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // Stage 2: Text tokenization
      stageStart = CFAbsoluteTimeGetCurrent()
      let normalizedText = puncNorm(Self.testText)
      var textTokens = model.textTokenizer!.textToTokens(normalizedText)
      textTokens.eval()

      // Add CFG duplication and start/end tokens
      let cfgWeight: Float = 0.5
      if cfgWeight > 0.0 {
        textTokens = MLX.concatenated([textTokens, textTokens], axis: 0)
      }
      let sot = model.config.t3Config.startTextToken
      let eot = model.config.t3Config.stopTextToken
      let sotTokens = MLXArray.full([textTokens.shape[0], 1], values: MLXArray(Int32(sot)))
      let eotTokens = MLXArray.full([textTokens.shape[0], 1], values: MLXArray(Int32(eot)))
      textTokens = MLX.concatenated([sotTokens, textTokens, eotTokens], axis: 1)
      textTokens.eval()
      stageTimes["text_tokenization"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // Stage 3: T3 inference (text -> speech tokens)
      stageStart = CFAbsoluteTimeGetCurrent()
      var speechTokens = model.t3.inference(
        t3Cond: &conds.t3,
        textTokens: textTokens,
        maxNewTokens: 1000,
        temperature: 0.8,
        topP: 1.0,
        minP: 0.05,
        repetitionPenalty: 1.2,
        cfgWeight: cfgWeight,
      )
      speechTokens.eval()
      stageTimes["t3_inference"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      let numSpeechTokens = speechTokens.shape[1]

      // Post-process tokens
      speechTokens = speechTokens[0 ..< 1]
      speechTokens = dropInvalidTokens(speechTokens)

      // Filter out tokens >= SPEECH_VOCAB_SIZE
      let mask = speechTokens .< ChatterboxSpeechVocabSize
      let maskValues = mask.asArray(Bool.self)
      let validIndices = maskValues.enumerated().compactMap { $0.element ? Int32($0.offset) : nil }
      if !validIndices.isEmpty {
        speechTokens = speechTokens[MLXArray(validIndices)]
      }
      speechTokens = speechTokens.expandedDimensions(axis: 0)

      // Stage 4: S3Gen waveform generation
      stageStart = CFAbsoluteTimeGetCurrent()
      var wav = model.s3gen(speechTokens: speechTokens, refDict: conds.gen, finalize: true)
      wav.eval()
      stageTimes["s3gen_waveform"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      stageTimes["total"]!.append(CFAbsoluteTimeGetCurrent() - totalStart)

      // Print per-run results
      if wav.ndim == 2 {
        wav = wav.squeezed(axis: 0)
      }
      let audioDuration = Double(wav.shape[0]) / Double(ChatterboxS3GenSr)
      lastAudioDuration = audioDuration
      let rtf = stageTimes["total"]!.last! / audioDuration

      print("  prepare_conditionals: \(String(format: "%.3f", stageTimes["prepare_conditionals"]!.last!))s")
      print("  text_tokenization:    \(String(format: "%.3f", stageTimes["text_tokenization"]!.last!))s")
      print("  t3_inference:         \(String(format: "%.3f", stageTimes["t3_inference"]!.last!))s (\(numSpeechTokens) tokens)")
      print("  s3gen_waveform:       \(String(format: "%.3f", stageTimes["s3gen_waveform"]!.last!))s")
      print("  total:                \(String(format: "%.3f", stageTimes["total"]!.last!))s")
      print("  audio_duration:       \(String(format: "%.2f", audioDuration))s")
      print("  RTF:                  \(String(format: "%.2f", rtf))")
      let memStats = MLXMemory.snapshot()
      print("  [Memory] active=\(memStats.activeMB)MB, cache=\(memStats.cacheMB)MB, peak=\(memStats.peakMB)MB")
    }

    // Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (averaged over \(Self.numRuns) runs)")
    print("=" * 60)

    let totalAvg = stageTimes["total"]!.reduce(0, +) / Double(Self.numRuns)
    for (stage, times) in stageTimes.sorted(by: { $0.key < $1.key }) {
      let avg = times.reduce(0, +) / Double(times.count)
      let variance = times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(times.count)
      let std = sqrt(variance)
      let pct = stage == "total" ? 100.0 : (avg / totalAvg) * 100.0
      print("\(stage.padding(toLength: 25, withPad: " ", startingAt: 0)): \(String(format: "%.3f", avg))s ± \(String(format: "%.3f", std))s (\(String(format: "%5.1f", pct))%)")
    }

    // RTF summary
    let avgRtf = totalAvg / lastAudioDuration
    print("\nAverage RTF: \(String(format: "%.2f", avgRtf))")
    print("Audio duration: \(String(format: "%.2f", lastAudioDuration))s")
  }

  // MARK: - Helpers

  private func downloadAndLoadAudio(from url: URL) async throws -> (samples: [Float], sampleRate: Int) {
    // Download audio file (cached)
    let fileURL = try await TestAudioCache.downloadToFile(from: url)

    // Load audio using AVFoundation
    let audioFile = try AVAudioFile(forReading: fileURL)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
      throw TestError(message: "Failed to create audio buffer")
    }

    try audioFile.read(into: buffer)

    guard let floatData = buffer.floatChannelData else {
      throw TestError(message: "Failed to read audio data")
    }

    let channelCount = Int(format.channelCount)
    let frameLength = Int(buffer.frameLength)

    // Convert to mono if needed
    var samples: [Float]
    if channelCount == 1 {
      samples = Array(UnsafeBufferPointer(start: floatData[0], count: frameLength))
    } else {
      samples = [Float](repeating: 0, count: frameLength)
      for frame in 0 ..< frameLength {
        var sum: Float = 0
        for channel in 0 ..< channelCount {
          sum += floatData[channel][frame]
        }
        samples[frame] = sum / Float(channelCount)
      }
    }

    return (samples, Int(format.sampleRate))
  }
}

// MARK: - String Repeat Operator

private extension String {
  static func * (left: String, right: Int) -> String {
    String(repeating: left, count: right)
  }
}
