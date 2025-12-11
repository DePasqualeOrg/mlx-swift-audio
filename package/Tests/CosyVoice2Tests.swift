import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import Testing

@testable import MLXAudio

// MARK: - End-to-End Integration Tests

@Suite(.serialized)
struct CosyVoice2IntegrationTests {
  /// HuggingFace repo ID for 4-bit model
  static let modelRepoId = "mlx-community/CosyVoice2-0.5B-4bit"

  /// HuggingFace repo ID for S3 tokenizer
  static let s3TokenizerRepoId = "mlx-community/S3TokenizerV2"

  /// Reference audio from LJ Speech dataset (public domain)
  /// This is a clear female voice reading: "The examination and testimony of the experts
  /// enabled the commission to conclude that five shots may have been fired"
  static let referenceAudioURL = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
  static let referenceTranscription =
    "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired"

  /// Output directory for generated audio
  static let outputDir = FileManager.default.temporaryDirectory.appendingPathComponent("cosyvoice2-test")

  /// Load S3TokenizerV2 from HuggingFace
  static func loadS3Tokenizer() async throws -> S3TokenizerV2 {
    let modelDirectory = try await HubConfiguration.shared.snapshot(from: s3TokenizerRepoId)
    let weightURL = modelDirectory.appendingPathComponent("model.safetensors")
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

  /// Download audio from URL and return as MLXArray at 24kHz
  static func downloadAudio(from url: URL) async throws -> MLXArray {
    print("Downloading audio from \(url.lastPathComponent)...")
    let (data, _) = try await URLSession.shared.data(from: url)

    // Save to temp file for AVAudioFile to read
    let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("temp_\(UUID().uuidString).wav")
    try data.write(to: tempURL)
    defer { try? FileManager.default.removeItem(at: tempURL) }

    return try loadAudioFile(at: tempURL)
  }

  /// Load audio file and resample to 24kHz mono
  static func loadAudioFile(at url: URL) throws -> MLXArray {
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

  /// End-to-end test: Generate speech in both modes and verify with Whisper
  /// Uses publicly available LJ Speech audio as reference voice
  @Test func testVoiceMatchingWithWhisperVerification() async throws {
    print("=== CosyVoice2 Voice Matching Test with Whisper Verification ===\n")

    // === Step 1: Load all models once ===
    print("Step 1: Loading models...")

    let ttsStart = CFAbsoluteTimeGetCurrent()
    let tts = try await CosyVoice2TTS.load(repoId: Self.modelRepoId)
    print("  CosyVoice2 loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - ttsStart))s")

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

    let crossLingualText = "Hello, this is a test of CosyVoice voice matching."
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

    let crossLingualURL = Self.outputDir.appendingPathComponent("cross_lingual_test.wav")
    try Self.saveAudio(crossLingualResult.audio, to: crossLingualURL, sampleRate: crossLingualResult.sampleRate)

    let crossLingualTranscription = try await whisper.transcribe(crossLingualURL, language: .english)
    let crossLingualAccuracy = Self.computeWordAccuracy(expected: crossLingualText, transcribed: crossLingualTranscription.text)

    print("  Output: \"\(crossLingualTranscription.text)\"")
    print("  Accuracy: \(String(format: "%.0f%%", crossLingualAccuracy * 100)), RTF: \(String(format: "%.2fx", crossLingualRTF))")
    print("  Saved: \(crossLingualURL.path)")

    #expect(crossLingualAccuracy >= 0.70, "Cross-lingual: Expected ≥70% accuracy, got \(String(format: "%.0f%%", crossLingualAccuracy * 100))")
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

    let zeroShotURL = Self.outputDir.appendingPathComponent("zero_shot_test.wav")
    try Self.saveAudio(zeroShotResult.audio, to: zeroShotURL, sampleRate: zeroShotResult.sampleRate)

    let zeroShotTranscription = try await whisper.transcribe(zeroShotURL, language: .english)
    let zeroShotAccuracy = Self.computeWordAccuracy(expected: zeroShotText, transcribed: zeroShotTranscription.text)

    print("  Output: \"\(zeroShotTranscription.text)\"")
    print("  Accuracy: \(String(format: "%.0f%%", zeroShotAccuracy * 100)), RTF: \(String(format: "%.2fx", zeroShotRTF))")
    print("  Saved: \(zeroShotURL.path)")

    #expect(zeroShotAccuracy >= 0.70, "Zero-shot: Expected ≥70% accuracy, got \(String(format: "%.0f%%", zeroShotAccuracy * 100))")
    #expect(zeroShotRTF < 5.0, "Zero-shot: RTF \(String(format: "%.1f", zeroShotRTF)) exceeds 5x threshold")

    // Cleanup
    await whisper.unload()
//    try? FileManager.default.removeItem(at: Self.outputDir)

    print("\n=== All voice matching tests passed! ===")
  }
}
