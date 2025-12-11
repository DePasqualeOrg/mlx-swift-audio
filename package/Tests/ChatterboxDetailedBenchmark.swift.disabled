import AVFoundation
import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXAudio

// MARK: - Memory Management

/// Configure GPU memory limits to prevent runaway memory growth during benchmarks.
private func configureMemoryLimits() {
  MLXMemory.configure(cacheLimit: 512 * 1024 * 1024)
  MLXMemory.logStats(prefix: "Initial")
}

/// Clear GPU cache between benchmark runs
private func clearMemoryBetweenRuns() {
  MLXMemory.clearCache()
}

/// Detailed Chatterbox Pipeline Benchmark
///
/// Measures timing for each sub-component of prepare_conditionals and T3 inference.
@Suite(.serialized)
struct ChatterboxDetailedBenchmark {
  // MARK: - Configuration

  static let testText = "Hello, this is a test of the Chatterbox text to speech system."
  static let seed: UInt64 = 42
  static let numRuns = 3
  static let defaultReferenceAudioURL = URL(
    string: "https://archive.org/download/short_poetry_001_librivox/dead_boche_graves_sm.mp3",
  )!

  // MARK: - Prepare Conditionals Detailed Benchmark

  @Test @MainActor func prepareConditionalsDetailedBenchmark() async throws {
    print("\n" + "=" * 60)
    print("PREPARE_CONDITIONALS DETAILED BENCHMARK (Swift)")
    print("=" * 60)

    // Configure memory limits
    configureMemoryLimits()

    // Load model (global shared)
    let model = try await ChatterboxTestHelper.getOrLoadModel()

    // Load reference audio
    let (samples, sampleRate) = try await downloadAndLoadAudio(from: Self.defaultReferenceAudioURL)
    let refWav = MLXArray(samples)
    let refSr = sampleRate

    print("\nReference audio: \(samples.count) samples at \(sampleRate) Hz")

    var stageTimes: [String: [Double]] = [
      "resample_to_24k": [],
      "resample_24k_to_16k": [],
      "resample_to_16k_full": [],
      "mel_spectrogram_s3gen": [],
      "mel_spectrogram_t3": [],
      "s3_tokenizer_s3gen": [],
      "s3_tokenizer_t3": [],
      "s3gen_embed_ref": [],
      "voice_encoder": [],
      "total": [],
    ]

    for run in 0 ..< Self.numRuns {
      print("\n--- Run \(run + 1)/\(Self.numRuns) ---")

      // Clear cache between runs
      if run > 0 {
        clearMemoryBetweenRuns()
      }

      var wav = refWav
      if wav.ndim == 2 {
        wav = wav.squeezed(axis: 0)
      }

      let totalStart = CFAbsoluteTimeGetCurrent()

      // Resample to 24kHz for S3Gen
      var stageStart = CFAbsoluteTimeGetCurrent()
      var refWav24k = wav
      if refSr != ChatterboxS3GenSr {
        refWav24k = resampleAudio(wav, origSr: refSr, targetSr: ChatterboxS3GenSr)
      }
      if refWav24k.shape[0] > ChatterboxModel.decCondLen {
        refWav24k = refWav24k[0 ..< ChatterboxModel.decCondLen]
      }
      refWav24k.eval()
      stageTimes["resample_to_24k"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // Resample 24kHz to 16kHz for S3Gen tokenization
      stageStart = CFAbsoluteTimeGetCurrent()
      let refWav16kFrom24k = resampleAudio(refWav24k, origSr: ChatterboxS3GenSr, targetSr: ChatterboxS3Sr)
      refWav16kFrom24k.eval()
      stageTimes["resample_24k_to_16k"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // Resample original to 16kHz for T3 encoder conditioning
      stageStart = CFAbsoluteTimeGetCurrent()
      var refWav16kFull = wav
      if refSr != ChatterboxS3Sr {
        refWav16kFull = resampleAudio(wav, origSr: refSr, targetSr: ChatterboxS3Sr)
      }
      var refWav16k = refWav16kFull
      if refWav16k.shape[0] > ChatterboxModel.encCondLen {
        refWav16k = refWav16k[0 ..< ChatterboxModel.encCondLen]
      }
      refWav16k.eval()
      refWav16kFull.eval()
      stageTimes["resample_to_16k_full"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // S3Gen mel spectrogram
      stageStart = CFAbsoluteTimeGetCurrent()
      let s3genMel = logMelSpectrogramChatterbox(audio: refWav16kFrom24k)
      let s3genMelBatch = s3genMel.expandedDimensions(axis: 0)
      let s3genMelLen = MLXArray([Int32(s3genMel.shape[1])])
      s3genMelBatch.eval()
      stageTimes["mel_spectrogram_s3gen"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // T3 mel spectrogram
      stageStart = CFAbsoluteTimeGetCurrent()
      let t3Mel = logMelSpectrogramChatterbox(audio: refWav16k)
      let t3MelBatch = t3Mel.expandedDimensions(axis: 0)
      let t3MelLen = MLXArray([Int32(t3Mel.shape[1])])
      t3MelBatch.eval()
      stageTimes["mel_spectrogram_t3"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // S3Gen tokenization
      stageStart = CFAbsoluteTimeGetCurrent()
      let (s3genTokens, _) = model.s3Tokenizer.quantize(mel: s3genMelBatch, melLen: s3genMelLen)
      s3genTokens.eval()
      stageTimes["s3_tokenizer_s3gen"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // T3 tokenization
      stageStart = CFAbsoluteTimeGetCurrent()
      let (t3Tokens, _) = model.s3Tokenizer.quantize(mel: t3MelBatch, melLen: t3MelLen)
      t3Tokens.eval()
      stageTimes["s3_tokenizer_t3"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // S3Gen embed_ref
      stageStart = CFAbsoluteTimeGetCurrent()
      let s3genRefDict = model.s3gen.embedRef(
        refWav: refWav24k.expandedDimensions(axis: 0),
        refSr: ChatterboxS3GenSr,
        refSpeechTokens: s3genTokens,
        refSpeechTokenLens: MLXArray([Int32(s3genTokens.shape[1])]),
      )
      s3genRefDict.embedding.eval()
      stageTimes["s3gen_embed_ref"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // Voice encoder
      stageStart = CFAbsoluteTimeGetCurrent()
      let veEmbed = model.ve.embedsFromWavs(wavs: [refWav16kFull])
      let veEmbedMean = veEmbed.mean(axis: 0, keepDims: true)
      veEmbedMean.eval()
      stageTimes["voice_encoder"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      stageTimes["total"]!.append(CFAbsoluteTimeGetCurrent() - totalStart)

      // Print per-run results
      for (stage, times) in stageTimes.sorted(by: { $0.key < $1.key }) {
        if let lastTime = times.last {
          print("  \(stage.padding(toLength: 25, withPad: " ", startingAt: 0)): \(String(format: "%.4f", lastTime))s")
        }
      }
    }

    // Print summary
    print("\n" + "-" * 40)
    print("SUMMARY (averaged)")
    print("-" * 40)
    let totalAvg = stageTimes["total"]!.reduce(0, +) / Double(Self.numRuns)
    for (stage, times) in stageTimes.sorted(by: { $0.key < $1.key }) {
      let avg = times.reduce(0, +) / Double(times.count)
      let variance = times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(times.count)
      let std = sqrt(variance)
      let pct = stage == "total" ? 100.0 : (avg / totalAvg) * 100.0
      print("\(stage.padding(toLength: 25, withPad: " ", startingAt: 0)): \(String(format: "%.4f", avg))s ± \(String(format: "%.4f", std))s (\(String(format: "%5.1f", pct))%)")
    }
  }

  // MARK: - T3 Inference Detailed Benchmark

  @Test @MainActor func t3InferenceDetailedBenchmark() async throws {
    print("\n" + "=" * 60)
    print("T3 INFERENCE DETAILED BENCHMARK (Swift)")
    print("=" * 60)

    // Configure memory limits
    configureMemoryLimits()

    // Load model (global shared)
    let model = try await ChatterboxTestHelper.getOrLoadModel()

    // Load reference audio
    let (samples, sampleRate) = try await downloadAndLoadAudio(from: Self.defaultReferenceAudioURL)
    let refWav = MLXArray(samples)
    let refSr = sampleRate

    // Prepare conditionals once
    var conds = model.prepareConditionals(refWav: refWav, refSr: refSr, exaggeration: 0.1)
    conds.t3.speakerEmb.eval()
    conds.gen.embedding.eval()

    // Prepare text tokens
    let normalizedText = puncNorm(Self.testText)
    var textTokens = model.textTokenizer!.textToTokens(normalizedText)

    let cfgWeight: Float = 0.5
    if cfgWeight > 0.0 {
      textTokens = MLX.concatenated([textTokens, textTokens], axis: 0)
    }

    let sot = model.config.t3Config.startTextToken
    let eot = model.config.t3Config.stopTextToken
    let sotTokens = MLXArray.full([textTokens.shape[0], 1], values: MLXArray(Int32(sot)))
    let eotTokens = MLXArray.full([textTokens.shape[0], 1], values: MLXArray(Int32(eot)))
    textTokens = MLX.concatenated([sotTokens, textTokens, eotTokens], axis: 1)

    var stageTimes: [String: [Double]] = [
      "prepare_conditioning": [],
      "text_embedding": [],
      "initial_forward": [],
      "generation_loop": [],
      "tokens_per_second": [],
      "total": [],
    ]

    for run in 0 ..< Self.numRuns {
      print("\n--- Run \(run + 1)/\(Self.numRuns) ---")

      // Clear cache between runs
      if run > 0 {
        clearMemoryBetweenRuns()
      }

      MLXRandom.seed(Self.seed + UInt64(run))

      let totalStart = CFAbsoluteTimeGetCurrent()

      // Prepare conditioning
      var stageStart = CFAbsoluteTimeGetCurrent()
      var condEmb = model.t3.prepareConditioning(&conds.t3)
      condEmb.eval()
      stageTimes["prepare_conditioning"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // Text embedding
      stageStart = CFAbsoluteTimeGetCurrent()
      var textEmbeddings = model.t3.textEmb(textTokens)
      if cfgWeight > 0.0 {
        textEmbeddings = MLX.concatenated([
          textEmbeddings[0 ..< 1],
          MLXArray.zeros(like: textEmbeddings[0 ..< 1]),
        ], axis: 0)
      }
      if model.config.t3Config.inputPosEmb == "learned" {
        textEmbeddings = textEmbeddings + model.t3.textPosEmb(textTokens)
      }
      textEmbeddings.eval()
      stageTimes["text_embedding"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // Initial forward pass
      stageStart = CFAbsoluteTimeGetCurrent()
      let bosToken = MLXArray([Int32(model.config.t3Config.startSpeechToken)]).reshaped([1, 1])
      var bosEmbed = model.t3.speechEmb(bosToken)
      bosEmbed = bosEmbed + model.t3.speechPosEmb.getFixedEmbedding(0)
      if cfgWeight > 0.0 {
        bosEmbed = MLX.concatenated([bosEmbed, bosEmbed], axis: 0)
      }

      if condEmb.shape[0] != textEmbeddings.shape[0] {
        condEmb = MLX.broadcast(
          condEmb,
          to: [textEmbeddings.shape[0]] + Array(condEmb.shape.dropFirst()),
        )
      }

      let inputEmbeddings = MLX.concatenated([condEmb, textEmbeddings, bosEmbed], axis: 1)

      let cache = model.t3.tfmr.newCache(quantized: false)
      var hidden = model.t3.tfmr(inputEmbeddings, cache: cache)
      hidden.eval()
      stageTimes["initial_forward"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // Generation loop
      stageStart = CFAbsoluteTimeGetCurrent()
      var generatedIds: [Int32] = [Int32(model.config.t3Config.startSpeechToken)]
      let maxNewTokens = 200 // Limit for benchmark

      for step in 0 ..< maxNewTokens {
        var logits = model.t3.speechHead(hidden[0..., -1 ..< hidden.shape[1], 0...])
        logits = logits.squeezed(axis: 1)

        if cfgWeight > 0.0, logits.shape[0] > 1 {
          let condLogits = logits[0 ..< 1, 0...]
          let uncondLogits = logits[1 ..< 2, 0...]
          logits = condLogits + cfgWeight * (condLogits - uncondLogits)
        } else {
          logits = logits[0 ..< 1, 0...]
        }

        // Sample (simplified for benchmark)
        let probs = softmax(logits / 0.8, axis: -1)
        let nextToken = MLXRandom.categorical(probs)
        let nextTokenId = nextToken.item(Int32.self)

        if nextTokenId == Int32(model.config.t3Config.stopSpeechToken) {
          generatedIds.append(nextTokenId)
          break
        }

        generatedIds.append(nextTokenId)

        // Next step
        var nextEmbed = model.t3.speechEmb(nextToken.reshaped([1, 1]))
        nextEmbed = nextEmbed + model.t3.speechPosEmb.getFixedEmbedding(step + 1)
        if cfgWeight > 0.0 {
          nextEmbed = MLX.concatenated([nextEmbed, nextEmbed], axis: 0)
        }

        hidden = model.t3.tfmr(nextEmbed, cache: cache)
        hidden.eval()
      }

      let genTime = CFAbsoluteTimeGetCurrent() - stageStart
      stageTimes["generation_loop"]!.append(genTime)
      stageTimes["tokens_per_second"]!.append(Double(generatedIds.count) / genTime)

      stageTimes["total"]!.append(CFAbsoluteTimeGetCurrent() - totalStart)

      // Print per-run results
      print("  prepare_conditioning:   \(String(format: "%.4f", stageTimes["prepare_conditioning"]!.last!))s")
      print("  text_embedding:         \(String(format: "%.4f", stageTimes["text_embedding"]!.last!))s")
      print("  initial_forward:        \(String(format: "%.4f", stageTimes["initial_forward"]!.last!))s")
      print("  generation_loop:        \(String(format: "%.4f", stageTimes["generation_loop"]!.last!))s (\(generatedIds.count) tokens)")
      print("  tokens_per_second:      \(String(format: "%.1f", stageTimes["tokens_per_second"]!.last!)) tok/s")
      print("  total:                  \(String(format: "%.4f", stageTimes["total"]!.last!))s")
    }

    // Print summary
    print("\n" + "-" * 40)
    print("SUMMARY (averaged)")
    print("-" * 40)
    for (stage, times) in stageTimes.sorted(by: { $0.key < $1.key }) {
      let avg = times.reduce(0, +) / Double(times.count)
      let variance = times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(times.count)
      let std = sqrt(variance)
      print("\(stage.padding(toLength: 25, withPad: " ", startingAt: 0)): \(String(format: "%.4f", avg)) ± \(String(format: "%.4f", std))")
    }
  }

  // MARK: - S3Gen Detailed Benchmark

  @Test @MainActor func s3genDetailedBenchmark() async throws {
    print("\n" + "=" * 60)
    print("S3GEN WAVEFORM DETAILED BENCHMARK (Swift)")
    print("=" * 60)

    // Configure memory limits
    configureMemoryLimits()

    // Load model (global shared)
    let model = try await ChatterboxTestHelper.getOrLoadModel()

    // Load reference audio
    let (samples, sampleRate) = try await downloadAndLoadAudio(from: Self.defaultReferenceAudioURL)
    let refWav = MLXArray(samples)
    let refSr = sampleRate

    // Prepare conditionals once
    var conds = model.prepareConditionals(refWav: refWav, refSr: refSr, exaggeration: 0.1)
    conds.t3.speakerEmb.eval()
    conds.gen.embedding.eval()

    // Generate speech tokens once (use T3 for realistic tokens)
    MLXRandom.seed(Self.seed)
    let normalizedText = puncNorm(Self.testText)
    var textTokens = model.textTokenizer!.textToTokens(normalizedText)
    let cfgWeight: Float = 0.5
    if cfgWeight > 0.0 {
      textTokens = MLX.concatenated([textTokens, textTokens], axis: 0)
    }
    let sot = model.config.t3Config.startTextToken
    let eot = model.config.t3Config.stopTextToken
    let sotTokens = MLXArray.full([textTokens.shape[0], 1], values: MLXArray(Int32(sot)))
    let eotTokens = MLXArray.full([textTokens.shape[0], 1], values: MLXArray(Int32(eot)))
    textTokens = MLX.concatenated([sotTokens, textTokens, eotTokens], axis: 1)

    var speechTokens = model.t3.inference(
      t3Cond: &conds.t3,
      textTokens: textTokens,
      maxNewTokens: 200,
      temperature: 0.8,
      topP: 1.0,
      minP: 0.05,
      repetitionPenalty: 1.2,
      cfgWeight: cfgWeight,
    )
    speechTokens.eval()

    // Post-process tokens
    speechTokens = speechTokens[0 ..< 1]
    speechTokens = dropInvalidTokens(speechTokens)
    let mask = speechTokens .< ChatterboxSpeechVocabSize
    let maskValues = mask.asArray(Bool.self)
    let validIndices = maskValues.enumerated().compactMap { $0.element ? Int32($0.offset) : nil }
    if !validIndices.isEmpty {
      speechTokens = speechTokens[MLXArray(validIndices)]
    }
    speechTokens = speechTokens.expandedDimensions(axis: 0)

    let numTokens = speechTokens.shape[1]
    print("Speech tokens: \(numTokens)")

    var stageTimes: [String: [Double]] = [
      "flow_matching": [],
      "hifi_gan_vocoder": [],
      "total": [],
    ]

    for run in 0 ..< Self.numRuns {
      print("\n--- Run \(run + 1)/\(Self.numRuns) ---")

      // Clear cache between runs
      if run > 0 {
        clearMemoryBetweenRuns()
      }

      let totalStart = CFAbsoluteTimeGetCurrent()

      // Flow matching (token-to-mel)
      var stageStart = CFAbsoluteTimeGetCurrent()
      var outputMels = model.s3gen.flowInference(
        speechTokens: speechTokens,
        refDict: conds.gen,
        finalize: true,
      )
      outputMels.eval()
      stageTimes["flow_matching"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      // HiFi-GAN vocoder (mel-to-wav)
      stageStart = CFAbsoluteTimeGetCurrent()
      var (outputWav, _) = model.s3gen.hiftInference(speechFeat: outputMels)
      outputWav.eval()
      stageTimes["hifi_gan_vocoder"]!.append(CFAbsoluteTimeGetCurrent() - stageStart)

      stageTimes["total"]!.append(CFAbsoluteTimeGetCurrent() - totalStart)

      // Print per-run results
      let flowPct = stageTimes["flow_matching"]!.last! / stageTimes["total"]!.last! * 100
      let vocPct = stageTimes["hifi_gan_vocoder"]!.last! / stageTimes["total"]!.last! * 100
      print("  flow_matching:     \(String(format: "%.4f", stageTimes["flow_matching"]!.last!))s (\(String(format: "%.1f", flowPct))%)")
      print("  hifi_gan_vocoder:  \(String(format: "%.4f", stageTimes["hifi_gan_vocoder"]!.last!))s (\(String(format: "%.1f", vocPct))%)")
      print("  total:             \(String(format: "%.4f", stageTimes["total"]!.last!))s")

      // Audio duration for context
      if outputWav.ndim == 2 {
        outputWav = outputWav.squeezed(axis: 0)
      }
      let audioDuration = Double(outputWav.shape[0]) / Double(ChatterboxS3GenSr)
      print("  audio_duration:    \(String(format: "%.2f", audioDuration))s")
    }

    // Print summary
    print("\n" + "-" * 40)
    print("SUMMARY (averaged)")
    print("-" * 40)
    let totalAvg = stageTimes["total"]!.reduce(0, +) / Double(Self.numRuns)
    for (stage, times) in stageTimes.sorted(by: { $0.key < $1.key }) {
      let avg = times.reduce(0, +) / Double(times.count)
      let variance = times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(times.count)
      let std = sqrt(variance)
      let pct = stage == "total" ? 100.0 : (avg / totalAvg) * 100.0
      print("\(stage.padding(toLength: 20, withPad: " ", startingAt: 0)): \(String(format: "%.4f", avg))s ± \(String(format: "%.4f", std))s (\(String(format: "%5.1f", pct))%)")
    }
  }

  // MARK: - Helpers

  private func downloadAndLoadAudio(from url: URL) async throws -> (samples: [Float], sampleRate: Int) {
    let (data, response) = try await URLSession.shared.data(from: url)

    guard let httpResponse = response as? HTTPURLResponse,
          (200 ... 299).contains(httpResponse.statusCode)
    else {
      throw NSError(domain: "ChatterboxBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to download audio"])
    }

    let tempURL = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString)
      .appendingPathExtension("mp3")

    try data.write(to: tempURL)
    defer { try? FileManager.default.removeItem(at: tempURL) }

    let audioFile = try AVAudioFile(forReading: tempURL)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
      throw NSError(domain: "ChatterboxBenchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
    }

    try audioFile.read(into: buffer)

    guard let floatData = buffer.floatChannelData else {
      throw NSError(domain: "ChatterboxBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to read audio data"])
    }

    let channelCount = Int(format.channelCount)
    let frameLength = Int(buffer.frameLength)

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
