import AVFoundation
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - MarvisTTS Actor

/// Marvis TTS actor providing thread-safe text-to-speech generation.
///
/// This actor directly manages the Marvis model and audio codec, following the same
/// pattern as OrpheusTTS. The model weights are loaded directly into `MarvisModel`,
/// and Mimi codec weights are loaded separately.
///
/// Use the static `load()` factory method to create an initialized instance.
actor MarvisTTS {
  // MARK: - Types

  enum MarvisTTSError: Error, LocalizedError {
    case invalidArgument(String)
    case voiceNotFound
    case invalidRefAudio(String)

    var errorDescription: String? {
      switch self {
        case let .invalidArgument(msg):
          msg
        case .voiceNotFound:
          "Requested voice not found or missing reference assets."
        case let .invalidRefAudio(msg):
          msg
      }
    }
  }

  // MARK: - Properties

  private let model: MarvisModel
  private let promptURLs: [URL]
  private let textTokenizer: any Tokenizer
  private let audioTokenizer: MimiTokenizer
  private let streamingDecoder: MimiStreamingDecoder

  let sampleRate: Double

  // MARK: - Initialization

  private init(
    model: MarvisModel,
    promptURLs: [URL],
    textTokenizer: any Tokenizer,
    audioTokenizer: MimiTokenizer,
    streamingDecoder: MimiStreamingDecoder,
    sampleRate: Double,
  ) {
    self.model = model
    self.promptURLs = promptURLs
    self.textTokenizer = textTokenizer
    self.audioTokenizer = audioTokenizer
    self.streamingDecoder = streamingDecoder
    self.sampleRate = sampleRate
  }

  /// Load and initialize a MarvisTTS instance.
  ///
  /// - Parameters:
  ///   - repoId: Hugging Face repository ID for the model
  ///   - progressHandler: Callback for download progress
  /// - Returns: A fully initialized MarvisTTS actor
  static func load(
    repoId: String = MarvisEngine.ModelVariant.default.repoId,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in },
  ) async throws -> MarvisTTS {
    let (config, prompts, weightFileURL) = try await snapshotAndConfig(
      repoId: repoId,
      progressHandler: progressHandler,
    )

    // Initialize model
    let model = try MarvisModel(config: config)

    // Load tokenizers
    let textTokenizer = try await loadTokenizer(
      configuration: ModelConfiguration(id: repoId),
      hub: HubApi.shared,
    )
    let mimi = try await Mimi.fromPretrained(progressHandler: progressHandler)
    let audioTokenizer = MimiTokenizer(mimi)
    let streamingDecoder = MimiStreamingDecoder(mimi)

    // Install weights into the model
    try installWeights(
      into: model,
      config: config,
      weightFileURL: weightFileURL,
    )

    try model.resetCaches()

    return MarvisTTS(
      model: model,
      promptURLs: prompts,
      textTokenizer: textTokenizer,
      audioTokenizer: audioTokenizer,
      streamingDecoder: streamingDecoder,
      sampleRate: mimi.config.sampleRate,
    )
  }

  // MARK: - Public API

  func generate(
    text: String,
    voice: MarvisEngine.Voice,
    quality: MarvisEngine.QualityLevel,
  ) throws -> TTSGenerationResult {
    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    let results = try generateCore(
      text: sentences,
      voice: voice,
      refAudio: nil,
      refText: nil,
      qualityLevel: quality,
      stream: false,
      streamingInterval: 0.5,
      onStreamingResult: nil,
    )
    return Self.mergeResults(results)
  }

  func generateStreaming(
    text: String,
    voice: MarvisEngine.Voice,
    quality: MarvisEngine.QualityLevel,
    interval: Double,
  ) -> AsyncThrowingStream<TTSGenerationResult, Error> {
    let (stream, continuation) = AsyncThrowingStream<TTSGenerationResult, Error>.makeStream()

    let task = Task {
      do {
        let sentences = SentenceTokenizer.splitIntoSentences(text: text)
        _ = try self.generateCore(
          text: sentences,
          voice: voice,
          refAudio: nil,
          refText: nil,
          qualityLevel: quality,
          stream: true,
          streamingInterval: interval,
          onStreamingResult: { continuation.yield($0) },
        )
        continuation.finish()
      } catch {
        continuation.finish(throwing: error)
      }
    }

    continuation.onTermination = { _ in
      task.cancel()
    }

    return stream
  }

  // MARK: - Model Loading

  private static func snapshotAndConfig(
    repoId: String,
    progressHandler: @escaping (Progress) -> Void,
  ) async throws -> (config: MarvisConfig, promptURLs: [URL], weightFileURL: URL) {
    let modelDirectoryURL = try await Hub.snapshot(from: repoId, progressHandler: progressHandler)
    let weightFileURL = modelDirectoryURL.appending(path: "model.safetensors")
    let promptDir = modelDirectoryURL.appending(path: "prompts", directoryHint: .isDirectory)

    var audioPromptURLs: [URL] = []
    for url in try FileManager.default.contentsOfDirectory(at: promptDir, includingPropertiesForKeys: nil)
      where url.pathExtension == "wav"
    {
      audioPromptURLs.append(url)
    }

    let configFileURL = modelDirectoryURL.appending(path: "config.json")
    let config = try JSONDecoder().decode(MarvisConfig.self, from: Data(contentsOf: configFileURL))
    return (config, audioPromptURLs, weightFileURL)
  }

  private static func installWeights(
    into model: MarvisModel,
    config: MarvisConfig,
    weightFileURL: URL,
  ) throws {
    var weights = try loadArrays(url: weightFileURL)

    func extractInt(from value: StringOrNumber?) -> Int? {
      guard let value else { return nil }
      switch value {
        case let .int(i):
          return i
        case let .float(f):
          return Int(f)
        case let .string(s):
          return Int(s)
        default:
          return nil
      }
    }

    // Strip "model." prefix from all weights - MarvisModel expects direct children
    weights = stripModelPrefix(weights: weights)

    if let quantization = config.quantization,
       let groupSize = extractInt(from: quantization["group_size"]),
       let bits = extractInt(from: quantization["bits"])
    {
      quantize(model: model, groupSize: groupSize, bits: bits) { path, _ in
        weights["\(path).scales"] != nil
      }
    } else {
      weights = sanitize(weights: weights)
    }

    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: .noUnusedKeys)
    eval(model)
  }

  /// Strip "model." prefix from weight keys since MarvisModel expects direct children
  private static func stripModelPrefix(weights: [String: MLXArray]) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    out.reserveCapacity(weights.count)

    for (key, value) in weights {
      let newKey = key.hasPrefix("model.") ? String(key.dropFirst(6)) : key
      out[newKey] = value
    }

    return out
  }

  private static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    out.reserveCapacity(weights.count)

    for (rawKey, v) in weights {
      var k = rawKey

      if k.contains("attn") && !k.contains("self_attn") {
        k = k.replacingOccurrences(of: "attn", with: "self_attn")
        k = k.replacingOccurrences(of: "output_proj", with: "o_proj")
      }

      if k.contains("mlp") {
        k = k.replacingOccurrences(of: "w1", with: "gate_proj")
        k = k.replacingOccurrences(of: "w2", with: "down_proj")
        k = k.replacingOccurrences(of: "w3", with: "up_proj")
      }

      if k.contains("sa_norm") || k.contains("mlp_norm") {
        k = k.replacingOccurrences(of: "sa_norm", with: "input_layernorm")
        k = k.replacingOccurrences(of: "scale", with: "weight")
        k = k.replacingOccurrences(of: "mlp_norm", with: "post_attention_layernorm")
        k = k.replacingOccurrences(of: "scale", with: "weight")
      }

      if k.contains("decoder.norm") || k.contains("backbone.norm") {
        k = k.replacingOccurrences(of: "scale", with: "weight")
      }

      out[k] = v
    }

    return out
  }

  // MARK: - Tokenization

  private func tokenizeTextSegment(text: String, speaker: Int) -> (MLXArray, MLXArray) {
    let K = model.args.audioNumCodebooks
    let frameW = K + 1

    let prompt = "[\(speaker)]" + text
    let ids = MLXArray(textTokenizer.encode(text: prompt))

    let T = ids.shape[0]
    var frame = MLXArray.zeros([T, frameW], type: Int32.self)
    var mask = MLXArray.zeros([T, frameW], type: Bool.self)

    let lastCol = frameW - 1
    do {
      let left = split(frame, indices: [lastCol], axis: 1)[0]
      let right = split(frame, indices: [lastCol], axis: 1)[1]
      let tail = split(right, indices: [1], axis: 1)
      let after = (tail.count > 1) ? tail[1] : MLXArray.zeros([T, 0], type: Int32.self)
      frame = concatenated([left, ids.reshaped([T, 1]), after], axis: 1)
    }

    do {
      let ones = MLXArray.ones([T, 1], type: Bool.self)
      let left = split(mask, indices: [lastCol], axis: 1)[0]
      let right = split(mask, indices: [lastCol], axis: 1)[1]
      let tail = split(right, indices: [1], axis: 1)
      let after = (tail.count > 1) ? tail[1] : MLXArray.zeros([T, 0], type: Bool.self)
      mask = concatenated([left, ones, after], axis: 1)
    }

    return (frame, mask)
  }

  private func tokenizeAudio(_ audio: MLXArray, addEOS: Bool = true) -> (MLXArray, MLXArray) {
    let K = model.args.audioNumCodebooks
    let frameW = K + 1

    let x = audio.reshaped([1, 1, audio.shape[0]])
    var codes = audioTokenizer.codec.encode(x)
    codes = split(codes, indices: [1], axis: 0)[0].reshaped([K, codes.shape[2]])

    if addEOS {
      let eos = MLXArray.zeros([K, 1], type: Int32.self)
      codes = concatenated([codes, eos], axis: 1)
    }

    let T = codes.shape[1]
    var frame = MLXArray.zeros([T, frameW], type: Int32.self)
    var mask = MLXArray.zeros([T, frameW], type: Bool.self)

    let codesT = swappedAxes(codes, 0, 1)
    if K > 0 {
      let leftLen = K
      let right = split(frame, indices: [leftLen], axis: 1)[1]
      frame = concatenated([codesT, right], axis: 1)
    }
    if K > 0 {
      let ones = MLXArray.ones([T, K], type: Bool.self)
      let right = MLXArray.zeros([T, 1], type: Bool.self)
      mask = concatenated([ones, right], axis: 1)
    }

    return (frame, mask)
  }

  private func tokenizeSegment(_ segment: MarvisSegment, addEOS: Bool = true) -> (MLXArray, MLXArray) {
    let (txt, txtMask) = tokenizeTextSegment(text: segment.text, speaker: segment.speaker)
    let (aud, audMask) = tokenizeAudio(segment.audio, addEOS: addEOS)
    return (concatenated([txt, aud], axis: 0), concatenated([txtMask, audMask], axis: 0))
  }

  private func tokenizeStart(for segment: MarvisSegment) -> (tokens: MLXArray, mask: MLXArray, pos: MLXArray) {
    let (st, sm) = tokenizeSegment(segment, addEOS: false)
    let promptTokens = concatenated([st], axis: 0).asType(Int32.self)
    let promptMask = concatenated([sm], axis: 0).asType(Bool.self)
    let currTokens = expandedDimensions(promptTokens, axis: 0)
    let currMask = expandedDimensions(promptMask, axis: 0)
    // TODO: Use MLX.arange after next mlx-swift release (see MLXArray+Extensions.swift)
    let currPos = expandedDimensions(MLXArray.arange(promptTokens.shape[0]), axis: 0)
    return (currTokens, currMask, currPos)
  }

  // MARK: - Generation Context

  private func makeContext(
    voice: MarvisEngine.Voice?,
    refAudio: MLXArray?,
    refText: String?,
  ) throws -> MarvisSegment {
    if let refAudio, let refText {
      return MarvisSegment(speaker: 0, text: refText, audio: refAudio)
    } else if let voice {
      var refAudioURL: URL?
      for promptURL in promptURLs {
        if promptURL.lastPathComponent == "\(voice.rawValue).wav" {
          refAudioURL = promptURL
          break
        }
      }
      guard let refAudioURL else { throw MarvisTTSError.voiceNotFound }

      let (loadedSampleRate, loadedAudio) = try loadAudioArray(from: refAudioURL)
      let audio = if Int(loadedSampleRate) != 24000 {
        AudioResampler.resample(loadedAudio, from: Int(loadedSampleRate), to: 24000)
      } else {
        loadedAudio
      }
      let refTextURL = refAudioURL.deletingPathExtension().appendingPathExtension("txt")
      let text = try String(data: Data(contentsOf: refTextURL), encoding: .utf8)
      guard let text else { throw MarvisTTSError.voiceNotFound }
      return MarvisSegment(speaker: 0, text: text, audio: audio)
    }
    throw MarvisTTSError.voiceNotFound
  }

  // MARK: - Core Generation

  private func generateCore(
    text: [String],
    voice: MarvisEngine.Voice?,
    refAudio: MLXArray?,
    refText: String?,
    qualityLevel: MarvisEngine.QualityLevel,
    stream: Bool,
    streamingInterval: Double,
    onStreamingResult: (@Sendable (TTSGenerationResult) -> Void)?,
  ) throws -> [TTSGenerationResult] {
    guard voice != nil || refAudio != nil else {
      throw MarvisTTSError.invalidArgument("`voice` or `refAudio`/`refText` must be specified.")
    }

    let base = try makeContext(voice: voice, refAudio: refAudio, refText: refText)
    let sampleFn = TopPSampler(temperature: 0.9, topP: 0.8).sample
    let intervalTokens = Int(streamingInterval * 12.5)
    var results: [TTSGenerationResult] = []

    for prompt in text {
      // Check for cancellation between sentences
      if Task.isCancelled {
        throw CancellationError()
      }

      let generationText = (base.text + " " + prompt).trimmingCharacters(in: .whitespaces)
      let seg = MarvisSegment(speaker: 0, text: generationText, audio: base.audio)

      try model.resetCaches()
      if stream { streamingDecoder.reset() }

      let (tok, msk, pos) = tokenizeStart(for: seg)
      let r = try decodePrompt(
        currTokens: tok,
        currMask: msk,
        currPos: pos,
        qualityLevel: qualityLevel,
        stream: stream,
        streamingIntervalTokens: intervalTokens,
        sampler: sampleFn,
        onStreamingResult: onStreamingResult,
      )
      results.append(contentsOf: r)
    }

    try model.resetCaches()
    if stream { streamingDecoder.reset() }
    return results
  }

  private func decodePrompt(
    currTokens startTokens: MLXArray,
    currMask startMask: MLXArray,
    currPos startPos: MLXArray,
    qualityLevel: MarvisEngine.QualityLevel,
    stream: Bool,
    streamingIntervalTokens: Int,
    sampler sampleFn: (MLXArray) -> MLXArray,
    onStreamingResult: (@Sendable (TTSGenerationResult) -> Void)?,
  ) throws -> [TTSGenerationResult] {
    var results: [TTSGenerationResult] = []

    var samplesFrames: [MLXArray] = []
    var currTokens = startTokens
    var currMask = startMask
    var currPos = startPos

    var generatedCount = 0
    var yieldedCount = 0
    let maxAudioFrames = Int(60000 / 80.0)
    let maxSeqLen = 2048 - maxAudioFrames
    precondition(currTokens.shape[1] < maxSeqLen, "Inputs too long, must be below max_seq_len - max_audio_frames: \(maxSeqLen)")

    var startTime = CFAbsoluteTimeGetCurrent()

    for frameIndex in 0 ..< maxAudioFrames {
      // Check for cancellation periodically
      if frameIndex % 25 == 0, Task.isCancelled {
        throw CancellationError()
      }

      let frame = try model.generateFrame(
        maxCodebooks: qualityLevel.codebookCount,
        tokens: currTokens,
        tokensMask: currMask,
        sampler: sampleFn,
      )

      if frame.sum().item(Int32.self) == 0 { break }

      samplesFrames.append(frame)

      let zerosText = MLXArray.zeros([1, 1], type: Int32.self)
      let nextFrame = concatenated([frame, zerosText], axis: 1)
      currTokens = expandedDimensions(nextFrame, axis: 1)

      let onesK = ones([1, frame.shape[1]], type: Bool.self)
      let zero1 = zeros([1, 1], type: Bool.self)
      let nextMask = concatenated([onesK, zero1], axis: 1)
      currMask = expandedDimensions(nextMask, axis: 1)

      currPos = split(currPos, indices: [currPos.shape[1] - 1], axis: 1)[1] + MLXArray(1)

      generatedCount += 1

      if stream, (generatedCount - yieldedCount) >= streamingIntervalTokens {
        yieldedCount = generatedCount
        let gr = generateResultChunk(samplesFrames, start: startTime, streaming: true)
        results.append(gr)
        onStreamingResult?(gr)
        samplesFrames.removeAll(keepingCapacity: true)
        startTime = CFAbsoluteTimeGetCurrent()
      }
    }

    if !samplesFrames.isEmpty {
      let gr = generateResultChunk(samplesFrames, start: startTime, streaming: stream)
      if stream { onStreamingResult?(gr) } else { results.append(gr) }
    }

    return results
  }

  private func generateResultChunk(
    _ frames: [MLXArray],
    start: CFTimeInterval,
    streaming: Bool,
  ) -> TTSGenerationResult {
    var stacked = MLX.stacked(frames, axis: 0)
    stacked = swappedAxes(stacked, 0, 1)
    stacked = swappedAxes(stacked, 1, 2)

    let audio1x1x = streaming
      ? streamingDecoder.decodeFrames(stacked)
      : audioTokenizer.codec.decode(stacked)

    let sampleCount = audio1x1x.shape[2]
    let audio = audio1x1x.reshaped([sampleCount])
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    return TTSGenerationResult(
      audio: audio.asArray(Float32.self),
      sampleRate: Int(sampleRate),
      processingTime: elapsed,
    )
  }

  // MARK: - Result Merging

  private static func mergeResults(_ parts: [TTSGenerationResult]) -> TTSGenerationResult {
    guard let first = parts.first else {
      return TTSGenerationResult(audio: [], sampleRate: 24000, processingTime: 0)
    }
    if parts.count == 1 { return first }

    var samples: [Float] = []
    samples.reserveCapacity(parts.reduce(0) { $0 + $1.audio.count })
    var processingTime: Double = 0

    for r in parts {
      samples += r.audio
      processingTime += r.processingTime
    }

    return TTSGenerationResult(
      audio: samples,
      sampleRate: first.sampleRate,
      processingTime: processingTime,
    )
  }
}

// MARK: - Supporting Types

private struct MarvisSegment {
  let speaker: Int
  let text: String
  let audio: MLXArray
}
