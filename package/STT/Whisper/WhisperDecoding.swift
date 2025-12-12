import Foundation
import MLX
import MLXNN

// MARK: - Decoding Options

/// Options for Whisper decoding
struct DecodingOptions {
  /// Task: transcribe or translate to English
  let task: TranscriptionTask

  /// Language code (e.g., "en", "zh"), nil for auto-detect
  let language: String?

  /// Sampling temperature (0.0 for greedy decoding)
  let temperature: Float

  /// Maximum number of tokens to generate
  let maxTokens: Int

  /// Whether to include timestamps
  let timestamps: TimestampGranularity

  static let `default` = DecodingOptions(
    task: .transcribe,
    language: nil,
    temperature: 0.0,
    maxTokens: 448,
    timestamps: .segment
  )
}

// MARK: - Decoding Result

/// Result from decoding a single audio segment
struct DecodingResult {
  /// Generated token sequence
  let tokens: [Int]

  /// Decoded text
  let text: String

  /// Average log probability
  let avgLogProb: Float

  /// No-speech probability (0-1)
  let noSpeechProb: Float

  /// Temperature used
  let temperature: Float

  /// Compression ratio (text length / gzip(text) length)
  let compressionRatio: Float
}

// MARK: - Greedy Decoder

/// Greedy decoder for Whisper
///
/// Implements simple greedy decoding with KV caching
class GreedyDecoder {
  let model: WhisperModel
  let tokenizer: WhisperTokenizer
  let options: DecodingOptions

  init(model: WhisperModel, tokenizer: WhisperTokenizer, options: DecodingOptions) {
    self.model = model
    self.tokenizer = tokenizer
    self.options = options
  }

  /// Decode an audio segment
  ///
  /// - Parameter mel: Mel spectrogram (batch=1, n_mels, n_frames)
  /// - Returns: Decoding result
  func decode(_ mel: MLXArray) -> DecodingResult {
    // Encode audio to features
    let audioFeatures = model.encode(mel)
    eval(audioFeatures) // Ensure audio features are fully computed

    // Verify audio features are valid (not zero/constant)
    let afStd = audioFeatures.variance().sqrt().item(Float.self)
    if afStd < 0.01 {
      Log.model.error("⚠️  Audio features have very low variance!")
    }

    // Build initial token sequence
    var tokens = tokenizer.sotSequence(language: options.language, task: options.task)

    // Only add no-timestamps token if timestamps are disabled
    // When timestamps are enabled, the first timestamp is the first GENERATED token
    if options.timestamps == .none {
      tokens.append(tokenizer.noTimestamps)
    }

    // Calculate how many tokens we can generate
    // We need to account for the initial SOT sequence
    let initialTokenCount = tokens.count
    let maxGenerateTokens = options.maxTokens - initialTokenCount

    // Autoregressive decoding
    var kvCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)]? = nil
    var sumLogProb: Float = 0.0
    var tokenCount = 0
    var noSpeechProb: Float = 0.0

    for iteration in 0 ..< maxGenerateTokens {
      // Convert tokens to MLXArray
      // With KV caching: only pass new token(s), not all tokens
      let tokensToProcess: MLXArray
      if kvCache != nil {
        // Pass only the last token (the new one)
        let lastToken = tokens.last!
        tokensToProcess = MLXArray([Int32(lastToken)]).expandedDimensions(axis: 0)
      } else {
        // First iteration: pass all initial tokens (SOT sequence)
        tokensToProcess = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
      }

      // Forward pass through decoder
      let (logits, newCache, _) = model.decode(
        tokensToProcess,
        audioFeatures: audioFeatures,
        kvCache: kvCache
      )
      eval(logits) // Ensure logits are computed before use

      // Compute no-speech probability from first forward pass
      // Extract logits at the last position (after seeing full initial sequence: SOT, language, task)
      if iteration == 0 {
        // logits shape: [batch=1, seq_len, vocab_size]
        // Use -1 to get the last position's logits (after processing all initial tokens)
        let initialLogits = logits[0, -1] // [vocab_size]

        // Apply softmax to get probabilities
        let probs = MLX.softmax(initialLogits, axis: -1)

        // Extract probability for no-speech token
        noSpeechProb = probs[tokenizer.noSpeech].item(Float.self)
      }

      // Update KV cache
      kvCache = newCache

      // Get logits for last token
      var lastLogits = logits[0, -1]

      // Suppress special tokens (same as Python implementation)
      // These tokens should never be generated during decoding
      // Base suppression: special tokens and non-speech tokens
      // Exactly mirrors Python's _get_suppress_tokens() implementation
      var suppressedTokens: [Int] = []

      // Add non-speech tokens (matches suppress_tokens="-1" default)
      suppressedTokens.append(contentsOf: tokenizer.nonSpeechTokens())

      // Add special tokens (matches Python's explicit list)
      suppressedTokens.append(contentsOf: [
        tokenizer.transcribe,
        tokenizer.translate,
        tokenizer.sot,
        tokenizer.sotPrev,
        tokenizer.sotLm,
        tokenizer.noSpeech,
      ])

      // Get number of generated tokens (excluding initial SOT sequence)
      let numGenerated = tokens.count - initialTokenCount

      // Apply timestamp rules (matches Python's ApplyTimestampRules)
      if options.timestamps != .none {
        // Check if we just generated a timestamp
        let lastWasTimestamp = numGenerated >= 1 && tokens.last! >= tokenizer.timestampBegin

        // Check if second-to-last was also a timestamp
        // Python: penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= timestamp_begin
        let penultimateWasTimestamp = numGenerated < 2 || tokens[tokens.count - 2] >= tokenizer.timestampBegin

        if numGenerated == 0 {
          // First generated token MUST be a timestamp
          // Suppress all non-timestamp tokens (0 to timestampBegin-1)
          for tokenId in 0 ..< tokenizer.timestampBegin {
            suppressedTokens.append(tokenId)
          }

          // Apply max_initial_timestamp option if set (limits first timestamp range)
          // For 30-second segments, this prevents timestamps beyond the segment boundary
          // Python uses 1500 frames @ 20ms = 30 seconds, so max_initial_timestamp = 1500
          let maxInitialTimestampIndex = 1500 // 30 seconds at 0.02s per token
          let lastAllowed = tokenizer.timestampBegin + maxInitialTimestampIndex
          if lastAllowed < lastLogits.shape[0] {
            for tokenId in (lastAllowed + 1) ..< lastLogits.shape[0] {
              suppressedTokens.append(tokenId)
            }
          }
        } else if lastWasTimestamp {
          if penultimateWasTimestamp {
            // Two timestamps in a row (or only 1 token generated): next MUST be text
            // Python: logits[:, :timestamp_begin] = -np.inf
            // Suppress all timestamps (allow only text and EOT)
            for tokenId in tokenizer.timestampBegin ..< lastLogits.shape[0] {
              suppressedTokens.append(tokenId)
            }
          } else {
            // Text then timestamp: cannot be normal text tokens (only timestamps or EOT allowed)
            // Python: mask[k, : self.tokenizer.eot] = -np.inf
            // This suppresses 0 to eot-1 (all text tokens), but NOT eot itself!
            // EOT is allowed here so the model can stop after generating the final timestamp
            for tokenId in 0 ..< tokenizer.eot {
              suppressedTokens.append(tokenId)
            }
          }
        }

        // Enforce timestamp monotonicity: timestamps shouldn't decrease
        // Find all timestamps in generated sequence and prevent going backward in time
        let generatedTokens = tokens.suffix(numGenerated)
        let timestamps = generatedTokens.enumerated().compactMap { _, token -> Int? in
          token > tokenizer.timestampBegin ? token : nil
        }

        if !timestamps.isEmpty {
          var lastTimestamp = timestamps.last!

          // Force each segment to have a nonzero length to prevent infinite looping
          // Python: if not last_timestamp or penultimate_was_timestamp: last_timestamp += 1
          if lastTimestamp == tokenizer.timestampBegin || penultimateWasTimestamp {
            lastTimestamp += 1
          }

          // Suppress all timestamps before the last one (prevents going backward)
          for tokenId in tokenizer.timestampBegin ..< min(lastTimestamp, lastLogits.shape[0]) {
            suppressedTokens.append(tokenId)
          }
        }

        // Suppress no-timestamps token during generation
        suppressedTokens.append(tokenizer.noTimestamps)
      }

      // Suppress blank at the beginning of sampling (first generated token only)
      if iteration == 0 {
        // Suppress tokens that encode blank/space and EOT
        let blankTokens = tokenizer.encode(" ")
        suppressedTokens.append(contentsOf: blankTokens)
        suppressedTokens.append(tokenizer.eot)
      }

      // Create suppression mask (must build as Swift array first, then convert)
      var maskValues = [Float](repeating: 0.0, count: lastLogits.shape[0])
      for token in suppressedTokens {
        maskValues[token] = -Float.infinity
      }
      var suppressionMask = MLXArray(maskValues)

      // Apply timestamp probability heuristic (Python lines 381-390)
      // If sum of timestamp probabilities > max text probability, force timestamp
      // NOTE: This must be computed AFTER applying base suppression mask!
      if options.timestamps != .none, numGenerated > 0 {
        // Apply base suppression first to get correct probabilities
        let logitsWithSuppression = lastLogits + suppressionMask

        // Compute log probabilities
        let logSumExp = MLX.logSumExp(logitsWithSuppression, axes: [-1], keepDims: true)
        let logProbs = logitsWithSuppression - logSumExp

        // Sum of log probabilities over all timestamp tokens
        let timestampLogProbs = logProbs[tokenizer.timestampBegin...]
        let timestampLogProbSum = MLX.logSumExp(timestampLogProbs, axes: [-1], keepDims: true)

        // Max log probability among text tokens (0 to timestampBegin-1)
        let textLogProbs = logProbs[0 ..< tokenizer.timestampBegin]
        let maxTextLogProb = textLogProbs.max(axes: [-1], keepDims: true)

        // If timestamp probability sum > max text probability, suppress all text tokens
        // This forces the model to generate a timestamp when it's clearly time for one
        if timestampLogProbSum.item(Float.self) > maxTextLogProb.item(Float.self) {
          for tokenId in 0 ..< tokenizer.timestampBegin {
            maskValues[tokenId] = -Float.infinity
          }
          suppressionMask = MLXArray(maskValues)
        }
      }

      // Apply suppression mask to logits
      lastLogits = lastLogits + suppressionMask

      // Sample next token
      let nextToken: Int
      if options.temperature == 0.0 {
        // Greedy decoding
        nextToken = Int(MLX.argMax(lastLogits).item(Int32.self))
      } else {
        // Temperature sampling
        let probs = MLX.softmax(lastLogits / options.temperature, axis: -1)
        nextToken = sampleFromDistribution(probs)
      }

      // Track log probability
      let logProbs = MLX.log(MLX.softmax(lastLogits, axis: -1))
      let logProb = logProbs[nextToken].item(Float.self)
      sumLogProb += logProb
      tokenCount += 1

      // Add token to sequence
      tokens.append(nextToken)

      // Check for end-of-text token
      if nextToken == tokenizer.eot {
        break
      }
    }

    // Compute statistics
    let avgLogProb = tokenCount > 0 ? sumLogProb / Float(tokenCount) : 0.0

    // Decode text (filter out special tokens)
    let text = tokenizer.decode(tokens)

    // Compute compression ratio (simple approximation)
    let compressionRatio = text.isEmpty ? 1.0 : Float(text.count) / Float(tokens.count)

    return DecodingResult(
      tokens: tokens,
      text: text,
      avgLogProb: avgLogProb,
      noSpeechProb: noSpeechProb,
      temperature: options.temperature,
      compressionRatio: compressionRatio
    )
  }

  /// Sample from a probability distribution
  ///
  /// - Parameter probs: Probability distribution
  /// - Returns: Sampled index
  private func sampleFromDistribution(_ probs: MLXArray) -> Int {
    // Simple categorical sampling
    let probsArray = probs.asArray(Float.self)
    let random = Float.random(in: 0 ..< 1)

    var cumsum: Float = 0.0
    for (i, p) in probsArray.enumerated() {
      cumsum += p
      if cumsum >= random {
        return i
      }
    }

    // Fallback to last token (shouldn't happen)
    return probsArray.count - 1
  }
}
