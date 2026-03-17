// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - CosyVoice3LM (Main LLM)

/// Qwen2-based Language Model for CosyVoice3 speech token generation
///
/// Key differences from Qwen2LM (CosyVoice2):
/// - Uses unified speech_embedding for all tokens including special tokens
/// - Special token indices: sos = speech_token_size + 0, eos = speech_token_size + 1, etc.
/// - llm_decoder has no bias and extended vocabulary (+200)
class CosyVoice3LM: Module {
  let llmInputSize: Int
  let llmOutputSize: Int
  let speechTokenSize: Int
  let extendedVocabSize: Int

  // CosyVoice3 special token indices (unified in speech_embedding)
  var sosToken: Int { speechTokenSize + 0 } // 6561
  var eosToken: Int { speechTokenSize + 1 } // 6562
  var taskIdToken: Int { speechTokenSize + 2 } // 6563
  var fillToken: Int { speechTokenSize + 3 } // 6564

  // Mix ratio for bidirectional streaming [text_tokens, speech_tokens]
  let mixRatio: [Int]

  // LLM backbone
  @ModuleInfo var llm: CosyVoiceQwen2Encoder

  // Output projection: no bias, extended vocabulary
  @ModuleInfo(key: "llm_decoder") var llmDecoder: Linear

  // Unified speech token embedding (includes special tokens)
  @ModuleInfo(key: "speech_embedding") var speechEmbedding: Embedding

  // Sampling function
  var sampling: ((MLXArray, [Int], Int) -> Int)?

  // Stop token IDs for generation (all extended vocab tokens >= speechTokenSize)
  var stopTokenIds: [Int] {
    (0 ..< extendedVocabSize).map { speechTokenSize + $0 }
  }

  init(
    llmInputSize: Int = 896,
    llmOutputSize: Int = 896,
    speechTokenSize: Int = 6561,
    extendedVocabSize: Int = 200,
    qwen2Config: CosyVoiceQwen2Config = CosyVoiceQwen2Config(),
    mixRatio: [Int] = [5, 15]
  ) {
    self.llmInputSize = llmInputSize
    self.llmOutputSize = llmOutputSize
    self.speechTokenSize = speechTokenSize
    self.extendedVocabSize = extendedVocabSize
    self.mixRatio = mixRatio

    // Unified speech embedding (no separate llm_embedding)
    _speechEmbedding.wrappedValue = Embedding(
      embeddingCount: speechTokenSize + extendedVocabSize,
      dimensions: llmInputSize
    )

    _llm.wrappedValue = CosyVoiceQwen2Encoder(config: qwen2Config)

    // Output projection: no bias, extended vocabulary
    _llmDecoder.wrappedValue = Linear(llmOutputSize, speechTokenSize + extendedVocabSize, bias: false)
  }

  /// Sample token IDs with optional EOS rejection
  private func samplingIds(
    weightedScores: MLXArray,
    decodedTokens: [Int],
    sampling: Int,
    ignoreEos: Bool = true
  ) throws -> Int {
    guard let samplingFn = self.sampling else {
      fatalError("Sampling function not set")
    }

    var numTrials = 0
    let maxTrials = 100

    while true {
      let topIds = samplingFn(weightedScores, decodedTokens, sampling)

      // If not ignoring EOS, or sampled token is valid speech token, accept it
      if !ignoreEos || topIds < speechTokenSize {
        return topIds
      }

      numTrials += 1
      if numTrials > maxTrials {
        throw CosyVoice3Error.maxSamplingTrialsExceeded
      }
    }
  }

  /// Generate speech tokens autoregressively
  /// - Parameters:
  ///   - text: Text token IDs (1, T_text)
  ///   - textLen: Text length (1,)
  ///   - promptText: Prompt text token IDs (1, T_prompt_text)
  ///   - promptTextLen: Prompt text length (1,)
  ///   - promptSpeechToken: Prompt speech tokens (1, T_prompt_speech)
  ///   - promptSpeechTokenLen: Prompt speech token length (1,)
  ///   - sampling: Top-k sampling parameter
  ///   - maxTokenTextRatio: Maximum speech/text token ratio
  ///   - minTokenTextRatio: Minimum speech/text token ratio
  /// - Returns: Array of generated speech token IDs
  func inference(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    embedding _: MLXArray? = nil,
    sampling: Int = 25,
    maxTokenTextRatio: Float = 20,
    minTokenTextRatio: Float = 2
  ) throws -> [Int] {
    // Concatenate prompt and input text
    let fullText = MLX.concatenated([promptText, text], axis: 1)
    let fullTextLen = textLen + promptTextLen

    // Embed text tokens using Qwen2's embedding
    let textEmb = llm.embedTokens(fullText)

    // Get special token embeddings from unified speech_embedding
    let sosEmb = speechEmbedding.weight[sosToken].reshaped(1, 1, -1)
    let taskIdEmb = speechEmbedding.weight[taskIdToken].reshaped(1, 1, -1)

    // Embed prompt speech tokens if provided
    let promptSpeechTokenEmb: MLXArray = if promptSpeechTokenLen[0].item(Int.self) != 0 {
      speechEmbedding(promptSpeechToken)
    } else {
      MLXArray.zeros([1, 0, llmInputSize])
    }

    // Construct initial LM input: [sos, text, task_id, prompt_speech]
    let lmInput = MLX.concatenated([sosEmb, textEmb, taskIdEmb, promptSpeechTokenEmb], axis: 1)

    // Calculate min/max generation length
    let textLenInt = Int(fullTextLen[0].item(Int32.self))
    let promptTextLenInt = Int(promptTextLen[0].item(Int32.self))
    let minLen = Int(Float(textLenInt - promptTextLenInt) * minTokenTextRatio)
    let maxLen = Int(Float(textLenInt - promptTextLenInt) * maxTokenTextRatio)

    // Generate tokens
    return try inferenceLoop(lmInput: lmInput, sampling: sampling, minLen: minLen, maxLen: maxLen)
  }

  /// Core inference loop with KV caching
  private func inferenceLoop(
    lmInput: MLXArray,
    sampling: Int,
    minLen: Int,
    maxLen: Int
  ) throws -> [Int] {
    var outTokens: [Int] = []
    var cache: [KVCacheSimple]? = nil
    var currentInput = lmInput

    for i in 0 ..< maxLen {
      // Forward pass
      let (yPred, newCache) = llm.forwardOneStep(currentInput, cache: cache)
      cache = newCache

      // Pipeline: start async eval immediately after forward
      // This allows GPU to work while CPU does logits/sampling below
      if let c = cache { asyncEval(yPred, c) } else { asyncEval(yPred) }

      // Get logits for last position (forces eval of yPred)
      let logits = llmDecoder(yPred[0..., yPred.shape[1] - 1, 0...])
      let logp = MLX.log(MLX.softmax(logits, axis: -1))

      // Sample next token (.item() forces eval)
      let topIds = try samplingIds(
        weightedScores: logp.reshaped(-1),
        decodedTokens: outTokens,
        sampling: sampling,
        ignoreEos: i < minLen
      )

      // Check for any stop token (EOS or any extended vocab token)
      if topIds >= speechTokenSize {
        break
      }

      // Add the token
      outTokens.append(topIds)

      // Prepare input for next step
      currentInput = speechEmbedding.weight[topIds].reshaped(1, 1, -1)
    }

    return outTokens
  }

  /// Streaming inference - yields tokens one by one as an AsyncStream
  ///
  /// This is the true streaming interface that yields tokens as they're generated,
  /// enabling chunked audio generation with lower latency.
  ///
  /// Uses the pull-based (unfolding) pattern for proper async token generation.
  func inferenceStreamAsync(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    embedding _: MLXArray? = nil,
    sampling: Int = 25,
    maxTokenTextRatio: Float = 20,
    minTokenTextRatio: Float = 2
  ) -> AsyncThrowingStream<Int, Error> {
    // Concatenate prompt and input text
    let fullText = MLX.concatenated([promptText, text], axis: 1)
    let fullTextLen = textLen + promptTextLen

    // Embed text tokens using Qwen2's embedding
    let textEmb = llm.embedTokens(fullText)

    // Get special token embeddings from unified speech_embedding
    let sosEmb = speechEmbedding.weight[sosToken].reshaped(1, 1, -1)
    let taskIdEmb = speechEmbedding.weight[taskIdToken].reshaped(1, 1, -1)

    // Embed prompt speech tokens if provided
    let promptSpeechTokenEmb: MLXArray = if promptSpeechTokenLen[0].item(Int.self) != 0 {
      speechEmbedding(promptSpeechToken)
    } else {
      MLXArray.zeros([1, 0, llmInputSize])
    }

    // Construct initial LM input: [sos, text, task_id, prompt_speech]
    let lmInput = MLX.concatenated([sosEmb, textEmb, taskIdEmb, promptSpeechTokenEmb], axis: 1)

    // Calculate min/max generation length
    let textLenInt = Int(fullTextLen[0].item(Int32.self))
    let promptTextLenInt = Int(promptTextLen[0].item(Int32.self))
    let minLen = Int(Float(textLenInt - promptTextLenInt) * minTokenTextRatio)
    let maxLen = Int(Float(textLenInt - promptTextLenInt) * maxTokenTextRatio)

    // Create state for pull-based streaming
    let state = TokenGeneratorState(
      lmInput: lmInput,
      llm: llm,
      llmDecoder: llmDecoder,
      speechEmbedding: speechEmbedding,
      speechTokenSize: speechTokenSize,
      sampling: sampling,
      minLen: minLen,
      maxLen: maxLen
    )

    return AsyncThrowingStream { try await state.next() }
  }
}

// MARK: - Token Generator State

/// Encapsulates state for pull-based token streaming.
/// Each call to `next()` generates one token, enabling true async streaming.
private final class TokenGeneratorState: @unchecked Sendable {
  private var outTokens: [Int] = []
  private var cache: [KVCacheSimple]?
  private var currentInput: MLXArray
  private var iteration = 0
  private var finished = false

  private let llm: CosyVoiceQwen2Encoder
  private let llmDecoder: Linear
  private let speechEmbedding: Embedding
  private let speechTokenSize: Int
  private let sampling: Int
  private let minLen: Int
  private let maxLen: Int

  init(
    lmInput: MLXArray,
    llm: CosyVoiceQwen2Encoder,
    llmDecoder: Linear,
    speechEmbedding: Embedding,
    speechTokenSize: Int,
    sampling: Int,
    minLen: Int,
    maxLen: Int
  ) {
    currentInput = lmInput
    self.llm = llm
    self.llmDecoder = llmDecoder
    self.speechEmbedding = speechEmbedding
    self.speechTokenSize = speechTokenSize
    self.sampling = sampling
    self.minLen = minLen
    self.maxLen = maxLen
  }

  func next() async throws -> Int? {
    // Check termination conditions.
    // Note: Task.isCancelled is checked but ongoing MLX operations cannot be interrupted.
    // Cancellation takes effect between token generations.
    guard !finished, iteration < maxLen, !Task.isCancelled else {
      return nil
    }

    // Forward pass
    let (yPred, newCache) = llm.forwardOneStep(currentInput, cache: cache)
    cache = newCache

    // Pipeline: start async eval immediately after forward
    if let c = cache { asyncEval(yPred, c) } else { asyncEval(yPred) }

    // Get logits for last position (forces eval of yPred)
    let logits = llmDecoder(yPred[0..., yPred.shape[1] - 1, 0...])
    let logp = MLX.log(MLX.softmax(logits, axis: -1))

    // Sample next token with proper EOS rejection when below min length
    let topIds = try cosyVoice3SamplingWithEosRejection(
      logits: logp.reshaped(-1),
      decodedTokens: outTokens,
      sampling: sampling,
      speechTokenSize: speechTokenSize,
      ignoreEos: iteration < minLen,
      topP: 0.8,
      topK: sampling,
      winSize: 10,
      tauR: 0.1
    )

    iteration += 1

    // Check for any stop token (EOS or any extended vocab token)
    if topIds >= speechTokenSize {
      finished = true
      return nil
    }

    outTokens.append(topIds)

    // Prepare input for next step
    currentInput = speechEmbedding.weight[topIds].reshaped(1, 1, -1)

    return topIds
  }
}

// MARK: - Sampling Functions for CosyVoice3

/// Nucleus (top-p) sampling with top-k cutoff
func cosyVoice3NucleusSampling(logits: MLXArray, topP: Float = 0.8, topK: Int = 25) -> Int {
  // Convert logits to probabilities
  let probs = MLX.softmax(logits)

  // Sort by probability (descending)
  let sortedIndices = MLX.argSort(-probs)
  let sortedProbs = probs[sortedIndices]

  // Compute cumulative probabilities
  let cumsumProbs = MLX.cumsum(sortedProbs)

  // Find cutoff: where cumsum first exceeds top_p, limited by top_k
  let belowThreshold = cumsumProbs .< topP

  // Count how many tokens are below threshold, but cap at topK
  let nTokens = min(Int(MLX.sum(belowThreshold).item(Int32.self)) + 1, topK)

  // Get top-n token indices and their probabilities
  let topIndices = sortedIndices[0 ..< nTokens]
  var topProbs = sortedProbs[0 ..< nTokens]

  // Renormalize and sample
  topProbs = topProbs / MLX.sum(topProbs)
  let idx = MLXRandom.categorical(MLX.log(topProbs))

  return Int(topIndices[idx].item(Int32.self))
}

/// Repetition-Aware Sampling (RAS) for CosyVoice3
func cosyVoice3RasSampling(
  logits: MLXArray,
  decodedTokens: [Int],
  sampling _: Int,
  topP: Float = 0.8,
  topK: Int = 25,
  winSize: Int = 10,
  tauR: Float = 0.1
) -> Int {
  // First, try nucleus sampling
  var topIds = cosyVoice3NucleusSampling(logits: logits, topP: topP, topK: topK)

  // Check for repetition in recent window
  if !decodedTokens.isEmpty {
    let recentTokens = Array(decodedTokens.suffix(winSize))
    let repNum = recentTokens.filter { $0 == topIds }.count

    // If repetition exceeds threshold, fall back to random sampling
    if Float(repNum) >= Float(winSize) * tauR {
      let probs = MLX.softmax(logits)
      topIds = Int(MLXRandom.categorical(MLX.log(probs)).item(Int32.self))
    }
  }

  return topIds
}

/// Simple top-k sampling from logits for CosyVoice3
func cosyVoice3TopKSampling(logits: MLXArray, decodedTokens _: [Int], topK: Int = 25) -> Int {
  // Get top-k indices using argpartition
  let topKIndices = MLX.argPartition(-logits, kth: topK - 1)[0 ..< topK]

  // Get the values at those indices
  let topKValues = logits[topKIndices]

  // Sample from top-k using softmax probabilities
  let probs = MLX.softmax(topKValues)
  let idx = MLXRandom.categorical(MLX.log(probs))

  return Int(topKIndices[idx].item(Int32.self))
}

/// Sample token IDs with optional EOS rejection (standalone version for async contexts)
///
/// When `ignoreEos` is true and a stop token (>= speechTokenSize) is sampled,
/// the function retries sampling up to `maxTrials` times until a valid speech token is found.
func cosyVoice3SamplingWithEosRejection(
  logits: MLXArray,
  decodedTokens: [Int],
  sampling: Int,
  speechTokenSize: Int,
  ignoreEos: Bool,
  topP: Float = 0.8,
  topK: Int = 25,
  winSize: Int = 10,
  tauR: Float = 0.1,
  maxTrials: Int = 100
) throws -> Int {
  var numTrials = 0

  while true {
    let topIds = cosyVoice3RasSampling(
      logits: logits,
      decodedTokens: decodedTokens,
      sampling: sampling,
      topP: topP,
      topK: topK,
      winSize: winSize,
      tauR: tauR
    )

    // If not ignoring EOS, or sampled token is valid speech token, accept it
    if !ignoreEos || topIds < speechTokenSize {
      return topIds
    }

    numTrials += 1
    if numTrials > maxTrials {
      throw CosyVoice3Error.maxSamplingTrialsExceeded
    }
  }
}

// MARK: - Error Types

enum CosyVoice3Error: LocalizedError {
  case maxSamplingTrialsExceeded
  case modelNotLoaded
  case invalidInput(String)
  case tokenizerNotFound(String)
  case audioProcessingFailed(String)

  var errorDescription: String? {
    switch self {
      case .maxSamplingTrialsExceeded:
        "Maximum sampling trials exceeded during token generation"
      case .modelNotLoaded:
        "CosyVoice3 model is not loaded"
      case let .invalidInput(message):
        "Invalid input: \(message)"
      case let .tokenizerNotFound(path):
        "Tokenizer not found at path: \(path)"
      case let .audioProcessingFailed(message):
        "Audio processing failed: \(message)"
    }
  }
}
