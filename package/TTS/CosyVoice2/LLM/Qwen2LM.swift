// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Qwen2LM (Main CosyVoice2 LLM)

/// Qwen2-based Language Model for CosyVoice2 speech token generation
/// Generates speech tokens autoregressively from text
class Qwen2LM: Module {
  let llmInputSize: Int
  let llmOutputSize: Int
  let speechTokenSize: Int

  // Special token indices
  let sosEos: Int = 0 // Start/end of sequence
  let taskId: Int = 1 // Task identifier
  let fillToken: Int = 2 // Fill token for streaming

  // Mix ratio for bidirectional streaming [text_tokens, speech_tokens]
  let mixRatio: [Int]

  // Embedding for special tokens (sos_eos, task_id)
  @ModuleInfo(key: "llm_embedding") var llmEmbedding: Embedding

  // LLM backbone
  @ModuleInfo var llm: CosyVoiceQwen2Encoder

  // Output projection: LLM hidden -> speech token logits
  @ModuleInfo(key: "llm_decoder") var llmDecoder: Linear

  // Speech token embedding
  @ModuleInfo(key: "speech_embedding") var speechEmbedding: Embedding

  // Sampling function
  var sampling: ((MLXArray, [Int], Int) -> Int)?

  init(
    llmInputSize: Int = 896,
    llmOutputSize: Int = 896,
    speechTokenSize: Int = 6561,
    qwen2Config: CosyVoiceQwen2Config = CosyVoiceQwen2Config(),
    mixRatio: [Int] = [5, 15]
  ) {
    self.llmInputSize = llmInputSize
    self.llmOutputSize = llmOutputSize
    self.speechTokenSize = speechTokenSize
    self.mixRatio = mixRatio

    _llmEmbedding.wrappedValue = Embedding(embeddingCount: 2, dimensions: llmInputSize)
    _llm.wrappedValue = CosyVoiceQwen2Encoder(config: qwen2Config)
    _llmDecoder.wrappedValue = Linear(llmOutputSize, speechTokenSize + 3) // +3 for special tokens
    _speechEmbedding.wrappedValue = Embedding(embeddingCount: speechTokenSize + 3, dimensions: llmInputSize)
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

      // If not ignoring EOS, or sampled token is not EOS, accept it
      if !ignoreEos || topIds != speechTokenSize {
        return topIds
      }

      numTrials += 1
      if numTrials > maxTrials {
        throw CosyVoice2Error.maxSamplingTrialsExceeded
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

    // Get special token embeddings
    let sosEosEmb = llmEmbedding.weight[sosEos].reshaped(1, 1, -1)
    let taskIdEmb = llmEmbedding.weight[taskId].reshaped(1, 1, -1)

    // Embed prompt speech tokens if provided
    let promptSpeechTokenEmb: MLXArray = if promptSpeechTokenLen[0].item(Int.self) != 0 {
      speechEmbedding(promptSpeechToken)
    } else {
      MLXArray.zeros([1, 0, llmInputSize])
    }

    // Construct initial LM input: [sos, text, task_id, prompt_speech]
    let lmInput = MLX.concatenated([sosEosEmb, textEmb, taskIdEmb, promptSpeechTokenEmb], axis: 1)

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

      // Check for EOS
      if topIds == speechTokenSize {
        break
      }

      // Prepare input for next step
      currentInput = speechEmbedding.weight[topIds].reshaped(1, 1, -1)

      // Skip special tokens (fill_token, etc.) - don't yield them
      if topIds > speechTokenSize {
        continue
      }

      // Add the token
      outTokens.append(topIds)
    }

    return outTokens
  }
}

// MARK: - Sampling Functions

/// Nucleus (top-p) sampling with top-k cutoff
func nucleusSampling(logits: MLXArray, topP: Float = 0.8, topK: Int = 25) -> Int {
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

/// Repetition-Aware Sampling (RAS)
/// Uses nucleus sampling but falls back to random sampling if repetition is detected
func rasSampling(
  logits: MLXArray,
  decodedTokens: [Int],
  sampling _: Int,
  topP: Float = 0.8,
  topK: Int = 25,
  winSize: Int = 10,
  tauR: Float = 0.1
) -> Int {
  // First, try nucleus sampling
  var topIds = nucleusSampling(logits: logits, topP: topP, topK: topK)

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

/// Simple top-k sampling from logits
func topKSampling(logits: MLXArray, decodedTokens _: [Int], topK: Int = 25) -> Int {
  // Get top-k indices using argpartition
  let topKIndices = MLX.argPartition(-logits, kth: topK - 1)[0 ..< topK]

  // Get the values at those indices
  let topKValues = logits[topKIndices]

  // Sample from top-k using softmax probabilities
  let probs = MLX.softmax(topKValues)
  let idx = MLXRandom.categorical(MLX.log(probs))

  return Int(topKIndices[idx].item(Int32.self))
}

// MARK: - Error Types

enum CosyVoice2Error: LocalizedError {
  case maxSamplingTrialsExceeded
  case modelNotLoaded
  case invalidInput(String)
  case tokenizerNotFound(String)

  var errorDescription: String? {
    switch self {
      case .maxSamplingTrialsExceeded:
        "Maximum sampling trials exceeded during token generation"
      case .modelNotLoaded:
        "CosyVoice2 model is not loaded"
      case let .invalidInput(message):
        "Invalid input: \(message)"
      case let .tokenizerNotFound(path):
        "Tokenizer not found at path: \(path)"
    }
  }
}
