// Word-level timestamp support for Whisper using DTW alignment

import Accelerate
import Dispatch
import Foundation
import MLX

// MARK: - Word Timing Result

/// Represents timing information for a single word
public struct WordTiming: Sendable {
  /// The word text
  public var word: String

  /// Token IDs that make up this word
  public var tokens: [Int]

  /// Start time in seconds
  public var start: Float

  /// End time in seconds
  public var end: Float

  /// Average probability/confidence for this word
  public var probability: Float
}

// MARK: - DTW Algorithm

/// Dynamic Time Warping alignment
///
/// Finds the optimal alignment path between text tokens and audio frames
/// using dynamic programming. This is the core algorithm for word-level timestamps.
///
/// - Parameters:
///   - costMatrix: Flat cost matrix of shape (N, M) where lower values indicate better alignment
///   - rows: Number of rows (N) in the cost matrix
///   - cols: Number of columns (M) in the cost matrix
/// - Returns: Tuple of aligned (textIndices, timeIndices)
@inlinable
func dtw(
  _ costMatrix: UnsafeBufferPointer<Float>,
  rows N: Int,
  cols M: Int
) -> (textIndices: [Int], timeIndices: [Int]) {
  // Flat arrays for cache efficiency (matches Numba's optimization approach)
  let costRows = N + 1
  let costCols = M + 1
  var cost = [Float](repeating: .infinity, count: costRows * costCols)
  var trace = [Int8](repeating: -1, count: costRows * costCols)

  // Helper for 2D indexing into flat array
  @inline(__always)
  func idx(_ i: Int, _ j: Int) -> Int { i * costCols + j }

  cost[idx(0, 0)] = 0

  // Main DTW loop - matches Python implementation exactly
  for j in 1 ... M {
    for i in 1 ... N {
      let c0 = cost[idx(i - 1, j - 1)] // diagonal
      let c1 = cost[idx(i - 1, j)] // vertical
      let c2 = cost[idx(i, j - 1)] // horizontal

      let (c, t): (Float, Int8)
      if c0 < c1, c0 < c2 {
        (c, t) = (c0, 0)
      } else if c1 < c0, c1 < c2 {
        (c, t) = (c1, 1)
      } else {
        (c, t) = (c2, 2)
      }

      cost[idx(i, j)] = costMatrix[(i - 1) * M + (j - 1)] + c
      trace[idx(i, j)] = t
    }
  }

  // Set boundary conditions before backtracing (matches Python exactly)
  // First row: horizontal movement (2) - when at top, can only go left
  for j in 0 ..< costCols {
    trace[idx(0, j)] = 2
  }
  // First column: vertical movement (1) - when at left edge, can only go up
  for i in 0 ..< costRows {
    trace[idx(i, 0)] = 1
  }

  return backtrace(trace, rows: costRows, cols: costCols)
}

/// Backtrace through the DTW trace matrix to recover the alignment path
///
/// - Parameters:
///   - trace: Trace matrix from DTW (flat array)
///   - rows: Number of rows in trace matrix
///   - cols: Number of columns in trace matrix
/// - Returns: Tuple of aligned (textIndices, timeIndices)
@inlinable
func backtrace(_ trace: [Int8], rows: Int, cols: Int) -> (textIndices: [Int], timeIndices: [Int]) {
  @inline(__always)
  func idx(_ i: Int, _ j: Int) -> Int { i * cols + j }

  var i = rows - 1
  var j = cols - 1
  var result: [(Int, Int)] = []
  result.reserveCapacity(i + j) // Pre-allocate for performance

  while i > 0 || j > 0 {
    result.append((i - 1, j - 1))

    switch trace[idx(i, j)] {
      case 0: i -= 1; j -= 1 // diagonal
      case 1: i -= 1 // vertical
      case 2: j -= 1 // horizontal
      default:
        // Should not happen with valid input, but handle gracefully
        if i > 0 { i -= 1 }
        if j > 0 { j -= 1 }
    }
  }

  result.reverse()
  return (result.map { $0.0 }, result.map { $0.1 })
}

// MARK: - Median Filter

/// Fast median of 7 elements using optimal comparison network
///
/// Uses 10 comparisons to find the median (vs 16 for full sort).
/// Based on the optimal comparison network for median selection.
@inline(__always)
@usableFromInline
func median7(_ a: Float, _ b: Float, _ c: Float, _ d: Float, _ e: Float, _ f: Float, _ g: Float) -> Float {
  // Sorting network approach - swap pairs to bubble median to center
  var v0 = a, v1 = b, v2 = c, v3 = d, v4 = e, v5 = f, v6 = g

  // Comparison network optimized for finding median of 7
  @inline(__always)
  func sortPair(_ x: inout Float, _ y: inout Float) {
    if x > y { swap(&x, &y) }
  }

  // Stage 1: Sort pairs
  sortPair(&v0, &v1)
  sortPair(&v2, &v3)
  sortPair(&v4, &v5)

  // Stage 2: Find smaller of the larger elements
  sortPair(&v1, &v3)
  sortPair(&v3, &v5)
  sortPair(&v1, &v3)

  // Stage 3: Ensure v6 is in right position
  sortPair(&v3, &v6)

  // Stage 4: Sort remaining to find median
  sortPair(&v0, &v2)
  sortPair(&v2, &v4)
  sortPair(&v0, &v2)

  // Stage 5: Final comparisons to isolate median
  sortPair(&v2, &v3)
  sortPair(&v3, &v4)
  sortPair(&v3, &v2)

  // v3 is now the median
  return max(v2, min(v3, v4))
}

/// Median filter for attention weight smoothing
///
/// Applies a 1D median filter along the frames dimension for each head and token.
/// Uses reflect padding to handle boundaries, matching scipy.signal.medfilt behavior.
/// Optimized with parallel processing and fast median-of-7.
///
/// - Parameters:
///   - weights: Flat attention weights array of shape (heads, tokens, frames)
///   - heads: Number of attention heads
///   - tokens: Number of tokens
///   - frames: Number of frames
///   - filterWidth: Filter width (must be odd, default 7)
/// - Returns: Filtered weights with same shape
@inlinable
func medianFilterAttention(
  _ weights: [Float],
  heads: Int,
  tokens: Int,
  frames: Int,
  filterWidth: Int = 7
) -> [Float] {
  precondition(filterWidth == 7, "Only filterWidth=7 is optimized")

  let totalRows = heads * tokens
  var result = [Float](repeating: 0, count: weights.count)

  // Process rows in parallel (each row is independent)
  weights.withUnsafeBufferPointer { weightsPtr in
    result.withUnsafeMutableBufferPointer { resultPtr in
      // Using nonisolated(unsafe) is appropriate here because:
      // 1. Performance-critical: tight loop processing audio data benefits from pointer arithmetic
      // 2. Provably safe: each iteration accesses non-overlapping memory (different rows)
      // 3. Synchronous: pointers don't escape; operation completes before buffer scope ends
      // 4. DispatchQueue.concurrentPerform is more efficient than withTaskGroup for CPU-bound work
      nonisolated(unsafe) let weightsBase = weightsPtr.baseAddress!
      nonisolated(unsafe) let resultBase = resultPtr.baseAddress!
      DispatchQueue.concurrentPerform(iterations: totalRows) { rowIdx in
        let rowStart = rowIdx * frames
        let rowPtr = weightsBase + rowStart
        let outPtr = resultBase + rowStart

        // Helper to get value with reflect padding
        @inline(__always)
        func getValue(_ idx: Int) -> Float {
          var i = idx
          if i < 0 { i = -i }
          if i >= frames { i = 2 * frames - i - 2 }
          i = max(0, min(frames - 1, i))
          return rowPtr[i]
        }

        // Apply median filter using optimized median7
        for f in 0 ..< frames {
          outPtr[f] = median7(
            getValue(f - 3),
            getValue(f - 2),
            getValue(f - 1),
            getValue(f),
            getValue(f + 1),
            getValue(f + 2),
            getValue(f + 3)
          )
        }
      }
    }
  }

  return result
}

// MARK: - Softmax Utilities

/// Apply softmax to a row of values in-place
///
/// - Parameters:
///   - values: Array to apply softmax to
///   - scale: Optional scaling factor (for QK scaling)
@inline(__always)
func softmaxInPlace(_ values: inout [Float], scale: Float = 1.0) {
  // Apply scaling
  if scale != 1.0 {
    for i in 0 ..< values.count {
      values[i] *= scale
    }
  }

  // Find max for numerical stability
  var maxVal: Float = -.infinity
  vDSP_maxv(values, 1, &maxVal, vDSP_Length(values.count))

  // Subtract max and exponentiate
  var negMax = -maxVal
  vDSP_vsadd(values, 1, &negMax, &values, 1, vDSP_Length(values.count))

  // Exponentiate
  var count = Int32(values.count)
  vvexpf(&values, values, &count)

  // Normalize
  var sum: Float = 0
  vDSP_sve(values, 1, &sum, vDSP_Length(values.count))
  if sum > 0 {
    var invSum = 1.0 / sum
    vDSP_vsmul(values, 1, &invSum, &values, 1, vDSP_Length(values.count))
  }
}

/// Standardize values: (x - mean) / std
///
/// - Parameters:
///   - values: Array to standardize in-place
///   - eps: Small value to prevent division by zero
@inline(__always)
func standardizeInPlace(_ values: inout [Float], eps: Float = 1e-8) {
  let count = vDSP_Length(values.count)

  // Calculate mean
  var mean: Float = 0
  vDSP_meanv(values, 1, &mean, count)

  // Subtract mean
  var negMean = -mean
  vDSP_vsadd(values, 1, &negMean, &values, 1, count)

  // Calculate std (sqrt of mean of squares after mean subtraction)
  var sumSq: Float = 0
  vDSP_svesq(values, 1, &sumSq, count)
  let std = sqrt(sumSq / Float(values.count)) + eps

  // Divide by std
  var invStd = 1.0 / std
  vDSP_vsmul(values, 1, &invStd, &values, 1, count)
}

// MARK: - Punctuation Merging

/// Default punctuation characters to prepend to following word
let defaultPrependPunctuations = #"\"'"¿([{-"#

/// Default punctuation characters to append to previous word
let defaultAppendPunctuations = #"\"'.。,，!！?？:：")]}、"#

/// Merge punctuation marks with adjacent words
///
/// This improves word timestamp accuracy by combining punctuation with
/// the words they belong to rather than treating them as separate words.
///
/// - Parameters:
///   - alignment: Array of word timings to modify in-place
///   - prepended: Characters that should be merged with the following word
///   - appended: Characters that should be merged with the previous word
func mergePunctuations(
  _ alignment: inout [WordTiming],
  prepended: String = defaultPrependPunctuations,
  appended: String = defaultAppendPunctuations
) {
  guard alignment.count > 1 else { return }

  // Merge prepended punctuations (iterate backwards)
  var i = alignment.count - 2
  var j = alignment.count - 1
  while i >= 0 {
    let previousWord = alignment[i].word
    if previousWord.hasPrefix(" "), prepended.contains(previousWord.trimmingCharacters(in: .whitespaces)) {
      // Prepend to the following word
      alignment[j].word = previousWord + alignment[j].word
      alignment[j].tokens = alignment[i].tokens + alignment[j].tokens
      alignment[j].start = alignment[i].start
      alignment[i].word = ""
      alignment[i].tokens = []
    } else {
      j = i
    }
    i -= 1
  }

  // Merge appended punctuations (iterate forwards)
  i = 0
  j = 1
  while j < alignment.count {
    let previousWord = alignment[i].word
    let followingWord = alignment[j].word
    if !previousWord.hasSuffix(" "), appended.contains(followingWord) {
      // Append to the previous word
      alignment[i].word = previousWord + followingWord
      alignment[i].tokens = alignment[i].tokens + alignment[j].tokens
      alignment[i].end = alignment[j].end
      alignment[j].word = ""
      alignment[j].tokens = []
    } else {
      i = j
    }
    j += 1
  }

  // Remove empty entries
  alignment.removeAll { $0.word.isEmpty && $0.tokens.isEmpty }
}

// MARK: - Constants

/// Tokens per second in Whisper (50 tokens = 1 second, i.e., 20ms per token)
let tokensPerSecond: Float = 50.0

/// Sentence-ending punctuation marks for duration clipping
private let sentenceEndMarks: Set<Character> = [".", "。", "!", "！", "?", "？"]

// MARK: - Duration Clipping

/// Calculate median and max duration for word timing clipping
///
/// - Parameter alignment: Array of word timings
/// - Returns: Tuple of (medianDuration, maxDuration) where maxDuration = min(0.7, median) * 2
func calculateDurationThresholds(_ alignment: [WordTiming]) -> (median: Float, max: Float) {
  // Get non-zero word durations
  let durations = alignment.compactMap { timing -> Float? in
    let duration = timing.end - timing.start
    return duration > 0 ? duration : nil
  }

  guard !durations.isEmpty else {
    return (median: 0.0, max: 0.0)
  }

  // Calculate median
  let sorted = durations.sorted()
  let median: Float = if sorted.count % 2 == 0 {
    (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
  } else {
    sorted[sorted.count / 2]
  }

  // Cap median at 0.7s and calculate max as 2x median
  let cappedMedian = min(0.7, median)
  let maxDuration = cappedMedian * 2

  return (median: cappedMedian, max: maxDuration)
}

/// Clip long word durations at sentence boundaries
///
/// Words at sentence boundaries (., 。, !, ！, ?, ？) are clipped to maxDuration
/// if they exceed it. This helps correct alignment errors at natural pause points.
///
/// - Parameters:
///   - alignment: Array of word timings to modify in-place
///   - maxDuration: Maximum allowed duration for words at sentence boundaries
func clipAtSentenceBoundaries(_ alignment: inout [WordTiming], maxDuration: Float) {
  guard alignment.count > 1, maxDuration > 0 else { return }

  // Sentence end marks as strings for exact word matching (matches Python behavior)
  let sentenceEndMarkStrings = Set(sentenceEndMarks.map { String($0) })

  for i in 1 ..< alignment.count {
    let duration = alignment[i].end - alignment[i].start
    guard duration > maxDuration else { continue }

    let currentWord = alignment[i].word
    let previousWord = alignment[i - 1].word

    // Check if current word IS a sentence-ending mark (exact match, like Python)
    if sentenceEndMarkStrings.contains(currentWord) {
      alignment[i].end = alignment[i].start + maxDuration
    }
    // Check if previous word IS a sentence-ending mark
    else if sentenceEndMarkStrings.contains(previousWord) {
      alignment[i].start = alignment[i].end - maxDuration
    }
  }
}

/// Clip long word durations at segment boundaries (after pauses)
///
/// Handles the first and second word after a pause to prevent unreasonably long durations.
/// This is a heuristic to correct alignment errors when speech resumes after silence.
///
/// - Parameters:
///   - words: Array of word dictionaries with start/end times (modified in-place)
///   - lastSpeechTimestamp: End time of the last speech segment
///   - medianDuration: Median word duration for the segment
///   - maxDuration: Maximum allowed duration (typically 2x median)
func clipAtSegmentBoundaries(
  _ words: inout [WordTiming],
  lastSpeechTimestamp: Float,
  medianDuration: Float,
  maxDuration: Float
) {
  guard !words.isEmpty, maxDuration > 0 else { return }

  let firstWord = words[0]

  // Check if there's a significant pause before the first word
  let pauseBeforeFirst = firstWord.end - lastSpeechTimestamp
  let firstDuration = firstWord.end - firstWord.start

  if pauseBeforeFirst > medianDuration * 4 {
    // Long pause detected - check if first/second words need clipping
    let needsClipping = firstDuration > maxDuration ||
      (words.count > 1 && words[1].end - firstWord.start > maxDuration * 2)

    if needsClipping {
      // Clip second word if it's also too long
      if words.count > 1 {
        let secondDuration = words[1].end - words[1].start
        if secondDuration > maxDuration {
          let boundary = max(words[1].end / 2, words[1].end - maxDuration)
          words[0].end = boundary
          words[1].start = boundary
        }
      }
      // Clip first word start
      words[0].start = max(0, words[0].end - maxDuration)
    }
  }
}

/// Adjust segment boundaries based on word timestamps
///
/// Ensures segment start/end times are consistent with the first/last word timestamps,
/// preferring segment-level timestamps when words appear misaligned.
///
/// - Parameters:
///   - words: Array of word timings for this segment
///   - segmentStart: Segment start time (may be adjusted)
///   - segmentEnd: Segment end time (may be adjusted)
///   - medianDuration: Median word duration
/// - Returns: Tuple of adjusted (segmentStart, segmentEnd, lastSpeechTimestamp)
func adjustSegmentBoundaries(
  _ words: [WordTiming],
  segmentStart: Float,
  segmentEnd: Float,
  medianDuration: Float
) -> (start: Float, end: Float, lastSpeech: Float) {
  guard !words.isEmpty else {
    return (segmentStart, segmentEnd, segmentEnd)
  }

  var adjustedStart = segmentStart
  var adjustedEnd = segmentEnd

  let firstWord = words[0]
  let lastWord = words[words.count - 1]

  // Prefer segment-level start if first word appears too early
  if segmentStart < firstWord.end, segmentStart - 0.5 > firstWord.start {
    adjustedStart = max(0, min(firstWord.end - medianDuration, segmentStart))
  } else {
    adjustedStart = firstWord.start
  }

  // Prefer segment-level end if last word appears too late
  if segmentEnd > lastWord.start, segmentEnd + 0.5 < lastWord.end {
    adjustedEnd = max(lastWord.start + medianDuration, segmentEnd)
  } else {
    adjustedEnd = lastWord.end
  }

  return (adjustedStart, adjustedEnd, adjustedEnd)
}

// MARK: - Find Alignment

/// Find word-level alignment using cross-attention weights and DTW
///
/// This is the core function for extracting word-level timestamps. It:
/// 1. Runs a forward pass with the text tokens to get cross-attention weights
/// 2. Extracts and normalizes attention from alignment heads
/// 3. Uses DTW to align text tokens to audio frames
/// 4. Maps token boundaries to word boundaries
///
/// - Parameters:
///   - model: Whisper model with alignment heads configured
///   - tokenizer: Whisper tokenizer
///   - textTokens: Text tokens to align (without special tokens)
///   - mel: Mel spectrogram (n_frames, n_mels) or (batch, n_frames, n_mels)
///   - numFrames: Number of audio frames
///   - language: Optional language code for SOT sequence
///   - task: Transcription task
///   - medfiltWidth: Median filter width (default 7)
///   - qkScale: QK scaling factor (default 1.0)
/// - Returns: Array of word timings
func findAlignment(
  model: WhisperModel,
  tokenizer: WhisperTokenizer,
  textTokens: [Int],
  mel: MLXArray,
  numFrames: Int,
  language: String?,
  task: TranscriptionTask,
  medfiltWidth: Int = 7,
  qkScale: Float = 1.0
) -> [WordTiming] {
  guard !textTokens.isEmpty else { return [] }

  // Check if alignment heads are available
  guard model.alignmentHeads.size > 0 else {
    Log.model.warning("No alignment heads configured - word timestamps unavailable")
    return []
  }

  // Build token sequence: [sot_sequence, no_timestamps, text_tokens, eot]
  var tokens = tokenizer.sotSequence(language: language, task: task)
  let noTimestampsIndex = tokens.count // Index of no_timestamps token
  tokens.append(tokenizer.noTimestamps)
  let textStartIndex = tokens.count // Where text tokens begin (after no_timestamps)
  tokens.append(contentsOf: textTokens)
  tokens.append(tokenizer.eot)

  let tokenArray = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)

  // Ensure mel has batch dimension
  var melBatched = mel
  if mel.ndim == 2 {
    melBatched = mel.expandedDimensions(axis: 0)
  }

  // Forward pass with cross-attention extraction
  let (logits, crossQK) = model.forwardWithCrossQK(melBatched, tokens: tokenArray)
  eval(logits) // Ensure computation is complete

  // Get text token probabilities for confidence scores
  // logits[i] predicts tokens[i+1], so to predict text_tokens we need logits starting
  // at position (textStartIndex - 1), which is the no_timestamps token position
  let logitsStartIndex = textStartIndex - 1
  let sampledLogits = logits[0, logitsStartIndex ..< (tokens.count - 2), 0 ..< tokenizer.eot]
  let tokenProbs = MLX.softmax(sampledLogits, axis: -1, precise: true)
  eval(tokenProbs)

  // Extract probabilities for actual text tokens
  var textTokenProbs = [Float](repeating: 0, count: textTokens.count)
  for (i, token) in textTokens.enumerated() {
    textTokenProbs[i] = tokenProbs[i, token].item(Float.self)
  }

  // Get alignment head indices
  let alignmentHeadsArray = model.alignmentHeads.asArray(Int32.self)
  let numAlignmentHeads = alignmentHeadsArray.count / 2

  guard numAlignmentHeads > 0 else {
    Log.model.warning("Alignment heads array is empty")
    return []
  }

  // Determine frame count (Whisper uses stride-2 convolution)
  let framesLen = numFrames / 2

  // Collect attention weights from alignment heads using MLX (GPU)
  // Python: weights = mx.stack([cross_qk[_l][0, _h] for _l, _h in model.alignment_heads.tolist()])
  var headWeightsArrays: [MLXArray] = []

  for h in 0 ..< numAlignmentHeads {
    let layerIdx = Int(alignmentHeadsArray[h * 2])
    let headIdx = Int(alignmentHeadsArray[h * 2 + 1])

    guard layerIdx < crossQK.count, let layerQK = crossQK[layerIdx] else {
      continue
    }

    // Extract weights for this head: shape (batch, heads, tokens, frames)
    let headWeights = layerQK[0, headIdx]
    headWeightsArrays.append(headWeights)
  }

  guard !headWeightsArrays.isEmpty else {
    Log.model.warning("No valid attention weights extracted from alignment heads")
    return []
  }

  // Stack heads and slice to frame count: (heads, tokens, frames)
  var weights = MLX.stacked(headWeightsArrays, axis: 0)
  weights = weights[0..., 0..., 0 ..< framesLen]

  // GPU preprocessing (matching Python exactly):
  // weights = mx.softmax(weights * qk_scale, axis=-1, precise=True)
  weights = MLX.softmax(weights * qkScale, axis: -1, precise: true)

  // mean = mx.mean(weights, axis=-2, keepdims=True)
  // std = mx.var(weights, axis=-2, keepdims=True, ddof=0).sqrt()
  // weights = (weights - mean) / std
  let mean = MLX.mean(weights, axis: -2, keepDims: true)
  let variance = MLX.variance(weights, axis: -2, keepDims: true)
  let std = MLX.sqrt(variance + 1e-8)
  weights = (weights - mean) / std
  weights = weights.asType(.float32)
  eval(weights)

  // Get actual dimensions
  let actualTokensLen = weights.shape[1]
  let actualFramesLen = weights.shape[2]
  let numHeads = weights.shape[0]

  // Convert to Swift array for median filter (CPU operation)
  // Shape: (heads, tokens, frames) flattened
  let weightsArray = weights.asArray(Float.self)

  // Apply median filter (still CPU - scipy.signal.medfilt equivalent)
  let filteredWeights = medianFilterAttention(
    weightsArray,
    heads: numHeads,
    tokens: actualTokensLen,
    frames: actualFramesLen,
    filterWidth: medfiltWidth
  )

  // Average across heads using vDSP (CPU)
  let matrixSize = actualTokensLen * actualFramesLen
  var avgMatrix = [Float](repeating: 0, count: matrixSize)
  let vLen = vDSP_Length(matrixSize)

  // Sum all heads using vDSP_vadd
  for h in 0 ..< numHeads {
    filteredWeights.withUnsafeBufferPointer { ptr in
      let headPtr = ptr.baseAddress! + h * matrixSize
      vDSP_vadd(avgMatrix, 1, headPtr, 1, &avgMatrix, 1, vLen)
    }
  }

  // Divide by number of heads
  var invNumHeads = 1.0 / Float(numHeads)
  vDSP_vsmul(avgMatrix, 1, &invNumHeads, &avgMatrix, 1, vLen)

  // Extract the text portion for DTW: [no_timestamps, text_tokens] (excludes sot_sequence and eot)
  // Python: matrix = matrix[len(tokenizer.sot_sequence) : -1]
  // This includes no_timestamps at the start, matching Python's behavior exactly
  let textMatrixStart = noTimestampsIndex * actualFramesLen
  let textMatrixEnd = min((textStartIndex + textTokens.count) * actualFramesLen, avgMatrix.count)
  guard textMatrixStart < textMatrixEnd else {
    Log.model.warning("Invalid text matrix range")
    return []
  }

  var textMatrix = Array(avgMatrix[textMatrixStart ..< textMatrixEnd])
  let textMatrixRows = 1 + textTokens.count // no_timestamps + text_tokens (matches Python)
  let textMatrixCols = actualFramesLen

  // Run DTW on negative cost matrix (we want to maximize attention, so negate for min-cost DTW)
  for i in 0 ..< textMatrix.count {
    textMatrix[i] = -textMatrix[i]
  }

  let (textIndices, timeIndices) = textMatrix.withUnsafeBufferPointer { ptr in
    dtw(ptr, rows: textMatrixRows, cols: textMatrixCols)
  }

  guard !textIndices.isEmpty else {
    Log.model.warning("DTW returned empty alignment")
    return []
  }

  // Split into words
  let (words, wordTokenGroups) = tokenizer.splitToWordTokens(textTokens + [tokenizer.eot])

  guard wordTokenGroups.count > 1 else {
    // Only EOT marker or empty
    return []
  }

  // Calculate word boundaries (cumulative token counts)
  var wordBoundaries = [0]
  var cumLen = 0
  for group in wordTokenGroups.dropLast() {
    cumLen += group.count
    wordBoundaries.append(cumLen)
  }

  // Find jump points (where text index changes) - matches Python's approach
  // Python: jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
  // This creates a boolean mask with True at every position where text_index changes,
  // plus a forced True at position 0.
  var jumpPositions = [0] // Start with position 0 (forced like Python's padding)
  for i in 1 ..< textIndices.count {
    if textIndices[i] != textIndices[i - 1] {
      jumpPositions.append(i)
    }
  }

  // Convert jump positions to times
  // Python: jump_times = time_indices[jumps] / TOKENS_PER_SECOND
  let jumpTimes = jumpPositions.map { idx -> Float in
    guard idx < timeIndices.count else { return 0 }
    return Float(timeIndices[idx]) / tokensPerSecond
  }

  // Calculate word timings
  // Python uses word_boundaries directly without offset:
  //   start_times = jump_times[word_boundaries[:-1]]
  //   end_times = jump_times[word_boundaries[1:]]
  var wordTimings: [WordTiming] = []
  for i in 0 ..< (words.count - 1) { // Skip EOT marker
    let startBoundary = wordBoundaries[i]
    let endBoundary = wordBoundaries[i + 1]

    // Map boundaries to times
    let startTime: Float
    let endTime: Float

    // Find the jump time for this boundary
    if startBoundary < jumpTimes.count {
      startTime = jumpTimes[startBoundary]
    } else if !jumpTimes.isEmpty {
      startTime = jumpTimes.last!
    } else {
      startTime = 0
    }

    if endBoundary < jumpTimes.count {
      endTime = jumpTimes[endBoundary]
    } else if !jumpTimes.isEmpty {
      endTime = jumpTimes.last!
    } else {
      endTime = startTime
    }

    // Calculate average probability for word
    // Use original word boundaries (without +1 offset) since textTokenProbs is indexed on text tokens
    var avgProb: Float = 0
    let probStart = wordBoundaries[i]
    let probEnd = min(wordBoundaries[i + 1], textTokenProbs.count)
    if probStart < probEnd {
      for j in probStart ..< probEnd {
        avgProb += textTokenProbs[j]
      }
      avgProb /= Float(probEnd - probStart)
    }

    wordTimings.append(WordTiming(
      word: words[i],
      tokens: wordTokenGroups[i],
      start: startTime,
      end: max(endTime, startTime), // Ensure end >= start
      probability: avgProb
    ))
  }

  return wordTimings
}

// MARK: - Batched Word Timestamps

/// Add word-level timestamps to segments using batched processing
///
/// This matches Python's `add_word_timestamps()` function which processes all segments
/// in a single forward pass for efficiency. Key optimizations:
/// - Single `findAlignment` call for all segments (vs per-segment in naive approach)
/// - Batched duration calculations and clipping
///
/// - Parameters:
///   - segments: Array of transcription segments to add word timestamps to (modified in-place)
///   - model: Whisper model with alignment heads
///   - tokenizer: Whisper tokenizer
///   - mel: Mel spectrogram for the current window
///   - numFrames: Number of audio frames (segment_size)
///   - language: Language code
///   - task: Transcription task
///   - timeOffset: Time offset for current window
///   - lastSpeechTimestamp: End time of last speech (for segment boundary clipping)
/// - Returns: Updated lastSpeechTimestamp
func addWordTimestamps(
  segments: inout [TranscriptionSegment],
  model: WhisperModel,
  tokenizer: WhisperTokenizer,
  mel: MLXArray,
  numFrames: Int,
  language: String?,
  task: TranscriptionTask,
  timeOffset: Float,
  lastSpeechTimestamp: Float
) -> Float {
  guard !segments.isEmpty else { return lastSpeechTimestamp }

  // Extract text tokens per segment (matching Python exactly)
  let textTokensPerSegment: [[Int]] = segments.map { segment in
    segment.tokens.filter { $0 < tokenizer.eot }
  }

  // Concatenate all text tokens for batched alignment (Python: itertools.chain.from_iterable)
  let allTextTokens = textTokensPerSegment.flatMap { $0 }

  guard !allTextTokens.isEmpty else { return lastSpeechTimestamp }

  // Single findAlignment call for ALL segments (key optimization)
  var alignment = findAlignment(
    model: model,
    tokenizer: tokenizer,
    textTokens: allTextTokens,
    mel: mel,
    numFrames: numFrames,
    language: language,
    task: task
  )

  guard !alignment.isEmpty else { return lastSpeechTimestamp }

  // Calculate duration thresholds across all words
  let (medianDuration, maxDuration) = calculateDurationThresholds(alignment)

  // Clip long words at sentence boundaries (Python lines 250-258)
  if maxDuration > 0 {
    clipAtSentenceBoundaries(&alignment, maxDuration: maxDuration)
  }

  // Merge punctuations
  mergePunctuations(&alignment)

  // Distribute words back to segments (Python lines 265-329)
  var wordIndex = 0
  var updatedLastSpeechTimestamp = lastSpeechTimestamp

  for (segmentIdx, textTokens) in textTokensPerSegment.enumerated() {
    var savedTokens = 0
    var words: [Word] = []

    // Consume words until we've covered this segment's tokens
    while wordIndex < alignment.count, savedTokens < textTokens.count {
      let timing = alignment[wordIndex]

      if !timing.word.isEmpty {
        words.append(Word(
          word: timing.word,
          start: TimeInterval(timeOffset + timing.start),
          end: TimeInterval(timeOffset + timing.end),
          probability: timing.probability
        ))
      }

      savedTokens += timing.tokens.count
      wordIndex += 1
    }

    if !words.isEmpty {
      // Clip at segment boundaries (Python lines 287-303)
      if maxDuration > 0 {
        var wordTimings = words.map {
          WordTiming(word: $0.word, tokens: [], start: Float($0.start), end: Float($0.end), probability: $0.probability)
        }
        clipAtSegmentBoundaries(
          &wordTimings,
          lastSpeechTimestamp: updatedLastSpeechTimestamp,
          medianDuration: medianDuration,
          maxDuration: maxDuration
        )
        // Apply clipped times back to words
        for i in 0 ..< min(words.count, wordTimings.count) {
          words[i] = Word(
            word: words[i].word,
            start: TimeInterval(wordTimings[i].start),
            end: TimeInterval(wordTimings[i].end),
            probability: words[i].probability
          )
        }
      }

      // Adjust segment boundaries based on word timestamps (Python lines 305-325)
      let segment = segments[segmentIdx]
      let segmentStart = Float(segment.start)
      let segmentEnd = Float(segment.end)

      var adjustedStart = segmentStart
      var adjustedEnd = segmentEnd

      if let firstWord = words.first {
        // Prefer segment-level start if first word appears too early
        if segmentStart < Float(firstWord.end), segmentStart - 0.5 > Float(firstWord.start) {
          adjustedStart = max(0, min(Float(firstWord.end) - medianDuration, segmentStart))
          words[0] = Word(
            word: firstWord.word,
            start: TimeInterval(adjustedStart),
            end: firstWord.end,
            probability: firstWord.probability
          )
        } else {
          adjustedStart = Float(firstWord.start)
        }
      }

      if let lastWord = words.last {
        // Prefer segment-level end if last word appears too late
        if segmentEnd > Float(lastWord.start), segmentEnd + 0.5 < Float(lastWord.end) {
          adjustedEnd = max(Float(lastWord.start) + medianDuration, segmentEnd)
          words[words.count - 1] = Word(
            word: lastWord.word,
            start: lastWord.start,
            end: TimeInterval(adjustedEnd),
            probability: lastWord.probability
          )
        } else {
          adjustedEnd = Float(lastWord.end)
        }
        updatedLastSpeechTimestamp = adjustedEnd
      }

      // Update segment with words and adjusted boundaries
      segments[segmentIdx] = TranscriptionSegment(
        text: segment.text,
        start: TimeInterval(adjustedStart),
        end: TimeInterval(adjustedEnd),
        tokens: segment.tokens,
        avgLogProb: segment.avgLogProb,
        noSpeechProb: segment.noSpeechProb,
        words: words
      )
    }
  }

  return updatedLastSpeechTimestamp
}

// MARK: - Hallucination Detection

/// Python's string.punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
/// Used to filter out single-character punctuation words in anomaly detection.
private let pythonPunctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

/// Calculate anomaly score for a word
///
/// Anomalous words are very long, very short, or have low probability.
/// This matches Python's `word_anomaly_score()` function exactly.
///
/// - Parameter word: Word timing to score
/// - Returns: Anomaly score (higher = more anomalous)
func wordAnomalyScore(_ word: WordTiming) -> Float {
  let duration = word.end - word.start
  var score: Float = 0.0

  // Low probability words are suspicious
  if word.probability < 0.15 {
    score += 1.0
  }

  // Very short words (< 133ms) are suspicious
  if duration < 0.133 {
    score += (0.133 - duration) * 15
  }

  // Very long words (> 2s) are suspicious
  if duration > 2.0 {
    score += duration - 2.0
  }

  return score
}

/// Check if a segment's words indicate a possible hallucination
///
/// A segment is considered anomalous if the first 8 non-punctuation words
/// have a combined anomaly score >= 3, or if almost all words are anomalous.
/// This matches Python's `is_segment_anomaly()` function exactly.
///
/// - Parameter words: Words from a segment
/// - Returns: True if the segment appears to be a hallucination
func isSegmentAnomaly(_ words: [WordTiming]?) -> Bool {
  guard let words, !words.isEmpty else {
    return false
  }

  // Filter out words that are entirely punctuation (matching Python's `w["word"] not in punctuation`)
  // Python checks if the word string is a substring of string.punctuation
  // This filters single-char punctuation like "." but keeps multi-char like "..."
  let filteredWords = words.filter { word in
    !pythonPunctuation.contains(word.word)
  }.prefix(8)

  guard !filteredWords.isEmpty else {
    return false
  }

  let score = filteredWords.reduce(Float(0)) { $0 + wordAnomalyScore($1) }

  // Anomalous if score >= 3 or if almost all words are anomalous
  return score >= 3 || score + 0.01 >= Float(filteredWords.count)
}

/// Get the end time of the last word across segments
///
/// - Parameter segments: Array of transcription segments
/// - Returns: End time of the last word, or nil if no words
func getLastWordEnd(_ segments: [TranscriptionSegment]) -> Float? {
  for segment in segments.reversed() {
    if let words = segment.words, let lastWord = words.last {
      return Float(lastWord.end)
    }
  }
  // Fall back to segment end time
  return segments.last.map { Float($0.end) }
}

/// Find the next segment that has words
///
/// - Parameters:
///   - segments: Array of segments to search
///   - startIndex: Index to start searching from
/// - Returns: The next segment with words, or nil
func nextWordsSegment(_ segments: [TranscriptionSegment], startIndex: Int) -> TranscriptionSegment? {
  for i in startIndex ..< segments.count {
    if let words = segments[i].words, !words.isEmpty {
      return segments[i]
    }
  }
  return nil
}

/// Filter out hallucinated segments based on anomaly detection
///
/// This implements hallucination detection matching Python's approach exactly.
/// Segments that appear anomalous and are surrounded by silence are filtered out.
///
/// Python's logic (whisper.py lines 781-820):
/// - Iterates through current_segments checking each for anomaly
/// - Uses hal_last_end to track last speech end (initialized to last_speech_timestamp)
/// - silence_before: gap from last speech > threshold OR near window start
/// - silence_after: gap to next segment > threshold OR next is anomaly OR near window end
/// - When both are true, segment is filtered as hallucination
///
/// - Parameters:
///   - segments: Array of transcription segments with word timestamps
///   - threshold: Silence threshold in seconds for hallucination detection
///   - audioDuration: Total audio duration in seconds
/// - Returns: Filtered array of segments with hallucinations removed
func filterHallucinatedSegments(
  _ segments: [TranscriptionSegment],
  threshold: Float,
  audioDuration: Float
) -> [TranscriptionSegment] {
  let chunkLength: Float = 30.0 // Whisper's 30-second chunk size

  guard threshold > 0, !segments.isEmpty else {
    return segments
  }

  var filteredSegments: [TranscriptionSegment] = []
  var lastSpeechTimestamp: Float = 0

  for (index, segment) in segments.enumerated() {
    guard let words = segment.words, !words.isEmpty else {
      // Keep segments without words (they won't affect hallucination detection)
      filteredSegments.append(segment)
      continue
    }

    let wordTimings = words.map { word in
      WordTiming(
        word: word.word,
        tokens: [],
        start: Float(word.start),
        end: Float(word.end),
        probability: word.probability
      )
    }

    // Check if this segment is anomalous
    if isSegmentAnomaly(wordTimings) {
      let segmentStart = Float(segment.start)
      let segmentEnd = Float(segment.end)

      // Compute which 30s window this segment belongs to (for relative time calculations)
      // Python uses time_offset and window_end_time within each processing window
      let windowIdx = Int(segmentStart / chunkLength)
      let timeOffset = Float(windowIdx) * chunkLength
      let windowEndTime = min(Float(windowIdx + 1) * chunkLength, audioDuration)

      // Find next segment with words
      let nextSeg = nextWordsSegment(segments, startIndex: index + 1)
      let halNextStart: Float = if let next = nextSeg, let nextWords = next.words, let firstWord = nextWords.first {
        Float(firstWord.start)
      } else {
        // Python: hal_next_start = time_offset + segment_duration (i.e., window end)
        timeOffset + chunkLength
      }

      // Check for silence before this segment (matching Python exactly)
      // Python: segment["start"] - hal_last_end > threshold
      //         or segment["start"] < threshold
      //         or segment["start"] - time_offset < 2.0
      let silenceBefore = (segmentStart - lastSpeechTimestamp > threshold) ||
        (segmentStart < threshold) ||
        (segmentStart - timeOffset < 2.0)

      // Check for silence after this segment (matching Python exactly)
      // Python: hal_next_start - segment["end"] > threshold
      //         or is_segment_anomaly(next_segment)
      //         or window_end_time - segment["end"] < 2.0
      let nextWordTimings: [WordTiming]? = nextSeg?.words?.map { word in
        WordTiming(
          word: word.word,
          tokens: [],
          start: Float(word.start),
          end: Float(word.end),
          probability: word.probability
        )
      }
      let silenceAfter = (halNextStart - segmentEnd > threshold) ||
        isSegmentAnomaly(nextWordTimings) ||
        (windowEndTime - segmentEnd < 2.0)

      // If surrounded by silence, this is likely a hallucination - skip it
      if silenceBefore, silenceAfter {
        Log.model.debug("Filtering hallucinated segment: \"\(segment.text)\" (\(segmentStart)-\(segmentEnd)s)")
        continue
      }
    }

    // Keep this segment and update last speech timestamp
    // Python: hal_last_end = segment["end"] (only for non-filtered segments)
    filteredSegments.append(segment)

    if let lastWord = words.last {
      lastSpeechTimestamp = Float(lastWord.end)
    }
  }

  return filteredSegments
}
