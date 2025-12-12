import Foundation
import MLX
import MLXNN

/// Make mask tensor containing indices of non-padded part.
///
/// The sequences in a batch may have different lengths. To enable
/// batch computing, padding is needed to make all sequences the same
/// size. To avoid the padding part passing value to context dependent
/// blocks such as attention or convolution, this padding part is masked.
///
/// - Parameters:
///   - lengths: Batch of lengths (B,)
///   - maxLen: Maximum length. If 0, use the maximum length in batch.
/// - Returns: Mask tensor containing indices of padded part (B, max_T). 1 for non-padded, 0 for padded.
func makeNonPadMask(lengths: MLXArray, maxLen: Int = 0) -> MLXArray {
  let batchSize = lengths.shape[0]
  let actualMaxLen = maxLen > 0 ? maxLen : Int(lengths.max().item(Int.self))
  let seqRange = MLXArray(0 ..< actualMaxLen)
  let seqRangeExpand = seqRange.expandedDimensions(axis: 0)
  let seqRangeBroadcast = MLX.broadcast(seqRangeExpand, to: [batchSize, actualMaxLen])
  let seqLengthExpand = lengths.expandedDimensions(axis: -1)
  let mask = seqRangeBroadcast .>= seqLengthExpand
  return MLX.logicalNot(mask)
}

/// Convert boolean mask to attention bias.
///
/// - Parameters:
///   - mask: Boolean mask tensor
///   - dtype: Output data type (must be floating point)
/// - Returns: Attention bias tensor with -1e10 for masked positions
func maskToBias(_ mask: MLXArray, dtype: DType = .float32) -> MLXArray {
  let maskFloat = mask.asType(dtype)
  return (1.0 - maskFloat) * -1.0e10
}

/// Padding the data into batch data
///
/// - Parameter data: List of arrays, shape of each array (128, T)
/// - Returns: Tuple of (padded_feats, feats_lengths)
func padSequences(_ data: [MLXArray]) -> (MLXArray, MLXArray) {
  let featsLengths = MLXArray(data.map { Int32($0.shape[1]) })
  let maxLen = data.map { $0.shape[1] }.max() ?? 0
  let batchSize = data.count
  let nMels = data[0].shape[0]

  let paddedFeats = MLXArray.zeros([batchSize, nMels, maxLen], dtype: data[0].dtype)

  for (i, feat) in data.enumerated() {
    let seqLen = feat.shape[1]
    // Copy the original data
    paddedFeats[i, 0..., 0 ..< seqLen] = feat
  }

  return (paddedFeats, featsLengths)
}

/// Merges tokenized outputs by keeping the middle and dropping half of the overlapped tokens.
///
/// - Parameters:
///   - tokenizedSegments: List of tokenized sequences.
///   - overlap: Overlapping duration in seconds.
///   - tokenRate: Number of tokens per second.
/// - Returns: A single merged token sequence.
func mergeTokenizedSegments(
  _ tokenizedSegments: [[Int]],
  overlap: Int,
  tokenRate: Int,
) -> [Int] {
  var mergedTokens: [Int] = []
  let overlapTokens = (overlap / 2) * tokenRate

  for (i, tokens) in tokenizedSegments.enumerated() {
    let left = i == 0 ? 0 : overlapTokens
    let right = i != tokenizedSegments.count - 1 ? tokens.count - overlapTokens : tokens.count
    if left < right {
      mergedTokens.append(contentsOf: tokens[left ..< right])
    }
  }

  return mergedTokens
}

/// Compute the log-Mel spectrogram using standard STFT.
///
/// This implementation matches the MLX S3Tokenizer which uses standard mel spectrogram computation.
///
/// - Parameters:
///   - audio: Audio waveform (T,) in 16 kHz
///   - sampleRate: Sample rate (default 16000)
///   - nMels: Number of Mel-frequency filters (default 128)
///   - nFft: FFT size (default 400)
///   - hopLength: Hop length (default 160)
///   - padding: Number of zero samples to pad to the right
/// - Returns: Log-Mel spectrogram (n_mels, T')
func logMelSpectrogram(
  audio: MLXArray,
  sampleRate: Int = 16000,
  nMels: Int = 128,
  nFft: Int = 400,
  hopLength: Int = 160,
  padding: Int = 0,
) -> MLXArray {
  var audioArray = audio

  if padding > 0 {
    audioArray = MLX.padded(audioArray, widths: [IntOrPair((0, padding))])
  }

  // Create Hann window
  let window = hanningWindow(length: nFft + 1)[0 ..< nFft]

  // Compute STFT
  let stftResult = stft(
    audioArray,
    window: window,
    nFft: nFft,
    hopLength: hopLength,
    winLength: nFft,
  )

  // Swap axes to get (F, T) format then compute magnitudes
  let freqs = stftResult.swappedAxes(0, 1)
  let magnitudes = MLX.pow(MLX.abs(freqs), 2)

  // Create mel filterbank
  let filters = melFilters(
    sampleRate: sampleRate,
    nFft: nFft,
    nMels: nMels,
  )

  // Apply mel filterbank: (T, F) @ (F, M) -> (T, M)
  // magnitudes is (T, nFft/2+1), filters is (nMels, nFft/2+1)
  let melSpec = MLX.matmul(magnitudes, filters.T)

  // Log compression with S3Tokenizer-style normalization
  var logSpec = MLX.log10(MLX.maximum(melSpec, MLXArray(1e-10)))
  logSpec = MLX.maximum(logSpec, logSpec.max() - 8.0)
  logSpec = (logSpec + 4.0) / 4.0

  return logSpec
}

/// Compute the log-Mel spectrogram for Chatterbox (drops last frame to match PyTorch).
///
/// This implementation matches the PyTorch S3Tokenizer which uses torch.stft behavior.
///
/// - Parameters:
///   - audio: Audio waveform (T,) in 16 kHz
///   - nMels: Number of Mel-frequency filters (default 128)
///   - padding: Number of zero samples to pad to the right
/// - Returns: Log-Mel spectrogram (n_mels, T')
func logMelSpectrogramChatterbox(
  audio: MLXArray,
  nMels: Int = 128,
  padding: Int = 0,
) -> MLXArray {
  var audioArray = audio

  if padding > 0 {
    audioArray = MLX.padded(audioArray, widths: [IntOrPair((0, padding))])
  }

  // Create Hann window
  let window = hanningWindow(length: 400 + 1)[0 ..< 400]

  // Compute STFT
  let stftResult = stft(
    audioArray,
    window: window,
    nFft: 400,
    hopLength: 160,
    winLength: 400,
  )

  // Drop last frame to match PyTorch torch.stft behavior
  let spec = stftResult[0 ..< (stftResult.shape[0] - 1), 0...]

  // Compute magnitudes
  let magnitudes = MLX.pow(MLX.abs(spec), 2)

  // Create mel filterbank with slaney normalization
  let filters = melFilters(
    sampleRate: 16000,
    nFft: 400,
    nMels: nMels,
  )

  // Apply mel filterbank: (T, F) @ (F, M) -> (T, M)
  let melSpec = MLX.matmul(magnitudes, filters.T)

  // Transpose to (M, T)
  let melSpecT = melSpec.transposed(1, 0)

  // Log compression with S3Tokenizer-style normalization
  var logSpec = MLX.log10(MLX.maximum(melSpecT, MLXArray(1e-10)))
  logSpec = MLX.maximum(logSpec, logSpec.max() - 8.0)
  logSpec = (logSpec + 4.0) / 4.0

  return logSpec
}

// MARK: - Helper functions

/// Create a Hanning window (symmetric)
func hanningWindow(length: Int) -> MLXArray {
  if length == 1 {
    return MLXArray([1.0])
  }

  let n = MLXArray(Array(stride(from: Float(1 - length), to: Float(length), by: 2.0)))
  let factor = Float.pi / Float(length - 1)
  return 0.5 + 0.5 * MLX.cos(n * factor)
}

/// Compute STFT of a signal
func stft(
  _ x: MLXArray,
  window: MLXArray,
  nFft: Int,
  hopLength: Int,
  winLength _: Int,
  center: Bool = true,
  padMode _: String = "reflect",
) -> MLXArray {
  var xArray = x

  // Pad window to nFft if needed (matches Python behavior)
  var w = window
  if w.shape[0] < nFft {
    let padSize = nFft - w.shape[0]
    w = MLX.concatenated([w, MLXArray.zeros([padSize])])
  }

  // Center padding
  if center {
    xArray = reflectPad(xArray, padding: nFft / 2)
  }

  // Frame the signal
  let numFrames = 1 + (xArray.shape[0] - nFft) / hopLength
  if numFrames <= 0 {
    fatalError("Input is too short for STFT")
  }

  // Create frames using as_strided equivalent
  let shape = [numFrames, nFft]
  let strides = [hopLength, 1]
  let frames = MLX.asStrided(xArray, shape, strides: strides)

  // Apply window and compute FFT
  let windowedFrames = frames * w
  let spec = MLX.rfft(windowedFrames)

  return spec
}

/// Reflect padding for 1D array
private func reflectPad(_ x: MLXArray, padding: Int) -> MLXArray {
  if padding == 0 {
    return x
  }

  // Handle edge cases
  let n = x.shape[0]
  if n == 1 {
    // If array has only one element, just repeat it
    return MLX.concatenated([
      MLXArray.full([padding], values: x[0]),
      x,
      MLXArray.full([padding], values: x[0]),
    ])
  }

  // Reflect at boundaries
  var prefixArray = reverseAlongAxis(x[1 ..< min(padding + 1, n)], axis: 0)
  var suffixArray = reverseAlongAxis(x[max(0, n - padding - 1) ..< (n - 1)], axis: 0)

  // Handle cases where array is shorter than padding
  while prefixArray.shape[0] < padding {
    let additional = min(padding - prefixArray.shape[0], n - 1)
    prefixArray = MLX.concatenated([reverseAlongAxis(x[1 ..< (additional + 1)], axis: 0), prefixArray])
  }

  while suffixArray.shape[0] < padding {
    let additional = min(padding - suffixArray.shape[0], n - 1)
    suffixArray = MLX.concatenated([suffixArray, reverseAlongAxis(x[(n - additional - 1) ..< (n - 1)], axis: 0)])
  }

  return MLX.concatenated([prefixArray[0 ..< padding], x, suffixArray[0 ..< padding]])
}

/// Create mel filterbank
func melFilters(
  sampleRate: Int,
  nFft: Int,
  nMels: Int,
  fMin: Float = 0.0,
  fMax: Float? = nil,
) -> MLXArray {
  let actualFMax = fMax ?? Float(sampleRate) / 2.0

  // Mel scale conversion functions (slaney style)
  func hzToMel(_ hz: Float) -> Float {
    let fSp: Float = 200.0 / 3.0
    let minLogHz: Float = 1000.0
    let minLogMel = minLogHz / fSp
    let logstep: Float = log(6.4) / 27.0

    if hz >= minLogHz {
      return minLogMel + log(hz / minLogHz) / logstep
    } else {
      return hz / fSp
    }
  }

  func melToHz(_ mel: Float) -> Float {
    let fSp: Float = 200.0 / 3.0
    let minLogHz: Float = 1000.0
    let minLogMel = minLogHz / fSp
    let logstep: Float = log(6.4) / 27.0

    if mel >= minLogMel {
      return minLogHz * exp(logstep * (mel - minLogMel))
    } else {
      return fSp * mel
    }
  }

  // Create mel points
  let melMin = hzToMel(fMin)
  let melMax = hzToMel(actualFMax)
  let melPoints = (0 ... nMels + 1).map { i in
    melToHz(melMin + Float(i) * (melMax - melMin) / Float(nMels + 1))
  }

  // Convert to FFT bin numbers
  let fftFreqs = (0 ..< (nFft / 2 + 1)).map { i in
    Float(i) * Float(sampleRate) / Float(nFft)
  }

  // Create filterbank
  var filterbank = [[Float]](repeating: [Float](repeating: 0, count: nFft / 2 + 1), count: nMels)

  for m in 0 ..< nMels {
    let fLeft = melPoints[m]
    let fCenter = melPoints[m + 1]
    let fRight = melPoints[m + 2]

    for k in 0 ..< (nFft / 2 + 1) {
      let freq = fftFreqs[k]

      if freq >= fLeft, freq <= fCenter {
        filterbank[m][k] = (freq - fLeft) / (fCenter - fLeft)
      } else if freq > fCenter, freq <= fRight {
        filterbank[m][k] = (fRight - freq) / (fRight - fCenter)
      }
    }

    // Slaney normalization
    let enorm = 2.0 / (melPoints[m + 2] - melPoints[m])
    for k in 0 ..< (nFft / 2 + 1) {
      filterbank[m][k] *= enorm
    }
  }

  return MLXArray(filterbank.flatMap { $0 }).reshaped([nMels, nFft / 2 + 1])
}
