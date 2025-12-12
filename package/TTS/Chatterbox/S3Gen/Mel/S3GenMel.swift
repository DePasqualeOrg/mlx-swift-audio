import Foundation
import MLX

/// Reflect pad a 2D array (B, T) along axis 1
private func reflectPad2D(_ x: MLXArray, padAmount: Int) -> MLXArray {
  if padAmount == 0 {
    return x
  }

  let T = x.shape[1]

  // Reflect at start: x[:, 1:padAmount+1][:, ::-1]
  let prefixEnd = min(padAmount + 1, T)
  let prefixSlice = x[0..., 1 ..< prefixEnd]
  let prefix = reverseAlongAxis(prefixSlice, axis: 1)

  // Reflect at end: x[:, -(padAmount+1):-1][:, ::-1]
  let suffixStart = max(0, T - padAmount - 1)
  let suffixSlice = x[0..., suffixStart ..< (T - 1)]
  let suffix = reverseAlongAxis(suffixSlice, axis: 1)

  return MLX.concatenated([prefix, x, suffix], axis: 1)
}

/// Extract mel-spectrogram from waveform for S3Gen
///
/// - Parameters:
///   - y: Waveform (B, T) or (T,)
///   - nFft: FFT size (default 1920)
///   - numMels: Number of mel bins (default 80)
///   - samplingRate: Sample rate (default 24000)
///   - hopSize: Hop size (default 480)
///   - winSize: Window size (default 1920)
///   - fmin: Minimum frequency (default 0)
///   - fmax: Maximum frequency (default 8000)
///   - center: Whether to center the window (default false)
/// - Returns: Mel-spectrogram (B, num_mels, T')
func s3genMelSpectrogram(
  y: MLXArray,
  nFft: Int = 1920,
  numMels: Int = 80,
  samplingRate: Int = 24000,
  hopSize: Int = 480,
  winSize: Int = 1920,
  fmin: Int = 0,
  fmax: Int = 8000,
  center _: Bool = false,
) -> MLXArray {
  var yArray = y
  let was1D = y.ndim == 1
  if was1D {
    yArray = y.expandedDimensions(axis: 0)
  }

  // Pad signal with reflection
  let padAmount = (nFft - hopSize) / 2
  yArray = reflectPad2D(yArray, padAmount: padAmount)

  // Process each batch item - independent operations can be parallelized by MLX
  let batchSize = yArray.shape[0]
  let window = hanningWindow(length: winSize + 1)[0 ..< winSize]

  let specs = (0 ..< batchSize).map { i -> MLXArray in
    stft(
      yArray[i],
      window: window,
      nFft: nFft,
      hopLength: hopSize,
      winLength: winSize,
      center: false,
    )
  }

  // Stack: each spec is (T', F) -> stack to (B, T', F)
  let spec = MLX.stacked(specs, axis: 0)

  // Magnitude spectrogram
  let magnitudes = MLX.abs(spec) // (B, T', F)

  // Create mel filterbank
  let filters = melFilters(
    sampleRate: samplingRate,
    nFft: nFft,
    nMels: numMels,
    fMin: Float(fmin),
    fMax: Float(fmax),
  )

  // Apply mel filterbank: (B, T', F) @ (F, M) -> (B, T', M)
  var melSpec = MLX.matmul(magnitudes, filters.T)
  melSpec = melSpec.transposed(0, 2, 1) // (B, M, T')

  // Log compression
  melSpec = MLX.log(MLX.maximum(melSpec, MLXArray(1e-5)))

  return was1D ? melSpec.squeezed(axis: 0) : melSpec
}
