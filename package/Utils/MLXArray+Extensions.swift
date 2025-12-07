import MLX

// MARK: - arange extension

// TODO: Remove after next mlx-swift release. `arange` has been added to MLX Swift:
// https://github.com/ml-explore/mlx-swift/pull/302

extension MLXArray {
  /// Generate values in the half-open interval `[0, stop)`.
  ///
  /// Example:
  /// ```swift
  /// let r = MLXArray.arange(10) // [0, 1, 2, ..., 9]
  /// ```
  ///
  /// - Parameter stop: End of the sequence (exclusive).
  /// - Returns: An array containing the generated range as int32.
  static func arange(_ stop: Int) -> MLXArray {
    guard stop > 0 else { return MLXArray([Int32]()) }
    return MLXArray(Array(0 ..< stop).map { Int32($0) })
  }
}
