import AVFoundation
import Foundation

/// Simple error type for test failures
struct TestError: LocalizedError {
  let message: String
  var errorDescription: String? { message }
}

/// Shared audio caching utility for tests
/// Downloads audio files once and caches them locally
enum TestAudioCache {
  /// Cache directory for downloaded audio files
  static var cacheDir: URL {
    let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return cacheDir.appendingPathComponent("mlx-swift-audio/test-audio", isDirectory: true)
  }

  /// Download audio data from URL with caching
  /// Returns the raw audio data (not resampled)
  static func downloadData(from url: URL) async throws -> Data {
    // Create cache directory if needed
    try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

    // Compute cache file path from URL
    let cacheFileName = url.lastPathComponent
    let cacheURL = cacheDir.appendingPathComponent(cacheFileName)

    // Check if cached file exists
    if FileManager.default.fileExists(atPath: cacheURL.path) {
      print("[TestAudioCache] Loading cached: \(cacheFileName)")
      return try Data(contentsOf: cacheURL)
    }

    // Download and cache
    print("[TestAudioCache] Downloading: \(url.lastPathComponent)...")
    let (data, _) = try await URLSession.shared.data(from: url)
    try data.write(to: cacheURL)
    print("[TestAudioCache] Cached to: \(cacheURL.path)")

    return data
  }

  /// Download audio and save to a temp file for AVAudioFile reading
  /// Returns URL to the cached or downloaded file
  static func downloadToFile(from url: URL) async throws -> URL {
    // Create cache directory if needed
    try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

    // Compute cache file path from URL
    let cacheFileName = url.lastPathComponent
    let cacheURL = cacheDir.appendingPathComponent(cacheFileName)

    // Check if cached file exists
    if FileManager.default.fileExists(atPath: cacheURL.path) {
      print("[TestAudioCache] Loading cached: \(cacheFileName)")
      return cacheURL
    }

    // Download and cache
    print("[TestAudioCache] Downloading: \(url.lastPathComponent)...")
    let (data, _) = try await URLSession.shared.data(from: url)
    try data.write(to: cacheURL)
    print("[TestAudioCache] Cached to: \(cacheURL.path)")

    return cacheURL
  }
}
