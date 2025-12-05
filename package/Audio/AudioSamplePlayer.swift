@preconcurrency import AVFoundation
import Foundation

/// Plays raw audio samples for TTS engines
@Observable
@MainActor
final class AudioSamplePlayer {
  // MARK: - Public State

  /// Whether audio is currently playing
  private(set) var isPlaying: Bool = false

  /// Number of samples currently queued for playback
  private(set) var queuedSampleCount: Int = 0

  // MARK: - Private Properties

  @ObservationIgnored private var engine: AVAudioEngine!
  @ObservationIgnored private var playerNode: AVAudioPlayerNode!
  @ObservationIgnored private var audioFormat: AVAudioFormat!
  @ObservationIgnored private var hasStartedPlayback: Bool = false
  @ObservationIgnored private var playbackCompletionContinuation: CheckedContinuation<Void, Never>?

  private let sampleRate: Int

  // MARK: - Initialization

  /// Create an audio engine with the specified sample rate
  /// - Parameter sampleRate: Sample rate in Hz (default: 24000)
  init(sampleRate: Int = 24000) {
    self.sampleRate = sampleRate
    setup()
  }

  deinit {
    NotificationCenter.default.removeObserver(self)
  }

  // MARK: - Setup

  private func setup() {
    engine = AVAudioEngine()
    playerNode = AVAudioPlayerNode()

    audioFormat = AVAudioFormat(
      standardFormatWithSampleRate: Double(sampleRate),
      channels: 1,
    )
    guard audioFormat != nil else {
      Log.audio.error("Failed to create audio format")
      return
    }

    engine.attach(playerNode)
    engine.connect(playerNode, to: engine.mainMixerNode, format: audioFormat)

    do {
      try engine.start()
      Log.audio.debug("Audio engine started successfully")
    } catch {
      Log.audio.error("Failed to start audio engine: \(error.localizedDescription)")
    }
  }

  // MARK: - Public API

  /// Play audio samples and wait for playback to complete
  /// - Parameters:
  ///   - samples: Float audio samples
  ///   - volumeBoost: Optional volume multiplier (default: 1.25)
  func play(samples: [Float], volumeBoost: Float = TTSConstants.Audio.volumeBoostFactor) async {
    guard !samples.isEmpty else { return }

    // Stop any current playback (but don't restart engine)
    await stop()
    resetEngineIfNeeded()

    guard let buffer = createBuffer(from: samples, volumeBoost: volumeBoost) else {
      Log.audio.error("Failed to create audio buffer")
      return
    }

    scheduleBuffer(buffer)

    playerNode.play()
    isPlaying = true
    hasStartedPlayback = true

    // Await completion (resumed by buffer completion callback)
    await withCheckedContinuation { continuation in
      self.playbackCompletionContinuation = continuation
    }
  }

  /// Enqueue audio samples for streaming playback
  /// - Parameters:
  ///   - samples: Float audio samples to enqueue
  ///   - prebufferSeconds: Seconds to buffer before starting playback (0 for immediate)
  func enqueue(samples: [Float], prebufferSeconds: Double = 0) {
    guard !samples.isEmpty else { return }

    resetEngineIfNeeded()

    // Schedule in small slices for smoother streaming
    let sliceSamples = max(1, Int(0.03 * Double(sampleRate))) // 30ms slices
    var offset = 0

    while offset < samples.count {
      let remaining = samples.count - offset
      let thisLength = min(sliceSamples, remaining)
      let slice = Array(samples[offset ..< (offset + thisLength)])

      guard let buffer = createBuffer(from: slice, volumeBoost: 1.0) else {
        offset += thisLength
        continue
      }

      queuedSampleCount += thisLength
      let decrementAmount = thisLength

      playerNode.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
        Task { @MainActor [weak self] in
          guard let self else { return }
          queuedSampleCount = max(0, queuedSampleCount - decrementAmount)
        }
      }

      // Start playback when prebuffer is reached
      if !hasStartedPlayback {
        let prebufferSamples = Int(prebufferSeconds * Double(sampleRate))
        if prebufferSamples == 0 || queuedSampleCount >= prebufferSamples {
          playerNode.play()
          hasStartedPlayback = true
          isPlaying = true

          // Retry if playback didn't start
          if !playerNode.isPlaying {
            Task {
              try? await Task.sleep(for: .milliseconds(100))
              self.playerNode.play()
            }
          }
        }
      } else if !playerNode.isPlaying {
        playerNode.play()
      }

      offset += thisLength
    }
  }

  /// Stop playback and reset state
  func stop() async {
    // Resume any waiting continuation before stopping
    playbackCompletionContinuation?.resume()
    playbackCompletionContinuation = nil

    // Dispatch to background QoS to avoid priority inversion
    // (audio threads run at Default QoS, so calling from Background avoids inversion)
    await withCheckedContinuation { continuation in
      DispatchQueue.global(qos: .background).async { [playerNode] in
        playerNode?.stop()
        playerNode?.reset()
        continuation.resume()
      }
    }

    isPlaying = false
    hasStartedPlayback = false
    queuedSampleCount = 0

    Log.audio.debug("Audio playback stopped")
  }

  /// Reset the audio engine (useful after interruptions)
  func reset() async {
    await stop()

    // Reconnect components
    if playerNode.engine != nil {
      engine.detach(playerNode)
    }
    engine.attach(playerNode)
    engine.connect(playerNode, to: engine.mainMixerNode, format: audioFormat)

    // Restart engine
    do {
      try engine.start()
      Log.audio.debug("Audio engine reset successfully")
    } catch {
      Log.audio.error("Failed to restart audio engine: \(error.localizedDescription)")
    }
  }

  // MARK: - Private

  private func createBuffer(from samples: [Float], volumeBoost: Float) -> AVAudioPCMBuffer? {
    let frameCount = AVAudioFrameCount(samples.count)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameCount) else {
      return nil
    }

    buffer.frameLength = frameCount

    guard let channelData = buffer.floatChannelData else {
      return nil
    }

    // Copy samples with volume boost and clipping prevention
    let maxValue = TTSConstants.Audio.maxSampleValue
    for i in 0 ..< samples.count {
      let boosted = samples[i] * volumeBoost
      channelData[0][i] = min(max(boosted, -maxValue), maxValue)
    }

    return buffer
  }

  private func scheduleBuffer(_ buffer: AVAudioPCMBuffer) {
    playerNode.scheduleBuffer(buffer, at: nil, options: [], completionCallbackType: .dataPlayedBack) { [weak self] _ in
      Task { @MainActor [weak self] in
        guard let self else { return }
        // Playback complete when player stops
        if hasStartedPlayback, !playerNode.isPlaying {
          isPlaying = false
          hasStartedPlayback = false
          playbackCompletionContinuation?.resume()
          playbackCompletionContinuation = nil
        }
      }
    }
  }

  private func resetEngineIfNeeded() {
    if !engine.isRunning {
      do {
        try engine.start()
      } catch {
        Log.audio.error("Failed to restart audio engine: \(error.localizedDescription)")
      }
    }
  }
}
