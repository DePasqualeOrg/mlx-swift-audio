import AVFoundation
import Foundation

/// Manages microphone recording for speech-to-text
@MainActor
@Observable
final class AudioRecorder: NSObject {
  // MARK: - State

  /// Whether currently recording
  private(set) var isRecording: Bool = false

  /// Recording duration in seconds
  private(set) var duration: TimeInterval = 0

  /// URL of the last recording
  private(set) var recordingURL: URL?

  /// Last error that occurred
  private(set) var error: Error?

  // MARK: - Audio Level Metering

  /// Average audio power level
  private(set) var averagePower: Float = -160

  /// Peak audio power level
  private(set) var peakPower: Float = -160

  // MARK: - Private Properties

  private var audioRecorder: AVAudioRecorder?
  private var durationTimer: Timer?
  private var meteringTimer: Timer?

  // MARK: - Permission

  /// Request microphone permission
  func requestPermission() async -> Bool {
    #if os(macOS)
    let status = AVCaptureDevice.authorizationStatus(for: .audio)
    switch status {
      case .authorized:
      return true
      case .notDetermined:
      return await AVCaptureDevice.requestAccess(for: .audio)
      default:
      return false
    }
    #else
    return await AVAudioApplication.requestRecordPermission()
    #endif
  }

  // MARK: - Recording

  /// Start recording audio
  func startRecording() throws {
    guard !isRecording else { return }

    // Create temporary file URL
    let tempDir = FileManager.default.temporaryDirectory
    let fileName = "recording_\(Date().timeIntervalSince1970).wav"
    let fileURL = tempDir.appendingPathComponent(fileName)

    // Configure recording settings for Whisper (16kHz mono)
    let settings: [String: Any] = [
      AVFormatIDKey: Int(kAudioFormatLinearPCM),
      AVSampleRateKey: 16000.0,
      AVNumberOfChannelsKey: 1,
      AVLinearPCMBitDepthKey: 16,
      AVLinearPCMIsFloatKey: false,
      AVLinearPCMIsBigEndianKey: false,
    ]

    do {
      #if os(iOS)
      // Configure audio session for recording
      let session = AVAudioSession.sharedInstance()
      try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
      try session.setActive(true)
      #endif

      audioRecorder = try AVAudioRecorder(url: fileURL, settings: settings)
      audioRecorder?.isMeteringEnabled = true
      audioRecorder?.delegate = self

      guard audioRecorder?.record() == true else {
        throw RecordingError.failedToStart
      }

      recordingURL = fileURL
      isRecording = true
      duration = 0
      error = nil

      startTimers()
    } catch {
      self.error = error
      throw error
    }
  }

  /// Stop recording
  func stopRecording() {
    guard isRecording else { return }

    stopTimers()
    audioRecorder?.stop()
    isRecording = false

    #if os(iOS)
    try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    #endif
  }

  /// Delete the recorded file
  func deleteRecording() {
    if let url = recordingURL {
      try? FileManager.default.removeItem(at: url)
      recordingURL = nil
    }
  }

  // MARK: - Private Helpers

  private func startTimers() {
    // Duration timer
    durationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
      Task { @MainActor in
        guard let self, self.isRecording else { return }
        self.duration = self.audioRecorder?.currentTime ?? 0
      }
    }

    // Metering timer for audio levels
    meteringTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
      Task { @MainActor in
        guard let self, let recorder = self.audioRecorder, self.isRecording else { return }
        recorder.updateMeters()
        self.averagePower = recorder.averagePower(forChannel: 0)
        self.peakPower = recorder.peakPower(forChannel: 0)
      }
    }
  }

  private func stopTimers() {
    durationTimer?.invalidate()
    durationTimer = nil
    meteringTimer?.invalidate()
    meteringTimer = nil
    averagePower = -160
    peakPower = -160
  }
}

// MARK: - AVAudioRecorderDelegate

extension AudioRecorder: AVAudioRecorderDelegate {
  nonisolated func audioRecorderDidFinishRecording(_: AVAudioRecorder, successfully flag: Bool) {
    Task { @MainActor in
      if !flag {
        error = RecordingError.recordingFailed
      }
      isRecording = false
      stopTimers()
    }
  }

  nonisolated func audioRecorderEncodeErrorDidOccur(_: AVAudioRecorder, error: Error?) {
    Task { @MainActor in
      self.error = error ?? RecordingError.encodingFailed
      isRecording = false
      stopTimers()
    }
  }
}

// MARK: - Errors

enum RecordingError: LocalizedError {
  case failedToStart
  case recordingFailed
  case encodingFailed
  case permissionDenied

  var errorDescription: String? {
    switch self {
      case .failedToStart:
        "Failed to start recording"
      case .recordingFailed:
        "Recording failed"
      case .encodingFailed:
        "Audio encoding failed"
      case .permissionDenied:
        "Microphone permission denied"
    }
  }
}
