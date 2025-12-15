import Foundation
import MLXAudio
import SwiftUI

/// Audio input source
enum AudioSource: String, CaseIterable {
  case file = "File"
  case microphone = "Microphone"
}

/// STT task type
enum STTTask: String, CaseIterable {
  case transcribe = "Transcribe"
  case translate = "Translate"
  case detectLanguage = "Detect Language"
}

/// Central state management for the STT application
@MainActor
@Observable
final class AppState {
  // MARK: - Dependencies

  let engineManager: EngineManager
  let audioRecorder: AudioRecorder

  // MARK: - Configuration

  /// Selected Whisper model size
  var selectedModelSize: WhisperModelSize = .base

  /// Selected quantization level
  var selectedQuantization: WhisperQuantization = .q4

  /// Selected task (transcribe, translate, detect)
  var selectedTask: STTTask = .transcribe

  /// Selected source language (nil = auto-detect)
  var selectedLanguage: Language?

  /// Timestamp granularity (segment recommended - .none can cause hallucinations)
  var timestampGranularity: TimestampGranularity = .segment

  // MARK: - Audio Source

  /// Current audio input source
  var audioSource: AudioSource = .file

  /// URL of imported audio file
  var importedFileURL: URL?

  // MARK: - UI State

  /// Whether settings inspector is visible
  var showInspector: Bool = true

  /// Status message to display
  var statusMessage: String = ""

  // MARK: - Results

  /// Last transcription result
  private(set) var lastResult: TranscriptionResult?

  /// Streaming segments during recording
  private(set) var streamingSegments: [TranscriptionSegment] = []

  /// Detected language result (for detect language task)
  private(set) var detectedLanguageResult: (language: Language, confidence: Float)?

  // MARK: - Delegated State (from EngineManager)

  var isLoaded: Bool { engineManager.isLoaded }
  var isTranscribing: Bool { engineManager.isTranscribing }
  var isModelLoading: Bool { engineManager.isLoading }
  var loadingProgress: Double { engineManager.loadingProgress }
  var error: STTError? { engineManager.error }
  var transcriptionTime: TimeInterval { engineManager.transcriptionTime }

  // MARK: - Recording State (from AudioRecorder)

  var isRecording: Bool { audioRecorder.isRecording }
  var recordingDuration: TimeInterval { audioRecorder.duration }
  var recordingURL: URL? { audioRecorder.recordingURL }

  // MARK: - Computed Properties

  var canTranscribe: Bool {
    guard !isTranscribing, !isRecording, !isModelLoading else { return false }
    switch audioSource {
      case .file:
        return importedFileURL != nil
      case .microphone:
        return recordingURL != nil
    }
  }

  var canStartRecording: Bool {
    !isTranscribing && !isRecording && !isModelLoading && audioSource == .microphone
  }

  var needsModelReload: Bool {
    engineManager.needsReload(modelSize: selectedModelSize, quantization: selectedQuantization)
  }

  var audioURLToProcess: URL? {
    switch audioSource {
      case .file:
        importedFileURL
      case .microphone:
        recordingURL
    }
  }

  // MARK: - Initialization

  init() {
    engineManager = EngineManager()
    audioRecorder = AudioRecorder()
  }

  // MARK: - Engine Operations

  /// Load the engine with current configuration
  func loadEngine() async throws {
    do {
      try await engineManager.loadEngine(
        modelSize: selectedModelSize,
        quantization: selectedQuantization
      )
      statusMessage = "\(selectedModelSize.displayName) (\(selectedQuantization.rawValue)) loaded"
    } catch {
      statusMessage = error.localizedDescription
      throw error
    }
  }

  /// Perform transcription/translation/detection based on selected task
  func performTask() async {
    guard let url = audioURLToProcess else {
      statusMessage = "No audio file selected"
      return
    }

    // Load engine if not loaded or config changed
    if !isLoaded || needsModelReload {
      do {
        try await loadEngine()
      } catch {
        return
      }
    }

    statusMessage = "Processing..."
    lastResult = nil
    detectedLanguageResult = nil

    do {
      switch selectedTask {
        case .transcribe:
          lastResult = try await engineManager.transcribe(
            url: url,
            language: selectedLanguage,
            timestamps: timestampGranularity
          )
          if let result = lastResult {
            statusMessage = formatResultStatus(result)
          }

        case .translate:
          lastResult = try await engineManager.translate(
            url: url,
            language: selectedLanguage,
            timestamps: timestampGranularity
          )
          if let result = lastResult {
            statusMessage = formatResultStatus(result)
          }

        case .detectLanguage:
          let (language, confidence) = try await engineManager.detectLanguage(url: url)
          detectedLanguageResult = (language, confidence)
          let percentage = Int(confidence * 100)
          statusMessage = "Detected: \(language.displayName) (\(percentage)% confidence)"
      }
    } catch is CancellationError {
      statusMessage = "Stopped"
    } catch {
      statusMessage = error.localizedDescription
    }
  }

  /// Stop current transcription
  func stop() async {
    await engineManager.stop()
    statusMessage = "Stopped"
  }

  // MARK: - Recording Operations

  private var streamingTask: Task<Void, Never>?

  /// Start recording from microphone
  func startRecording() async {
    let hasPermission = await audioRecorder.requestPermission()
    guard hasPermission else {
      statusMessage = "Microphone permission denied"
      return
    }

    // Load model if needed for streaming transcription
    if !isLoaded || needsModelReload {
      do {
        try await loadEngine()
      } catch {
        // Continue without streaming - model will load when transcribing
      }
    }

    do {
      try audioRecorder.startRecording()
      statusMessage = "Recording..."
      lastResult = nil
      streamingSegments = []

      // Start streaming transcription if model is loaded
      if isLoaded {
        startStreamingTranscription()
      }
    } catch {
      statusMessage = "Failed to start recording: \(error.localizedDescription)"
    }
  }

  /// Stop recording
  func stopRecording() {
    streamingTask?.cancel()
    streamingTask = nil
    audioRecorder.stopRecording()
    statusMessage = "Recording stopped. Ready to transcribe."
  }

  /// Start periodic transcription during recording
  private func startStreamingTranscription() {
    streamingTask = Task {
      // Wait for initial audio accumulation
      try? await Task.sleep(for: .seconds(3))

      while !Task.isCancelled, isRecording {
        // Transcribe current recording buffer
        if let url = audioRecorder.recordingURL {
          do {
            let result = try await engineManager.transcribe(
              url: url,
              language: selectedLanguage,
              timestamps: .segment
            )
            // Update streaming segments
            if !Task.isCancelled {
              streamingSegments = result.segments
              statusMessage = "Recording... (\(Int(recordingDuration))s)"
            }
          } catch {
            // Ignore errors during streaming - file may be in use
          }
        }

        // Wait before next transcription
        try? await Task.sleep(for: .seconds(5))
      }
    }
  }

  /// Append new segments from streaming transcription
  func appendStreamingSegments(_ newSegments: [TranscriptionSegment]) {
    // Merge segments avoiding duplicates based on timing
    for segment in newSegments {
      if !streamingSegments.contains(where: { abs($0.start - segment.start) < 0.5 }) {
        streamingSegments.append(segment)
      }
    }
    streamingSegments.sort { $0.start < $1.start }
  }

  // MARK: - File Import

  /// Set imported file URL
  func setImportedFile(_ url: URL) {
    importedFileURL = url
    statusMessage = "Selected: \(url.lastPathComponent)"
    lastResult = nil
    detectedLanguageResult = nil
  }

  /// Clear imported file
  func clearImportedFile() {
    importedFileURL = nil
    lastResult = nil
    detectedLanguageResult = nil
    statusMessage = ""
  }

  // MARK: - Configuration Changes

  /// Handle model size change
  func setModelSize(_ size: WhisperModelSize) {
    guard size != selectedModelSize else { return }
    selectedModelSize = size
    lastResult = nil
    detectedLanguageResult = nil
  }

  /// Handle quantization change
  func setQuantization(_ quantization: WhisperQuantization) {
    guard quantization != selectedQuantization else { return }
    selectedQuantization = quantization
    lastResult = nil
    detectedLanguageResult = nil
  }

  // MARK: - Private Helpers

  private func formatResultStatus(_ result: TranscriptionResult) -> String {
    let timeStr = String(format: "%.2f", result.processingTime)
    let durationStr = String(format: "%.2f", result.duration)
    let rtfStr = String(format: "%.2f", result.realTimeFactor)
    return "Processed \(durationStr)s audio in \(timeStr)s (RTF: \(rtfStr)x)"
  }
}

// MARK: - Model Size Display

extension WhisperModelSize {
  var displayName: String {
    switch self {
      case .tiny: "Tiny (39M)"
      case .tinyEn: "Tiny.en (39M)"
      case .base: "Base (74M)"
      case .baseEn: "Base.en (74M)"
      case .small: "Small (244M)"
      case .smallEn: "Small.en (244M)"
      case .medium: "Medium (769M)"
      case .mediumEn: "Medium.en (769M)"
      case .large: "Large-v3 (1.5B)"
      case .largeTurbo: "Large-v3-Turbo (809M)"
    }
  }

  var isEnglishOnly: Bool {
    switch self {
      case .tinyEn, .baseEn, .smallEn, .mediumEn:
        true
      default:
        false
    }
  }
}
