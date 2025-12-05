import MLXAudio
import SwiftUI

/// Central state management for the TTS application
@MainActor
@Observable
final class AppState {
  // MARK: - Dependencies

  let engineManager: EngineManager

  // MARK: - User Input

  var inputText: String = "How are you doing today?"

  /// Speech speed multiplier (Kokoro only)
  var speed: Float = TTSConstants.Speed.default

  // MARK: - UI State

  var showInspector: Bool = true

  var autoPlay: Bool = true

  var statusMessage: String = ""

  // MARK: - Generated Output

  private(set) var lastResult: AudioResult?

  /// Flag to prevent auto-play after user stops
  private var stopRequested: Bool = false

  // MARK: - Delegated State (from EngineManager)

  var selectedProvider: TTSProvider { engineManager.selectedProvider }
  var isLoaded: Bool { engineManager.isLoaded }
  var isGenerating: Bool { engineManager.isGenerating }
  var isPlaying: Bool { engineManager.isPlaying }
  var isModelLoading: Bool { engineManager.isLoading }
  var loadingProgress: Double { engineManager.loadingProgress }
  var error: TTSError? { engineManager.error }
  var generationTime: TimeInterval { engineManager.generationTime }
  var lastGeneratedAudioURL: URL? { engineManager.lastGeneratedAudioURL }
  var streamingGranularity: StreamingGranularity { engineManager.currentEngine.streamingGranularity }

  var canGenerate: Bool {
    isLoaded && !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isGenerating
  }

  // MARK: - Initialization

  init() {
    engineManager = EngineManager()
  }

  // MARK: - Provider Management

  func selectProvider(_ provider: TTSProvider) async {
    guard provider != selectedProvider else { return }

    await engineManager.selectProvider(provider)
    lastResult = nil
    statusMessage = provider.statusMessage
  }

  // MARK: - Engine Operations

  func loadEngine() async throws {
    do {
      try await engineManager.loadEngine()
    } catch {
      statusMessage = error.localizedDescription
      throw error
    }
  }

  /// Generate audio from the current input text
  func generate() async {
    guard canGenerate else { return }

    stopRequested = false
    statusMessage = "Generating..."

    do {
      lastResult = try await engineManager.generate(text: inputText, speed: speed)

      guard !stopRequested else { return }

      if let result = lastResult {
        statusMessage = formatResultStatus(result)
      }

      if autoPlay, let result = lastResult {
        await engineManager.play(result)
      }
    } catch {
      statusMessage = error.localizedDescription
    }
  }

  /// Generate with streaming
  func generateStreaming() async {
    guard canGenerate else { return }

    stopRequested = false
    statusMessage = "Streaming..."

    do {
      lastResult = try await engineManager.sayStreaming(text: inputText, speed: speed)

      if let result = lastResult {
        statusMessage = formatResultStatus(result)
      }
    } catch {
      if !stopRequested { statusMessage = error.localizedDescription }
    }
  }

  /// Play the last generated audio
  func play() async {
    guard let result = lastResult else { return }
    await engineManager.play(result)
  }

  /// Stop generation and playback
  func stop() async {
    stopRequested = true
    await engineManager.stop()
    statusMessage = "Stopped"
  }

  // MARK: - Private Helpers

  private func formatResultStatus(_ result: AudioResult) -> String {
    let timeStr = result.processingTime.formatted(decimals: 2)

    if let duration = result.duration {
      let durationStr = duration.formatted(decimals: 2)
      if let rtf = result.realTimeFactor {
        let rtfStr = rtf.formatted(decimals: 2)
        return "Generated \(durationStr)s audio in \(timeStr)s (RTF: \(rtfStr)x)"
      }
      return "Generated \(durationStr)s audio in \(timeStr)s"
    }

    return "Generated in \(timeStr)s"
  }
}

// MARK: - Voice Selection

extension AppState {
  var kokoroVoice: KokoroEngine.Voice {
    get { engineManager.kokoroVoice }
    set { engineManager.kokoroVoice = newValue }
  }

  var orpheusVoice: OrpheusEngine.Voice {
    get { engineManager.orpheusVoice }
    set { engineManager.orpheusVoice = newValue }
  }

  var marvisVoice: MarvisEngine.Voice {
    get { engineManager.marvisVoice }
    set { engineManager.marvisVoice = newValue }
  }
}

// MARK: - Engine-Specific Property Access

extension AppState {
  /// Quality level for Marvis
  var marvisQualityLevel: MarvisEngine.QualityLevel {
    get { engineManager.marvisEngine.qualityLevel }
    set { engineManager.marvisEngine.qualityLevel = newValue }
  }

  /// Streaming interval for Marvis
  var streamingInterval: Double {
    get { engineManager.marvisEngine.streamingInterval }
    set { engineManager.marvisEngine.streamingInterval = newValue }
  }

  /// Temperature for Orpheus
  var orpheusTemperature: Float {
    get { engineManager.orpheusEngine.temperature }
    set { engineManager.orpheusEngine.temperature = newValue }
  }

  /// Top-P for Orpheus
  var orpheusTopP: Float {
    get { engineManager.orpheusEngine.topP }
    set { engineManager.orpheusEngine.topP = newValue }
  }
}

// MARK: - Chatterbox Reference Audio

extension AppState {
  /// Current prepared reference audio for Chatterbox
  var chatterboxReferenceAudio: ChatterboxReferenceAudio? {
    get { engineManager.chatterboxReferenceAudio }
    set { engineManager.chatterboxReferenceAudio = newValue }
  }

  /// Whether reference audio is prepared for Chatterbox
  var isChatterboxReferenceAudioLoaded: Bool {
    engineManager.chatterboxReferenceAudio != nil
  }

  /// Description of current reference audio
  var chatterboxReferenceAudioDescription: String {
    engineManager.chatterboxReferenceAudio?.description ?? "No reference audio"
  }

  /// Prepare reference audio from a URL (local file or remote)
  func prepareChatterboxReferenceAudio(from url: URL) async throws {
    let ref = try await engineManager.chatterboxEngine.prepareReferenceAudio(from: url)
    engineManager.chatterboxReferenceAudio = ref
  }

  /// Prepare the default reference audio
  func prepareDefaultChatterboxReferenceAudio() async throws {
    let ref = try await engineManager.chatterboxEngine.prepareDefaultReferenceAudio()
    engineManager.chatterboxReferenceAudio = ref
  }

  /// Emotion exaggeration for Chatterbox
  var chatterboxExaggeration: Float {
    get { engineManager.chatterboxEngine.exaggeration }
    set { engineManager.chatterboxEngine.exaggeration = newValue }
  }

  /// Temperature for Chatterbox
  var chatterboxTemperature: Float {
    get { engineManager.chatterboxEngine.temperature }
    set { engineManager.chatterboxEngine.temperature = newValue }
  }

  /// CFG weight for Chatterbox
  var chatterboxCFGWeight: Float {
    get { engineManager.chatterboxEngine.cfgWeight }
    set { engineManager.chatterboxEngine.cfgWeight = newValue }
  }
}
