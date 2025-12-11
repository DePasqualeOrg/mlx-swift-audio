import Kokoro
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

    statusMessage = "Generating..."

    do {
      lastResult = try await engineManager.generate(text: inputText, speed: speed)

      if let result = lastResult {
        statusMessage = formatResultStatus(result)
      }

      if autoPlay, let result = lastResult {
        await engineManager.play(result)
      }
    } catch is CancellationError {
      statusMessage = "Stopped"
    } catch {
      statusMessage = error.localizedDescription
    }
  }

  /// Generate with streaming
  func generateStreaming() async {
    guard canGenerate else { return }

    statusMessage = "Streaming..."

    do {
      lastResult = try await engineManager.sayStreaming(text: inputText, speed: speed)

      if let result = lastResult {
        statusMessage = formatResultStatus(result)
      }
    } catch is CancellationError {
      statusMessage = "Stopped"
    } catch {
      statusMessage = error.localizedDescription
    }
  }

  /// Play the last generated audio
  func play() async {
    guard let result = lastResult else { return }
    await engineManager.play(result)
  }

  /// Stop generation and playback
  func stop() async {
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
        return "Generated \(durationStr) sec. audio in \(timeStr) sec. (RTF: \(rtfStr)x)"
      }
      return "Generated \(durationStr) sec. audio in \(timeStr) sec."
    }

    return "Generated in \(timeStr) sec."
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

// MARK: - CosyVoice2 Speaker

extension AppState {
  /// Current prepared speaker for CosyVoice2
  var cosyVoice2Speaker: CosyVoice2Speaker? {
    get { engineManager.cosyVoice2Speaker }
    set { engineManager.cosyVoice2Speaker = newValue }
  }

  /// Whether speaker is prepared for CosyVoice2
  var isCosyVoice2SpeakerLoaded: Bool {
    engineManager.cosyVoice2Speaker != nil
  }

  /// Description of current speaker
  var cosyVoice2SpeakerDescription: String {
    engineManager.cosyVoice2Speaker?.description ?? "No speaker"
  }

  /// Prepare speaker from a URL (local file or remote)
  func prepareCosyVoice2Speaker(from url: URL, transcription: String? = nil) async throws {
    let speaker = try await engineManager.cosyVoice2Engine.prepareSpeaker(from: url, transcription: transcription)
    engineManager.cosyVoice2Speaker = speaker
  }

  /// Prepare the default speaker
  func prepareDefaultCosyVoice2Speaker() async throws {
    let speaker = try await engineManager.cosyVoice2Engine.prepareDefaultSpeaker()
    engineManager.cosyVoice2Speaker = speaker
  }

  /// Generation mode for CosyVoice2
  var cosyVoice2GenerationMode: CosyVoice2Engine.GenerationMode {
    get { engineManager.cosyVoice2Engine.generationMode }
    set { engineManager.cosyVoice2Engine.generationMode = newValue }
  }

  /// Instruct text for CosyVoice2 instruct mode
  var cosyVoice2InstructText: String {
    get { engineManager.cosyVoice2Engine.instructText }
    set { engineManager.cosyVoice2Engine.instructText = newValue }
  }

  /// Sampling parameter for CosyVoice2
  var cosyVoice2Sampling: Int {
    get { engineManager.cosyVoice2Engine.sampling }
    set { engineManager.cosyVoice2Engine.sampling = newValue }
  }

  /// Number of flow matching timesteps for CosyVoice2
  var cosyVoice2NTimesteps: Int {
    get { engineManager.cosyVoice2Engine.nTimesteps }
    set { engineManager.cosyVoice2Engine.nTimesteps = newValue }
  }

  /// Whether source audio is loaded for voice conversion
  var isCosyVoice2SourceAudioLoaded: Bool {
    engineManager.cosyVoice2Engine.isSourceAudioLoaded
  }

  /// Description of the loaded source audio
  var cosyVoice2SourceAudioDescription: String {
    engineManager.cosyVoice2Engine.sourceAudioDescription
  }

  /// Prepare source audio for voice conversion
  func prepareCosyVoice2SourceAudio(from url: URL) async throws {
    try await engineManager.cosyVoice2Engine.prepareSourceAudio(from: url)
  }

  /// Clear source audio
  func clearCosyVoice2SourceAudio() async {
    await engineManager.cosyVoice2Engine.clearSourceAudio()
  }

  /// Generate voice conversion
  func generateCosyVoice2VoiceConversion() async throws -> AudioResult {
    try await engineManager.cosyVoice2Engine.generateVoiceConversion(
      speaker: engineManager.cosyVoice2Speaker
    )
  }
}

// MARK: - OuteTTS Speaker Profile

extension AppState {
  /// Current speaker profile for OuteTTS
  var outeTTSSpeaker: OuteTTSSpeakerProfile? {
    get { engineManager.outeTTSSpeaker }
    set { engineManager.outeTTSSpeaker = newValue }
  }

  /// Whether a speaker profile is loaded for OuteTTS
  var isOuteTTSSpeakerLoaded: Bool {
    engineManager.outeTTSSpeaker != nil
  }

  /// Description of current speaker profile
  var outeTTSSpeakerDescription: String {
    if let speaker = engineManager.outeTTSSpeaker {
      let wordCount = speaker.words.count
      return "Custom speaker (\(wordCount) words)"
    }
    return "Default speaker"
  }

  /// Load speaker profile from a URL (local file)
  func loadOuteTTSSpeaker(from url: URL) async throws {
    let speaker = try await OuteTTSSpeakerProfile.load(from: url.path)
    engineManager.outeTTSSpeaker = speaker
  }

  /// Load the default speaker profile
  func loadDefaultOuteTTSSpeaker() async throws {
    // Setting to nil means the engine will use the bundled default
    engineManager.outeTTSSpeaker = nil
  }

  /// Create and load speaker profile from an audio file
  func createOuteTTSSpeaker(from url: URL) async throws {
    let speaker = try await engineManager.outeTTSEngine.createSpeakerProfile(from: url)
    engineManager.outeTTSSpeaker = speaker
  }

  /// Temperature for OuteTTS
  var outeTTSTemperature: Float {
    get { engineManager.outeTTSEngine.temperature }
    set { engineManager.outeTTSEngine.temperature = newValue }
  }

  /// Top-P for OuteTTS
  var outeTTSTopP: Float {
    get { engineManager.outeTTSEngine.topP }
    set { engineManager.outeTTSEngine.topP = newValue }
  }
}
