import MLXAudio
import SwiftUI

struct SettingsSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    VStack(alignment: .leading, spacing: 20) {
      // Speed (Kokoro only)
      if appState.selectedProvider.supportsSpeed {
        SpeedSliderView(
          speed: $appState.speed,
          isDisabled: appState.isGenerating,
        )
      }

      // Quality Level (Marvis only)
      if appState.selectedProvider.supportsQualityLevels {
        QualityLevelSection()
      }

      // Reference Audio (OuteTTS)
      if appState.selectedProvider == .outetts {
        ReferenceAudioPicker(
          config: ReferenceAudioPickerConfig(
            title: "Speaker Profile",
            infoText: "Create a speaker profile from reference audio with automatic transcription. Use 5-15 seconds of clear speech.",
            loadingText: "Processing audio...",
          ),
          statusDescription: appState.outeTTSSpeakerDescription,
          isLoaded: appState.isOuteTTSSpeakerLoaded,
          onLoadDefault: {
            try await appState.loadDefaultOuteTTSSpeaker()
          },
          onLoadFromFile: { url in
            try await appState.createOuteTTSSpeaker(from: url)
          },
          onLoadFromURL: { url in
            try await appState.createOuteTTSSpeaker(from: url)
          },
        )
      }

      // Reference Audio (Chatterbox)
      if appState.selectedProvider == .chatterbox {
        ReferenceAudioPicker(
          config: ReferenceAudioPickerConfig(
            title: "Reference Audio",
            infoText: "Chatterbox uses reference audio to match voice characteristics. For best results, use 10+ seconds of clear speech."
          ),
          statusDescription: appState.chatterboxReferenceAudioDescription,
          isLoaded: appState.isChatterboxReferenceAudioLoaded,
          onLoadDefault: {
            try await appState.prepareDefaultChatterboxReferenceAudio()
          },
          onLoadFromFile: { url in
            try await appState.prepareChatterboxReferenceAudio(from: url)
          },
          onLoadFromURL: { url in
            try await appState.prepareChatterboxReferenceAudio(from: url)
          }
        )

        // Emotion exaggeration slider
        VStack(alignment: .leading, spacing: 4) {
          HStack {
            Text("Emotion")
            Spacer()
            Text(String(format: "%.1f", appState.chatterboxExaggeration))
              .foregroundStyle(.secondary)
          }
          Slider(value: $appState.chatterboxExaggeration, in: 0 ... 2, step: 0.1)
        }
      }

      // Speaker (CosyVoice2)
      if appState.selectedProvider == .cosyVoice2 {
        ReferenceAudioPicker(
          config: ReferenceAudioPickerConfig(
            title: "Speaker",
            infoText: "CosyVoice2 matches voice characteristics from reference audio. Use 5-30 seconds of clear speech for best results."
          ),
          statusDescription: appState.cosyVoice2SpeakerDescription,
          isLoaded: appState.isCosyVoice2SpeakerLoaded,
          onLoadDefault: {
            try await appState.prepareDefaultCosyVoice2Speaker()
          },
          onLoadFromFile: { url in
            try await appState.prepareCosyVoice2Speaker(from: url)
          },
          onLoadFromURL: { url in
            try await appState.prepareCosyVoice2Speaker(from: url)
          }
        )

        // Generation mode picker
        CosyVoice2ModeSection()
      }

      // Provider Status Message
      if !appState.selectedProvider.statusMessage.isEmpty {
        Text(appState.selectedProvider.statusMessage)
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }
  }
}

/// Quality level section for Marvis
private struct QualityLevelSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    HStack {
      Picker("Quality", selection: $appState.marvisQualityLevel) {
        ForEach(MarvisEngine.QualityLevel.allCases, id: \.self) { level in
          Text("\(level.rawValue.capitalized) (\(level.codebookCount) codebooks)")
            .tag(level)
        }
      }
      .pickerStyle(.menu)
    }
  }
}

/// Generation mode section for CosyVoice2
private struct CosyVoice2ModeSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    VStack(alignment: .leading, spacing: 12) {
      // Mode picker
      HStack {
        Picker("Mode", selection: $appState.cosyVoice2GenerationMode) {
          ForEach(CosyVoice2Engine.GenerationMode.allCases, id: \.self) { mode in
            Text(mode.rawValue)
              .tag(mode)
          }
        }
        .pickerStyle(.menu)
      }

      // Mode description
      Text(appState.cosyVoice2GenerationMode.description)
        .font(.caption)
        .foregroundStyle(.secondary)

      // Mode-specific UI
      switch appState.cosyVoice2GenerationMode {
        case .instruct:
          CosyVoice2InstructSection()
        case .voiceConversion:
          CosyVoice2SourceAudioSection()
        case .zeroShot:
          CosyVoice2ZeroShotSection()
        case .crossLingual:
          EmptyView()
      }
    }
  }
}

/// Instruct text input for CosyVoice2 instruct mode
private struct CosyVoice2InstructSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    VStack(alignment: .leading, spacing: 4) {
      Text("Style Instructions")
        .font(.caption)
        .foregroundStyle(.secondary)
      TextField("e.g., Speak slowly and calmly", text: $appState.cosyVoice2InstructText)
        .textFieldStyle(.roundedBorder)
    }
  }
}

/// Zero-shot mode section showing transcription status
private struct CosyVoice2ZeroShotSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      if let speaker = appState.cosyVoice2Speaker {
        if let transcription = speaker.transcription {
          Text("Speaker Transcription")
            .font(.caption)
            .foregroundStyle(.secondary)
          Text(transcription)
            .font(.caption)
            .foregroundStyle(.primary)
            .lineLimit(3)
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.background.secondary)
            .clipShape(RoundedRectangle(cornerRadius: 6))
        } else {
          Text("No transcription available. Speaker audio will be auto-transcribed when loaded.")
            .font(.caption)
            .foregroundStyle(.orange)
        }
      }
    }
  }
}

/// Source audio picker for CosyVoice2 voice conversion mode
private struct CosyVoice2SourceAudioSection: View {
  @Environment(AppState.self) private var appState
  @State private var isLoading = false
  @State private var showFilePicker = false
  @State private var errorMessage: String?

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text("Source Audio")
          .font(.subheadline.weight(.medium))
        Spacer()
        if isLoading {
          ProgressView()
            .scaleEffect(0.7)
        }
      }

      Text("Audio to convert to the target speaker's voice")
        .font(.caption)
        .foregroundStyle(.secondary)

      HStack {
        Text(appState.cosyVoice2SourceAudioDescription)
          .font(.caption)
          .foregroundStyle(appState.isCosyVoice2SourceAudioLoaded ? .primary : .secondary)
        Spacer()
        Button("Choose File") {
          showFilePicker = true
        }
        .buttonStyle(.bordered)
        .disabled(isLoading)
      }

      if let errorMessage {
        Text(errorMessage)
          .font(.caption)
          .foregroundStyle(.red)
      }
    }
    .fileImporter(
      isPresented: $showFilePicker,
      allowedContentTypes: [.audio],
      allowsMultipleSelection: false
    ) { result in
      Task {
        await handleFileSelection(result)
      }
    }
  }

  private func handleFileSelection(_ result: Result<[URL], Error>) async {
    do {
      let urls = try result.get()
      guard let url = urls.first else { return }

      isLoading = true
      errorMessage = nil

      // Start accessing the security-scoped resource
      guard url.startAccessingSecurityScopedResource() else {
        throw NSError(domain: "CosyVoice2", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unable to access the selected file"])
      }
      defer { url.stopAccessingSecurityScopedResource() }

      try await appState.prepareCosyVoice2SourceAudio(from: url)
      isLoading = false
    } catch {
      isLoading = false
      errorMessage = error.localizedDescription
    }
  }
}
