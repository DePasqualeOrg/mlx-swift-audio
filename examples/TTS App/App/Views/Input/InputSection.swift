import SwiftUI

struct InputSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    VStack(spacing: 16) {
      // Text Input
      TextInputView(
        text: $appState.inputText,
        isDisabled: appState.isGenerating,
      )

      // Auto-play toggle
      Toggle("Auto-play", isOn: $appState.autoPlay)
        .toggleStyle(.switch)

      // Generate Button
      GenerateButton(
        isLoading: appState.isModelLoading,
        isGenerating: appState.isGenerating,
        canGenerate: !appState.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
        onGenerate: {
          Task {
            if !appState.isLoaded {
              try? await appState.loadEngine()
            }
            await appState.generate()
          }
        },
        onStop: {
          Task { await appState.stop() }
        },
      )

      // Streaming option
      VStack {
        Button {
          Task {
            if !appState.isLoaded {
              try? await appState.loadEngine()
            }
            await appState.generateStreaming()
          }
        } label: {
          HStack(spacing: 8) {
            Image(systemName: "waveform.path")
            Text("Stream")
          }
          .padding(.vertical, 4)
        }
        .buttonStyle(.bordered)
        .disabled(appState.isModelLoading || appState.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        Text(appState.streamingGranularity.shortDescription)
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }
  }
}
