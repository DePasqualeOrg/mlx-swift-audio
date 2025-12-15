import SwiftUI

/// Combined input section with audio source selection and controls
struct InputSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState

    VStack(spacing: 16) {
      // Audio source toggle
      AudioSourcePicker(
        audioSource: $appState.audioSource,
        isDisabled: appState.isRecording || appState.isTranscribing
      )

      // Source-specific controls
      switch appState.audioSource {
        case .file:
          FileImportButton(
            selectedFileName: appState.importedFileURL?.lastPathComponent,
            onFileSelected: { url in
              appState.setImportedFile(url)
            },
            onClear: {
              appState.clearImportedFile()
            }
          )

        case .microphone:
          RecordButton(
            isRecording: appState.isRecording,
            duration: appState.recordingDuration,
            averagePower: appState.audioRecorder.averagePower,
            onStart: {
              Task {
                await appState.startRecording()
              }
            },
            onStop: {
              appState.stopRecording()
            }
          )

          // Show recorded file info
          if appState.recordingURL != nil, !appState.isRecording {
            HStack {
              Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
              Text("Recording ready")
              Spacer()
              Text(formatDuration(appState.recordingDuration))
                .foregroundStyle(.secondary)
                .monospacedDigit()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
              RoundedRectangle(cornerRadius: 8)
                .fill(Color.green.opacity(0.1))
            )
          }
      }

      // Transcribe button
      Button(action: {
        Task {
          await appState.performTask()
        }
      }) {
        HStack {
          if appState.isTranscribing || appState.isModelLoading {
            ProgressView()
              .controlSize(.small)
          } else {
            Image(systemName: appState.selectedTask.icon)
          }
          Text(appState.selectedTask.rawValue)
        }
      }
      .buttonStyle(.borderedProminent)
      .disabled(!appState.canTranscribe)

      // Model loading progress
      if appState.isModelLoading {
        VStack(spacing: 4) {
          ProgressView(value: appState.loadingProgress)
          Text("Loading \(appState.selectedModelSize.displayName)...")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
      }
    }
    .padding()
  }

  private func formatDuration(_ duration: TimeInterval) -> String {
    let minutes = Int(duration) / 60
    let seconds = Int(duration) % 60
    return String(format: "%d:%02d", minutes, seconds)
  }
}
