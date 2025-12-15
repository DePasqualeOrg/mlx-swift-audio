import MLXAudio
import SwiftUI

/// Combined settings view for STT configuration
struct SettingsSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState

    Form {
      Section {
        ModelPickerView(
          selectedModel: Binding(
            get: { appState.selectedModelSize },
            set: { appState.setModelSize($0) }
          ),
          isDisabled: appState.isTranscribing
        )

        QuantizationPickerView(
          selectedQuantization: Binding(
            get: { appState.selectedQuantization },
            set: { appState.setQuantization($0) }
          ),
          isDisabled: appState.isTranscribing
        )
      }

      Section {
        TaskPickerView(
          selectedTask: $appState.selectedTask,
          isEnglishOnlyModel: appState.selectedModelSize.isEnglishOnly,
          isDisabled: appState.isTranscribing
        )

        Text(appState.selectedTask.description)
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      Section {
        LanguagePickerView(
          selectedLanguage: $appState.selectedLanguage,
          isDisabled: appState.isTranscribing || appState.selectedTask == .detectLanguage
        )

        if appState.selectedTask != .detectLanguage {
          TimestampPickerView(
            selectedTimestamps: $appState.timestampGranularity,
            isDisabled: appState.isTranscribing
          )

          Text(appState.timestampGranularity.description)
            .font(.caption)
            .foregroundStyle(.secondary)
        }
      }
    }
    .formStyle(.grouped)
  }
}
