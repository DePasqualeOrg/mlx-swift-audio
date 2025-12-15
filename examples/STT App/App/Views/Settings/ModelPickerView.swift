import MLXAudio
import SwiftUI

/// Picker for selecting Whisper model size
struct ModelPickerView: View {
  @Binding var selectedModel: WhisperModelSize
  let isDisabled: Bool

  var body: some View {
    Picker("Model", selection: $selectedModel) {
      Section("Multilingual") {
        ForEach(multilingualModels, id: \.self) { model in
          Text(model.displayName).tag(model)
        }
      }
      Section("English Only") {
        ForEach(englishOnlyModels, id: \.self) { model in
          Text(model.displayName).tag(model)
        }
      }
    }
    .disabled(isDisabled)
  }

  private var multilingualModels: [WhisperModelSize] {
    [.tiny, .base, .small, .medium, .large, .largeTurbo]
  }

  private var englishOnlyModels: [WhisperModelSize] {
    [.tinyEn, .baseEn, .smallEn, .mediumEn]
  }
}
