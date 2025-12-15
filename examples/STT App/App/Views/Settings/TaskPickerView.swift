import MLXAudio
import SwiftUI

/// Picker for selecting STT task type
struct TaskPickerView: View {
  @Binding var selectedTask: STTTask
  let isEnglishOnlyModel: Bool
  let isDisabled: Bool

  var body: some View {
    Picker("Task", selection: $selectedTask) {
      ForEach(availableTasks, id: \.self) { task in
        Text(task.rawValue).tag(task)
      }
    }
    .pickerStyle(.menu)
    .disabled(isDisabled)
  }

  private var availableTasks: [STTTask] {
    if isEnglishOnlyModel {
      // English-only models can't translate
      return [.transcribe, .detectLanguage]
    }
    return STTTask.allCases
  }
}

extension STTTask {
  var icon: String {
    switch self {
      case .transcribe: "text.alignleft"
      case .translate: "globe"
      case .detectLanguage: "questionmark.circle"
    }
  }

  var description: String {
    switch self {
      case .transcribe:
        "Transcribe speech in the original language"
      case .translate:
        "Translate speech to English"
      case .detectLanguage:
        "Detect the spoken language"
    }
  }
}
