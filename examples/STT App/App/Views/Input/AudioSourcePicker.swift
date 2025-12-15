import SwiftUI

/// Toggle between file import and microphone recording
struct AudioSourcePicker: View {
  @Binding var audioSource: AudioSource
  let isDisabled: Bool

  var body: some View {
    Picker("Audio Source", selection: $audioSource) {
      ForEach(AudioSource.allCases, id: \.self) { source in
        Label(source.rawValue, systemImage: source.icon).tag(source)
      }
    }
    .pickerStyle(.segmented)
    .disabled(isDisabled)
  }
}

extension AudioSource {
  var icon: String {
    switch self {
      case .file: "doc.fill"
      case .microphone: "mic.fill"
    }
  }
}
