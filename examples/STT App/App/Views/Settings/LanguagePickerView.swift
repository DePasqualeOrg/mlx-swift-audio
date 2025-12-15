import MLXAudio
import SwiftUI

/// Picker for selecting source language
struct LanguagePickerView: View {
  @Binding var selectedLanguage: Language?
  let isDisabled: Bool

  @State private var searchText = ""

  var body: some View {
    Picker("Language", selection: $selectedLanguage) {
      Text("Auto-detect").tag(nil as Language?)

      Section("Common") {
        ForEach(commonLanguages, id: \.self) { lang in
          Text(lang.displayName).tag(lang as Language?)
        }
      }

      Section("All Languages") {
        ForEach(filteredLanguages, id: \.self) { lang in
          Text(lang.displayName).tag(lang as Language?)
        }
      }
    }
    .disabled(isDisabled)
  }

  private var commonLanguages: [Language] {
    [.english, .chinese, .spanish, .french, .german, .japanese, .korean, .portuguese, .russian, .italian]
  }

  private var filteredLanguages: [Language] {
    let common = Set(commonLanguages)
    return Language.allCases
      .filter { !common.contains($0) }
      .sorted { $0.displayName < $1.displayName }
  }
}
