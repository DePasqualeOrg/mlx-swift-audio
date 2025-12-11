import Kokoro
import MLXAudio
import SwiftUI

struct VoicePickerView: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    switch appState.selectedProvider {
      case .kokoro:
        KokoroVoicePicker(voice: $appState.kokoroVoice)
      case .orpheus:
        OrpheusVoicePicker(voice: $appState.orpheusVoice)
      case .marvis:
        MarvisVoicePicker(voice: $appState.marvisVoice)
      case .outetts:
        EmptyView()
      case .chatterbox:
        EmptyView()
      case .cosyVoice2:
        EmptyView()
    }
  }
}

// MARK: - Kokoro Voice Picker

private struct KokoroVoicePicker: View {
  @Binding var voice: KokoroEngine.Voice

  var body: some View {
    Menu {
      ForEach(KokoroEngine.Voice.allCases, id: \.self) { v in
        Button {
          voice = v
        } label: {
          if v == voice {
            Label(v.displayName, systemImage: "checkmark")
          } else {
            Text(v.displayName)
          }
        }
      }
    } label: {
      voiceLabel(voice.displayName)
    }
    .buttonStyle(.plain)
  }
}

// MARK: - Orpheus Voice Picker

private struct OrpheusVoicePicker: View {
  @Binding var voice: OrpheusEngine.Voice

  var body: some View {
    Menu {
      ForEach(OrpheusEngine.Voice.allCases, id: \.self) { v in
        Button {
          voice = v
        } label: {
          if v == voice {
            Label(v.rawValue.capitalized, systemImage: "checkmark")
          } else {
            Text(v.rawValue.capitalized)
          }
        }
      }
    } label: {
      voiceLabel(voice.rawValue.capitalized)
    }
    .buttonStyle(.plain)
  }
}

// MARK: - Marvis Voice Picker

private struct MarvisVoicePicker: View {
  @Binding var voice: MarvisEngine.Voice

  var body: some View {
    Menu {
      ForEach(MarvisEngine.Voice.allCases, id: \.self) { v in
        Button {
          voice = v
        } label: {
          if v == voice {
            Label(displayName(for: v), systemImage: "checkmark")
          } else {
            Text(displayName(for: v))
          }
        }
      }
    } label: {
      voiceLabel(displayName(for: voice))
    }
    .buttonStyle(.plain)
  }

  private func displayName(for voice: MarvisEngine.Voice) -> String {
    switch voice {
      case .conversationalA: "Conversational A"
      case .conversationalB: "Conversational B"
    }
  }
}

// MARK: - Shared Styling

@ViewBuilder
private func voiceLabel(_ text: String) -> some View {
  HStack(spacing: 6) {
    Text(text)
      .lineLimit(1)
    Image(systemName: "chevron.up.chevron.down")
      .font(.caption)
  }
  .padding(.horizontal, 12)
  .padding(.vertical, 8)
  .background(.background.secondary)
  .clipShape(RoundedRectangle(cornerRadius: 8))
}

// MARK: - Voice Display Name Extension

extension KokoroEngine.Voice {
  var displayName: String {
    // Format: afHeart -> Heart
    let name = String(describing: self)
    guard name.count > 2 else { return name.capitalized }
    return String(name.dropFirst(2)).capitalized
  }
}
