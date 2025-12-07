import SwiftUI
import UniformTypeIdentifiers

/// View for selecting and managing reference audio for Chatterbox TTS
struct ReferenceAudioView: View {
  @Environment(AppState.self) private var appState

  @State private var isShowingFilePicker = false
  @State private var isShowingURLInput = false
  @State private var customURL = ""
  @State private var isLoading = false
  @State private var errorMessage: String?

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Reference Audio")
        .font(.headline)

      // Current selection display
      HStack {
        Image(systemName: audioStatusIcon)
          .foregroundStyle(audioStatusColor)

        VStack(alignment: .leading, spacing: 2) {
          Text(appState.chatterboxReferenceAudioDescription)
            .font(.subheadline)

          if isLoading {
            Text("Preparing...")
              .font(.caption)
              .foregroundStyle(.secondary)
          } else if let error = errorMessage {
            Text(error)
              .font(.caption)
              .foregroundStyle(.red)
          }
        }

        Spacer()
      }
      .padding(8)
      .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))

      // Action buttons
      HStack(spacing: 12) {
        // Default button
        Button {
          Task { await loadDefaultAudio() }
        } label: {
          Label("Default", systemImage: "waveform")
        }
        .buttonStyle(.bordered)
        .disabled(isLoading)

        // File picker button
        Button {
          isShowingFilePicker = true
        } label: {
          Label("File", systemImage: "folder")
        }
        .buttonStyle(.bordered)
        .disabled(isLoading)

        // URL input button
        Button {
          isShowingURLInput = true
        } label: {
          Label("URL", systemImage: "link")
        }
        .buttonStyle(.bordered)
        .disabled(isLoading)
      }

      // Info text
      Text("Chatterbox uses reference audio to match voice characteristics. For best results, use 10+ seconds of clear speech.")
        .font(.caption)
        .foregroundStyle(.secondary)
    }
    .fileImporter(
      isPresented: $isShowingFilePicker,
      allowedContentTypes: [.audio, .wav, .mp3, .aiff],
      allowsMultipleSelection: false,
    ) { result in
      handleFileSelection(result)
    }
    .alert("Enter Audio URL", isPresented: $isShowingURLInput) {
      TextField("https://...", text: $customURL)
        .textContentType(.URL)
      #if os(iOS)
        .autocapitalization(.none)
      #endif

      Button("Cancel", role: .cancel) {
        customURL = ""
      }

      Button("Load") {
        Task { await loadFromURL() }
      }
      .disabled(customURL.isEmpty)
    } message: {
      Text("Enter the URL of an audio file (MP3, WAV, etc.)")
    }
  }

  // MARK: - Computed Properties

  private var audioStatusIcon: String {
    if isLoading {
      "arrow.triangle.2.circlepath"
    } else if appState.isChatterboxReferenceAudioLoaded {
      "checkmark.circle.fill"
    } else {
      "circle"
    }
  }

  private var audioStatusColor: Color {
    if isLoading {
      .orange
    } else if appState.isChatterboxReferenceAudioLoaded {
      .green
    } else {
      .secondary
    }
  }

  // MARK: - Actions

  private func loadDefaultAudio() async {
    isLoading = true
    errorMessage = nil

    do {
      try await appState.prepareDefaultChatterboxReferenceAudio()
    } catch {
      errorMessage = error.localizedDescription
    }

    isLoading = false
  }

  private func handleFileSelection(_ result: Result<[URL], Error>) {
    switch result {
      case let .success(urls):
        guard let url = urls.first else { return }

        // Start accessing the security-scoped resource
        guard url.startAccessingSecurityScopedResource() else {
          errorMessage = "Permission denied"
          return
        }

        Task {
          defer { url.stopAccessingSecurityScopedResource() }

          isLoading = true
          errorMessage = nil

          do {
            try await appState.prepareChatterboxReferenceAudio(from: url)
          } catch {
            errorMessage = error.localizedDescription
          }

          isLoading = false
        }

      case let .failure(error):
        errorMessage = error.localizedDescription
    }
  }

  private func loadFromURL() async {
    guard let url = URL(string: customURL) else {
      errorMessage = "Invalid URL"
      return
    }

    isLoading = true
    errorMessage = nil

    do {
      try await appState.prepareChatterboxReferenceAudio(from: url)
    } catch {
      errorMessage = error.localizedDescription
    }

    isLoading = false
    customURL = ""
  }
}

// MARK: - UTType Extensions

extension UTType {
  static var wav: UTType { UTType(filenameExtension: "wav")! }
  static var mp3: UTType { UTType(filenameExtension: "mp3")! }
  static var aiff: UTType { UTType(filenameExtension: "aiff")! }
}
