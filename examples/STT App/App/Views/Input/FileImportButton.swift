import SwiftUI
import UniformTypeIdentifiers

/// Button to import audio files
struct FileImportButton: View {
  let selectedFileName: String?
  let onFileSelected: (URL) -> Void
  let onClear: () -> Void

  @State private var isShowingFilePicker = false

  var body: some View {
    VStack(spacing: 12) {
      Button(action: { isShowingFilePicker = true }) {
        HStack(spacing: 12) {
          Image(systemName: "doc.badge.plus")
            .font(.title2)

          VStack(alignment: .leading, spacing: 2) {
            Text("Select Audio File")
              .fontWeight(.medium)
            Text("WAV, MP3, M4A, and more")
              .font(.caption)
              .foregroundStyle(.secondary)
          }

          Spacer()

          Image(systemName: "chevron.right")
            .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(
          RoundedRectangle(cornerRadius: 12)
            .fill(Color.secondary.opacity(0.1))
        )
      }
      .buttonStyle(.plain)
      .fileImporter(
        isPresented: $isShowingFilePicker,
        allowedContentTypes: supportedAudioTypes,
        allowsMultipleSelection: false
      ) { result in
        switch result {
          case let .success(urls):
            if let url = urls.first {
              // Start accessing security-scoped resource
              if url.startAccessingSecurityScopedResource() {
                onFileSelected(url)
              }
            }
          case .failure:
            break
        }
      }

      if let fileName = selectedFileName {
        HStack {
          Image(systemName: "waveform")
            .foregroundStyle(.blue)
          Text(fileName)
            .lineLimit(1)
            .truncationMode(.middle)
          Spacer()
          Button(action: onClear) {
            Image(systemName: "xmark.circle.fill")
              .foregroundStyle(.secondary)
          }
          .buttonStyle(.plain)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
          RoundedRectangle(cornerRadius: 8)
            .fill(Color.blue.opacity(0.1))
        )
      }
    }
  }

  private var supportedAudioTypes: [UTType] {
    [.audio, .wav, .mp3, .mpeg4Audio, .aiff]
  }
}
