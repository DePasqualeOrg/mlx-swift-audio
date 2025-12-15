import SwiftUI

struct CompactLayoutView: View {
  @State private var showSettings = false

  var body: some View {
    NavigationStack {
      ScrollView {
        VStack(alignment: .leading, spacing: 20) {
          InputSection()
          OutputSection()
        }
        .padding(.vertical)
      }
      .navigationTitle("STT App")
      #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
      #endif
        .toolbar {
          ToolbarItem(placement: .primaryAction) {
            Button {
              showSettings = true
            } label: {
              Image(systemName: "gearshape")
            }
          }
        }
        .sheet(isPresented: $showSettings) {
          SettingsSheet()
        }
    }
  }
}

/// Full settings sheet for compact layout
private struct SettingsSheet: View {
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    NavigationStack {
      SettingsSection()
        .navigationTitle("Settings")
      #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
      #endif
        .toolbar {
          ToolbarItem(placement: .confirmationAction) {
            Button {
              dismiss()
            } label: {
              Image(systemName: "xmark.circle.fill")
            }
          }
        }
    }
  }
}
