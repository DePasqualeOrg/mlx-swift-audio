import SwiftUI

struct RegularLayoutView: View {
  var body: some View {
    NavigationSplitView {
      ScrollView {
        SettingsSection()
          .padding()
      }
      .navigationTitle("Settings")
      .navigationSplitViewColumnWidth(min: 250, ideal: 300, max: 400)
    } detail: {
      VStack(spacing: 0) {
        ScrollView {
          VStack(spacing: 24) {
            InputSection()
            OutputSection()
          }
          .padding()
        }
      }
      .navigationTitle("STT App")
      #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
      #endif
    }
  }
}
