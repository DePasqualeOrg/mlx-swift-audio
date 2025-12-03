import SwiftUI

struct ContentView: View {
    @State private var appState = AppState()

    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    var body: some View {
        Group {
            if horizontalSizeClass == .regular {
                RegularLayoutView()
            } else {
                CompactLayoutView()
            }
        }
        .environment(appState)
    }
}
