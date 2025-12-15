import AVFoundation
import SwiftUI

@main
struct STTApp: App {
  init() {
    #if os(iOS)
    configureAudioSession()
    #endif
  }

  var body: some Scene {
    WindowGroup {
      ContentView()
    }
    #if os(macOS)
    .windowStyle(.automatic)
    .defaultSize(width: 1000, height: 700)
    #endif
  }

  #if os(iOS)
  private func configureAudioSession() {
    do {
      let session = AVAudioSession.sharedInstance()
      try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
      try session.setActive(true)
    } catch {
      print("Failed to configure audio session: \(error)")
    }
  }
  #endif
}
