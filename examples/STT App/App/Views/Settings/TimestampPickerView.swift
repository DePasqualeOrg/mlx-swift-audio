import MLXAudio
import SwiftUI

/// Picker for selecting timestamp granularity
struct TimestampPickerView: View {
  @Binding var selectedTimestamps: TimestampGranularity
  let isDisabled: Bool

  private let options: [TimestampGranularity] = [.none, .segment, .word]

  var body: some View {
    Picker("Timestamps", selection: $selectedTimestamps) {
      ForEach(options, id: \.self) { granularity in
        Text(granularity.displayName).tag(granularity)
      }
    }
    .disabled(isDisabled)
  }
}

extension TimestampGranularity {
  var displayName: String {
    switch self {
      case .none: "None (not recommended)"
      case .segment: "Segment"
      case .word: "Word"
    }
  }

  var description: String {
    switch self {
      case .none:
        "No timestamps - may cause hallucinations"
      case .segment:
        "Start/end time for each segment (recommended)"
      case .word:
        "Individual word timing (slower but precise)"
    }
  }
}
