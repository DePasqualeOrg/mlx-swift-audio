import MLXAudio
import SwiftUI

/// Picker for selecting quantization level
struct QuantizationPickerView: View {
  @Binding var selectedQuantization: WhisperQuantization
  let isDisabled: Bool

  var body: some View {
    Picker("Quantization", selection: $selectedQuantization) {
      ForEach(WhisperQuantization.allCases, id: \.self) { quant in
        Text(quant.displayName).tag(quant)
      }
    }
    .disabled(isDisabled)
  }
}

extension WhisperQuantization {
  var displayName: String {
    switch self {
      case .fp16: "FP16 (Best Quality)"
      case .q8: "8-bit (Balanced)"
      case .q4: "4-bit (Fastest)"
    }
  }
}
