import SwiftUI

/// Recording button with visual feedback
struct RecordButton: View {
  let isRecording: Bool
  let duration: TimeInterval
  let averagePower: Float
  let onStart: () -> Void
  let onStop: () -> Void

  @State private var isPulsing = false

  var body: some View {
    Button(action: {
      if isRecording {
        onStop()
      } else {
        onStart()
      }
    }) {
      HStack(spacing: 12) {
        ZStack {
          Circle()
            .fill(isRecording ? Color.red : Color.red.opacity(0.2))
            .frame(width: 24, height: 24)
            .scaleEffect(isPulsing ? 1.2 : 1.0)

          if isRecording {
            // Audio level indicator
            Circle()
              .stroke(Color.red.opacity(0.5), lineWidth: 2)
              .frame(width: 24, height: 24)
              .scaleEffect(audioLevelScale)
          }
        }

        VStack(alignment: .leading, spacing: 2) {
          Text(isRecording ? "Stop Recording" : "Start Recording")
            .fontWeight(.medium)

          if isRecording {
            Text(formattedDuration)
              .font(.caption)
              .foregroundStyle(.secondary)
              .monospacedDigit()
          }
        }
      }
      .frame(maxWidth: .infinity, alignment: .leading)
      .padding()
      .background(
        RoundedRectangle(cornerRadius: 12)
          .fill(isRecording ? Color.red.opacity(0.1) : Color.secondary.opacity(0.1))
      )
      .overlay(
        RoundedRectangle(cornerRadius: 12)
          .stroke(isRecording ? Color.red.opacity(0.3) : Color.clear, lineWidth: 1)
      )
    }
    .buttonStyle(.plain)
    .onChange(of: isRecording) { _, newValue in
      withAnimation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true)) {
        isPulsing = newValue
      }
      if !newValue {
        isPulsing = false
      }
    }
  }

  private var formattedDuration: String {
    let minutes = Int(duration) / 60
    let seconds = Int(duration) % 60
    let tenths = Int((duration.truncatingRemainder(dividingBy: 1)) * 10)
    return String(format: "%d:%02d.%d", minutes, seconds, tenths)
  }

  private var audioLevelScale: CGFloat {
    // Convert dB to scale (0.0 to 2.0)
    // averagePower typically ranges from -160 (silence) to 0 (max)
    let normalized = (averagePower + 60) / 60 // -60 to 0 -> 0 to 1
    let clamped = max(0, min(1, normalized))
    return 1.0 + CGFloat(clamped) * 0.5
  }
}
