import Foundation
import MLX

/// Creates an AsyncThrowingStream that generates audio sentence-by-sentence.
///
/// This utility enables sentence-level streaming for any TTS engine by splitting
/// the input text into sentences and generating audio for each one sequentially.
///
/// - Parameters:
///   - text: The text to synthesize
///   - sampleRate: Sample rate for the audio chunks
///   - generate: Closure that generates audio samples for a single sentence
/// - Returns: An async stream of audio chunks, one per sentence
@MainActor
func sentenceStreamingGenerate(
  text: String,
  sampleRate: Int,
  generate: @escaping (String) async throws -> [Float],
) -> AsyncThrowingStream<AudioChunk, Error> {
  let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

  guard !trimmedText.isEmpty else {
    return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Text cannot be empty")) }
  }

  let sentences = SentenceTokenizer.splitIntoSentences(text: trimmedText)

  guard !sentences.isEmpty else {
    return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Failed to split text into sentences")) }
  }

  return AsyncThrowingStream { continuation in
    Task { @MainActor in
      let startTime = Date()

      for (index, sentence) in sentences.enumerated() {
        do {
          let samples = try await generate(sentence)

          let chunk = AudioChunk(
            samples: samples,
            sampleRate: sampleRate,
            isLast: index == sentences.count - 1,
            processingTime: Date().timeIntervalSince(startTime),
          )
          continuation.yield(chunk)

          MLX.GPU.clearCache()
        } catch {
          continuation.finish(throwing: error)
          return
        }
      }

      continuation.finish()
    }
  }
}
