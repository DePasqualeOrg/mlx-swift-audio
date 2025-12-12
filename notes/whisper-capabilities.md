# Whisper Model Capabilities

This document catalogs all the capabilities of the Whisper model and what is currently exposed through the MLXAudio API.

## Core Operations

### 1. Transcribe
Speech-to-text in the original language.

**Status**: ✅ Implemented and exposed
**API**:
```swift
func transcribe(
  _ url: URL,
  language: Language? = nil,          // nil = auto-detect
  temperature: Float = 0.0,           // greedy decoding
  timestamps: TimestampGranularity = .segment
) async throws -> TranscriptionResult
```

### 2. Translate
Translate audio to English only (Whisper model limitation - only translates to English).

**Status**: ✅ Implemented and exposed
**API**:
```swift
func translate(
  _ url: URL,
  language: Language? = nil           // source language hint
) async throws -> TranscriptionResult
```

### 3. Detect Language
Identify the spoken language without full transcription.

**Status**: ✅ Implemented and exposed
**API**:
```swift
func detectLanguage(_ url: URL) async throws -> (Language, Float)
// Returns (language, confidence)
```

## Configuration Options

### 4. Language Specification
Type-safe language specification with enum for all 99 supported languages.

**Status**: ✅ Implemented and exposed
**API**: `language: Language?` parameter (nil = auto-detect)
**Implementation**: `Language` enum in `Language.swift` with all 99 languages
**Languages supported**: 99 languages with type-safe enum (e.g., `.english`, `.spanish`, `.chinese`)

### 5. Timestamps
Control timestamp granularity in transcription output.

**Status**: 🟡 Partially implemented
**API**: `timestamps: TimestampGranularity` parameter (default: `.segment`)
- `.none` - No timestamps ✅
- `.segment` - Segment-level timestamps ✅
- `.word` - Word-level timestamps ❌ (planned, not implemented)

**Implementation**: Timestamp rules in `GreedyDecoder` enforce:
- Monotonic timestamps (no going backward)
- Max initial timestamp constraint (30s segments)
- Timestamp probability heuristics

### 6. Temperature Control
Control sampling randomness vs greedy decoding.

**Status**: ✅ Implemented and exposed
**API**: `temperature: Float` parameter (default: 0.0 = greedy, higher = more random)
**Default**: 0.0 (greedy decoding)

### 7. Model Size Selection
Choose between different Whisper model sizes for speed/accuracy tradeoff.

**Status**: ✅ Implemented and exposed
**API**: `WhisperModelSize` enum
- `.tiny` - 39M parameters
- `.base` - 74M parameters
- `.small` - 244M parameters
- `.medium` - 769M parameters
- `.large` - 1550M parameters (whisper-large-v3)
- `.largeTurbo` - 809M parameters (whisper-large-v3-turbo)

## Quality Metrics (Exposed in Output)

The `TranscriptionResult` and `TranscriptionSegment` structs expose:

### Per-Result Metrics
- `text: String` - Full transcription
- `language: String` - Detected/specified language
- `segments: [TranscriptionSegment]` - Individual segments
- `processingTime: TimeInterval` - Time taken
- `duration: TimeInterval` - Audio duration
- `realTimeFactor: Double` - processingTime / duration (< 1.0 = faster than real-time)

### Per-Segment Metrics
- `text: String` - Segment text
- `start: TimeInterval` - Start timestamp
- `end: TimeInterval` - End timestamp
- `tokens: [Int]` - Token IDs
- `avgLogProb: Float` - Average log probability (confidence)
- `noSpeechProb: Float` - Probability of silence/no speech (0-1, higher = likely silence)
- `words: [Word]?` - Word-level timestamps (not implemented yet)

## Internal Capabilities (Not Currently Exposed)

### 7. Compression Ratio
Ratio of text length to compressed text length. Used internally to detect hallucination.

**Status**: ❌ Computed but not exposed
**Implementation**: `DecodingResult.compressionRatio`
**Use case**: Values significantly > 1.0 can indicate model hallucination

### 8. Voice Activity Detection (VAD)
Using `noSpeechProb` threshold to filter silent segments.

**Status**: 🟡 Raw data exposed (`noSpeechProb`) but no built-in filtering
**Implementation**: `noSpeechProb` computed in `GreedyDecoder`
**Potential enhancement**: Add option to auto-filter segments with high noSpeechProb

### 9. Beam Search
Alternative to greedy decoding for potentially better quality.

**Status**: ❌ Not implemented
**Current**: Only `GreedyDecoder` available
**Potential enhancement**: Add `BeamSearchDecoder` for higher quality at cost of speed

### 10. Max Initial Timestamp
Controls the maximum timestamp for the first token (currently hardcoded to 1500 frames = 30s).

**Status**: ✅ Implemented but hardcoded
**Implementation**: `maxInitialTimestampIndex = 1500` in `GreedyDecoder`
**Potential enhancement**: Make configurable for different segment lengths

### 11. Token Suppression
Custom suppression of specific tokens during decoding.

**Status**: ✅ Implemented with defaults (suppress non-speech, special tokens)
**Implementation**: Suppression logic in `GreedyDecoder.decode()`
**Current behavior**:
- Suppress non-speech tokens
- Suppress special tokens (SOT, EOT, etc.)
- Suppress blank at start
- Enforce timestamp rules
**Potential enhancement**: Allow custom suppression lists

### 12. Max Tokens
Maximum number of tokens to generate per segment.

**Status**: ✅ Implemented but hardcoded
**Implementation**: `maxTokens: 448` in `DecodingOptions`
**Potential enhancement**: Make configurable via API

## Audio Processing

### 13. Audio Segmentation
Automatic splitting of long audio into 30-second chunks.

**Status**: ✅ Implemented automatically
**Implementation**: `segmentAudio()` in `WhisperSTT`
**Behavior**: Audio longer than 30s is automatically split and processed in chunks

### 14. Audio Resampling
Automatic resampling to 16kHz if needed.

**Status**: ✅ Implemented automatically
**Implementation**: `AudioResampler.resample()` in `WhisperEngine`
**Input formats**: Supports WAV, MP3, M4A, etc. (via AVFoundation)

### 15. Mel Spectrogram Generation
Converts audio to mel spectrogram (required Whisper input format).

**Status**: ✅ Implemented automatically
**Implementation**: `whisperLogMelSpectrogram()` in `WhisperAudio.swift`
**Configuration**: Uses model-specific n_mels from config.json

## Future Enhancements to Consider

1. **Word-level timestamps** - Complete implementation for `.word` granularity
2. **Beam search decoder** - Add for improved quality
3. **VAD filtering** - Option to auto-filter silent segments
4. **Streaming transcription** - Process audio in real-time (noted in STTProvider as future feature)
5. **Expose compression ratio** - Add to result for hallucination detection
6. **Configurable max tokens** - Allow users to control generation length
7. **Custom token suppression** - Allow users to specify tokens to suppress
8. **Prompt/context** - Allow previous text as context (Whisper supports this)
9. **Multi-file batch processing** - Optimize for processing multiple files

## Summary

### Currently Exposed (Good API)
- ✅ Transcribe vs translate
- ✅ Language detection (auto or specified)
- ✅ Segment-level timestamps
- ✅ Temperature control
- ✅ Model size selection
- ✅ Quality metrics (noSpeechProb, avgLogProb, etc.)
- ✅ Automatic audio preprocessing (resample, segment, mel spectrogram)

### Not Exposed (Could Enhance API)
- ⚠️ Word-level timestamps (in progress)
- ⚠️ Compression ratio
- ⚠️ Beam search
- ⚠️ VAD filtering options
- ⚠️ Configurable max tokens
- ⚠️ Custom token suppression
- ⚠️ Context/prompt for continued transcription
- ⚠️ Streaming transcription

The current API exposes the most important Whisper capabilities with sensible defaults for the advanced features.
