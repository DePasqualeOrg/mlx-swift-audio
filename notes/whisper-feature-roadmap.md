# MLX Whisper Feature Roadmap

**Last Updated:** December 15, 2025
**Purpose:** Track feature gaps and potential improvements for MLX Whisper based on WhisperKit analysis

---

## Executive Summary

WhisperKit is a production-grade Swift implementation of Whisper by Argmax, designed for on-device deployment with CoreML. The MLX Whisper implementation in this repository is a well-optimized port using Apple's MLX framework. This report identifies key features from WhisperKit that could improve the MLX implementation.

---

## Repository Overview

### WhisperKit
- **Location:** `../forked/WhisperKit`
- **Framework:** CoreML
- **Lines of Code:** ~8,000+ in core modules
- **Key Files:**
  - `Sources/WhisperKit/Core/WhisperKit.swift` (1046 lines)
  - `Sources/WhisperKit/Core/Models.swift` (1799 lines)
  - `Sources/WhisperKit/Core/TranscribeTask.swift`
  - `Sources/WhisperKit/Core/Audio/AudioStreamTranscriber.swift`
  - `Sources/WhisperKit/Core/Audio/EnergyVAD.swift`
  - `Sources/WhisperKit/Core/Text/SegmentSeeker.swift`

### MLX Whisper
- **Location:** `package/STT/Whisper/`
- **Framework:** MLX (Apple's ML framework)
- **Lines of Code:** ~3,200 in core modules
- **Key Files:**
  - `WhisperSTT.swift` (657 lines)
  - `WhisperEngine.swift` (373 lines)
  - `WhisperDecoding.swift` (448 lines)
  - `WhisperTiming.swift` (1000+ lines)
  - `WhisperModel.swift` (269 lines)

---

## Feature Comparison Matrix

| Feature | WhisperKit | MLX Whisper | Gap |
|---------|------------|-------------|-----|
| Basic Transcription | ✅ | ✅ | None |
| Language Detection | ✅ | ✅ | None |
| Translation | ✅ | ✅ | None |
| Segment Timestamps | ✅ | ✅ | None |
| Word Timestamps | ✅ | ✅ | None |
| Real-Time Streaming | ✅ | ❌ | **Major** |
| Voice Activity Detection | ✅ (Energy-based) | ⚠️ (Segment-level only) | **Major** |
| Progress Callbacks | ✅ | ❌ | Medium |
| Concurrent Processing | ✅ | ❌ | Medium |
| Model Prefill | ✅ | ❌ | Medium |
| Modular Logits Filtering | ✅ | ⚠️ (Inline) | Minor |
| Multi-Channel Audio | ✅ | ⚠️ (Mono only) | Minor |
| Detailed Timing Metrics | ✅ (30+ metrics) | ⚠️ (Basic) | Minor |
| Quantization Support | ✅ | ✅ | None |
| KV Cache | ✅ | ✅ | None |

---

## Detailed Feature Analysis

### 1. Real-Time Streaming Transcription

**WhisperKit Implementation:**

WhisperKit provides `AudioStreamTranscriber`, an actor-based streaming system:

```swift
// WhisperKit/Sources/WhisperKit/Core/Audio/AudioStreamTranscriber.swift
public actor AudioStreamTranscriber {
    struct State {
        var isRecording: Bool
        var currentFallbacks: Int
        var lastBufferSize: Int
        var lastConfirmedSegmentEndSeconds: Float
        var bufferEnergy: [Float]
        var currentText: String
        var confirmedSegments: [TranscriptionSegment]
        var unconfirmedSegments: [TranscriptionSegment]
        var unconfirmedText: [String]
    }
}
```

Key features:
- **Confirmed vs Unconfirmed Segments:** Prevents mid-speech corrections from appearing to the user
- **Buffer-Based Processing:** Processes when buffer ≥ 1 second
- **Energy Tracking:** Monitors audio loudness in real-time
- **Configurable Confirmation Count:** Last N segments kept unconfirmed (default 2)

**MLX Whisper Status:** Not implemented. Only batch processing available.

**Recommendation:** Implement streaming transcriber for real-time use cases.

---

### 2. Voice Activity Detection (VAD)

**WhisperKit Implementation:**

Protocol-based VAD system with energy-based implementation:

```swift
// WhisperKit/Sources/WhisperKit/Core/Audio/VoiceActivityDetector.swift
public protocol VoiceActivityDetecting {
    func voiceActivity(in audioArray: [Float]) -> [Bool]
}

// WhisperKit/Sources/WhisperKit/Core/Audio/EnergyVAD.swift
public class EnergyVAD: VoiceActivityDetecting {
    public var energyThreshold: Float = 0.02
    public var frameLengthSamples: Int = 1600  // 0.1s at 16kHz

    public func voiceActivity(in audioArray: [Float]) -> [Bool] {
        // Chunk audio into frames
        // Calculate per-frame energy
        // Compare to threshold
        // Return boolean array
    }
}
```

Additional utilities:
- `calculateActiveChunks()` - Finds voice segments
- `findLongestSilence()` - Identifies pause points
- `voiceActivityClipTimestamps()` - Generates seek clips
- `ChunkingStrategy.vad` - Process only voice regions

**MLX Whisper Status:** Only segment-level no-speech detection via Whisper's internal `no_speech_prob`.

**Recommendation:** Port `EnergyVAD` for frame-level VAD to skip silent regions entirely.

---

### 3. Logits Filtering Architecture

**WhisperKit Implementation:**

Modular protocol-based system:

```swift
// WhisperKit/Sources/WhisperKit/Core/Text/LogitsFilter.swift
public protocol LogitsFiltering {
    func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray
}

// Implementations:
class TimestampRulesFilter: LogitsFiltering { ... }
class SuppressTokensFilter: LogitsFiltering { ... }
class SuppressBlankFilter: LogitsFiltering { ... }
```

Filters are applied in sequence, allowing easy customization and testing.

**MLX Whisper Status:** Inline suppression logic in `WhisperDecoding.swift` - functional but not modular.

**Recommendation:** Refactor to protocol-based system for extensibility.

---

### 4. Progress Tracking & Callbacks

**WhisperKit Implementation:**

```swift
// TranscriptionCallback for per-token updates
public typealias TranscriptionCallback = (TranscriptionProgress) -> Bool?

public struct TranscriptionProgress: Sendable {
    public var timings: TranscriptionTimings
    public var text: String
    public var tokens: [Int]
    public var temperature: Float
    // ... more fields
}
```

30+ timing metrics tracked:
- `audioLoading`, `audioProcessing`, `logmels`, `encoding`
- `decodingInit`, `prefill`, `decodingPredictions`
- `decodingFiltering`, `decodingSampling`, `decodingKvCaching`
- `decodingWordTimestamps`, `decodingWindowing`
- Computed: `tokensPerSecond`, `realTimeFactor`, `speedFactor`

**MLX Whisper Status:** Only basic `processingTime` tracking.

**Recommendation:** Add callback system and detailed timing for debugging/optimization.

---

### 5. Concurrent Window Processing

**WhisperKit Implementation:**

```swift
// Configurable concurrent workers
public var concurrentWorkerCount: Int = 16  // macOS default

// Task group-based parallel processing
await withTaskGroup(of: TranscriptionSegment?.self) { group in
    for window in windows {
        group.addTask { await self.transcribeWindow(window) }
    }
}
```

**MLX Whisper Status:** Sequential segment processing only.

**Recommendation:** Add parallel processing for independent segments (when not using prompt conditioning).

---

### 6. Model Prefill Optimization

**WhisperKit Implementation:**

Separate `TextDecoderContextPrefill` model that:
- Pre-generates task/language tokens
- Caches KV states for fast initial tokens
- Reduces initial decoding steps significantly

**MLX Whisper Status:** No prefill optimization - starts fresh each segment.

**Recommendation:** Implement KV cache prefilling for task/language tokens.

---

### 7. Segment Seeking Logic

**WhisperKit Implementation:**

`SegmentSeeker` class with sophisticated logic:

```swift
// WhisperKit/Sources/WhisperKit/Core/Text/SegmentSeeker.swift
func findSeekPointAndSegments(
    decodingResult: DecodingResult,
    options: DecodingOptions,
    allSegmentsCount: Int,
    currentSeek: Int,
    segmentSize: Int,
    sampleRate: Int,
    timeOffset: Float,
    lastSpeechTimestamp: Float
) -> (Int?, [TranscriptionSegment]?, Bool)
```

Features:
- Handles multiple timestamp pairs in single output
- Sub-segment splitting for multi-timestamp windows
- Silent window skipping based on threshold
- `windowClipTime` padding to prevent end-of-clip hallucinations

**MLX Whisper Status:** Basic seek advancement based on last timestamp.

**Recommendation:** Consider porting edge case handling.

---

### 8. Audio Multi-Channel Handling

**WhisperKit Implementation:**

```swift
public enum AudioChannelMode {
    case sumChannels(peakNormalization: Double?)
    case specificChannel(index: Int)
}
```

**MLX Whisper Status:** Simple channel averaging to mono.

**Recommendation:** Add configurable channel handling for better audio quality.

---

### 9. Configuration Options

**WhisperKit `DecodingOptions`:**

```swift
public struct DecodingOptions {
    // Basic
    var verbose: Bool
    var task: DecodingTask  // .transcribe | .translate
    var language: String?

    // Sampling
    var temperature: Float = 0.0
    var temperatureIncrementOnFallback: Float = 0.2
    var temperatureFallbackCount: Int = 5
    var sampleLength: Int = 224
    var topK: Int = 5

    // Prefill
    var usePrefillPrompt: Bool = true
    var usePrefillCache: Bool = true

    // Timestamps
    var withoutTimestamps: Bool = false
    var wordTimestamps: Bool = false
    var maxInitialTimestamp: Float?

    // Seeking
    var maxWindowSeek: Int?
    var clipTimestamps: [Float] = []
    var windowClipTime: Float = 1.0  // Padding to prevent hallucinations

    // Quality thresholds
    var compressionRatioThreshold: Float? = 2.4
    var logProbThreshold: Float? = -1.0
    var firstTokenLogProbThreshold: Float? = -1.5
    var noSpeechThreshold: Float? = 0.6

    // Performance
    var concurrentWorkerCount: Int
    var chunkingStrategy: ChunkingStrategy?

    // Token control
    var suppressBlank: Bool = true
    var supressTokens: [Int]
    var promptTokens: [Int]?
    var prefixTokens: [Int]?
}
```

**MLX Whisper Status:** Smaller `WhisperGenerateOptions` with fewer options.

**Recommendation:** Consider adding `windowClipTime`, `maxWindowSeek`, `firstTokenLogProbThreshold`.

---

## Resolved Issues

### Alignment Heads Configuration (RESOLVED - December 2025)

**Background:** Alignment heads are specific cross-attention heads in the Whisper decoder that OpenAI identified as highly correlated with word-level timing. Only a sparse subset of heads provides good timing information.

**The Issue (now fixed):** The mlx-community models were using **DEFAULT** alignment heads (all heads in the last half of decoder layers) instead of OpenAI's **OPTIMAL** heads.

**Resolution:**
1. Fixed `convert.py` in both mlx-examples and mlx-audio-plus to use optimal alignment heads when loading from Hugging Face repos
2. Re-converted and uploaded all 30 Whisper models to mlx-community with optimal alignment heads:
   - 10 model variants: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v3, large-v3-turbo
   - 3 quantizations each: fp16, 8bit, 4bit

**Optimal heads per model (now in mlx-community):**

| Model | Optimal Heads |
|-------|---------------|
| tiny | 6 |
| tiny.en | 8 |
| base | 8 |
| base.en | 5 |
| small | 10 |
| small.en | 19 |
| medium | 6 |
| medium.en | 18 |
| large-v3 | 10 |
| large-v3-turbo | 6 |

**Technical Details:**
- Alignment heads stored in `model.safetensors` as `alignment_heads` parameter
- Swift code loads from weights automatically via `@ParameterInfo(key: "alignment_heads")`
- Falls back to default heads only if not present in weights (legacy models)

---

## Priority Recommendations

| Priority | Feature | Impact | Effort |
|----------|---------|--------|--------|
| 1 | Real-Time Streaming | High | High |
| 2 | Energy-Based VAD | High | Medium |
| 3 | Progress Callbacks | Medium | Low |
| 4 | Logits Filter Architecture | Medium | Medium |
| 5 | Model Prefill | Medium | Medium |
| 6 | Concurrent Processing | Medium | Medium |
| 7 | Audio Channel Options | Low | Low |
| 8 | Enhanced Config Options | Low | Low |

---

## Appendix: File References

### WhisperKit Key Files
- `Sources/WhisperKit/Core/WhisperKit.swift` - Main entry point
- `Sources/WhisperKit/Core/Models.swift` - Type definitions
- `Sources/WhisperKit/Core/Configurations.swift` - Configuration classes
- `Sources/WhisperKit/Core/TranscribeTask.swift` - Pipeline orchestration
- `Sources/WhisperKit/Core/TextDecoder.swift` - Decoding protocol
- `Sources/WhisperKit/Core/FeatureExtractor.swift` - Mel spectrogram
- `Sources/WhisperKit/Core/Audio/AudioProcessor.swift` - Audio loading
- `Sources/WhisperKit/Core/Audio/AudioStreamTranscriber.swift` - Streaming
- `Sources/WhisperKit/Core/Audio/EnergyVAD.swift` - VAD implementation
- `Sources/WhisperKit/Core/Text/TokenSampler.swift` - Sampling strategies
- `Sources/WhisperKit/Core/Text/LogitsFilter.swift` - Logits filtering
- `Sources/WhisperKit/Core/Text/SegmentSeeker.swift` - Segment detection

### MLX Whisper Key Files
- `package/STT/Whisper/WhisperSTT.swift` - Main actor wrapper
- `package/STT/Whisper/WhisperEngine.swift` - Public API
- `package/STT/Whisper/WhisperModel.swift` - Model loading
- `package/STT/Whisper/WhisperDecoding.swift` - Token decoding
- `package/STT/Whisper/WhisperTiming.swift` - Word timestamps (DTW)
- `package/STT/Whisper/WhisperAudio.swift` - Audio preprocessing
- `package/STT/Whisper/WhisperTokenizer.swift` - BPE tokenization
- `package/STT/Whisper/Layers/AudioEncoder.swift` - Audio encoding
- `package/STT/Whisper/Layers/TextDecoder.swift` - Text decoding

---

## Resources for Porting Whisper Functionality to Swift

### Primary Reference Implementations

| Resource | Location | Use For |
|----------|----------|---------|
| **Python MLX Reference** | `../forked/mlx-audio-plus/mlx_audio/stt/models/whisper/` | Authoritative source for MLX-based implementation |
| **WhisperKit** | `../forked/WhisperKit/Sources/WhisperKit/` | Production Swift patterns, streaming, VAD |
| **OpenAI Whisper** | https://github.com/openai/whisper | Original implementation, alignment heads, tokenizer |

### Feature-Specific References

#### Real-Time Streaming
- **WhisperKit:** `Sources/WhisperKit/Core/Audio/AudioStreamTranscriber.swift`
- **Concepts:** Actor-based state management, confirmed vs unconfirmed segments, buffer handling

#### Voice Activity Detection (VAD)
- **WhisperKit:** `Sources/WhisperKit/Core/Audio/EnergyVAD.swift`, `VoiceActivityDetector.swift`
- **Python:** Frame-level energy calculation with Accelerate/vDSP is straightforward to port

#### Word-Level Timestamps (DTW)
- **Python MLX:** `mlx_audio/stt/models/whisper/timing.py` - `find_alignment()`, `add_word_timestamps()`
- **MLX Swift:** `package/STT/Whisper/WhisperTiming.swift` (already implemented)
- **Paper:** "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)

#### Alignment Heads
- **OpenAI Whisper:** https://github.com/openai/whisper/blob/main/whisper/__init__.py (`_ALIGNMENT_HEADS`)
- **Python MLX:** `mlx_audio/stt/models/whisper/scripts/convert.py:47-64`
- **Note:** Base85-encoded boolean arrays indicating which heads correlate with word timing

#### Tokenization
- **OpenAI:** https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
- **Tiktoken files:** `multilingual.tiktoken`, `gpt2.tiktoken` (BPE vocabularies)

#### Audio Preprocessing
- **Python MLX:** `mlx_audio/stt/models/whisper/audio.py` - Mel spectrogram, padding
- **Constants:** 16kHz sample rate, 80/128 mel bins, hop_length=160, n_fft=400

#### Logits Filtering
- **WhisperKit:** `Sources/WhisperKit/Core/Text/LogitsFilter.swift`
- **Python MLX:** `mlx_audio/stt/models/whisper/decoding.py` - `SuppressBlank`, `ApplyTimestampRules`

### Documentation & Papers

1. **Whisper Paper:** "Robust Speech Recognition via Large-Scale Weak Supervision"
   - https://arxiv.org/abs/2212.04356
   - Describes model architecture, training, and capabilities

2. **OpenAI Whisper Blog:** https://openai.com/research/whisper
   - High-level overview and use cases

3. **MLX Documentation:** https://ml-explore.github.io/mlx/build/html/index.html
   - Swift MLX framework reference

4. **WhisperKit Documentation:** https://github.com/argmaxinc/WhisperKit
   - Production deployment patterns for Apple devices

### Model Weights

| Source | Format | Notes |
|--------|--------|-------|
| `mlx-community/whisper-*` | safetensors | MLX-optimized, quantized variants |
| OpenAI `.pt` files | PyTorch | Original weights with optimal alignment heads |
| HuggingFace `openai/whisper-*` | safetensors | Transformers format (different key names) |

### Useful Code Patterns

#### Swift Accelerate/vDSP for Audio
```swift
import Accelerate

// Fast softmax
func softmax(_ values: inout [Float]) {
    var maxVal: Float = 0
    vDSP_maxv(values, 1, &maxVal, vDSP_Length(values.count))
    var negMax = -maxVal
    vDSP_vsadd(values, 1, &negMax, &values, 1, vDSP_Length(values.count))
    var count = Int32(values.count)
    vvexpf(&values, values, &count)
    var sum: Float = 0
    vDSP_sve(values, 1, &sum, vDSP_Length(values.count))
    vDSP_vsdiv(values, 1, &sum, &values, 1, vDSP_Length(values.count))
}
```

#### Actor-Based Streaming Pattern
```swift
public actor StreamingTranscriber {
    private var confirmedSegments: [Segment] = []
    private var unconfirmedSegments: [Segment] = []
    private var audioBuffer: [Float] = []

    func processAudioChunk(_ chunk: [Float]) async -> StreamingResult {
        audioBuffer.append(contentsOf: chunk)
        guard audioBuffer.count >= minBufferSize else { return .waiting }
        // Process and return confirmed + unconfirmed
    }
}
```

### Testing Resources

- **Test audio files:** LJ Speech dataset, LibriSpeech, Common Voice
- **WhisperKit tests:** `Tests/WhisperKitTests/` - Good patterns for accuracy testing
- **MLX Whisper tests:** `package/Tests/WhisperTests.swift`

### Community Resources

- **MLX Community Discord:** https://discord.gg/mlx-community
- **WhisperKit Issues:** https://github.com/argmaxinc/WhisperKit/issues
- **MLX Swift Issues:** https://github.com/ml-explore/mlx-swift/issues
