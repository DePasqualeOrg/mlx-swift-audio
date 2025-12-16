# MLX Whisper Feature Roadmap

**Last Updated:** December 16, 2025
**Purpose:** Track feature gaps and potential improvements for MLX Whisper based on WhisperKit and mlx-audio-plus analysis

---

## Executive Summary

WhisperKit is a production-grade Swift implementation of Whisper by Argmax, designed for on-device deployment with CoreML. The MLX Whisper implementation in this repository is a well-optimized port using Apple's MLX framework. This report identifies key features from WhisperKit that could improve the MLX implementation.

---

## Repository Overview

### WhisperKit (Swift/CoreML)
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
  - `Sources/WhisperKit/Core/Text/TokenSampler.swift` (greedy + beam search stub)

### mlx-audio-plus (Python MLX)
- **Location:** `../forked/mlx-audio-plus`
- **Framework:** MLX (Python)
- **Key Files:**
  - `mlx_audio/stt/models/whisper/whisper.py` - Model definition & transcription
  - `mlx_audio/stt/models/whisper/decoding.py` - Decoding strategies (greedy only)
  - `mlx_audio/stt/models/whisper/audio.py` - Audio preprocessing
  - `mlx_audio/stt/models/whisper/tokenizer.py` - Token vocabulary
  - `mlx_audio/stt/models/whisper/timing.py` - Word-level timestamps

### MLX Whisper (This Repo)
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
| Timestamp Rules | ✅ | ✅ | None |
| Temperature Fallback | ✅ | ✅ | None |
| Compression Ratio Filter | ✅ | ✅ | None |
| Log Prob Threshold | ✅ | ✅ | None |
| No Speech Detection | ✅ | ✅ | None |
| Prompt/Context Conditioning | ✅ | ✅ | None |
| Quantization Support | ✅ | ✅ | None |
| KV Cache | ✅ | ✅ | None |
| Real-Time Streaming | ✅ | ❌ | **Major** |
| Voice Activity Detection | ✅ (Energy-based) | ⚠️ (Segment-level only) | **Major** |
| Progress Callbacks | ✅ | ❌ | Medium |
| Concurrent Processing | ✅ | ❌ | Medium |
| Model Prefill | ✅ | ❌ | Medium |
| Transcription Cancellation | ✅ | ❌ | Medium |
| Beam Search | ❌ (stub) | ❌ | Low* |
| Modular Logits Filtering | ✅ | ⚠️ (Inline) | Minor |
| Multi-Channel Audio | ✅ | ⚠️ (Mono only) | Minor |
| Detailed Timing Metrics | ✅ (30+ metrics) | ⚠️ (Basic) | Minor |
| Custom Token Suppression | ✅ | ❌ | Minor |

*Beam search is not implemented in either WhisperKit or mlx-audio-plus (Python MLX). See "Beam Search Investigation" section below.

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

## Beam Search Investigation (December 2025)

### Executive Summary

**Finding: Beam search is NOT implemented in either reference repository.**

Both `mlx-audio-plus` (Python MLX) and `WhisperKit` (Swift) have beam search infrastructure in place but neither has a functional implementation.

---

### mlx-audio-plus (Python MLX)

**Location:** `../forked/mlx-audio-plus/mlx_audio/stt/models/whisper/decoding.py`

**Status:** Infrastructure exists, throws NotImplementedError

```python
# Lines 436-439
if options.beam_size is not None:
    raise NotImplementedError("Beam search decoder is not yet implemented")
else:
    self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)
```

**Available Parameters (DecodingOptions, lines 94-95):**
- `beam_size: Optional[int] = None` - number of beams
- `patience: Optional[float] = None` - early stopping patience (references arxiv:2204.05424)

**What IS Implemented:**
- `GreedyDecoder` (lines 255-283): Full implementation with temperature sampling
- `MaximumLikelihoodRanker` (lines 165-188): Length penalty normalization for sequence ranking
- KV cache infrastructure has `rearrange_kv_cache()` method (lines 144-148) ready for beam search
- Validation logic for beam_size/best_of mutual exclusivity (lines 466-472)

---

### WhisperKit (Swift)

**Location:** `../forked/WhisperKit/Sources/WhisperKit/Core/Text/TokenSampler.swift`

**Status:** Stub class exists, throws fatalError

```swift
// Lines 254-290
open class BeamSearchTokenSampler: TokenSampling {
    public var beamSize: Int
    public var eotToken: Int
    public var patience: Float
    var maxCandidates: Int
    var finishedSequences: [Float]

    public init(beamSize: Int, eotToken: Int, patience: Float = 1) {
        self.beamSize = beamSize
        self.eotToken = eotToken
        self.patience = patience
        self.maxCandidates = Int(Float(beamSize) * patience)
        self.finishedSequences = []
    }

    public func update(tokens: [Int], logits: MLMultiArray, logProbs: [Float]) -> SamplingResult {
        // TODO: Implement
        fatalError("Not implemented: \(#function)")
    }

    public func finalize(tokens: [Int], logProbs: [Float]) -> SamplingResult {
        // TODO: Implement
        fatalError("Not implemented: \(#function)")
    }
}
```

**What IS Implemented:**
- `GreedyTokenSampler` (lines 29-252): Full implementation with dual backends
  - MLTensor (macOS 15+, iOS 18+): Native Core ML tensor operations
  - BNNS fallback: Accelerate framework (marked deprecated, needs vDSP/MLX replacement)
- Temperature-based sampling: argmax when T=0, top-K multinomial when T>0

---

### Decoding Features Comparison

| Feature | mlx-audio-plus | WhisperKit | This Repo |
|---------|----------------|------------|-----------|
| Greedy Decoding | ✅ | ✅ | ✅ |
| Temperature Sampling | ✅ | ✅ | ✅ |
| Beam Search | ❌ (NotImplementedError) | ❌ (fatalError) | ❌ |
| Best-of-N Sampling | ✅ (infrastructure) | ❌ | ❌ |
| Temperature Fallback | ✅ | ✅ | ✅ |
| Compression Ratio Filter | ✅ (threshold: 2.4) | ✅ (threshold: 2.4) | ✅ (threshold: 2.4) |
| Log Prob Threshold | ✅ (threshold: -1.0) | ✅ (threshold: -1.0) | ✅ (threshold: -1.0) |
| First Token Log Prob | ❌ | ✅ (threshold: -1.5) | ❌ |
| No Speech Detection | ✅ (threshold: 0.6) | ✅ (threshold: 0.6) | ✅ (threshold: 0.6) |
| Hallucination Filtering | ❌ | ⚠️ (windowClipTime) | ✅ (timestamp + confidence) |
| Length Penalty | ✅ | ❌ | ❌ |

---

### Beam Search Implementation Requirements

To implement beam search in MLX Swift, the following components would be needed:

#### 1. Core Algorithm
```swift
class BeamSearchDecoder {
    var beamSize: Int           // Number of hypotheses to maintain
    var patience: Float         // Early stopping patience (default 1.0)
    var maxCandidates: Int      // beamSize * patience

    struct Hypothesis {
        var tokens: [Int]
        var logProb: Float
        var isFinished: Bool
    }

    var activeHypotheses: [Hypothesis]
    var finishedHypotheses: [Hypothesis]
}
```

#### 2. Key Operations
- **Expand**: For each hypothesis, get top-K next tokens
- **Prune**: Keep only top beamSize hypotheses by log probability
- **Finish**: Move hypotheses ending with EOT to finished set
- **Early Stop**: Stop when patience condition met (all top candidates finished)

#### 3. KV Cache Handling
- Must duplicate/rearrange KV cache for each beam
- Reference: `rearrange_kv_cache()` in mlx-audio-plus (lines 144-148)
- Memory consideration: beamSize × original cache size

#### 4. Length Normalization (Optional)
```swift
// Google NMT paper length penalty
func lengthPenalty(length: Int, alpha: Float) -> Float {
    return pow((5.0 + Float(length)) / 6.0, alpha)
}

func normalizedScore(logProb: Float, length: Int) -> Float {
    return logProb / lengthPenalty(length: length, alpha: self.lengthPenalty)
}
```

---

### Priority Assessment

**Should we implement beam search?**

Arguments **against** prioritizing beam search:
1. Neither major reference implementation has it working
2. Greedy decoding with temperature fallback produces good results for most use cases
3. Beam search significantly increases memory usage (beamSize × KV cache)
4. Modern large models (large-v3, turbo) perform well with greedy

Arguments **for** implementing beam search:
1. Can improve accuracy on difficult/ambiguous audio
2. Required for some research applications
3. Would make this repo more feature-complete than references

**Recommendation:** Lower priority than streaming, VAD, and temperature fallback. Consider implementing after those features are complete.

---

### Additional Findings: Quality Control Features

Both reference implementations have sophisticated quality control that we lack:

#### Temperature Fallback (High Value)
```python
# mlx-audio-plus pattern (whisper.py lines 521-557)
temperatures = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
for t in temperatures:
    result = decode(segment, temperature=t)
    if result.compression_ratio <= 2.4 and result.avg_logprob >= -1.0:
        break  # Good enough quality
    # Otherwise retry with higher temperature
```

This prevents:
- Repetitive/hallucinated output (compression ratio check)
- Low-confidence gibberish (log prob check)

#### Compression Ratio Calculation
```python
def compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))
```

High compression ratio (>2.4) indicates repetitive text, often a sign of hallucination.

---

## Resolved Issues

### Quality Control Features (RESOLVED - December 2025)

The following features from the Python reference implementation have been ported:

**Temperature Fallback:**
- Location: `WhisperSTT.swift` lines 180-230
- Automatically retries decoding with higher temperatures (0.0 → 0.2 → 0.4 → ...) when quality checks fail
- Triggers on: compression ratio > 2.4, avg log prob < -1.0

**Compression Ratio Filtering:**
- Location: `WhisperDecoding.swift` lines 390-410 (`computeCompressionRatio`)
- Uses zlib compression to detect repetitive/hallucinated text
- Threshold: 2.4 (matches Python default)

**Log Probability Threshold:**
- Location: `WhisperSTT.swift` lines 195-210
- Skips segments with avg_logprob < -1.0
- Configurable via `logprobThreshold` parameter

**No Speech Detection:**
- Location: `WhisperDecoding.swift` line 155 (`noSpeechProb` calculation)
- Uses probability of EOT token at first decoding step
- Threshold: 0.6 (matches Python default)

**Timestamp Rules (ApplyTimestampRules):**
- Location: `WhisperDecoding.swift` lines 215-330
- Enforces timestamp/text alternation
- Ensures monotonically increasing timestamps
- Implements `max_initial_timestamp` (1.0s default)
- Uses probability heuristic for timestamp forcing

**Prompt/Context Conditioning:**
- Location: `WhisperSTT.swift` lines 174-177, `WhisperDecoding.swift` lines 103-111
- `conditionOnPreviousText` parameter enables using previous segment output as prompt
- Prepends `<|startofprev|>` token followed by previous tokens
- Resets prompt when temperature > 0.5 (matches Python behavior)

---

### Hallucination Prevention (RESOLVED - December 2025)

Additional filters to catch and prevent hallucinations, particularly for short/final audio segments:

**Seek Sanity Check:**
- Location: `WhisperSTT.swift` line 347-352
- Prevents hallucinated timestamps from causing seek to jump beyond the current segment
- The model can hallucinate timestamps pointing far into the future (e.g., 25s when only 2s of audio remains)
- Uses `min(timestampSeek, segmentSize)` to cap seek advancement

**Timestamp Window Filter:**
- Location: `WhisperSTT.swift` lines 410-418
- Filters out segments where end timestamp exceeds the segment window (+ 1s tolerance)
- Catches impossible timestamps like 20s into a 2s segment
- These are clear indicators of hallucination

**Low-Confidence Filter:**
- Location: `WhisperSTT.swift` lines 421-429
- Discards all segments when temperature >= 0.8 AND avg_logprob < -2.0
- High temperature + low confidence after fallback exhaustion indicates hallucination
- Particularly effective for silence/unclear audio at end of content

**Short Segment Temperature Optimization:**
- Location: `WhisperSTT.swift` lines 189-191
- Uses fewer temperature steps for very short segments (< 2s): [0.0, 0.5, 1.0] instead of [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- Short segments are unlikely to benefit from extensive fallbacks and often produce hallucinations anyway
- Reduces processing time significantly for final audio segments

**Note:** These improvements were also backported to the Python mlx-audio-plus implementation.

---

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

| Priority | Feature | Impact | Effort | Status |
|----------|---------|--------|--------|--------|
| ~~1~~ | ~~Temperature Fallback~~ | ~~High~~ | ~~Low~~ | ✅ Done |
| ~~2~~ | ~~Compression Ratio / Log Prob Filters~~ | ~~High~~ | ~~Low~~ | ✅ Done |
| ~~3~~ | ~~Timestamp Rules (ApplyTimestampRules)~~ | ~~High~~ | ~~Medium~~ | ✅ Done |
| ~~4~~ | ~~Prompt/Context Conditioning~~ | ~~Medium~~ | ~~Medium~~ | ✅ Done |
| 1 | Real-Time Streaming | High | High | ❌ |
| 2 | Energy-Based VAD | High | Medium | ❌ |
| 3 | Progress Callbacks | Medium | Low | ❌ |
| 4 | Transcription Cancellation | Medium | Low | ❌ |
| 5 | Logits Filter Architecture | Medium | Medium | ❌ |
| 6 | Model Prefill | Medium | Medium | ❌ |
| 7 | Concurrent Processing | Medium | Medium | ❌ |
| 8 | Audio Channel Options | Low | Low | ❌ |
| 9 | Custom Token Suppression API | Low | Low | ❌ |
| 10 | Detailed Timing Metrics | Low | Low | ❌ |
| 11 | Beam Search | Low | High | ❌ |

**Note on Beam Search:** Neither WhisperKit nor mlx-audio-plus has implemented beam search. Given that greedy decoding with temperature fallback works well for most use cases, beam search should be considered low priority unless specifically needed for research applications.

**Recently Completed:** Temperature fallback, compression ratio filtering, log prob thresholds, timestamp rules (ApplyTimestampRules), prompt/context conditioning, and hallucination prevention (seek sanity check, timestamp window filter, low-confidence filter, short segment optimization) have all been implemented.

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
