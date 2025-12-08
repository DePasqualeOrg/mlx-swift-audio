# MLX Swift Audio Porting Roadmap

This document outlines the roadmap for porting functionality from [mlx-audio-plus](https://github.com/DePasqualeOrg/mlx-audio-plus) (Python) to [mlx-swift-audio](https://github.com/DePasqualeOrg/mlx-swift-audio) (Swift).

---

## Current State Comparison

### What's Already in Swift

| Component | Status | Notes |
|-----------|--------|-------|
| **TTS Models** | | |
| Chatterbox | ✅ Complete | Full voice cloning support |
| Kokoro | ✅ Complete | GPLv3 via espeak-ng |
| OuteTTS | ✅ Complete | Speaker profiles |
| Orpheus | ✅ Complete | SNAC codec integration |
| Marvis | ✅ Complete | Mimi codec, streaming |
| **Codecs** | | |
| DAC | ✅ Complete | Standalone codec package |
| SNAC | ✅ Complete | Embedded in Orpheus |
| Mimi | ✅ Complete | Embedded in Marvis |
| **Infrastructure** | | |
| Audio playback | ✅ Complete | AudioSamplePlayer, AudioFilePlayer |
| Model loading | ✅ Complete | Via swift-transformers |
| Sentence tokenization | ✅ Complete | SentenceTokenizer utility |

### What's Missing from Swift

| Component | Priority | Complexity | Dependencies |
|-----------|----------|------------|--------------|
| **STT Models** | | | |
| Whisper | 🔴 High | Medium | None |
| Voxtral | 🟡 Medium | Medium | None |
| Parakeet | 🟢 Low | High | Conformer impl |
| Wav2Vec | 🟢 Low | Medium | None |
| **TTS Models** | | | |
| Dia | 🟡 Medium | Medium | DAC (exists) |
| Spark | 🟡 Medium | High | Qwen2, BiCodec |
| Bark | 🟢 Low | High | Encodec |
| Sesame | 🟢 Low | High | Mimi, Llama |
| IndexTTS | 🟢 Low | High | BigVGAN, Conformer |
| **Codecs** | | | |
| Encodec | 🟡 Medium | Medium | LSTM layers |
| Vocos | 🟡 Medium | Low | ConvNeXt blocks |
| BigVGAN | 🟢 Low | Medium | Snake activations |
| S3 Tokenizer | 🟢 Low | Medium | Already partial |
| **Infrastructure** | | | |
| API Server | 🟢 Low | Medium | Vapor or similar |

---

## Phased Implementation Plan

### Phase 1: Speech-to-Text Foundation

**Goal**: Add STT capability to enable full voice cloning workflows and transcription features.

#### 1.1 Whisper Implementation
**Priority**: 🔴 High
**Estimated Effort**: 2-3 weeks

**Why First?**
- Foundational for voice cloning (auto-transcription of reference audio)
- Well-documented architecture
- Large community and model availability
- OpenAI standard for STT

**Components to Port**:
1. `whisper.py` - Main model architecture
   - Encoder (AudioEncoder)
   - Decoder (TextDecoder)
   - Multi-head attention

2. `audio.py` - Audio preprocessing
   - Log-mel spectrogram generation
   - Audio padding/trimming

3. `tokenizer.py` - Whisper tokenizer
   - Tiktoken-based BPE
   - Special tokens handling

4. `decoding.py` - Decoding strategies
   - Greedy decoding
   - Beam search
   - Temperature sampling

5. `timing.py` - Timestamp generation
   - Word-level timestamps
   - DTW alignment

**Swift Architecture**:
```
package/
├── STT/
│   ├── Whisper/
│   │   ├── WhisperModel.swift
│   │   ├── WhisperEncoder.swift
│   │   ├── WhisperDecoder.swift
│   │   ├── WhisperTokenizer.swift
│   │   ├── WhisperAudio.swift
│   │   ├── WhisperDecoding.swift
│   │   └── WhisperTiming.swift
│   └── STTEngine.swift  # Common STT interface
```

**Testing**:
- Unit tests for each component
- Integration tests with sample audio files
- Accuracy comparison with Python implementation

---

### Phase 2: Additional Codecs

**Goal**: Add codec support required for future TTS models.

#### 2.1 Encodec Implementation
**Priority**: 🟡 Medium
**Estimated Effort**: 1-2 weeks

**Why Important?**
- Required for Bark TTS
- Meta's foundational neural codec
- Widely used in audio ML

**Components**:
1. Encoder (CNN-based)
2. Decoder (transposed convolutions)
3. Residual Vector Quantizer (RVQ)
4. LSTM layers for temporal modeling

**Challenges**:
- LSTM implementation in MLX (may need custom layer)
- RVQ codebook management

#### 2.2 Vocos Implementation
**Priority**: 🟡 Medium
**Estimated Effort**: 1 week

**Why Important?**
- High-quality vocoder
- Lightweight and fast
- Used by multiple models

**Components**:
1. ConvNeXt blocks
2. iSTFT head
3. Feature upsampling

---

### Phase 3: Dialogue TTS

**Goal**: Add models optimized for conversational speech.

#### 3.1 Dia Implementation
**Priority**: 🟡 Medium
**Estimated Effort**: 2 weeks

**Why Dia?**
- Dialogue-focused (multi-speaker conversations)
- Non-verbal sounds support
- DAC codec already available in Swift
- Good quality for conversational AI

**Components**:
1. DiaModel (custom transformer)
2. KV Cache for efficient generation
3. Speaker tag handling ([S1], [S2])
4. DAC integration (already exists)

**Configuration**:
- Model config parsing
- Weight loading from safetensors

---

### Phase 4: LLM-Based TTS

**Goal**: Add models leveraging large language models.

#### 4.1 Spark TTS Implementation
**Priority**: 🟡 Medium
**Estimated Effort**: 3-4 weeks

**Why Spark?**
- Qwen2-based (LLM architecture)
- High-quality voice cloning
- Configurable pitch/speed/gender

**Components**:
1. Qwen2 model integration (via MLXLLM)
2. BiCodec tokenizer
3. Global/semantic token handling
4. Voice cloning pipeline

**Dependencies**:
- mlx-swift-lm integration
- BiCodec implementation

---

### Phase 5: Advanced STT

**Goal**: Expand STT capabilities with specialized models.

#### 5.1 Voxtral Implementation
**Priority**: 🟢 Low
**Estimated Effort**: 2 weeks

**Components**:
- Mistral-style audio encoder
- Custom attention mechanisms

#### 5.2 Parakeet Implementation
**Priority**: 🟢 Low
**Estimated Effort**: 3-4 weeks

**Components**:
- Conformer encoder
- CTC/RNN-T/TDT decoders
- Alignment utilities

---

### Phase 6: Complex TTS Models

**Goal**: Add remaining complex TTS models.

#### 6.1 Bark Implementation
**Priority**: 🟢 Low
**Estimated Effort**: 4-5 weeks

**Why Complex?**
- Three-stage pipeline (semantic → coarse → fine)
- Multiple transformer models
- Encodec integration required

**Components**:
1. Semantic model (GPT-like)
2. Coarse acoustics model
3. Fine acoustics model
4. Encodec integration

#### 6.2 Sesame Implementation
**Priority**: 🟢 Low
**Estimated Effort**: 3-4 weeks

**Components**:
- Llama backbone
- Mimi codec (extract from Marvis)
- Depth decoder
- Watermarking (optional)

#### 6.3 IndexTTS Implementation
**Priority**: 🟢 Low
**Estimated Effort**: 4-5 weeks

**Components**:
- GPT-2 backbone
- Conformer encoder
- Perceiver resampler
- BigVGAN vocoder
- ECAPA-TDNN speaker encoder

---

### Phase 7: Infrastructure

**Goal**: Add API server and developer tools.

#### 7.1 API Server (Optional)
**Priority**: 🟢 Low
**Estimated Effort**: 2-3 weeks

**Framework Options**:
- Vapor (Swift web framework)
- Native HTTP server

**Endpoints**:
- `POST /v1/audio/speech` - TTS
- `POST /v1/audio/transcriptions` - STT
- `GET /v1/models` - List models

---

## Recommended Implementation Order

```
Phase 1: Whisper STT (Foundation)
    ↓
Phase 2: Encodec + Vocos (Codec Support)
    ↓
Phase 3: Dia (Dialogue TTS)
    ↓
Phase 4: Spark (LLM-based TTS)
    ↓
Phase 5: Voxtral, Parakeet (Advanced STT)
    ↓
Phase 6: Bark, Sesame, IndexTTS (Complex TTS)
    ↓
Phase 7: API Server (Infrastructure)
```

---

## Technical Considerations

### 1. Swift Package Structure

```swift
// Package.swift additions
.library(name: "MLXAudioSTT", targets: ["MLXAudioSTT"]),
.library(name: "Whisper", targets: ["Whisper"]),

.target(
    name: "MLXAudioSTT",
    dependencies: ["MLXAudio", "Whisper"],
    path: "package/STT"
),
.target(
    name: "Whisper",
    dependencies: ["MLXAudio"],
    path: "package/STT/Whisper"
),
```

### 2. Common Components to Extract

The following utilities should be shared across models:

```swift
// package/Shared/
├── AudioProcessing.swift   // STFT, mel spectrograms
├── WindowFunctions.swift   // Hanning, Hamming, etc.
├── Resampling.swift        // Audio resampling
└── AttentionLayers.swift   // Common attention patterns
```

### 3. Model Loading Pattern

Follow the existing pattern from TTS models:

```swift
public protocol STTEngine {
    func load() async throws
    func transcribe(_ audio: URL) async throws -> TranscriptionResult
    func transcribe(_ samples: MLXArray, sampleRate: Int) async throws -> TranscriptionResult
}

public struct TranscriptionResult {
    let text: String
    let segments: [TranscriptionSegment]?
    let language: String?
}
```

### 4. Memory Management

For large models like Whisper large-v3:
- Implement lazy loading
- Use MLX's memory optimization features
- Support model offloading

### 5. Testing Strategy

Each phase should include:
1. Unit tests for individual components
2. Integration tests with real audio
3. Performance benchmarks
4. Accuracy validation against Python reference

---

## Success Metrics

### Phase 1 Complete When:
- [ ] Whisper tiny/base/small models working
- [ ] Word-level timestamps accurate
- [ ] Real-time factor < 0.5x for base model
- [ ] Memory usage within 2GB for base model

### Phase 2 Complete When:
- [ ] Encodec encode/decode round-trip working
- [ ] Vocos vocoder producing quality audio
- [ ] Both integrated with existing infrastructure

### Phase 3-6 Complete When:
- [ ] Each model produces audio matching Python quality
- [ ] Voice cloning working where applicable
- [ ] Documentation and examples provided

### Project Complete When:
- [ ] Feature parity with mlx-audio-plus
- [ ] All models tested and documented
- [ ] Example apps demonstrating usage
- [ ] API server (optional) functional

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| MLX Swift API differences | Medium | Reference mlx-swift-lm patterns |
| LSTM support in MLX | Medium | Custom layer or workaround |
| Memory constraints | High | Lazy loading, offloading |
| Tokenizer compatibility | Medium | Use swift-transformers |
| Audio I/O differences | Low | AVFoundation well-supported |

---

## Resources

### Reference Implementations
- [mlx-audio-plus](https://github.com/DePasqualeOrg/mlx-audio-plus) - Python reference
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) - LLM patterns
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - C++ Whisper reference

### Documentation
- [MLX Swift Documentation](https://ml-explore.github.io/mlx-swift/)
- [MLX Python Documentation](https://ml-explore.github.io/mlx/)
- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)

### Model Weights
- [mlx-community](https://huggingface.co/mlx-community) - Pre-converted weights
- [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3)
