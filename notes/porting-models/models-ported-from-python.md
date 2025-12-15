# MLX Audio Plus Investigation Report

This document provides a comprehensive analysis of the [mlx-audio-plus](https://github.com/DePasqualeOrg/mlx-audio-plus) repository and its functionality for potential porting to mlx-swift-audio.

## Repository Overview

**mlx-audio-plus** is a Python library for audio inference on Apple Silicon using MLX. It builds upon [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) and adds additional models.

### Core Structure

```
mlx-audio-plus/
├── mlx_audio/
│   ├── tts/              # Text-to-Speech models
│   │   ├── models/
│   │   │   ├── bark/       # Suno Bark TTS
│   │   │   ├── chatterbox/ # Resemble AI Chatterbox
│   │   │   ├── dia/        # NariLabs Dia
│   │   │   ├── indextts/   # IndexTTS (GPT-2 + Conformer)
│   │   │   ├── kokoro/     # Kokoro TTS
│   │   │   ├── llama/      # Llama-based TTS
│   │   │   ├── outetts/    # OuteTTS
│   │   │   ├── sesame/     # Sesame TTS (CSM)
│   │   │   └── spark/      # Spark TTS
│   │   ├── generate.py     # Main TTS generation entry point
│   │   └── utils.py        # Model loading utilities
│   ├── stt/              # Speech-to-Text models
│   │   ├── models/
│   │   │   ├── whisper/    # OpenAI Whisper
│   │   │   ├── voxtral/    # Mistral Voxtral
│   │   │   ├── parakeet/   # NVIDIA Parakeet
│   │   │   └── wav2vec/    # Meta Wav2Vec
│   │   └── generate.py     # Main STT generation entry point
│   ├── codec/            # Audio codecs
│   │   ├── models/
│   │   │   ├── descript/   # DAC (Descript Audio Codec)
│   │   │   ├── encodec/    # Meta Encodec
│   │   │   ├── mimi/       # Kyutai Mimi
│   │   │   ├── snac/       # SNAC
│   │   │   ├── vocos/      # Vocos vocoder
│   │   │   ├── s3/         # S3 tokenizer
│   │   │   └── bigvgan/    # BigVGAN vocoder
│   │   └── __init__.py
│   ├── sts/              # Speech-to-Speech
│   ├── server.py         # FastAPI OpenAI-compatible server
│   └── utils.py          # Common utilities (STFT, mel filters, etc.)
```

---

## TTS Models Analysis

### 1. Chatterbox (Resemble AI)

**Status in Swift**: ✅ Already implemented

**Architecture**:
- T3: LLaMA-based text-to-speech-token generator
- S3Gen: Flow matching decoder with HiFi-GAN vocoder
- VoiceEncoder: Speaker embedding extractor
- S3Tokenizer: Speech tokenizer for reference audio

**Key Features**:
- Voice cloning from reference audio (6-10 seconds recommended)
- Emotion exaggeration control
- Classifier-free guidance
- Sample rate: 24kHz

**Python Implementation**: `mlx_audio/tts/models/chatterbox/chatterbox.py`

---

### 2. Kokoro

**Status in Swift**: ✅ Already implemented

**Architecture**:
- CustomAlbert: BERT-like encoder for text processing
- ProsodyPredictor: Duration and F0/N prediction
- iSTFTNet Decoder: Audio synthesis
- Pipeline with language-specific phonemization

**Key Features**:
- Multiple voice presets
- Speed control
- Language code support (American English, British English, etc.)
- Sample rate: 24kHz

**Python Implementation**: `mlx_audio/tts/models/kokoro/kokoro.py`

---

### 3. Bark (Suno)

**Status in Swift**: ❌ Not implemented

**Architecture**:
- Semantic Model: GPT-like model for semantic token generation
- Coarse Acoustics Model: Generates coarse audio tokens
- Fine Acoustics Model: Refines to fine audio tokens
- Encodec Codec: Converts tokens to audio

**Key Features**:
- Multi-speaker support with speaker prompts
- Multilingual capabilities
- Non-speech sounds (laughter, music)
- Sample rate: 24kHz

**Complexity**: High - requires 3 cascaded transformer models + codec

**Python Implementation**: `mlx_audio/tts/models/bark/bark.py`

---

### 4. Spark TTS

**Status in Swift**: ❌ Not implemented

**Architecture**:
- Qwen2 backbone (LLM-based)
- BiCodec tokenizer for audio encoding/decoding
- Voice cloning support

**Key Features**:
- LLM-based generation using Qwen2
- Global and semantic token generation
- Voice cloning from reference audio
- Configurable pitch, speed, and gender
- Sample rate: 16kHz

**Complexity**: Medium - relies on mlx-lm for Qwen2

**Python Implementation**: `mlx_audio/tts/models/spark/spark.py`

---

### 5. Dia (NariLabs)

**Status in Swift**: ❌ Not implemented

**Architecture**:
- DiaModel: Custom transformer-based model
- DAC (Descript Audio Codec): Audio tokenization
- KV Cache for efficient generation

**Key Features**:
- Dialogue generation with speaker tags [S1], [S2]
- Non-verbal sound tags (laughs, sighs, etc.)
- Uses DAC codec at 44kHz
- Sample rate: 44kHz

**Complexity**: Medium - requires DAC codec (already in Swift)

**Python Implementation**: `mlx_audio/tts/models/dia/dia.py`

---

### 6. Sesame (CSM)

**Status in Swift**: ❌ Not implemented

**Architecture**:
- LlamaModel backbone (from mlx-lm)
- Mimi codec for audio encoding/decoding
- Depth decoder for multi-codebook generation
- Optional watermarking

**Key Features**:
- Llama-based speech generation
- Mimi codec integration
- Streaming decoder support
- Voice cloning via reference audio
- Sample rate: 24kHz

**Complexity**: High - requires Mimi codec (partially in Swift via Marvis) + LLM

**Python Implementation**: `mlx_audio/tts/models/sesame/sesame.py`

---

### 7. OuteTTS

**Status in Swift**: ✅ Already implemented

**Architecture**:
- Custom transformer model
- DAC codec for audio
- Speaker profile system

**Python Implementation**: `mlx_audio/tts/models/outetts/outetts.py`

---

### 8. IndexTTS

**Status in Swift**: ❌ Not implemented

**Architecture**:
- GPT-2 backbone
- Conformer encoder for conditioning
- Perceiver resampler
- BigVGAN vocoder
- ECAPA-TDNN speaker encoder

**Key Features**:
- Reference audio conditioning
- SentencePiece tokenization
- BigVGAN vocoder for high-quality synthesis
- Sample rate: 24kHz

**Complexity**: High - multiple complex components

**Python Implementation**: `mlx_audio/tts/models/indextts/indextts.py`

---

## STT Models Analysis

### 1. Whisper (OpenAI)

**Status in Swift**: ❌ Not implemented

**Architecture**:
- Encoder-decoder transformer
- Log-mel spectrogram input (80 or 128 channels)
- BPE tokenization
- Sinusoidal positional embeddings

**Key Features**:
- Multi-language transcription
- Word-level timestamps
- Language detection
- Multiple output formats (txt, srt, vtt, json)

**Complexity**: Medium - well-documented architecture

**Python Implementation**: `mlx_audio/stt/models/whisper/whisper.py`

---

### 2. Voxtral (Mistral)

**Status in Swift**: ❌ Not implemented

**Architecture**:
- Custom audio encoder
- Mistral-style attention mechanisms
- Multi-scale feature extraction

**Key Features**:
- State-of-the-art accuracy
- Efficient processing
- Streaming support

**Complexity**: Medium-High

**Python Implementation**: `mlx_audio/stt/models/voxtral/voxtral.py`

---

### 3. Parakeet (NVIDIA)

**Status in Swift**: ❌ Not implemented

**Architecture**:
- Conformer encoder
- Multiple decoder variants:
  - CTC (Connectionist Temporal Classification)
  - RNN-T (Transducer)
  - TDT (Token-and-Duration Transducer)

**Key Features**:
- High accuracy speech recognition
- Alignment and timestamp support
- Multiple decoding strategies

**Complexity**: High - requires Conformer + various decoders

**Python Implementation**: `mlx_audio/stt/models/parakeet/parakeet.py`

---

### 4. Wav2Vec (Meta)

**Status in Swift**: ❌ Not implemented

**Architecture**:
- CNN feature extractor
- Transformer encoder
- Contrastive learning framework

**Key Features**:
- Self-supervised pre-training
- Feature extraction
- Fine-tuned for ASR

**Python Implementation**: `mlx_audio/stt/models/wav2vec/wav2vec.py`

---

## Codec Models Analysis

### Currently Available in Swift

| Codec | Status | Location |
|-------|--------|----------|
| DAC | ✅ | `package/Codec/DAC/` |
| SNAC | ✅ | `package/TTS/Orpheus/SNAC/` |
| Mimi | ✅ | `package/TTS/Marvis/Mimi/` |

### Missing Codecs

| Codec | Status | Notes |
|-------|--------|-------|
| Encodec | ❌ | Used by Bark, foundational codec |
| Vocos | ❌ | High-quality vocoder |
| S3 Tokenizer | ❌ | Used by Chatterbox (available but embedded) |
| BigVGAN | ❌ | Used by IndexTTS, high-quality vocoder |

---

## Server & API

**Status in Swift**: ❌ Not implemented

The Python version includes a FastAPI-based server (`mlx_audio/server.py`) with:

- OpenAI-compatible endpoints:
  - `POST /v1/audio/speech` - TTS generation
  - `POST /v1/audio/transcriptions` - STT transcription
  - `GET /v1/models` - List available models
  - `POST /v1/models` - Load a model
  - `DELETE /v1/models` - Unload a model

- Features:
  - CORS support
  - Async model loading
  - Streaming audio responses
  - Multi-worker support

---

## Common Utilities

### Audio Processing (`mlx_audio/utils.py`)

Key functions available in Python that may need Swift equivalents:

1. **Window Functions**: hanning, hamming, blackman, bartlett
2. **STFT/iSTFT**: Short-time Fourier transform
3. **Mel Filterbank**: mel_filters with HTK and Slaney scales
4. **Audio Loading**: Via soundfile library

### Generation Base (`mlx_audio/tts/models/base.py`)

**GenerationResult** dataclass:
```python
@dataclass
class GenerationResult:
    audio: mx.array
    samples: int
    sample_rate: int
    segment_idx: int
    token_count: int
    audio_duration: str
    real_time_factor: float
    prompt: dict
    audio_samples: dict
    processing_time_seconds: float
    peak_memory_usage: float
```

---

## Dependencies Analysis

### Python Dependencies (requirements.txt)

| Package | Purpose | Swift Alternative |
|---------|---------|-------------------|
| mlx | Core MLX operations | MLX, MLXFFT |
| mlx-vlm | Vision-language models | - |
| transformers | Model loading, tokenizers | swift-transformers |
| sentencepiece | Tokenization | - |
| huggingface_hub | Model downloads | Built into swift-transformers |
| sounddevice | Audio playback | AVFoundation |
| soundfile | Audio I/O | AVFoundation |
| fastapi/uvicorn | API server | Vapor (if needed) |
| tiktoken | BPE tokenization | - |
| espeak-ng | Phonemization | espeak-ng-spm |

---

## Key Observations

### 1. Architecture Patterns

- Most TTS models follow: Text → Tokens → Audio pattern
- LLM-based models (Spark, Sesame) leverage existing mlx-lm infrastructure
- Codec models are reusable across multiple TTS/STT systems

### 2. Code Reuse Opportunities

- Many models share common components (attention, transformer layers)
- Codec implementations can be standalone packages
- Utility functions (STFT, mel filters) are generic

### 3. Porting Considerations

- Python uses NumPy/SciPy for some audio operations (resampling)
- MLX operations translate directly to Swift MLX
- Configuration/weight loading patterns are similar

### 4. Testing Strategy

The Python repo includes tests in:
- `mlx_audio/tts/tests/`
- `mlx_audio/stt/tests/`
- `mlx_audio/codec/tests/`

---

## Recommendations

### Immediate Priorities

1. **Whisper STT**: Foundational for voice cloning workflows
2. **Encodec**: Required for Bark and other models
3. **Vocos**: High-quality vocoder for multiple models

### Medium-Term Goals

4. **Dia TTS**: Dialogue generation, uses existing DAC
5. **Spark TTS**: Requires Qwen2 integration
6. **Parakeet STT**: High-accuracy alternative to Whisper

### Long-Term Goals

7. **Bark TTS**: Complex multi-stage pipeline
8. **Sesame TTS**: Requires full Mimi codec extraction
9. **IndexTTS**: Multiple complex components
10. **API Server**: OpenAI-compatible endpoints

---

## Meta Framework Comparison

This section compares the overall framework architecture between mlx-audio-plus (Python) and mlx-swift-audio (Swift).

### Feature Comparison Matrix

| Feature | Python (mlx-audio-plus) | Swift (mlx-swift-audio) |
|---------|------------------------|-------------------------|
| **Unified Model Loading** | ✅ Single `load_model()` for TTS & STT | ❌ No unified loader |
| **Auto Model Detection** | ✅ Detects model type from config/name | ❌ Must specify engine type |
| **OpenAI-Compatible API** | ✅ FastAPI server with `/v1/audio/*` | ❌ Not implemented |
| **Common Audio Utilities** | ✅ Shared STFT, mel filters, resampling | ⚠️ Scattered per-model |
| **Model Conversion Tools** | ✅ `convert.py` with quantization | ❌ Not applicable |
| **CLI Entry Points** | ✅ `mlx_audio.tts.generate`, `mlx_audio.stt.generate` | ❌ Library only |
| **Model Registry** | ✅ Dynamic discovery via `get_available_models()` | ⚠️ Hardcoded `TTSProvider` enum |
| **Common Base Classes** | ✅ `BaseModelArgs`, `GenerationResult` | ⚠️ `TTSEngine` protocol, `AudioResult` |
| **STT Support** | ✅ 4 models (Whisper, Voxtral, Parakeet, Wav2Vec) | ❌ None |
| **Streaming Support** | ✅ Generator-based streaming | ✅ `AsyncStream` + `StreamingGranularity` |
| **Memory Management** | ⚠️ Manual `mx.clear_cache()` | ✅ Explicit `unload()`, `cleanup()` |
| **Type Safety** | ⚠️ Duck typing, runtime errors | ✅ Protocol conformance, compile-time |

### Python Framework Architecture

**Unified Model Loading** (`mlx_audio/utils.py`):
```python
from mlx_audio.utils import load_model

# Auto-detects TTS vs STT from config.json "model_type" field
model = load_model("mlx-community/Chatterbox-TTS-4bit")  # → TTS
model = load_model("mlx-community/whisper-large-v3")    # → STT

# Dynamic model discovery
available = get_available_models()  # Scans models/ directory

# Unified generate interface - all models yield GenerationResult
for result in model.generate(text="Hello", ...):
    sf.write("output.wav", result.audio, result.sample_rate)
```

**Key Infrastructure Components**:

1. **`mlx_audio/utils.py`** - Shared audio DSP:
   - `stft()` / `istft()` - Short-time Fourier transform
   - `mel_filters()` - Mel filterbank with HTK/Slaney scales
   - `hanning()`, `hamming()`, `blackman()`, `bartlett()` - Window functions

2. **`mlx_audio/tts/utils.py`** - TTS model loading:
   - `load_model()` - Loads any TTS model by path/repo
   - `get_model_and_args()` - Resolves model type from config
   - `MODEL_REMAPPING` - Aliases for model types

3. **`mlx_audio/stt/utils.py`** - STT model loading:
   - Parallel structure to TTS utils
   - `load_audio()` - Audio file loading with resampling

4. **`mlx_audio/server.py`** - OpenAI-compatible API:
   - `ModelProvider` - Async model management
   - REST endpoints for TTS/STT

### Swift Framework Architecture

**Engine-Based Design** (`package/Protocols/TTSEngine.swift`):
```swift
// Must explicitly create specific engine
let engine = TTS.chatterbox()  // Factory method
try await engine.load()
try await engine.say("Hello", referenceAudio: ref)

// Feature discovery via provider
if engine.provider.supportsReferenceAudio {
    // Use reference audio feature
}
```

**Key Infrastructure Components**:

1. **`TTSEngine` Protocol** - Common interface:
   - `load()`, `stop()`, `unload()`, `cleanup()` - Lifecycle
   - `isLoaded`, `isGenerating`, `isPlaying` - State
   - `play(_ audio: AudioResult)` - Playback

2. **`TTSProvider` Enum** - Feature discovery:
   - `supportsSpeed`, `supportsExpressions`, `supportsReferenceAudio`
   - `sampleRate`, `displayName`

3. **`TTS` Factory Enum** - Engine creation:
   - `TTS.orpheus()`, `TTS.marvis()`, `TTS.chatterbox()`, etc.
   - Returns concrete types for full autocomplete

4. **`AudioResult` Enum** - Generation output:
   - `.samples(data:sampleRate:processingTime:)`
   - `.file(url:processingTime:)`

### What Python Has That Swift Lacks

1. **Unified `load_model()` Function**
   - Python: Single function loads ANY model, auto-detects category
   - Swift: Must know which `TTS.xyz()` factory to call

2. **Dynamic Model Discovery**
   - Python: Scans filesystem, supports `MODEL_REMAPPING` dict
   - Swift: `TTSProvider` enum requires code changes for new models

3. **Common Signal Processing Module**
   - Python: `mlx_audio/utils.py` has shared DSP functions
   - Swift: STFT/mel implementations duplicated in Kokoro, Chatterbox, Marvis

4. **STT Infrastructure**
   - Python: Full `stt/` module with models, utils, generate
   - Swift: No STT support

5. **Server/API Layer**
   - Python: Production-ready FastAPI server
   - Swift: Library only

6. **CLI Tools**
   - Python: `python -m mlx_audio.tts.generate --model X --text "Y"`
   - Swift: No command-line interface

### What Swift Does Better

1. **Type Safety**
   - Protocol conformance enforced at compile time
   - Engine-specific methods with typed parameters

2. **Swift Concurrency**
   - Native `async/await` throughout
   - `@MainActor` isolation for UI safety
   - `Sendable` conformance for thread safety

3. **Memory Management**
   - Explicit `unload()` preserves cached data
   - `cleanup()` for full resource release
   - Clear ownership semantics

4. **Observable State**
   - `@Observable` macro integration
   - `isLoaded`, `isGenerating`, `isPlaying` for UI binding

5. **Streaming Architecture**
   - `StreamingGranularity` enum (sentence vs frame)
   - `AudioChunk` struct for incremental playback

### Recommendations for Swift Framework Improvements

To achieve feature parity with Python's meta framework:

1. **Add Unified Model Registry**
   ```swift
   public enum MLXAudio {
       static func loadModel(_ identifier: String) async throws -> any AudioModel
       static func availableModels() -> [ModelInfo]
   }
   ```

2. **Extract Shared Audio DSP Module**
   ```swift
   // package/Audio/DSP/
   public struct AudioDSP {
       static func stft(_ signal: MLXArray, nFFT: Int, hopLength: Int, window: WindowFunction) -> MLXArray
       static func istft(_ stft: MLXArray, hopLength: Int, window: WindowFunction) -> MLXArray
       static func melSpectrogram(_ signal: MLXArray, sampleRate: Int, nMels: Int) -> MLXArray
       static func melFilterbank(sampleRate: Int, nFFT: Int, nMels: Int, scale: MelScale) -> MLXArray
   }
   ```

3. **Add STT Protocol**
   ```swift
   @MainActor
   public protocol STTEngine: Observable {
       var provider: STTProvider { get }
       func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws
       func transcribe(_ audio: URL) async throws -> TranscriptionResult
       func transcribe(_ samples: MLXArray, sampleRate: Int) async throws -> TranscriptionResult
   }
   ```

4. **Create Unified AudioModel Protocol**
   ```swift
   public protocol AudioModel {
       var modelType: AudioModelType { get }  // .tts or .stt
       var isLoaded: Bool { get }
       func load() async throws
       func unload() async
   }
   ```

### Summary

| Aspect | Python Advantage | Swift Advantage |
|--------|------------------|-----------------|
| Flexibility | Dynamic loading, runtime discovery | — |
| Type Safety | — | Compile-time guarantees |
| Feature Breadth | TTS + STT + Server | TTS only (currently) |
| Code Reuse | Shared utilities module | — |
| Memory Control | — | Explicit lifecycle methods |
| UI Integration | — | Observable, MainActor |
| Developer Experience | CLI tools, quick iteration | Autocomplete, type hints |

The Python framework is more **comprehensive** with unified loading, CLI tools, and server support. The Swift framework is more **type-safe** with better memory management and UI integration. The ideal Swift implementation would adopt Python's unified architecture while preserving Swift's type safety benefits.
