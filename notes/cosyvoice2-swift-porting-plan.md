# CosyVoice2 Swift Port Plan

This document catalogs all components needed to port CosyVoice2 from Python MLX to Swift MLX.

## References

| Resource | URL |
|----------|-----|
| CosyVoice2 PyTorch (official) | https://github.com/FunAudioLLM/CosyVoice |
| CosyVoice2 Python MLX | `../forked/mlx-audio-plus/mlx_audio/tts/models/cosyvoice2` |
| CosyVoice2 Paper | https://arxiv.org/html/2412.10117v2 |
| HuggingFace Models | https://huggingface.co/mlx-community/CosyVoice2-0.5B-fp16 |

## Swift Repo Architecture

The existing Swift MLX Audio repo follows this pattern for TTS engines:

```
package/
├── Protocols/
│   └── TTSEngine.swift          # Core protocol all engines implement
├── Models/
│   ├── TTSProvider.swift        # Engine metadata (name, sample rate, features)
│   ├── AudioResult.swift        # Generated audio container
│   └── Voice.swift              # Voice definitions
├── Audio/
│   ├── AudioResampler.swift     # AVFoundation-based resampling
│   └── AudioSamplePlayer.swift  # Playback
├── Utils/
│   └── SentenceTokenizer.swift  # Text chunking
└── TTS/
    └── {EngineName}/
        ├── {EngineName}Engine.swift  # Public API (@Observable, @MainActor, TTSEngine)
        ├── {EngineName}TTS.swift     # Actor wrapper for thread-safety (optional)
        ├── {EngineName}Model.swift   # Core model (Module subclass)
        └── Config/                   # Configuration structs
```

### Layer Responsibilities

**1. Engine Layer** (`{EngineName}Engine.swift`)
- `@Observable @MainActor` class conforming to `TTSEngine` protocol
- Public API: `load()`, `generate()`, `say()`, `generateStreaming()`, `sayStreaming()`, `stop()`, `unload()`
- Owns `TTSPlaybackController` for audio playback
- Manages state: `isLoaded`, `isGenerating`, `isPlaying`, `generationTime`
- Engine-specific parameters as mutable properties (e.g., `temperature`, `topP`, `cfgWeight`)
- Auto-loads model on first generation if not already loaded

**2. TTS Layer** (`{EngineName}TTS.swift`) - Optional actor wrapper
- `actor` for thread-safe model access (when needed)
- Wraps Model layer, provides `generate()` and `generateStreaming()`
- Handles text chunking (sentence splitting + max character limits)
- Returns `AsyncThrowingStream<[Float], Error>` for streaming

**3. Model Layer** (`{EngineName}Model.swift`)
- `Module` subclass with `@ModuleInfo` property wrappers for sub-models
- Weight loading with `sanitizeWeights()` for key remapping
- `prepareConditionals()` - pre-compute reference audio embeddings
- `generate()` - core inference (text → tokens → mel → audio)
- Constants for special tokens, sample rates, vocab sizes

### Key Patterns

**TTSProvider enum** (`Models/TTSProvider.swift`):
- Register new engine in `TTSProvider` enum
- Define `sampleRate`, `displayName`
- Set feature flags: `supportsReferenceAudio`, `supportsExpressions`, etc.

**Reference Audio → Conditionals**:
- `prepareConditionals(refWav:refSr:)` extracts embeddings once
- Conditionals struct is `@unchecked Sendable` (contains MLXArray)
- Reusable across multiple generations for same speaker

**Weight Loading**:
- `Module.update(parameters:verify:)` for hierarchical weight distribution
- `sanitizeWeights()` handles key remapping (Python → Swift naming)
- Quantization detection via `.scales` keys
- S3Tokenizer loaded from separate repo (shared across models)

**Streaming**:
- Engine returns `AsyncThrowingStream<AudioChunk, Error>`
- Model returns `AsyncThrowingStream<[Float], Error>`
- `TTSPlaybackController` handles stream → playback conversion
- `MLXMemory.clearCache()` between chunks

**Text Processing**:
- `SentenceTokenizer.splitIntoSentences()` for natural breaks
- `TextSplitter.splitToMaxLength()` for overflow chunks
- `puncNorm()` for text normalization (punctuation, capitalization)

## CosyVoice2 Architecture Overview

CosyVoice2 is a high-quality TTS model with:
1. **Qwen2 LLM** - Speech token generation from text (Qwen2-0.5B-Instruct)
2. **Flow Matching** - Mel spectrogram synthesis from speech tokens
3. **HiFi-GAN** - Waveform generation from mel spectrograms (24kHz)
4. **CAMPlus Speaker Encoder** - Speaker embedding extraction
5. **S3 Tokenizer** - Speech tokenization for voice cloning

## Shared Components (Already Ported)

These components are shared with Chatterbox and already exist in Swift:

### S3 Tokenizer (from codec module)
| Component | Swift Location | Status |
|-----------|----------------|--------|
| S3TokenizerV2 | TTS/Chatterbox/S3Tokenizer/S3Tokenizer.swift | ✅ Ported |
| AudioEncoderV2 | TTS/Chatterbox/S3Tokenizer/S3Tokenizer.swift | ✅ Ported |
| FSMNMultiHeadAttention | TTS/Chatterbox/S3Tokenizer/S3Tokenizer.swift | ✅ Ported |
| FSQVectorQuantization | TTS/Chatterbox/S3Tokenizer/S3Tokenizer.swift | ✅ Ported |
| FSQCodebook | TTS/Chatterbox/S3Tokenizer/S3Tokenizer.swift | ✅ Ported |
| logMelSpectrogram | TTS/Chatterbox/S3Tokenizer/S3TokenizerUtils.swift | ✅ Ported |

### Speaker Encoder
| Component | Swift Location | Status |
|-----------|----------------|--------|
| CAMPPlus | TTS/Chatterbox/S3Gen/CAMPPlus.swift | ✅ Ported |
| kaldiFbankCAMPPlus | TTS/Chatterbox/S3Gen/CAMPPlus.swift | ✅ Ported |

### Flow Matching / Conformer Encoder
| Component | Swift Location | Status |
|-----------|----------------|--------|
| UpsampleConformerEncoder | TTS/Chatterbox/S3Gen/Transformer/UpsampleConformerEncoder.swift | ✅ Ported |
| ConformerEncoderLayer | TTS/Chatterbox/S3Gen/Transformer/ConformerEncoderLayer.swift | ✅ Ported |
| MultiHeadedAttention | TTS/Chatterbox/S3Gen/Transformer/Attention.swift | ✅ Ported |
| RelPositionMultiHeadedAttention | TTS/Chatterbox/S3Gen/Transformer/Attention.swift | ✅ Ported |
| ConvolutionModule | TTS/Chatterbox/S3Gen/Transformer/ConvolutionModule.swift | ✅ Ported |
| PositionwiseFeedForward | TTS/Chatterbox/S3Gen/Transformer/PositionwiseFeedForward.swift | ✅ Ported |
| EspnetRelPositionalEncoding | TTS/Chatterbox/S3Gen/Transformer/Embedding.swift | ✅ Ported |
| LinearNoSubsampling | TTS/Chatterbox/S3Gen/Transformer/Subsampling.swift | ✅ Ported |

### Flow Matching / Decoder
| Component | Swift Location | Status |
|-----------|----------------|--------|
| CausalMaskedDiffWithXvec | TTS/Chatterbox/S3Gen/Flow.swift | ✅ Ported |
| ConditionalDecoder | TTS/Chatterbox/S3Gen/S3GenDecoder.swift | ✅ Ported |
| BASECFM | TTS/Chatterbox/S3Gen/FlowMatching.swift | ✅ Ported |
| CFMParams | TTS/Chatterbox/S3Gen/FlowMatching.swift | ✅ Ported |

### HiFi-GAN Shared Components
| Component | Swift Location | Status |
|-----------|----------------|--------|
| ResBlock | TTS/Chatterbox/S3Gen/HiFiGAN.swift | ✅ Ported |
| Snake | TTS/Chatterbox/S3Gen/HiFiGAN.swift | ✅ Ported |
| hannWindowPeriodic | TTS/Chatterbox/S3Gen/HiFiGAN.swift | ✅ Ported |
| stftHiFiGAN | TTS/Chatterbox/S3Gen/HiFiGAN.swift | ✅ Ported |
| istftHiFiGAN | TTS/Chatterbox/S3Gen/HiFiGAN.swift | ✅ Ported |

### Mel Spectrogram
| Component | Swift Location | Status |
|-----------|----------------|--------|
| s3genMelSpectrogram | TTS/Chatterbox/S3Gen/Mel/S3GenMel.swift | ✅ Ported |

---

## Components to Port

> **Verification Process:** After porting, each component must be verified line-by-line against the Python MLX implementation. Add ✅ to the "Verified" column only after confirming the Swift implementation matches the Python behavior exactly.

### 1. Configuration (`config.py`)

| Done | Verified | Component | Python Location | Priority | Complexity |
|:----:|:--------:|-----------|-----------------|----------|------------|
| ✅ | | `LLMConfig` | cosyvoice2/config.py:14 | High | Low |
| ✅ | | `FlowConfig` | cosyvoice2/config.py:33 | High | Low |
| ✅ | | `HiFiGANConfig` | cosyvoice2/config.py:91 | High | Low |
| ✅ | | `CosyVoice2Config` | cosyvoice2/config.py:118 | High | Low |
| ✅ | | `ModelConfig` | cosyvoice2/config.py:181 | High | Low |

**Swift Location:** `CosyVoice2/Config/CosyVoice2Config.swift`

---

### 2. LLM Module (`llm/llm.py`)

**Note:** CosyVoice2 uses Qwen2-0.5B-Instruct as backbone (not LLaMA like Chatterbox).

| Done | Verified | Component | Python Location | Priority | Complexity | Notes |
|:----:|:--------:|-----------|-----------------|----------|------------|-------|
| ✅ | | `Qwen2Config` | llm/llm.py:28 | High | Low | Config dataclass |
| ✅ | | `Qwen2Encoder` | llm/llm.py:45 | High | Medium | Wraps mlx-lm Qwen2 model |
| ✅ | | `Qwen2LM` | llm/llm.py:133 | High | High | Main LLM with speech embeddings |
| ✅ | | `nucleus_sampling` | llm/llm.py:490 | High | Low | Sampling function |
| ✅ | | `ras_sampling` | llm/llm.py:534 | High | Low | Repetition-aware sampling |
| ✅ | | `top_k_sampling` | llm/llm.py:577 | Medium | Low | Alternative sampling |

**Swift Location:** `CosyVoice2/LLM/Qwen2LM.swift`

**Key differences from Chatterbox T3:**
- Uses Qwen2 instead of LLaMA
- Has special tokens: `sos_eos`, `task_id`, `fill_token`
- Supports bidirectional streaming (`inference_bistream`)
- Speech token size: 6561 + 3 special tokens

---

### 3. Flow Matching (`flow_matching.py`)

| Done | Verified | Component | Python Location | Priority | Complexity | Notes |
|:----:|:--------:|-----------|-----------------|----------|------------|-------|
| ✅ | | `CosyVoice2ConditionalCFM` | flow_matching.py:15 | High | Medium | CosyVoice2-specific CFM |
| ✅ | | `CFM_PARAMS` | flow_matching.py:12 | High | Low | Default params |

**Swift Location:** `CosyVoice2/Flow/CosyVoice2CFM.swift`

**Bug Fixed During Verification:** `dt` calculation in Euler solver was using `tSpan[step]` instead of `t`.

**Key differences from Chatterbox CFM:**
- Different `in_channels` (240 vs 320)
- Identical Euler solver with CFG
- Noise is generated dynamically at inference time (more memory efficient than pre-allocated buffer)

---

### 4. HiFi-GAN Vocoder (`hifigan.py`) - **New Implementation Required**

**Note:** CosyVoice2 HiFi-GAN is different from Chatterbox and requires a separate Swift implementation.

| Done | Verified | Component | Python Location | Priority | Complexity | Notes |
|:----:|:--------:|-----------|-----------------|----------|------------|-------|
| ✅ | | `linear_interpolate_1d` | hifigan.py:24 | High | Low | Linear interpolation helper |
| ✅ | | `SineGen2` | hifigan.py:75 | High | Medium | 24kHz sine generator with interpolation |
| ✅ | | `SourceModuleHnNSF2` | hifigan.py:176 | High | Medium | 24kHz NSF source module |
| ✅ | | `CosyF0Predictor` | hifigan.py:231 | High | Medium | Built-in F0 predictor (ConvRNNF0Predictor) |
| ✅ | | `CosyHiFTGenerator` | hifigan.py:311 | High | High | Main HiFi-GAN generator |

**Swift Location:** `CosyVoice2/HiFiGAN/CosyHiFTGenerator.swift`

**Key differences from Chatterbox HiFi-GAN:**
- **24kHz** output (vs 22.05kHz for Chatterbox)
- Built-in F0 predictor as submodule
- Different upsample rates: `[8, 5, 3]` vs `[10, 6, 4, 2]`
- Uses `SineGen2` with interpolation for phase calculation
- Uses `SourceModuleHnNSF2` for 24kHz mode
- ISTFT params: `n_fft=16, hop_len=4`

**Shared with Chatterbox (can reuse):**
- `ResBlock` (HiFiGAN residual block)
- `Snake` activation
- `hann_window_periodic`
- `stft` / `istft` functions

---

### 5. Speaker Encoder (`speaker_encoder.py`)

| Done | Verified | Component | Python Location | Priority | Complexity | Notes |
|:----:|:--------:|-----------|-----------------|----------|------------|-------|
| ✅ | | `CAMPlusSpeakerEncoder` | speaker_encoder.py:12 | High | Low | Wrapper around CAMPPlus |

**Swift Location:** `CosyVoice2/SpeakerEncoder/CAMPlusSpeakerEncoder.swift`

**Note:** This is a thin wrapper around the already-ported `CAMPPlus`. Just needs Swift wrapper class.

---

### 6. Main Model (`cosyvoice2.py`)

| Done | Verified | Component | Python Location | Priority | Complexity | Notes |
|:----:|:--------:|-----------|-----------------|----------|------------|-------|
| ✅ | | `CosyVoice2` | cosyvoice2.py:26 | High | High | Main model class |
| ✅ | | `load_cosyvoice2` | cosyvoice2.py:648 | High | High | Model loading with weight mapping |
| ✅ | | `Model` | cosyvoice2.py:877 | High | High | API wrapper for generate() |

**Swift Locations:**
- `CosyVoice2/CosyVoice2Model.swift` - Main model class
- `CosyVoice2/CosyVoice2TTS.swift` - Actor wrapper for thread-safety

**Core methods:**
- `generate_tokens` - LLM token generation
- `tokens_to_mel` - Flow matching
- `mel_to_audio` - HiFi-GAN vocoder

**Synthesis modes:** See [Inference Modes](#inference-modes) section below.

---

## Inference Modes

CosyVoice2 supports four inference modes, selected based on which inputs are provided. **`ref_audio` is required for all modes.**

### Mode Selection Table

| Mode | source_audio | ref_audio | ref_text | instruct_text | Method |
|------|--------------|-----------|----------|---------------|--------|
| Cross-lingual | - | ✓ | - | - | `synthesize_cross_lingual` |
| Zero-shot | - | ✓ | ✓ | - | `synthesize_zero_shot` |
| Instruct | - | ✓ | - | ✓ | `synthesize_instruct` |
| Voice Conversion | ✓ | ✓ | - | - | `synthesize_vc` |

### Mode Details

#### 1. Cross-lingual Mode (Default)
- **Use case**: Zero-shot TTS when you don't have a transcription of the reference audio
- **Inputs**: `text` + `ref_audio`
- **How it works**:
  1. LLM receives only the target text (no prompt speech tokens)
  2. LLM generates speech tokens based purely on text
  3. Flow model uses `ref_audio` for speaker identity (mel + speech tokens + embedding)
- **Best for**: Different language than reference, or when transcription unavailable

#### 2. Zero-shot Mode
- **Use case**: Voice cloning with semantic alignment
- **Inputs**: `text` + `ref_audio` + `ref_text` (transcription of reference)
- **How it works**:
  1. LLM receives `ref_text` + `ref_speech_tokens` as prompt context
  2. This teaches the LLM the alignment between text and speech for this speaker
  3. LLM then generates speech tokens for target text
  4. Flow model uses `ref_audio` for speaker identity
- **Best for**: Same language as reference, highest quality voice cloning

#### 3. Instruct Mode
- **Use case**: Control speech style with natural language instructions
- **Inputs**: `text` + `ref_audio` + `instruct_text`
- **How it works**:
  1. LLM receives `instruct_text` (ending with `<|endofprompt|>`) but NO speech tokens
  2. Instructions guide the speaking style (e.g., "Speak slowly and calmly")
  3. Flow model uses `ref_audio` for speaker identity
- **Examples**: "Read with excitement", "Speak in a whisper", "Sound professional"

#### 4. Voice Conversion (VC) Mode
- **Use case**: Convert source speech to target speaker's voice
- **Inputs**: `source_audio` + `ref_audio` (no text involved)
- **How it works**:
  1. Extract speech tokens from `source_audio` (content to convert)
  2. **Skip LLM entirely** - use source tokens directly
  3. Flow model converts source tokens to target voice using `ref_audio` conditioning
- **Limitation**: Source audio truncated to 30 seconds (S3 tokenizer constraint)
- **Future**: Chunked processing at silence points for longer audio

### Fine-Grained Speech Control Tokens

Users can embed special tokens in input text:

| Token | Description |
|-------|-------------|
| `[breath]` | Insert a breath |
| `[laughter]` | Insert laughter |
| `[cough]` | Insert a cough |
| `[sigh]` | Insert a sigh |
| `<strong>text</strong>` | Emphasize/stress text |
| `<laughter>text</laughter>` | Speak while laughing |

### Streaming Mode

Additionally, `synthesize_streaming` provides chunked generation:
- Generates speech tokens incrementally
- Processes chunks through flow matching and HiFi-GAN
- Yields audio chunks as they're ready
- Useful for low-latency applications

---

## Port Order (Recommended)

### Phase 1: Configuration & Infrastructure
1. [x] `CosyVoice2Config` and sub-configs
2. [x] `CAMPlusSpeakerEncoder` wrapper

### Phase 2: LLM Module
3. [x] `Qwen2Config`
4. [x] `Qwen2Encoder` (integrate with mlx-swift-transformers Qwen2)
5. [x] `Qwen2LM` with speech embeddings
6. [x] Sampling functions (`nucleus_sampling`, `ras_sampling`)

### Phase 3: Flow Matching
7. [x] `CosyVoice2ConditionalCFM`

### Phase 4: HiFi-GAN (New Implementation)
8. [x] `linear_interpolate_1d`
9. [x] `SineGen2`
10. [x] `SourceModuleHnNSF2`
11. [x] `CosyF0Predictor`
12. [x] `CosyHiFTGenerator`

### Phase 5: Main Model
13. [x] `CosyVoice2` class
14. [x] `load_cosyvoice2` weight loading
15. [x] `Model` wrapper class
16. [x] Synthesis modes implementation

### Phase 6: Integration & Testing
17. [x] Weight loading verification ✅
18. [ ] Integration tests
19. [ ] Voice cloning tests
20. [ ] Streaming tests
21. [ ] Performance optimization

---

## Phase 6 Detailed Tasks

### 6.1 Weight Loading Verification ✅ COMPLETED
- [x] Test `CosyVoice2TTS.load()` with `mlx-community/CosyVoice2-0.5B-fp16` weights
- [x] Verify all weight keys map correctly from Python to Swift model structure
- [x] Test weight loading for all sub-models (LLM, Flow, HiFi-GAN, CAMPlus, S3Tokenizer)
- [x] Verify quantized model loading (4-bit variant tested, loads in ~0.28s)

**Issues Fixed:**
1. **KVCache naming conflict** - Renamed `KVCache` to `CosyVoice2KVCache` in `Qwen2LM.swift` to avoid conflict with MLXLMCommon's KVCache protocol
2. **Flow decoder weight mapping** - Updated `CosyVoice2TTS.createModel()` to properly initialize `ConditionalDecoder` (estimator) with full config parameters. Weights use path `flow.decoder.estimator.*`
3. **CAMPlus weight loading** - Added `loadWeights(from:)` method to load from pre-filtered dictionary. The campplus weights are embedded in main `model.safetensors` with `campplus.*` prefix (815 keys)

**Test Results (all pass):**
- `testConfigLoading()` - Config loads with correct defaults
- `testWeightKeyAnalysis()` - All 2477 weight keys found (campplus: 815, flow: 1121, hift: 246, llm: 4, qwen2: 291)
- `testFp16ModelLoading()` - fp16 model loads in ~0.18s with speaker encoder verified
- `testFourBitModelLoading()` - 4-bit model loads in ~0.28s

### 6.2 Text Tokenizer Integration ✅ COMPLETED
- [x] Integrate Qwen2 tokenizer from `swift-transformers`
- [x] Add special tokens for speech control (`<|endofprompt|>`, `[breath]`, etc.)
- [x] Test tokenizer encode/decode roundtrip
- [x] Verify token IDs match Python implementation

**Implementation:**
- Modified Python conversion script to generate `tokenizer.json` (fast tokenizer format)
- Updated HuggingFace repos (fp16, 4-bit, 8-bit) with clean tokenizer format
- Added `PreTrainedTokenizer` to `CosyVoice2TTS` actor
- Implemented `encode()`, `decode()`, `tokenToId()` methods
- Updated `prepareConditionals()` to use built-in tokenizer

**Test Results:**
- English "Hello, world!" → 4 tokens: `[9707, 11, 1879, 0]`
- Chinese "你好世界" → 4 tokens: `[56568, 52801, 99244, 97120]`
- Round-trip decode matches original text
- Special tokens `<|im_end|>`, `<|endoftext|>` recognized

### 6.3 Integration Tests
- [ ] End-to-end inference: text → audio waveform
- [ ] Test with sample text inputs in English and Chinese
- [ ] Compare output audio quality against Python MLX output
- [ ] Test batch processing (multiple utterances)

### 6.4 Voice Cloning Tests
- [ ] Zero-shot mode: with reference audio transcription
- [ ] Cross-lingual mode: without reference transcription
- [ ] Instruct mode: with style instructions
- [ ] Voice conversion mode: source audio → target voice

### 6.5 Numerical Accuracy Verification
- [ ] Compare intermediate tensors (LLM output, flow output, mel spectrogram)
- [ ] Verify speaker embeddings match Python output for same reference audio
- [ ] Test with fixed random seed for reproducible noise in CFM
- [ ] Measure output similarity (mel spectrogram MSE, audio waveform correlation)

### 6.6 Streaming Tests
- [ ] Test chunk-based LLM token generation
- [ ] Test streaming flow matching with cache
- [ ] Test incremental HiFi-GAN synthesis
- [ ] Measure latency for first audio chunk

### 6.7 Performance Optimization
- [ ] Profile inference pipeline to identify bottlenecks
- [ ] Optimize memory usage with `MLX.eval()` placement
- [ ] Test `MLXMemory.clearCache()` for long sessions
- [ ] Benchmark against Python MLX implementation (tokens/sec, RTF)

### 6.8 Error Handling & Edge Cases
- [ ] Empty text input
- [ ] Very short reference audio (<1 second)
- [ ] Very long text input (>1000 tokens)
- [ ] Invalid audio format handling
- [ ] Graceful failure when model weights not found

### 6.9 Audio I/O Integration
- [ ] Load reference audio from file (WAV, MP3, etc.)
- [ ] Resample audio to required sample rates (16kHz, 24kHz)
- [ ] Save generated audio to file
- [ ] Support for audio trimming/silence removal

---

## Weight Files

**Pre-converted MLX weights available on HuggingFace:**

| Variant | Repo | Size |
|---------|------|------|
| 4-bit | `mlx-community/CosyVoice2-0.5B-4bit` | ~742 MB |
| 8-bit | `mlx-community/CosyVoice2-0.5B-8bit` | ~913 MB |
| fp16 | `mlx-community/CosyVoice2-0.5B-fp16` | ~1.5 GB |

**Contents of `mlx-community/CosyVoice2-0.5B-*` repos:**
```
├── config.json              # Model configuration
├── model.safetensors        # All model weights (~778 MB for 4-bit)
├── vocab.json               # Qwen2 tokenizer vocabulary
├── merges.txt               # BPE merge rules
├── tokenizer_config.json    # Tokenizer configuration
└── README.md
```

**Weight prefixes in `model.safetensors`:**
- `qwen2.*` - Qwen2 LLM weights
- `llm.*` - Speech embedding and decoder weights
- `flow.*` - Flow matching weights
- `hift.*` - HiFi-GAN weights
- `campplus.*` - CAMPlus speaker encoder weights

**S3 Tokenizer** (separate repo):
- `mlx-community/S3TokenizerV2` - Shared speech tokenizer weights

---

## Key Constants

| Constant | Value | Notes |
|----------|-------|-------|
| Sample Rate | 24000 | Output audio rate |
| S3 Tokenizer Rate | 16000 | Input to S3 tokenizer |
| Speech Token Size | 6561 | FSQ vocabulary size |
| Special Tokens | +3 | sos/eos, task_id, fill_token |
| Mel Bins (Flow) | 80 | For flow matching |
| Mel Bins (S3) | 128 | For S3 tokenizer |
| Speaker Embed Dim | 192 | CAMPlus output |
| Qwen2 Hidden Size | 896 | LLM hidden dimension |
| Qwen2 Layers | 24 | Number of transformer layers |
| Token-Mel Ratio | 2 | mel_len = token_len * 2 |

---

## Dependencies

### Swift Packages

| Package | Description | Local Path |
|---------|-------------|------------|
| mlx-swift | Core MLX framework | `../forked/mlx-swift` |
| mlx-swift-lm | Qwen2 model (`MLXLLM/Models/Qwen2.swift`) | `../forked/mlx-swift-lm` |
| swift-transformers | Qwen2 tokenizer (BPE) | `../forked/swift-transformers` |

### Text Tokenizer

CosyVoice2 uses the **Qwen2 tokenizer** (BPE-based) to convert input text to token IDs.

**Already available in Swift:**
- `swift-transformers` has `Qwen2Tokenizer` registered as a `BPETokenizer` (`Tokenizer.swift:173`)
- Can load from HuggingFace tokenizer files (`vocab.json`, `merges.txt`, `tokenizer.json`)

**Tokenizer files** (in model's `tokenizer/` directory):
- `vocab.json` - Token vocabulary
- `merges.txt` - BPE merge rules
- `tokenizer.json` - Combined tokenizer config
- `tokenizer_config.json` - Tokenizer settings

**Special tokens** to add at runtime for speech control:
```
<|endofprompt|>, [breath], <strong>, </strong>, [noise], [laughter],
[cough], [clucking], [accent], [quick_breath], <laughter>, </laughter>,
[hissing], [sigh], [vocalized-noise], [lipsmack], [mn]
```

**Note:** Need to verify `swift-transformers` supports adding special tokens dynamically, or pre-add them to the tokenizer config.

The tokenizer is used to:
- Encode input text → token IDs (for LLM input)
- Encode `ref_text` → token IDs (for zero-shot mode)
- Encode `instruct_text` → token IDs (for instruct mode)

### Python → Swift Replacements

The Python implementation uses `scipy` and `librosa` for audio processing. In Swift, replace with:

| Python | Swift Replacement | Usage in CosyVoice2 |
|--------|-------------------|---------------------|
| `scipy.signal.resample` | AVFoundation (`AVAudioConverter`) | Resampling ref audio to 16kHz for S3 tokenizer |
| `librosa.effects.trim` | vDSP + custom implementation | Trimming silence from reference audio |

**Notes:**
- **AVFoundation** (preferred): Use for audio I/O, resampling, format conversion
- **vDSP**: Use for signal processing operations (FFT, filtering, etc.)
- `AudioResampler.resample(_:from:to:)` already exists in `Audio/AudioResampler.swift` using `AVAudioConverter`
- `librosa.effects.trim` needs a new Swift implementation using vDSP for RMS calculation and silence detection

---

## Component Count Summary

| Category | Count | Status |
|----------|-------|--------|
| Shared Components (S3 Tokenizer) | 6 | ✅ Already ported |
| Shared Components (Speaker Encoder) | 2 | ✅ Already ported |
| Shared Components (Conformer) | 8 | ✅ Already ported |
| Shared Components (Flow/Decoder) | 4 | ✅ Already ported |
| Shared Components (HiFi-GAN) | 5 | ✅ Already ported |
| Shared Components (Mel) | 1 | ✅ Already ported |
| **Total Shared** | **26** | ✅ |
| | | |
| Config Classes | 5 | ✅ Ported |
| LLM Components | 6 | ✅ Ported |
| Flow Matching | 2 | ✅ Ported |
| HiFi-GAN (CosyVoice2-specific) | 5 | ✅ Ported |
| Speaker Encoder | 1 | ✅ Ported |
| Main Model | 3 | ✅ Ported |
| **Total New** | **22** | ✅ |

---

## Proposed Swift Directory Structure

```
package/TTS/CosyVoice2/
├── CosyVoice2Engine.swift           # Public API (TTSEngine conformance)
├── CosyVoice2TTS.swift              # Actor wrapper for thread-safety
├── CosyVoice2Model.swift            # Main model class (Module)
├── Config/
│   └── CosyVoice2Config.swift       # All config structs
├── LLM/
│   ├── Qwen2LM.swift                # Qwen2 with speech embeddings
│   └── Sampling.swift               # nucleus_sampling, ras_sampling
├── Flow/
│   └── CosyVoice2CFM.swift          # CosyVoice2-specific CFM
├── HiFiGAN/
│   ├── CosyHiFTGenerator.swift      # Main vocoder
│   ├── SineGen2.swift               # 24kHz sine generator
│   ├── SourceModuleHnNSF2.swift     # NSF source module
│   └── CosyF0Predictor.swift        # F0 predictor
└── SpeakerEncoder/
    └── CAMPlusSpeakerEncoder.swift  # Wrapper (uses shared CAMPPlus)
```

**Shared components** (import from Chatterbox):
- `S3TokenizerV2` from `TTS/Chatterbox/S3Tokenizer/`
- `CAMPPlus` from `TTS/Chatterbox/S3Gen/CAMPPlus.swift`
- `UpsampleConformerEncoder` from `TTS/Chatterbox/S3Gen/Transformer/`
- `BASECFM`, `CFMParams` from `TTS/Chatterbox/S3Gen/FlowMatching.swift`
- `ResBlock`, `Snake`, `stft`, `istft` from `TTS/Chatterbox/S3Gen/HiFiGAN.swift`

---

## Potential Challenges

1. **Qwen2 Integration**
   - Need to adapt `mlx-swift-lm` Qwen2 for speech token generation
   - Custom embeddings (text + speech) instead of standard LM head
   - May need to extract/modify Qwen2 internals rather than using as black box

2. **Special Token Handling**
   - `swift-transformers` may not support dynamic token addition
   - Fallback: Pre-modify tokenizer config or handle token IDs manually

3. **Bidirectional Streaming**
   - Python uses generator-based streaming with `inference_bistream`
   - Swift needs careful `AsyncStream` design for incremental token generation

4. **Weight Key Mapping** ✅ RESOLVED
   - CosyVoice2 has different weight prefixes (`qwen2.*`, `llm.*`, `flow.*`, `hift.*`, `campplus.*`)
   - Weight loading verified - all 2477 keys map correctly
   - Fixed: Flow decoder uses nested path `flow.decoder.estimator.*`
   - Fixed: CAMPlus weights embedded in main model.safetensors

5. **Numerical Precision**
   - ✅ Resolved: Both Python and Swift now generate noise dynamically at inference time
   - No pre-generated noise tensors needed
   - May still have minor numerical differences due to random seed handling

6. **Memory Management**
   - Qwen2-0.5B + Flow + HiFi-GAN is larger than Chatterbox
   - Monitor memory usage, may need aggressive `MLXMemory.clearCache()`

---

## Porting Guidelines

### Faithful Implementation

**Do NOT use placeholders.** Every component must be ported faithfully to match the original Python MLX implementation exactly:

1. **Complete implementations only** - No placeholder functions that return zeros or dummy values
2. **Match behavior exactly** - Output should match Python MLX when given the same inputs and weights
3. **Preserve all functionality** - All modes (zero-shot, cross-lingual, instruct, VC) must work correctly
4. **Verify weight loading** - All weight keys must map correctly from Python to Swift

### MLX Built-ins for Efficiency

**Use MLX vectorized operations instead of Python-style loops** for performance:

| Avoid | Use Instead |
|-------|-------------|
| `for i in 0..<T { for j in 0..<S { ... } }` | Vectorized MLX operations |
| Element-by-element array construction | `MLX.linspace`, `MLXArray(0..<n)` |
| Manual cumsum with loops | `MLX.cumsum()` |
| Sequential mask building | Broadcasting with comparison operators |
| Loop-based attention masks | `MLX.triu()`, `MLX.tril()`, broadcasting |

**Examples:**

```swift
// ❌ Bad: Python-style loop for causal mask
for i in 0..<T {
    for j in 0..<(offset + i + 1) {
        mask[i, j] = MLXArray(Float(0))
    }
}

// ✅ Good: Vectorized causal mask
let rows = MLXArray(0..<Int32(T)).expandedDimensions(axis: 1)
let cols = MLXArray(0..<Int32(totalLen)).expandedDimensions(axis: 0)
let mask = MLX.where(cols .<= (rows + offset), MLXArray(Float(0)), MLXArray(Float(-1e9)))
```

**Performance impact:** Vectorized operations leverage Metal GPU acceleration. Loops execute sequentially on CPU and can be orders of magnitude slower.

---

## Remaining Work (Placeholders Replaced ✅)

All placeholder functions have been replaced with full implementations:

| Location | Function | Implementation |
|----------|----------|----------------|
| `CosyVoice2TTS.swift` | `computeMelSpectrogram80()` | ✅ Uses `s3genMelSpectrogram()` with CosyVoice2 params (n_fft=1920, hop=480, 80 mels, 24kHz) |
| `CosyVoice2TTS.swift` | `logMelSpectrogramCAMPPlus()` | ✅ Uses `logMelSpectrogramChatterbox()` (n_fft=400, hop=160, 128 mels, 16kHz) |
| `Qwen2LM.swift` | `createCausalMask()` | ✅ Vectorized using MLX broadcasting (`cols .<= rows + offset`) |

---

## Shared Component Parameterization Analysis ✅

Compared Python MLX shared components with Swift implementation. **No changes needed** - Swift is already properly parameterized:

### Already Parameterized in Swift:
| Component | Parameters | Notes |
|-----------|------------|-------|
| `S3TokenizerModelConfig` | nMels, nAudioState, nAudioHead, nAudioLayer | All model hyperparameters |
| `S3TokenizerConstants` | s3Sr, s3Hop, s3TokenRate, speechVocabSize | Exported constants |
| `CAMPPlus` | featDim, embeddingSize, growthRate, etc. | Full speaker encoder config |
| `kaldiFbankCAMPPlus` | sampleRate, numMelBins, frameLength, frameShift | Fbank extraction |
| `s3genMelSpectrogram` | nFft, numMels, samplingRate, hopSize, winSize, fmin, fmax | Flow model mel |
| `logMelSpectrogram` | sampleRate, nMels, nFft, hopLength | General mel extraction |

### Intentionally Hardcoded (Architectural Constants):
| Component | Hardcoded Values | Reason |
|-----------|------------------|--------|
| `logMelSpectrogramChatterbox` | n_fft=400, hop=160, sr=16000 | S3 tokenizer architecture requires these exact values |
| S3TokenizerV2 sliding window | window=30s, overlap=4s | Standard long-audio handling for S3 |

The Python MLX implementation uses the **same hardcoded values** in `chatterbox/s3tokenizer/utils.py`. These are not meant to be configurable - they're part of the S3 tokenizer's trained architecture.

---

## Notes

1. **HiFi-GAN is Different**: The CosyVoice2 HiFi-GAN operates at 24kHz with different architecture than Chatterbox's 22.05kHz version. A separate Swift implementation is required.

2. **Qwen2 Integration**: `mlx-swift-lm` has Qwen2 (`Libraries/MLXLLM/Models/Qwen2.swift`). Need to adapt for speech token generation with custom embeddings.

3. **Text Tokenizer**: CosyVoice2 uses Qwen2's tokenizer with additional special tokens. The Swift implementation will need to handle these.

4. **Streaming Support**: The model supports bidirectional streaming which may require careful async/await handling in Swift.

5. **Voice Conversion**: VC mode skips LLM and uses source speech tokens directly with flow matching.

---

## Lessons Learned for Future Ports

### Loading Quantized Models (4-bit, 8-bit)

MLX-Swift supports both fp16 and quantized models, but quantized models require an extra step: **call `quantize()` before `update(parameters:)`**.

**The Pattern (from mlx-swift-lm):**
```swift
// 1. Create model with standard Linear layers
let model = MyModel(config)

// 2. Filter weights for this component
let weights = allWeights.filter { $0.key.hasPrefix("mymodel.") }
  .reduce(into: [:]) { result, pair in
    result[String(pair.key.dropFirst("mymodel.".count))] = pair.value
  }

// 3. Quantize layers where .scales keys exist (BEFORE loading weights)
quantize(model: model) { path, _ in
  if weights["\(path).scales"] != nil {
    return (groupSize: 64, bits: 4, mode: .affine)
  }
  return nil
}

// 4. Load weights (now QuantizedLinear layers will receive quantized weights)
let weightsList = weights.map { (key: $0.key, value: $0.value) }
try model.update(parameters: ModuleParameters.unflattened(weightsList), verify: [])
```

**Why this works:**
- `quantize()` walks the model and replaces `Linear` → `QuantizedLinear` for layers where quantized weights are detected
- `QuantizedLinear` expects packed weights with `scales` and `biases` tensors
- The `update(parameters:)` call then loads the quantized weights into the correct layer type

**Error signature if you skip quantization:**
```
Fatal error: [addmm] Last dimension of first input with shape (1,19,896)
must match second to last dimension of second input with shape (112,896)
```
The `112` is the packed weight size (896 / 64 * 4 / 8 = 7 groups packed into uint32).

**Weight shape differences:**
- Non-quantized: `weight` shape is `(out_features, in_features)` e.g., `(896, 896)`
- 4-bit quantized: `weight` shape is `(out_features, packed_size)` e.g., `(896, 112)`
  - Plus `scales` shape `(out_features, num_groups)` e.g., `(896, 14)`
  - Plus `biases` shape `(out_features, num_groups)` e.g., `(896, 14)`

### Config File Parsing

The HuggingFace `config.json` may have a minimal structure that differs from the full model config:

```json
// Minimal HuggingFace config
{
  "model_type": "cosyvoice2",
  "sample_rate": 24000,
  "speech_token_size": 6561
}
```

vs

```json
// Full nested config (what model code expects)
{
  "llm": { "speech_token_size": 6561, "hidden_size": 896, ... },
  "flow": { ... },
  "hifigan": { ... }
}
```

**Solution:** Use sensible defaults in config structs and only override from JSON when keys are present. The `CosyVoice2Config.fromPretrained()` pattern handles this by initializing with defaults first.

### Weight Loading Best Practices

1. **Print weight key counts** during development to verify all expected prefixes are found
2. **Use `verify: []`** (empty verification) initially, then add `.noUnusedKeys` once weight mapping is confirmed
3. **Call `quantize()` before `update(parameters:)`** for each model component
4. **Document weight prefix conventions** (e.g., `qwen2.*`, `llm.*`, `flow.*`, `hift.*`)

### Testing with xcodebuild

**IMPORTANT:**
- Use `xcodebuild` (not `swift test`) for tests requiring Metal GPU access
- **Run one test at a time** - running all tests in parallel can hang or be extremely slow
- Tests may take 30-40+ seconds - be patient before canceling

**Run a single test suite:**
```bash
xcodebuild test-without-building -scheme mlx-audio -destination 'platform=macOS,arch=arm64' \
  -only-testing:MLXAudioTests/CosyVoice2IntegrationTests \
  2>&1 | grep -E "(DEBUG|passed|failed)"
```
