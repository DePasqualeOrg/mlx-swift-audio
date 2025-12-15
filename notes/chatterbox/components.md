# Chatterbox TTS Model Components

This document catalogs all components of the Chatterbox TTS Swift MLX implementation for comparison with Python MLX and PyTorch equivalents.

## Directory Structure

```
TTS/Chatterbox/
├── ChatterboxModel.swift           # Core model
├── ChatterboxUtils.swift           # Utilities
├── Config/ChatterboxConfig.swift   # Configuration
├── T3/                             # Text-to-Speech-Token
├── S3Gen/                          # Token-to-Waveform
├── S3Tokenizer/                    # Speech tokenization
├── VoiceEncoder/                   # Speaker embedding
└── Tokenizer/                      # Text tokenization
```

---

## 1. Core Model

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `ChatterboxModel` | ChatterboxModel.swift:137 | | | Equivalent |
| ✓ | `ChatterboxConditionals` | ChatterboxModel.swift:105 | | | Equivalent |
| ✓ | `puncNorm` (function) | ChatterboxModel.swift:29 | | | Equivalent |
| ✓ | `dropInvalidTokens` (function) | ChatterboxModel.swift:75 | | | Equivalent |

---

## 2. Configuration

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `T3LlamaConfig` | Config/ChatterboxConfig.swift:4 | | | Config only |
| ✓ | `T3Config` | Config/ChatterboxConfig.swift:64 | | | Config only |
| ✓ | `VoiceEncConfig` | Config/ChatterboxConfig.swift:134 | | | Config only |
| ✓ | `ChatterboxModelConfig` | Config/ChatterboxConfig.swift:175 | | | Config only |
| ✓ | `ChatterboxConstants` (enum) | Config/ChatterboxConfig.swift:231 | | | Config only |
| ✓ | `RopeScaling` (nested) | Config/ChatterboxConfig.swift:28 | | | Config only |

---

## 3. T3 Model (Text-to-Speech-Token)

### Core T3 Components

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `T3` | T3/T3.swift:14 | | | Has asyncEval pipelining |
| ✓ | `T3Cond` | T3/T3CondEnc.swift:11 | | | Equivalent |
| ✓ | `T3CondEnc` | T3/T3CondEnc.swift:46 | | | Equivalent |

### T3 LLaMA Backbone

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `T3LlamaBackbone` | T3/T3LlamaBackbone.swift:143 | | | Equivalent |
| ✓ | `T3LlamaModel` | T3/T3LlamaBackbone.swift:130 | | | Equivalent |
| ✓ | `T3TransformerBlock` | T3/T3LlamaBackbone.swift:97 | | | Equivalent |
| ✓ | `T3LlamaAttention` | T3/T3LlamaBackbone.swift:11 | | | **OPTIMIZED**: Uses attentionWithCacheUpdate |

### T3 Conditioning & Position Embeddings

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `Perceiver` | T3/Perceiver.swift:137 | | | Equivalent |
| ✓ | `AttentionBlock` | T3/Perceiver.swift:74 | | | Equivalent |
| ✓ | `AttentionQKV` | T3/Perceiver.swift:10 | | | Equivalent |
| ✓ | `LearnedPositionEmbeddings` | T3/LearnedPositionEmbeddings.swift:7 | | | Equivalent |

---

## 4. S3Gen Model (Token-to-Waveform)

### Core S3Gen Components

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `S3Token2Wav` | S3Gen/S3Gen.swift:282 | | | Equivalent |
| ✓ | `S3Token2Mel` | S3Gen/S3Gen.swift:109 | | | Equivalent |
| ✓ | `S3GenRefDict` | S3Gen/S3Gen.swift:24 | | | Equivalent |
| ✓ | `resampleAudio` (function) | S3Gen/S3Gen.swift:59 | | | Equivalent |

### Flow Matching / Diffusion

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `CausalMaskedDiffWithXvec` | S3Gen/Flow.swift:11 | | | Equivalent |
| ✓ | `CausalConditionalCFM` | S3Gen/FlowMatching.swift:214 | | | Pre-allocates zero arrays |
| ✓ | `ConditionalCFM` | S3Gen/FlowMatching.swift:92 | | | Pre-allocates zero arrays |
| ✓ | `BASECFM` | S3Gen/FlowMatching.swift:27 | | | Equivalent |
| ✓ | `CFMParams` | S3Gen/FlowMatching.swift:10 | | | Config only |

### Conditional Decoder (UNet-like)

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `ConditionalDecoder` | S3Gen/S3GenDecoder.swift:141 | | | Equivalent |
| ✓ | `DownBlock` | S3Gen/S3GenDecoder.swift:102 | | | Equivalent |
| ✓ | `MidBlock` | S3Gen/S3GenDecoder.swift:115 | | | Equivalent |
| ✓ | `UpBlock` | S3Gen/S3GenDecoder.swift:126 | | | Equivalent |
| ✓ | `CausalResnetBlock1D` | S3Gen/S3GenDecoder.swift:72 | | | Equivalent |
| ✓ | `CausalBlock1D` | S3Gen/S3GenDecoder.swift:49 | | | Equivalent |
| ✓ | `CausalConv1d` | S3Gen/S3GenDecoder.swift:10 | | | Equivalent |

### Matcha Decoder Components

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `SinusoidalPosEmb` | S3Gen/Matcha/MatchaDecoder.swift:10 | | | Equivalent |
| ✓ | `TimestepEmbedding` | S3Gen/Matcha/MatchaDecoder.swift:40 | | | Equivalent |
| ✓ | `Block1D` | S3Gen/Matcha/MatchaDecoder.swift:62 | | | Equivalent |
| ✓ | `ResnetBlock1D` | S3Gen/Matcha/MatchaDecoder.swift:85 | | | Equivalent |
| ✓ | `Downsample1D` | S3Gen/Matcha/MatchaDecoder.swift:116 | | | Equivalent |
| ✓ | `MatchaUpsample1D` | S3Gen/Matcha/MatchaDecoder.swift:135 | | | Equivalent |
| ✓ | `mish` (function) | S3Gen/Matcha/MatchaDecoder.swift:176 | | | Equivalent |
| ✓ | `softplus` (function) | S3Gen/Matcha/MatchaDecoder.swift:181 | | | Equivalent |

### Matcha Transformer Components

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `DiffusersAttention` | S3Gen/Matcha/MatchaTransformer.swift:11 | | | Equivalent |
| ✓ | `FeedForward` | S3Gen/Matcha/MatchaTransformer.swift:80 | | | Equivalent |
| ✓ | `BasicTransformerBlock` | S3Gen/Matcha/MatchaTransformer.swift:101 | | | Equivalent |

### HiFi-GAN Vocoder

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `HiFTGenerator` | S3Gen/HiFiGAN.swift:353 | | | Equivalent |
| ✓ | `SourceModuleHnNSF` | S3Gen/HiFiGAN.swift:192 | | | Equivalent |
| ✓ | `SineGen` | S3Gen/HiFiGAN.swift:121 | | | Uses vectorized harmonics |
| ✓ | `HiFiGANResBlock` | S3Gen/HiFiGAN.swift:59 | | | Computes resblocks in parallel |
| ✓ | `Snake` (activation) | S3Gen/HiFiGAN.swift:26 | | | Equivalent |
| ✓ | `hannWindowPeriodic` (function) | S3Gen/HiFiGAN.swift:11 | | | Equivalent |
| ✓ | `getPadding` (function) | S3Gen/HiFiGAN.swift:19 | | | Equivalent |
| ✓ | `stftHiFiGAN` (function) | S3Gen/HiFiGAN.swift:242 | | | Vectorized frame extraction |
| ✓ | `istftHiFiGAN` (function) | S3Gen/HiFiGAN.swift:279 | | | Vectorized overlap-add |

### F0 Predictor

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `ConvRNNF0Predictor` | S3Gen/F0Predictor.swift:10 | | | Equivalent |

### Speaker Encoder (CAMPPlus)

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `CAMPPlus` | S3Gen/CAMPPlus.swift:655 | | | Equivalent |
| ✓ | `CAMDenseTDNNBlock` | S3Gen/CAMPPlus.swift:539 | | | Equivalent |
| ✓ | `CAMDenseTDNNLayer` | S3Gen/CAMPPlus.swift:475 | | | Equivalent |
| ✓ | `CAMLayer` | S3Gen/CAMPPlus.swift:388 | | | Equivalent |
| ✓ | `TDNNLayer` | S3Gen/CAMPPlus.swift:313 | | | Equivalent |
| ✓ | `StatsPool` | S3Gen/CAMPPlus.swift:304 | | | Equivalent |
| ✓ | `FCM` | S3Gen/CAMPPlus.swift:214 | | | Equivalent |
| ✓ | `BasicResBlock` | S3Gen/CAMPPlus.swift:148 | | | Equivalent |
| ✓ | `TransitLayer` | S3Gen/CAMPPlus.swift:581 | | | Equivalent |
| ✓ | `DenseLayer` | S3Gen/CAMPPlus.swift:610 | | | Equivalent |
| ✓ | `ReLUActivation` | S3Gen/CAMPPlus.swift:363 | | | Equivalent |
| ✓ | `poveyWindow` (function) | S3Gen/CAMPPlus.swift:11 | | | Vectorized MLX |
| ✓ | `nextPowerOf2` (function) | S3Gen/CAMPPlus.swift:18 | | | Equivalent |
| ✓ | `kaldiFbankCAMPPlus` (function) | S3Gen/CAMPPlus.swift:28 | | | Vectorized frame extraction |
| ✓ | `melFiltersHTK` (function) | S3Gen/CAMPPlus.swift:105 | | | Equivalent |
| ✓ | `statisticsPooling` (function) | S3Gen/CAMPPlus.swift:296 | | | Equivalent |
| ✓ | `getNonlinear` (function) | S3Gen/CAMPPlus.swift:370 | | | Equivalent |

### Conformer Encoder

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `UpsampleConformerEncoder` | S3Gen/Transformer/UpsampleConformerEncoder.swift:195 | | | Equivalent |
| ✓ | `Upsample1D` | S3Gen/Transformer/UpsampleConformerEncoder.swift:10 | | | Equivalent |
| ✓ | `PreLookaheadLayer` | S3Gen/Transformer/UpsampleConformerEncoder.swift:60 | | | Equivalent |
| ✓ | `makePadMask` (function) | S3Gen/Transformer/UpsampleConformerEncoder.swift:107 | | | Equivalent |
| ✓ | `subsequentChunkMask` (function) | S3Gen/Transformer/UpsampleConformerEncoder.swift:122 | | | Equivalent |
| ✓ | `addOptionalChunkMask` (function) | S3Gen/Transformer/UpsampleConformerEncoder.swift:130 | | | Equivalent |

### Conformer Encoder Layer

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `ConformerEncoderLayer` | S3Gen/Transformer/ConformerEncoderLayer.swift:12 | | | Equivalent |

### Attention Modules

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `MultiHeadedAttention` | S3Gen/Transformer/Attention.swift:10 | | | Equivalent |
| ✓ | `RelPositionMultiHeadedAttention` | S3Gen/Transformer/Attention.swift:114 | | | Equivalent |

### Convolution Module

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `ConvolutionModule` | S3Gen/Transformer/ConvolutionModule.swift:10 | | | Equivalent |
| ✓ | `glu` (function) | S3Gen/Transformer/ConvolutionModule.swift:150 | | | Equivalent |

### Feed Forward

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `PositionwiseFeedForward` | S3Gen/Transformer/PositionwiseFeedForward.swift:12 | | | Equivalent |

### Positional Encodings

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `PositionalEncoding` | S3Gen/Transformer/Embedding.swift:12 | | | Equivalent |
| ✓ | `RelPositionalEncoding` | S3Gen/Transformer/Embedding.swift:69 | | | Equivalent |
| ✓ | `EspnetRelPositionalEncoding` | S3Gen/Transformer/Embedding.swift:86 | | | Equivalent |
| ✓ | `NoPositionalEncoding` | S3Gen/Transformer/Embedding.swift:159 | | | Equivalent |

### Subsampling

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `BaseSubsampling` | S3Gen/Transformer/Subsampling.swift:10 | | | Equivalent |
| ✓ | `LinearNoSubsampling` | S3Gen/Transformer/Subsampling.swift:29 | | | Equivalent |

### Mel Spectrogram

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `s3genMelSpectrogram` (function) | S3Gen/Mel/S3GenMel.swift:39 | | | Equivalent |

---

## 5. S3 Tokenizer (Speech Tokenization)

### Configuration

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `S3TokenizerConstants` | S3Tokenizer/S3TokenizerConfig.swift:4 | | | Config only |
| ✓ | `S3TokenizerModelConfig` | S3Tokenizer/S3TokenizerConfig.swift:13 | | | Config only |

### Core Tokenizer

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `S3TokenizerV2` | S3Tokenizer/S3Tokenizer.swift:430 | | | Equivalent |
| ✓ | `AudioEncoderV2` | S3Tokenizer/S3Tokenizer.swift:354 | | | Equivalent |
| ✓ | `S3ResidualAttentionBlock` | S3Tokenizer/S3Tokenizer.swift:316 | | | Equivalent |
| ✓ | `FSMNMultiHeadAttention` | S3Tokenizer/S3Tokenizer.swift:189 | | | Equivalent |
| ✓ | `FSQVectorQuantization` | S3Tokenizer/S3Tokenizer.swift:170 | | | Equivalent |
| ✓ | `FSQCodebook` | S3Tokenizer/S3Tokenizer.swift:128 | | | Equivalent |
| ✓ | `MultiHeadAttention` (S3) | S3Tokenizer/S3Tokenizer.swift:69 | | | Equivalent |
| ✓ | `precomputeFreqsCis` (function) | S3Tokenizer/S3Tokenizer.swift:9 | | | Equivalent |
| ✓ | `applyRotaryEmb` (function) | S3Tokenizer/S3Tokenizer.swift:36 | | | Equivalent |

### Tokenizer Utilities

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `makeNonPadMask` (function) | S3Tokenizer/S3TokenizerUtils.swift:17 | | | Equivalent |
| ✓ | `maskToBias` (function) | S3Tokenizer/S3TokenizerUtils.swift:34 | | | Equivalent |
| ✓ | `padSequences` (function) | S3Tokenizer/S3TokenizerUtils.swift:43 | | | Equivalent |
| ✓ | `mergeTokenizedSegments` (function) | S3Tokenizer/S3TokenizerUtils.swift:67 | | | Equivalent |
| ✓ | `logMelSpectrogram` (function) | S3Tokenizer/S3TokenizerUtils.swift:98 | | | Equivalent |
| ✓ | `logMelSpectrogramChatterbox` (function) | S3Tokenizer/S3TokenizerUtils.swift:156 | | | Equivalent |
| ✓ | `hanningWindow` (function) | S3Tokenizer/S3TokenizerUtils.swift:209 | | | Equivalent |
| ✓ | `stft` (function) | S3Tokenizer/S3TokenizerUtils.swift:220 | | | Vectorized |
| ✓ | `melFilters` (function) | S3Tokenizer/S3TokenizerUtils.swift:290 | | | Equivalent |

---

## 6. Voice Encoder (Speaker Embedding)

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `VoiceEncoder` | VoiceEncoder/VoiceEncoder.swift:47 | | | Equivalent |
| ✓ | `ChatterboxLSTM` | VoiceEncoder/ChatterboxLSTM.swift:74 | | | Custom impl (could use MLXNN.LSTM) |
| ✓ | `LSTMCell` | VoiceEncoder/ChatterboxLSTM.swift:11 | | | Custom impl (could use MLXNN.LSTM) |
| ✓ | `voiceEncoderMelspectrogram` (function) | VoiceEncoder/VoiceEncoderMelspec.swift:12 | | | Equivalent |
| ✓ | `getNumWins` (function) | VoiceEncoder/VoiceEncoder.swift:8 | | | Equivalent |
| ✓ | `getFrameStep` (function) | VoiceEncoder/VoiceEncoder.swift:27 | | | Equivalent |

---

## 7. Text Tokenizer

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `EnTokenizer` | Tokenizer/EnTokenizer.swift:29 | | | Equivalent |
| ✓ | `TokenizerError` (enum) | Tokenizer/EnTokenizer.swift:219 | | | Equivalent |

---

## 8. Utility Functions

| ✓ | Swift Component | File:Line | Python MLX | PyTorch | Notes |
|---|-----------------|-----------|------------|---------|-------|
| ✓ | `reverseAlongAxis` | ChatterboxUtils.swift:10 | | | Equivalent |

---

## 9. Key Constants

| Constant | Swift Value | Python MLX | PyTorch | Notes |
|----------|-------------|------------|---------|-------|
| S3 Tokenizer Sample Rate | 16000 | | | |
| S3Gen Output Sample Rate | 24000 | | | |
| Speech Vocab Size | 6561 (3^8) | | | |
| SOS Token | 6561 | | | |
| EOS Token | 6562 | | | |
| Mel Bins (S3Gen) | 80 | | | |
| Mel Bins (S3 Tokenizer) | 128 | | | |
| Mel Bins (Voice Encoder) | 40 | | | |
| T3 Hidden Size | 1024 | | | |
| T3 Num Layers | 30 | | | |
| T3 Attention Heads | 16 | | | |
| Speaker Embed Size | 256 (VE), 192 (CAM++) | | | |
| Conformer Blocks | 6 (main) + 4 (up) | | | |
| CFM Mid Blocks | 12 | | | |

---

## 10. Component Count Summary

| Category | Count |
|----------|-------|
| Classes | 60 |
| Structs | 9 |
| Enums | 3 |
| Functions | 47 |
| **Total Components** | **119** |

---

## 11. Generation Pipeline

```
Text Input
    ↓
EnTokenizer → Text Token IDs (vocab 704)
    ↓
T3 (LLaMA 520M backbone):
  ├── Text embedding + Learned position encoding
  ├── Speaker conditioning (VoiceEncoder → 256-dim)
  ├── Perceiver resampling (32 queries)
  └── Autoregressive generation with CFG
    ↓
Speech Token IDs (vocab 6561)
    ↓
S3Gen:
  ├── UpsampleConformerEncoder (6+4 blocks)
  ├── Flow Matching (CausalConditionalCFM)
  │   └── ConditionalDecoder (12 mid blocks)
  └── Speaker conditioning (CAMPPlus → 192-dim)
    ↓
Mel-Spectrogram (80 bins @ 24kHz)
    ↓
HiFTGenerator (HiFi-GAN + NSF):
  ├── F0 prediction (ConvRNNF0Predictor)
  ├── Harmonic synthesis (SineGen)
  └── Upsampling (8x → 5x → 3x = 120x)
    ↓
Waveform (24kHz)
```

---

## 12. Optimization Progress

- **Current Performance:** ~1.0 RTF (Swift MLX, 4-bit, M3)
- **Target Performance:** ~0.5 RTF (Python MLX, 4-bit, M3)

### Completed Optimizations
- [x] **T3LlamaAttention**: Now uses `attentionWithCacheUpdate` for automatic quantized KV cache support
- [x] **T3 generation loop**: Already has `asyncEval` pipelining
- [x] **HiFiGAN ResBlocks**: Already computes blocks in parallel via map
- [x] **STFT/ISTFT**: Already vectorized frame extraction and overlap-add
- [x] **CAMPPlus kaldi_fbank**: Already uses vectorized frame extraction

### Potential Further Optimizations
- [ ] **ChatterboxLSTM**: Uses custom implementation - could use MLXNN.LSTM (low priority, runs once per generation)
- [ ] **Memory layout**: Check for unnecessary transpositions
- [ ] **Kernel fusion**: Look for operations that could be fused

### Notes
Most components are equivalent between Swift and Python MLX. The main optimization applied was using `attentionWithCacheUpdate` in T3LlamaAttention for potential quantized KV cache benefits.
