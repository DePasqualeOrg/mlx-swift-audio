# VibeVoice Swift MLX Porting Progress

## Overview

VibeVoice is a real-time streaming TTS model that uses a Qwen2-based language model backbone with a diffusion-based prediction head. The architecture features:

- **Split Language Model**: Lower transformer layers encode text, upper layers handle TTS
- **Acoustic Tokenizer**: VAE-style decoder for converting latents to audio
- **Diffusion Head**: FFN-based architecture with AdaLN modulation for speech latent prediction
- **DPM-Solver**: Multi-step scheduler for efficient diffusion sampling
- **Voice Caching**: Pre-computed KV caches for voice conditioning

## Component List

| Component | Initial Port | Verification | Notes |
|-----------|--------------|--------------|-------|
| **Configuration** | | | |
| AcousticTokenizerConfig | ✓ | ✓✓✓ | All defaults match Python |
| DiffusionHeadConfig | ✓ | ✓✓✓ | All defaults match Python |
| Qwen2DecoderConfig | ✓ | ✓✓✓ | All defaults match Python |
| VibeVoiceConfig (main) | ✓ | ✓✓✓ | Combines all sub-configs |
| **Language Model** | | | |
| RMSNorm | ✓ | ✓✓✓ | Using built-in MLXNN.RMSNorm |
| RotaryEmbedding | ✓ | ✓✓✓ | Fixed: Added batch dim expansion |
| rotate_half | ✓ | ✓✓✓ | Matches Python exactly |
| apply_rotary_pos_emb | ✓ | ✓✓✓ | Fixed: cos/sin shape handling |
| Attention (GQA) | ✓ | ✓✓✓ | Uses MLXFast.scaledDotProductAttention |
| MLP (SwiGLU) | ✓ | ✓✓✓ | gate_proj, up_proj, down_proj |
| DecoderLayer | ✓ | ✓✓✓ | Pre-norm architecture |
| SpeechConnector | ✓ | ✓✓✓ | fc1 -> norm -> fc2 |
| BinaryClassifier | ✓ | ✓✓✓ | fc1 -> relu -> fc2 |
| Qwen2Model | ✓ | ✓✓✓ | Configurable norm layer |
| **Acoustic Tokenizer** | | | |
| ConvRMSNorm | ✓ | ✓✓✓ | Handles (B,C,T) format |
| CausalConv1d | ✓ | ✓✓✓ | Left padding, handles PyTorch<->MLX format |
| CausalConvTranspose1d | ✓ | ✓✓✓ | Trim padding for causal |
| DepthwiseConv | ✓ | ✓✓✓ | Wraps CausalConv1d with groups=dim |
| Mixer | ✓ | ✓✓✓ | Wraps DepthwiseConv |
| FeedForward | ✓ | ✓✓✓ | linear1 -> gelu -> linear2 |
| Block1D | ✓ | ✓✓✓ | mixer + ffn with layer scale |
| StemConv | ✓ | ✓✓✓ | Input convolution |
| UpsampleLayer | ✓ | ✓✓✓ | Transposed conv upsample |
| HeadConv | ✓ | ✓✓✓ | Output convolution |
| TokenizerDecoder | ✓ | ✓✓✓ | Full decoder pipeline |
| AcousticTokenizer | ✓ | ✓✓✓ | VAE decoder wrapper |
| **Diffusion Head** | | | |
| modulate | ✓ | ✓✓✓ | x * (1 + scale) + shift |
| TimestepEmbedder | ✓ | ✓✓✓ | Sinusoidal + MLP |
| FeedForwardNetwork | ✓ | ✓✓✓ | SwiGLU for diffusion |
| HeadLayer | ✓ | ✓✓✓ | AdaLN FFN block |
| FinalLayer | ✓ | ✓✓✓ | Final output layer |
| DiffusionHead | ✓ | ✓✓✓ | Main diffusion module |
| **Scheduler** | | | |
| betas_for_alpha_bar | ✓ | ✓✓✓ | Cosine beta schedule |
| SchedulerOutput | ✓ | ✓✓✓ | Step output struct |
| DPMSolverMultistepScheduler | ✓ | ✓✓✓ | First and second order updates |
| **Main Model** | | | |
| VibeVoiceModel | ✓ | ✓✓✓ | Main model class |
| loadVoice | ✓ | ✓✓✓ | Voice cache loading |
| sampleSpeechTokens | ✓ | ✓✓✓ | Diffusion with CFG |
| generate | ✓ | ✓✓✓ | Generation entry point |
| **TTS/Engine** | | | |
| VibeVoiceTTS | ✓ | ✓✓✓ | Actor for thread-safe gen |
| VibeVoiceEngine | ✓ | ✓✓✓ | TTSEngine protocol impl |

## Issues and Follow-up

1. **RoPE shape handling** - Fixed in Round 1: Added batch dimension expansion after getting rotary embeddings, matching Python implementation.

## Verification Notes

### Round 1
- Verified all configuration structs match Python defaults
- Fixed RoPE implementation to properly expand batch and head dimensions
- Verified attention uses correct shapes for Q, K, V projections
- Verified causal mask creation matches Python logic
- All component structures match Python MLX implementation

### Round 2
- Verified diffusion head architecture matches Python exactly
- Verified DPM-Solver scheduler logic for first/second order updates
- Verified acoustic tokenizer convolution padding and format conversions
- Verified main model layer split and voice cache loading

### Round 3
- Final verification of all components complete
- Added MIT license file for VibeVoice
- Added TTSProvider case for vibeVoice
- Port complete and ready for testing
