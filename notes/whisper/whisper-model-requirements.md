# Whisper Model Requirements for Swift Port

## Required Models

All models need to be available in **SafeTensors format** (`.safetensors`) for compatibility with MLX Swift's `MLX.loadArrays()` function.

### Model List

| Model Size | HF Repo ID | Parameters | Current Status | Files Needed |
|------------|-----------|------------|----------------|--------------|
| **Tiny** | `mlx-community/whisper-tiny-mlx` | 39M | ❌ NPZ only | `weights.safetensors` or `model.safetensors` + `config.json` |
| **Base** | `mlx-community/whisper-base-mlx` | 74M | ❌ NPZ only | `weights.safetensors` or `model.safetensors` + `config.json` |
| **Small** | `mlx-community/whisper-small-mlx` | 244M | ❌ NPZ only | `weights.safetensors` or `model.safetensors` + `config.json` |
| **Medium** | `mlx-community/whisper-medium-mlx` | 769M | ❌ NPZ only | `weights.safetensors` or `model.safetensors` + `config.json` |
| **Large v3** | `mlx-community/whisper-large-v3-mlx` | 1550M | ⚠️ Unknown | `weights.safetensors` or `model.safetensors` + `config.json` |
| **Large v3 Turbo** | `mlx-community/whisper-large-v3-turbo` | 809M | ✅ Has SafeTensors | `weights.safetensors` + `config.json` |

## File Format Details

### Required Files per Model

Each model repository must contain:

1. **Weight file** (one of):
   - `weights.safetensors` (preferred, used by newer models)
   - `model.safetensors` (alternative, used by some 4-bit models)

2. **Configuration**:
   - `config.json` (model dimensions and settings)

### Current File Formats

| Repo | weights.npz | weights.safetensors | model.safetensors |
|------|-------------|---------------------|-------------------|
| whisper-tiny-mlx | ✅ 74.4 MB | ❌ | ❌ |
| whisper-base-mlx | ✅ 144 MB | ❌ | ❌ |
| whisper-small-mlx | ⚠️ Unknown | ⚠️ Unknown | ⚠️ Unknown |
| whisper-medium-mlx | ⚠️ Unknown | ⚠️ Unknown | ⚠️ Unknown |
| whisper-large-v3-mlx | ⚠️ Unknown | ⚠️ Unknown | ⚠️ Unknown |
| whisper-large-v3-turbo | ❌ (deleted) | ✅ 1.61 GB | ❌ |

## Existing SafeTensors Conversions

### Community Conversions

**jkrukowski/whisper-tiny-mlx-safetensors**
- Format: Split files (`encoder.safetensors` + `decoder.safetensors`)
- Not suitable - our code expects unified weights file

### Quantized Models (4-bit)

Some 4-bit quantized models have safetensors:
- `mlx-community/whisper-tiny-mlx-4bit` - Has `model.safetensors` ✅
- `mlx-community/whisper-base-mlx-4bit` - Only `weights.npz` ❌
- `mlx-community/whisper-large-v3-mlx-4bit` - Unknown

**Note**: We need full precision (fp16/fp32) models, not quantized versions.

## Model Verification Checklist

Before uploading converted models, verify:

- [ ] Model loads successfully with `MLX.loadArrays(url: weightFileURL)`
- [ ] `config.json` contains all required fields:
  - `n_mels` (should be 80 for Whisper)
  - `n_audio_ctx` (should be 1500)
  - `n_audio_state` (varies by model size)
  - `n_audio_head` (varies by model size)
  - `n_audio_layer` (varies by model size)
  - `n_vocab` (should be 51865 for multilingual)
  - `n_text_ctx` (should be 448)
  - `n_text_state` (varies by model size)
  - `n_text_head` (varies by model size)
  - `n_text_layer` (varies by model size)
- [ ] Weight keys match expected format (e.g., `encoder.blocks.0.attn.query.weight`)
- [ ] Model produces same output as Python mlx-whisper for test audio

## Conversion Instructions

### Option 1: Using MLX Python

```python
import mlx.core as mx

# Load NPZ weights
weights = mx.load("weights.npz")

# Save as SafeTensors
mx.save_safetensors("weights.safetensors", weights)
```

### Option 2: Using safetensors Python Library

```python
import numpy as np
from safetensors.numpy import save_file

# Load NPZ
weights = np.load("weights.npz")

# Convert to dict of numpy arrays
weight_dict = {k: v for k, v in weights.items()}

# Save as SafeTensors
save_file(weight_dict, "weights.safetensors")
```

## Upload Strategy

### Recommended Approach

Upload to your own HF namespace initially for testing:
- `DePasqualeOrg/whisper-tiny-mlx-safetensors`
- `DePasqualeOrg/whisper-base-mlx-safetensors`
- `DePasqualeOrg/whisper-small-mlx-safetensors`
- `DePasqualeOrg/whisper-medium-mlx-safetensors`
- `DePasqualeOrg/whisper-large-v3-mlx-safetensors` (if needed)

Once verified, can:
1. Submit PR to mlx-community repos to add safetensors alongside npz
2. Keep separate repos if mlx-community doesn't want to maintain both formats

### Update Code After Upload

In `/Users/anthony/files/projects/mlx-swift-audio/package/Models/TranscriptionResult.swift` (lines 169-171):

```swift
/// HuggingFace repository ID
public var repoId: String {
    "DePasqualeOrg/\(rawValue)-mlx-safetensors"  // Or keep as mlx-community if they accept PRs
}
```

## Testing Priority

Test models in this order:

1. ✅ **whisper-large-v3-turbo** - Already has safetensors, test first
2. **whisper-tiny** - Smallest, fastest iteration for debugging
3. **whisper-base** - Good balance for development
4. **whisper-small** - Production quality
5. **whisper-medium** - Better quality, slower
6. **whisper-large-v3** - Best quality (if different from turbo)

## Expected Model Sizes

After conversion to SafeTensors:

| Model | Approx Size |
|-------|-------------|
| Tiny | ~75 MB |
| Base | ~145 MB |
| Small | ~490 MB |
| Medium | ~1.5 GB |
| Large v3 | ~3.1 GB |
| Large v3 Turbo | ~1.6 GB |

## References

- Original repos: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc
- MLX examples: https://github.com/ml-explore/mlx-examples/tree/main/whisper
- SafeTensors format: https://huggingface.co/docs/safetensors/index
