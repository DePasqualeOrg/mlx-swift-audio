# CLAUDE.md

This project includes a Swift MLX library and example apps for speech-to-text and text-to-speech models running on Apple Silicon.

## Project Structure

- `package/` - Main library source code
  - `Audio/` - Audio loading and processing
  - `Codec/DAC/` - DAC audio codec for neural audio compression
  - `Models/` - Model configuration types
  - `Protocols/` - Shared protocols
  - `STT/` - Speech-to-text (Whisper, FunASR)
  - `TTS/` - Text-to-speech (Chatterbox, CosyVoice2, CosyVoice3, Kokoro, Marvis, Orpheus, OuteTTS)
  - `TTS/Shared/` - Shared TTS components (RoPE, SwiGLU, etc.)
  - `Utils/` - Utility functions
  - `Tests/` - Test files
- `examples/` - Example apps (STT App, TTS App)
- `licenses/` - Third-party licenses for ported models
- `scripts/` - Build and utility scripts

## Cloning Reference Repositories

When referring to code from other repositories, for easy access to the source code, make a local clone of the repository in `/tmp` rather than using the web fetch or web search tools, which can be unreliable.

## Python MLX Implementations of Models

The Python MLX implementations of all models in this repo can be found at [DePasqualeOrg/mlx-audio-plus](https://github.com/DePasqualeOrg/mlx-audio-plus). Use these when porting or debugging Swift implementations. The original implementations of the models are usually in PyTorch and can be downloaded from the original model's GitHub repository.

## Other Reference Repositories

- [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm): LLMs in Python MLX
- [ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples): LLMs in Swift MLX
- [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift): Swift MLX built-ins
- [huggingface/transformers](https://github.com/huggingface/transformers): models, tokenizers, and pipelines in Python
- [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers): Hugging Face Hub API client and tokenizers in Swift

## Hugging Face Model Repositories

When doing work that requires referring to a model's weights, config files, and other files in the model's Hugging Face repository, check for a local copy of that repository. You can clone the Hugging Face repository into a directory in `/tmp` if it does not already exist there.

Models that have been converted to MLX format are usually found in the mlx-community organization on Hugging Face. The original model weights (usually in PyTorch format) are found in the respective organizations' repositories on Hugging Face.

Cached repositories downloaded by the mlx-swift-audio and mlx-audio-plus libraries on macOS:

- **Swift (this repo)**: Models downloaded by the mlx-swift-audio library using the swift-transformers library are stored in `~/Library/Caches/huggingface/models/`
- **Python (mlx-audio-plus)**: Models downloaded by mlx-audio-plus using the Python transformers library are stored in `~/.cache/huggingface/hub/`
