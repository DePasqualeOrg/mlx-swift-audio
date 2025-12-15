# Chatterbox Swift MLX Optimization

## Context

- **This repo**: Swift MLX implementation of Chatterbox (ported from Python MLX)
- **Python MLX source**: `mlx-audio-plus` repo (user's Python MLX port)
- **Original PyTorch**: https://github.com/resemble-ai/chatterbox

## Local Reference Repositories

All reference implementations are available locally at `../forked/`:

- **../forked/mlx-audio-plus**: Python MLX Chatterbox implementation
- **../forked/mlx-lm**: Python MLX LLM reference (for T3 LLaMA optimization patterns)
- **../forked/mlx-swift-lm**: Swift MLX LLM reference (for T3 LLaMA optimization patterns)

## Performance

| Implementation | RTF (4-bit, M3) |
|----------------|-----------------|
| Python MLX     | ~0.5            |
| Swift MLX      | ~1.0            |

## Goal

Optimize Swift MLX implementation to match Python MLX performance (~2x speedup needed).

## Optimization Strategies

- **Use MLX built-ins**: Prefer fast, efficient MLX built-in operations over manual implementations
- **Avoid Python-style loops**: Eliminate inefficient element-wise loops, especially in hot paths—use vectorized MLX operations instead
