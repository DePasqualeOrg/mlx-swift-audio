# Model Optimization

When optimizing a model that has been ported to MLX in Swift, use "plan mode" to gain an overview of the MLX implementation of the model in Python, which usually serves as the reference for the Swift port, as well as the ported model in Swift. The original PyTorch implementation of the model can also be used for reference.

Investigate the model's components in detail in the Swift port as well as the reference implementations in Python.

Look for any code that might need to be cleaned up or optimized.

Compile a list of potential optimizations and other improvements, and discuss each with the user to carefully consider any potential trade-offs between efficiency gains and accurate and clean code.

## Optimization Techniques

### MLX Built-ins
- Use optimized kernels: `RoPE`, `layerNorm`, `rmsNorm`, `scaledDotProductAttention` (available as free functions in the MLX module)

### Vectorization
- Avoid Python-style loops; use vectorized MLX operations (`MLX.where`, `MLX.split`, etc.)
- Use broadcast operations instead of explicit tiling
- Use scatter-add (`.at[indices].add()`) for sparse updates instead of loops

### Pre-computation
- Compute constants (frequencies, embeddings, masks) once during initialization
- Move invariant computations outside iteration loops

### KV Cache for Attention
- Cache key/value pairs across autoregressive decoding steps
- Cache encoder outputs for cross-attention (computed once, reused every step)
- Implement context window limiting to bound memory for long sequences

### Memory Management
- Configure GPU cache limits with `MLXMemory.configure(cacheLimit:)`
- Clear cache between heavy operations with `MLXMemory.clearCache()`
- Use platform-appropriate memory limits (iOS vs macOS)

### Computation Graph Control
- Use strategic `.eval()` calls to force computation and prevent graph explosion
- Evaluate intermediate results before caching or reuse
- Avoid building unbounded lazy computation graphs in loops

### Zero-Copy Operations
- Use view operations (`reshaped`, `transposed`, `expandedDimensions`) which don't copy
- Use `MLX.asStrided` for efficient sliding window operations
- Remap weight keys without duplicating array data

### Audio-Specific (GPU/MLX)
- Use strided convolutions for downsampling (stride-2 reduces sequence length by half)
- Use `MLX.rfft` for real-valued audio (2x faster than full FFT)
- Pre-norm residual connections for better gradient flow

### Audio-Specific (CPU/Accelerate)

For CPU-bound audio processing, use Apple's Accelerate framework for vectorized operations:

**vDSP for signal processing:**
- `vDSP_hann_window` - Efficient window function generation
- `vDSP_fft_zrip` / `vDSP_create_fftsetup` - Hardware-accelerated FFT
- `vDSP_vmul` - Vectorized multiplication (windowing, gain)
- `vDSP_vsmul` - Scalar multiplication (volume boost)
- `vDSP_vclip` - Clipping to range (peak limiting)
- `vDSP_svesq` - Sum of squares (RMS energy)
- `vDSP_maxv` / `vDSP_maxvi` - Find maximum value/index
- `vDSP_meanv` - Calculate mean
- `vDSP_vramp` - Generate frequency ramps
- `vDSP_dotpr` - Dot product (spectral centroid)
- `vDSP_zvmags` - Complex magnitude

**vvecLib for element-wise math:**
- `vvexpf` - Vectorized exponential (softmax)
- `vvsqrtf` - Vectorized square root

**AVFoundation for audio I/O:**
- `AVAudioConverter` - High-quality resampling with anti-aliasing
- `AVAudioEngine` / `AVAudioPlayerNode` - Low-latency playback
- `AVAudioPCMBuffer` - Efficient buffer management

### Algorithm Optimizations
- Use sorting networks for small fixed-size operations (e.g., median-of-7 with 10 comparisons)
- Use flat arrays with inline indexing for cache efficiency (DTW, attention)
- Use `DispatchQueue.concurrentPerform` for parallel CPU work on independent rows
- Use `@inline(__always)` and `@inlinable` for hot paths
- Pre-allocate arrays with `reserveCapacity` to avoid reallocations
