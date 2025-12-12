# Whisper Configuration Pattern

## Dynamic Configuration Loading

Following mlx-swift-lm patterns, ALL Whisper model configurations are loaded dynamically from HuggingFace `config.json` files. **No hardcoded model dimensions.**

## Implementation

### ✅ ModelDimensions (WhisperConfig.swift)

```swift
public struct ModelDimensions: Codable, Sendable {
  public let n_mels: Int              // Varies: 80 (most models), 128 (large-v3-turbo)
  public let n_audio_ctx: Int         // 1500 for all models
  public let n_audio_state: Int       // Varies by model size
  public let n_audio_head: Int        // Varies by model size
  public let n_audio_layer: Int       // Varies by model size
  public let n_vocab: Int             // 51865 (most), 51866 (large-v3-turbo)
  public let n_text_ctx: Int          // 448 for all models
  public let n_text_state: Int        // Varies by model size
  public let n_text_head: Int         // Varies by model size
  public let n_text_layer: Int        // Varies by model size

  /// Load model dimensions from config.json file
  ///
  /// This is the canonical way to get model dimensions. All dimensions are loaded
  /// dynamically from HuggingFace config.json to avoid hardcoded mismatches.
  public static func load(from url: URL) throws -> ModelDimensions {
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    return try decoder.decode(ModelDimensions.self, from: data)
  }
}
```

**Key Points:**
- ✅ Pure `Codable` struct - mirrors mlx-swift-lm pattern
- ✅ NO hardcoded default values
- ✅ NO `static let tiny/base/small/medium/large` properties
- ✅ All dimensions loaded from HuggingFace at runtime

### ✅ WhisperAudio Constants

```swift
public enum WhisperAudio {
  public static let sampleRate = 16000       // ✅ Constant across all models
  public static let nFft = 400               // ✅ Constant across all models
  public static let hopLength = 160          // ✅ Constant across all models
  public static let chunkLength = 30         // ✅ Constant across all models
  public static let nSamples = 480_000       // ✅ Derived constant
  public static let nFrames = 3000           // ✅ Derived constant

  // ❌ REMOVED: public static let nMels = 80
  // n_mels varies by model and MUST be loaded from config.json
}
```

**Key Points:**
- ✅ Only truly constant values are included
- ✅ `n_mels` removed - it varies by model (80 vs 128)
- ✅ Documented why n_mels is not a constant

### ✅ Mel Spectrogram Function

```swift
func whisperLogMelSpectrogram(
  audio: MLXArray,
  nMels: Int,  // ✅ REQUIRED parameter, no default
  padding: Int = 0
) -> MLXArray
```

**Key Points:**
- ✅ `nMels` is a required parameter
- ✅ No hardcoded default value
- ✅ Always called with `model.dims.n_mels`

### ✅ Model Loading Flow

```swift
// WhisperModel.swift
static func load(
  modelSize: WhisperModelSize,
  progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
) async throws -> WhisperModel {
  let repoId = modelSize.repoId

  // 1. Download from HuggingFace
  let modelDirectory = try await Hub.snapshot(
    from: repoId,
    matching: ["weights.safetensors", "model.safetensors", "config.json"],
    progressHandler: progressHandler
  )

  // 2. Load config from file (NOT hardcoded)
  let configURL = modelDirectory.appending(path: "config.json")
  let dims = try ModelDimensions.load(from: configURL)  // ✅ Dynamic loading

  // 3. Initialize model with loaded dims
  let model = WhisperModel(dims: dims)

  // 4. Load weights
  let weights = try MLX.loadArrays(url: weightFileURL)
  try model.update(parameters: parameters, verify: [.noUnusedKeys])

  return model
}
```

**Key Points:**
- ✅ Config loaded from downloaded `config.json`
- ✅ No fallback to hardcoded values
- ✅ Dimensions passed to model initialization
- ✅ Matches mlx-swift-lm pattern exactly

### ✅ Usage in WhisperSTT

```swift
// WhisperSTT.swift
func transcribe(...) -> TranscriptionResult {
  // ...

  // Use model's n_mels (loaded from config.json)
  let mel = whisperLogMelSpectrogram(
    audio: paddedAudio,
    nMels: model.dims.n_mels  // ✅ Dynamic from config
  )

  // ...
}
```

**Key Points:**
- ✅ Always uses `model.dims.n_mels`
- ✅ No hardcoded values
- ✅ Works for all model sizes automatically

## Model Configuration Examples

From HuggingFace:

### whisper-tiny-mlx
```json
{
  "n_mels": 80,
  "n_audio_ctx": 1500,
  "n_audio_state": 384,
  "n_audio_head": 6,
  "n_audio_layer": 4,
  "n_vocab": 51865,
  "n_text_ctx": 448,
  "n_text_state": 384,
  "n_text_head": 6,
  "n_text_layer": 4
}
```

### whisper-large-v3-turbo
```json
{
  "n_mels": 128,           // ← Different!
  "n_audio_ctx": 1500,
  "n_audio_state": 1280,
  "n_audio_head": 20,
  "n_audio_layer": 32,
  "n_vocab": 51866,        // ← Different!
  "n_text_ctx": 448,
  "n_text_state": 1280,
  "n_text_head": 20,
  "n_text_layer": 4        // ← Different!
}
```

## Benefits of This Pattern

1. **No Hardcoded Mismatches**: Impossible to have config drift between code and actual models
2. **Future-Proof**: New model sizes work automatically without code changes
3. **Consistent with mlx-swift-lm**: Follows established patterns from sister project
4. **Easy Debugging**: Config issues immediately show as JSON decode errors, not silent shape mismatches
5. **Single Source of Truth**: HuggingFace config.json is the canonical source

## Comparison with Previous Implementation

### ❌ Before (Hardcoded)
```swift
// WhisperConfig.swift
public static let largeTurbo = ModelDimensions(
  n_mels: 80,  // ❌ WRONG! Should be 128
  n_vocab: 51865,  // ❌ WRONG! Should be 51866
  n_text_layer: 32  // ❌ WRONG! Should be 4
)

// Result: Shape mismatch crashes at runtime
// Error: conv expects 128 input channels, got 80
```

### ✅ After (Dynamic)
```swift
// WhisperConfig.swift
// NO hardcoded configs - all loaded from HuggingFace

// WhisperModel.swift
let dims = try ModelDimensions.load(from: configURL)
let model = WhisperModel(dims: dims)

// Result: Always correct, no shape mismatches
```

## Testing Pattern

```swift
@Test func whisperConfigurationsLoadCorrectly() async throws {
  let engine = STT.whisper(model: .largeTurbo)
  try await engine.load()

  // Config is loaded from HuggingFace, not hardcoded
  #expect(engine.isLoaded == true)

  // The model's dims are whatever HuggingFace config.json specifies
  // (no hardcoded assertions about specific values)
}
```

## Summary

✅ **All Whisper configurations follow the dynamic loading pattern**
- No hardcoded model dimensions
- All configs loaded from HuggingFace config.json at runtime
- Matches mlx-swift-lm architecture
- Future-proof and maintainable
