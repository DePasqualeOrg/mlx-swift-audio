# Whisper Swift Port - Implementation Status

## Overview

This document tracks the Swift port of Whisper STT from the Python MLX implementation.

## Comparison Against Python Implementation (Dec 12, 2024)

### Remaining Known Gaps

#### 1. **Timestamp Rules** ❌ NOT IMPLEMENTED (HIGH PRIORITY)

Python has a complex `ApplyTimestampRules` logit filter (325-395 lines) that enforces:

1. **Timestamp/Text Alternation**: After a timestamp, must generate text. After text, must generate timestamp.
2. **Monotonic Increase**: Timestamps must never decrease
3. **First Token Constraint**: When timestamps enabled, first generated token must be a timestamp
4. **max_initial_timestamp**: Limits the first timestamp to ≤1.0s by default
5. **Probability Heuristic**: If sum(P(timestamp tokens)) > max(P(text tokens)), force timestamp

**Impact**: Without this, segment-level timestamps may be:
- Missing entirely
- Out of order
- Interspersed incorrectly with text
- Not monotonically increasing

**Current behavior**: We add `<|0.00|>` to the SOT sequence but don't enforce any rules during decoding.

**Estimated effort**: ~150 lines of Swift to port ApplyTimestampRules logic

#### 2. **Compression Ratio** ❌ SIMPLIFIED

- **Python**: Uses `zlib.compress()` to measure text compressibility
- **Swift**: Simple ratio of `text.count / tokens.count`
- **Impact**: Low - only affects quality metrics, not transcription

## Testing Strategy

### Before Fixes
- Model generated repetitive pattern: `" OMaking<|startoflm|>"` in a loop
- Root cause: `<|startoflm|>` (sotLm) token was not being suppressed

### After Fixes (Current)
- Need to run tests to verify:
  1. Special tokens no longer appear in output
  2. Non-speech tokens suppressed
  3. Text output is coherent
  4. Timestamps are present (but may be malformed without timestamp rules)

### Future Testing
- Once timestamp rules implemented:
  - Verify timestamps are monotonically increasing
  - Verify timestamp/text alternation
  - Compare segment boundaries with Python output

## Files Modified

1. **WhisperDecoding.swift**
   - Lines 137-165: Added logit suppression (special + non-speech + blank)
   - Uses tokenizer.nonSpeechTokens() to get full suppression list

2. **WhisperTokenizer.swift**
   - Lines 312-359: New `nonSpeechTokens()` method
   - Returns ~100+ token IDs for annotations/symbols

3. **Documentation**
   - `whisper-missing-features.md`: Detailed comparison with Python
   - This file: High-level status summary

## Next Steps

### Immediate
1. ✅ Run tests to verify special token suppression works
2. ✅ Verify output is now coherent text instead of repetitive patterns

### Near-term (if timestamps needed)
1. ❌ Implement `ApplyTimestampRules` logit filter
2. ❌ Test segment-level timestamps against Python output
3. ❌ Add max_initial_timestamp option to DecodingOptions

### Nice-to-have
1. ❌ Implement proper compression ratio using compression library
2. ❌ Calculate no-speech probability from initial logits
3. ❌ Add word-level timestamps (requires attention weights)

## References

- **Python Source**: `/Users/anthony/files/projects/forked/mlx-audio-plus/mlx_audio/stt/models/whisper/`
  - `decoding.py`: Lines 302-535 for logit filters
  - `tokenizer.py`: Lines 244-277 for non_speech_tokens

- **Swift Implementation**: `/Users/anthony/files/projects/mlx-swift-audio/package/STT/Whisper/`
  - `WhisperDecoding.swift`: Greedy decoder with logit suppression
  - `WhisperTokenizer.swift`: Tokenization + special token handling
