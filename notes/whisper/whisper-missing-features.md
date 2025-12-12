# Missing Features in Swift Whisper Implementation

Compared to the Python MLX Whisper implementation, our Swift version is missing several important features:

## 1. ❌ Non-Speech Token Suppression

**Python**: `tokenizer.non_speech_tokens` property (lines 244-277 in tokenizer.py)
- Suppresses ~100+ tokens for speaker tags and non-speech annotations
- Includes symbols: `♪♪♪`, `( SPEAKING FOREIGN LANGUAGE )`, `[DAVID]`, brackets, etc.
- These tokens are added to suppress_tokens list by default (when suppress_tokens="-1")

**Swift**: Not implemented
- We don't have a `nonSpeechTokens` property in WhisperTokenizer
- These tokens can still be generated, potentially causing garbage output

**Impact**: Medium - Can cause unwanted speaker tags or annotations in output

---

## 2. ❌ Timestamp Rules Enforcement

**Python**: `ApplyTimestampRules` logit filter (lines 325-395 in decoding.py)

Complex logic that:
1. **Suppresses `<|notimestamps|>` token** - handled separately
2. **Forces timestamp/text alternation** - timestamps must alternate with text tokens
3. **Prevents timestamp decrease** - timestamps must be monotonically increasing
4. **Forces first token to be timestamp** - when timestamps enabled, first generated token must be timestamp
5. **Applies max_initial_timestamp** - limits the maximum timestamp for the first token (default 1.0s)
6. **Timestamp probability heuristic** - if sum of timestamp probabilities > max text probability, force timestamp

**Swift**: Not implemented at all
- We add either timestampBegin or noTimestamps to SOT sequence, but don't enforce rules during decoding
- No alternation enforcement
- No monotonic increase check
- No max_initial_timestamp logic

**Impact**: HIGH for segment-level timestamps
- Without this, timestamps can be malformed
- May generate invalid timestamp sequences
- May not generate timestamps at all

---

## 3. ✅ Special Token Suppression (JUST ADDED)

**Python**: Lines 522-535 in decoding.py
```python
suppress_tokens.extend([
    self.tokenizer.transcribe,
    self.tokenizer.translate,
    self.tokenizer.sot,
    self.tokenizer.sot_prev,
    self.tokenizer.sot_lm,
])
if self.tokenizer.no_speech is not None:
    suppress_tokens.append(self.tokenizer.no_speech)
```

**Swift**: ✅ NOW IMPLEMENTED in WhisperDecoding.swift lines 137-164
- Suppresses sot, sotPrev, sotLm, translate, transcribe
- Missing noSpeech token suppression (should add if needed)

---

## 4. ✅ SuppressBlank (JUST ADDED)

**Python**: `SuppressBlank` logit filter (lines 302-312 in decoding.py)
- Applied only when `tokens.shape[1] == sample_begin` (first generated token)
- Suppresses space tokens and EOT to prevent blank output

**Swift**: ✅ NOW IMPLEMENTED in WhisperDecoding.swift lines 150-158
- Applied when `iteration == 0`

---

## 5. ❌ Compression Ratio Calculation

**Python**: Uses zlib compression (lines 14-16 in decoding.py)
```python
def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))
```

**Swift**: Simple approximation (WhisperDecoding.swift:178)
```swift
let compressionRatio = text.isEmpty ? 1.0 : Float(text.count) / Float(tokens.count)
```

**Impact**: Low - compression ratio is used for quality assessment but doesn't affect decoding

---

## 6. ❌ No-Speech Probability Calculation

**Python**: Lines 595-599 in decoding.py
```python
if self.tokenizer.no_speech is not None:
    probs_at_sot = mx.softmax(pre_logits[:, self.sot_index], axis=-1)
    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech]
```

**Swift**: Hardcoded to 0.0 (WhisperDecoding.swift:189)
```swift
noSpeechProb: 0.0, // TODO: Compute from initial logits
```

**Impact**: Low - useful for detecting silent audio but not critical for transcription

---

## 7. ❌ Decode Method (Filter Special Tokens)

**Python**: `tokenizer.decode()` filters out timestamp tokens (line 167)
```python
def decode(self, token_ids: List[int], **kwargs) -> str:
    token_ids = [t for t in token_ids if t < self.timestamp_begin]
    return self.encoding.decode(token_ids, **kwargs)
```

**Swift**: Relies on TiktokenSwift's decode which may not filter timestamps
- Need to verify if our decode() properly filters timestamp tokens (>= 50364)

**Impact**: Medium - timestamps might appear in output text

---

## Priority for Fixes

### HIGH PRIORITY (breaks basic functionality)
1. **Timestamp Rules** - Without this, segment-level timestamps won't work correctly
2. **Non-Speech Token Suppression** - Can cause garbage output

### MEDIUM PRIORITY (improves quality)
3. **Decode filtering** - Verify timestamps are filtered from text output
4. **Compression ratio** - Better quality metric

### LOW PRIORITY (nice to have)
5. **No-speech probability** - Useful for detecting silence but not critical

---

## Current Status After Latest Fixes

✅ Special token suppression (sot, sot_lm, sot_prev, transcribe, translate, noSpeech)
✅ Blank suppression at first token (space tokens + EOT)
✅ Non-speech token suppression (~100+ tokens for speaker tags, music, etc.)
✅ Decode filtering (timestamps filtered correctly in decode() method)
❌ Timestamp rules (ApplyTimestampRules) - **HIGH PRIORITY**
❌ Proper compression ratio (using zlib)
❌ No-speech probability calculation

## Summary of Fixes Applied (2024-12-12)

### 1. Special Token Suppression
**File**: `WhisperDecoding.swift:137-155`
- Added suppression for: sot, sotPrev, sotLm, translate, transcribe, noSpeech
- Creates -inf mask for these tokens to prevent sampling

### 2. Non-Speech Token Suppression
**File**: `WhisperTokenizer.swift:312-359`
- New method: `nonSpeechTokens()` returning ~100+ token IDs
- Suppresses speaker tags: `[DAVID]`, `(SPEAKING FOREIGN LANGUAGE)`
- Suppresses music symbols: `♪♪♪`, `♫`, `♬`
- Suppresses brackets/parens when used for annotations
- Integrated into WhisperDecoding.swift:149

### 3. Blank Suppression
**File**: `WhisperDecoding.swift:157-165`
- Applied only on first generated token (iteration == 0)
- Suppresses space tokens and EOT to prevent blank/empty output

### 4. Decode Filtering (Already Present)
**File**: `WhisperTokenizer.swift:187-193`
- Filters timestamp tokens (>= 50364) from output text
- Matches Python behavior

## Remaining Critical Issue: Timestamp Rules

The `ApplyTimestampRules` logit filter is the most complex missing piece. It enforces:

1. **Alternation**: Timestamps must alternate with text tokens
2. **Monotonicity**: Timestamps must increase or stay same, never decrease
3. **First token constraint**: When timestamps enabled, first token must be timestamp
4. **max_initial_timestamp**: Limits first timestamp (default 1.0s)
5. **Probability heuristic**: Forces timestamp when P(timestamp) > P(text)

Without this, segment-level timestamps may be malformed or missing entirely.

**Estimated complexity**: ~150 lines of Swift code to port the logic from Python's ApplyTimestampRules class.
