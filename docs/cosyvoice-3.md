# CosyVoice3 Reference Audio Guide

This document describes the Swift `CosyVoice3Engine` behavior that intentionally tracks the standard frontend from the original [`FunAudioLLM/CosyVoice`](https://github.com/FunAudioLLM/CosyVoice) PyTorch implementation.

## Zero-shot vs cross-lingual

- Zero-shot uses reference audio plus an explicit transcript of that same reference clip.
- Cross-lingual uses the reference audio without a transcript and is the safer mode when you do not have a clean matching transcript.

In the standard frontend from the original PyTorch implementation, zero-shot goes through `cosyvoice/cli/frontend.py:frontend_zero_shot`, which consumes caller-supplied `prompt_text`, and speech-token extraction goes through `cosyvoice/cli/frontend.py:_extract_speech_token`.

## Reference audio limit

- Reference audio must be 30 seconds or shorter.
- The standard frontend from the original PyTorch implementation asserts on longer prompt audio in `cosyvoice/cli/frontend.py:_extract_speech_token`.
- The Swift CosyVoice3 port follows that behavior and rejects longer reference audio instead of silently clipping it.

## Recommended workflow for zero-shot

1. Trim the reference audio to 30 seconds or less before calling Swift.
2. Make sure the clip starts and ends cleanly, without cut-off words.
3. Provide the exact transcript for that exact clipped segment.
4. Use cross-lingual instead if you do not have a clean transcript.

The reason for this is prompt alignment. Zero-shot conditions the model on both prompt audio and prompt text. If the clip is cut mid-word or the transcript does not match the exact segment, the model can produce startup artifacts, garbage syllables, or early silence.

## Auto-transcription in Swift

`CosyVoice3Engine.autoTranscribe` is a Swift convenience for metadata and display only. It does not promote a speaker into the standard zero-shot mode used by the original PyTorch implementation.

That behavior is intentional: using STT output as zero-shot prompt text is less reliable than using an explicit user-provided transcript, especially when the reference clip is noisy, long, or trimmed imperfectly.

## Swift API summary

- `prepareSpeaker(from:transcription:)` with `transcription:` supplied:
  uses zero-shot-compatible conditioning
- `prepareSpeaker(from:)` without `transcription:`:
  prepares a speaker for cross-lingual use
- `generate(..., instruction:)` or `say(..., instruction:)`:
  uses instruct mode when an instruction is present
