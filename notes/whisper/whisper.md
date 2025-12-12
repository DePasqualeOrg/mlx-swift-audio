# Whisper

The Whisper implementation is based on [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/whisper) but uses a class-based API instead of functional.

## Synced Files

These files are kept in sync with upstream:

- `decoding.py` - Core decoding logic
- `tokenizer.py` - GPT-2 tokenizer
- `timing.py` - Word-level timestamps
- `writers.py` - Output formats

## Custom Files (Don't Sync)

- `whisper.py` - Custom `Model` class
- `audio.py` - Uses mlx-audio utilities
- `__init__.py` - Custom exports

## Usage

```bash
# Check for upstream changes
./scripts/sync-whisper-from-upstream.sh

# Sync and test
./scripts/sync-whisper-from-upstream.sh --test
```
