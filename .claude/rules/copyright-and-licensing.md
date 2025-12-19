# Copyright and Licensing

Directions for how to handle licenses and copyright notices for original code created in this repository as well as derivative works based on third-party code.

This repository is MIT licensed. Ported models created in this repository, which are derivative works, retain their original licenses, which are stored in `licenses/`.

### Ported Model Files

All files containing code ported from another implementation must have a header identifying:

1. Original copyright holder and source repository URL
2. This repository's contributor
3. License file reference

**Header format for ported code:**

```swift
// Copyright © <year> <author> (original model implementation)
// Ported to MLX from https://github.com/original/repo
// Copyright © Anthony DePasquale (MLX port)
// License: licenses/modelname.txt
```

**When porting a new model:**

1. Find the original repository and its license.
2. Create `licenses/modelname.txt` with the original license text. Download the text file directly from the original model's GitHub repository.
3. Add the header to all ported Swift files.

### Original Code Created in This Repository

Files with original implementations (utilities, tests, standard algorithms like SwiGLU/RoPE) use a simple header:

```swift
// Copyright © Anthony DePasquale
```

Standard algorithms widely used across LLM implementations don't require attribution to any particular implementation—they're treated as original code.

### License Directory

- `licenses/README.md` - Explains the multi-license structure
- `licenses/modelname.txt` - Original license for each ported model or model family