# Porting Models from Python to Swift MLX

1. Clone the Python MLX and PyTorch implementations of the model to be ported into `/tmp` for easy access to the source code of the reference implementations.
2. Create a markdown document in the root of this repository called `porting-progress.md`, which you will use throughout the porting process.
3. Thoroughly investigate the Python MLX and PyTorch implementations of the model. Form a complete understanding of the overall architecture.
4. Add a full list of the model's components to the Porting Progress document. The list should be formatted as a table with columns titled "Component", "Initial port", "Verification", and "Notes". The list must be comprehensive, because it will guide the porting process. After your initial draft of the list, check all the model files one more time to ensure that the list is complete.
5. After completing the list, begin porting each component to Swift MLX. Investigate each component and understand its structure, purpose, and inner workings. You should port the each component fully and faithfully, without using placeholders, even if the components are complex. The end result must be a faithful, accurate, and well-optimized Swift port.
6. Ensure that your Swift port follows the best practices outlined in the `model-optimization.md` document in this directory, following Swift MLX conventions for style and efficiency. Refer to other models that have already been ported to Swift MLX in this repository, and use them as a guide.
7. After completing your initial port of all the components, use the Porting Progress document to slowly, methodically check each part of each component line by line against the original. If you find inaccuracies in the Swift port, correct them immediately. After your thorough verification of each component, add a checkmark to the "Verification" column in the Porting Progress document.
8. After checking all the components, complete two more rounds of verification and corrections. Remove the "Verification" checkmarks from the Porting Progress document before starting the next round of verification. Each round should be equally meticulous and methodical, checking each component line by line to ensure that the Swift port is accurate and efficient. Take your time to be thorough and accurate.
9. If you encounter problems or issues that require follow-up from the user during the porting process, keep track of them in the Porting Progress document.
10. Finally, create an end-to-end test in `package/Tests` using a reference audio file (see `audio-files.md` for URLs and transcriptions). Models that generate audio should save the resulting audio to `/tmp/test`. 



