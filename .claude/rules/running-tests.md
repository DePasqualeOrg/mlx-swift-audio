# Running Tests

Use `xcodebuild` for tests that require Metal GPU access (MLX operations).

**Important:** `swift test` does not have Metal library access and will fail with MLX operations.

### Quick Iteration Strategy (Recommended)

**Key Principles for Fast Testing:**

1. **Use explicit Mac device ID** - Completely bypasses iOS device discovery (no DTDKRemoteDeviceConnection errors)
2. **Pre-build once, run many** - Use `build-for-testing` then `test-without-building`
3. **Run long tests in background** - Use `&` and check progress with `tail`
4. **Use backticks for substitution** - Use `` `cmd` `` not `$(cmd)` for compatibility

**Time breakdown for typical test runs:**

- xcodebuild startup: ~3-5 seconds (unavoidable)
- Metal shader compilation: ~30 seconds (first run only per session)
- Model loading: varies by model (skip for unit tests)
- Actual test execution: depends on test

For fast debugging cycles, use this approach:

**Step 0: Get and save your Mac's device ID** (run once per machine)

```bash
# Extract MAC_ID and save to file (only need to do this once)
xcodebuild -scheme mlx-audio-Package -showdestinations 2>/dev/null | grep "platform:macOS, arch:arm64, id:" | head -1 | awk -F'id:' '{print $2}' | awk -F',' '{print $1}' > /tmp/mac_id.txt
cat /tmp/mac_id.txt
```

**Step 1: Pre-build once** (avoids recompilation during test runs)

```bash
# Build in background, check progress
MAC_ID=`cat /tmp/mac_id.txt` && xcodebuild build-for-testing -scheme mlx-audio-Package -destination "id=$MAC_ID,arch=arm64" -destination-timeout 1 -derivedDataPath /tmp/DerivedData > /tmp/build.txt 2>&1 &

# Check progress
tail -20 /tmp/build.txt
```

**Step 2: Run tests** (uses cached build)

```bash
# Run test in background, saving output
MAC_ID=`cat /tmp/mac_id.txt` && xcodebuild test-without-building -scheme mlx-audio-Package -destination "id=$MAC_ID,arch=arm64" -destination-timeout 1 -derivedDataPath /tmp/DerivedData -only-testing:MLXAudioTests/YourTestClass/yourTestMethod > /tmp/test.txt 2>&1 &

# Check progress (run multiple times)
tail -30 /tmp/test.txt

# Check if still running
pgrep -f xcodebuild
```

**Syntax Notes**

- Use **backticks** `` `cat /tmp/mac_id.txt` `` for command substitution (not `$(...)`)
- Use **`-destination "id=$MAC_ID,arch=arm64" -destination-timeout 1`** to bypass iOS device discovery
- **Test method syntax**: `-only-testing:MLXAudioTests/SuiteName/testMethodName` (no parentheses needed)
- Run **in background** with `&` for long tests, check with `tail`

**Step 3: Check results** (filter out iOS device noise)

```bash
# Filter out DTDKRemoteDeviceConnection errors (from connected iOS devices)
FILTER='grep -v "DTDKRemote\|MobileDevice\|stacktrace\|DVT\|libdispatch\|libsystem_pthread\|0x0\|Domain=\|UserInfo\|NSLocal\|passcode"'

# Check progress (filtered)
cat /tmp/test.txt | eval $FILTER | tail -30

# Quick error check (filtered)
cat /tmp/test.txt | eval $FILTER | grep -E "(fatal|Fatal|error:|failed)"

# See debug prints (format: [ComponentName] message)
grep "^\[" /tmp/test.txt

# Check test passed/failed
grep -E "(passed|failed|Test)" /tmp/test.txt | tail -10
```

### Test Commands Reference

```bash
# Get MAC_ID (once per machine)
xcodebuild -scheme mlx-audio-Package -showdestinations 2>/dev/null | grep "platform:macOS, arch:arm64, id:" | head -1 | awk -F'id:' '{print $2}' | awk -F',' '{print $1}' > /tmp/mac_id.txt

# Build for testing (do once, or after code changes)
MAC_ID=`cat /tmp/mac_id.txt` && xcodebuild build-for-testing -scheme mlx-audio-Package -destination "id=$MAC_ID,arch=arm64" -destination-timeout 1 -derivedDataPath /tmp/DerivedData > /tmp/build.txt 2>&1 &

# Run specific test
MAC_ID=`cat /tmp/mac_id.txt` && xcodebuild test-without-building -scheme mlx-audio-Package -destination "id=$MAC_ID,arch=arm64" -destination-timeout 1 -derivedDataPath /tmp/DerivedData -only-testing:MLXAudioTests/WhisperTests/whisperBasicTranscribe > /tmp/test.txt 2>&1 &

# Run all tests in a suite
MAC_ID=`cat /tmp/mac_id.txt` && xcodebuild test-without-building -scheme mlx-audio-Package -destination "id=$MAC_ID,arch=arm64" -destination-timeout 1 -derivedDataPath /tmp/DerivedData -only-testing:MLXAudioTests/WhisperTests > /tmp/test.txt 2>&1 &

# Monitor progress (filtered)
grep -v "DTDKRemote\|MobileDevice\|stacktrace" /tmp/test.txt | tail -30
```

### Debugging Test Failures

```bash
# See full context around errors
grep -B5 -A10 "fatal\|Fatal\|error\|Error" /tmp/test.txt

# Check what the test was doing when it failed
tail -100 /tmp/test.txt

# Find debug print statements (format: [ComponentName] message)
grep "^\[" /tmp/test.txt
```

### Best Practices

1. **Save MAC_ID to file** - Run the extraction once, read with backticks thereafter
2. **Use -derivedDataPath /tmp/DerivedData** - Keeps build artifacts in known location for reuse
3. **Run in background** - Use `&` for tests, check with `tail -30 /tmp/test.txt`
4. **Save all output to /tmp** - Tests can take time and terminal output can be lost
5. **Add debug prints** - Use `print("[ComponentName] message")` format for easy grep