# MLX Swift Audio

**This package is in early development. Expect breaking changes.**

- Text to speech
  - [Chatterbox](https://github.com/resemble-ai/chatterbox)
  - [OuteTTS](https://github.com/edwko/OuteTTS)
  - [Kokoro](https://github.com/hexgrad/kokoro)
  - [Orpheus](https://github.com/canopyai/Orpheus-TTS)
  - [Marvis](https://github.com/Marvis-Labs/marvis-tts)

### Installation

```swift
.package(url: "https://github.com/DePasqualeOrg/mlx-swift-audio.git", branch: "main")
```

### Usage

```swift
import MLXAudio

// Kokoro - 50+ voices, speed control
let kokoro = TTS.kokoro()
try await kokoro.load()
try await kokoro.say("Hello, world!", voice: .afHeart)
try await kokoro.say("Speaking faster", voice: .afNova, speed: 1.5)

// Orpheus - emotional expressions
let orpheus = TTS.orpheus()
try await orpheus.load()
try await orpheus.say("Ha! <laugh> That's funny.", voice: .tara)

// Marvis - streaming audio
let marvis = TTS.marvis()
try await marvis.load()
try await marvis.sayStreaming("This plays as it generates.", voice: .conversationalA)

// OuteTTS - custom voices with reference audio
let outetts = TTS.outetts()
try await outetts.load()
let speaker = try await OuteTTSSpeakerProfile.load(from: "speaker.json")
try await outetts.say("Using reference audio.", speaker: speaker)

// Chatterbox - custom reference audio with emotion control
let chatterbox = TTS.chatterbox()
try await chatterbox.load()
let referenceAudio = try await chatterbox.prepareReferenceAudio(from: audioFileURL)
try await chatterbox.say("Speaking with your reference audio.", referenceAudio: referenceAudio)
```

For more control over playback:

```swift
let kokoro = TTS.kokoro()
try await kokoro.load()
let audio = try await kokoro.generate("Hello!", voice: .afHeart)
await audio.play()
```

### Building

Build the library:

```sh
xcodebuild -scheme mlx-audio -destination 'platform=macOS' build
```

Build the example app:

```sh
xcodebuild -project 'examples/TTS App/TTS App.xcodeproj' -scheme 'TTS App' -destination 'platform=macOS' build
```

### Legal and ethical considerations

Voice synthesis technology should be used responsibly. Obtain consent before using voice recordings, respect intellectual property and personality rights, never use synthetic voices for deception or fraud, and comply with applicable laws in your jurisdiction.

### License

This project is licensed under the MIT License.

The Kokoro TTS engine imports [espeak-ng-spm](https://github.com/espeak-ng/espeak-ng-spm) as a Swift package, which is licensed under GPLv3. This may have implications for binaries built with this library.

### History

Commit [22b498c](https://github.com/DePasqualeOrg/mlx-swift-audio/commit/22b498ceaf01fa2ee138bb36c62799172efbd6ab) in this repository corresponds to commit [0ee931b](https://github.com/DePasqualeOrg/mlx-audio/commit/0ee931b6971a338f7c48176a86db217a434a0036) ([PR #279](https://github.com/Blaizzy/mlx-audio/pull/279)) in [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio), in which the Swift library and example app were completely rewritten. The commit history of files from mlx-audio has been preserved.
