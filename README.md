# MLX Swift Audio

**This package is in early development. Expect breaking changes.**

- Text to speech
  - [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice)
  - [Chatterbox](https://github.com/resemble-ai/chatterbox)
  - [OuteTTS](https://github.com/edwko/OuteTTS)
  - [Kokoro](https://github.com/hexgrad/kokoro)
  - [Orpheus](https://github.com/canopyai/Orpheus-TTS)
  - [Marvis](https://github.com/Marvis-Labs/marvis-tts)

- Speech to text
  - [Whisper](https://github.com/openai/whisper)

## Installation

In Xcode, go to File > Add Package Dependencies and enter `https://github.com/DePasqualeOrg/mlx-swift-audio`. Select the `main` branch, then add `MLXAudio` to your target. If you want to use Kokoro (which has GPLv3 dependencies), also add the `Kokoro` library.

## Usage

### Text to speech

```swift
import MLXAudio

// CosyVoice2 - voice matching with zero-shot and cross-lingual modes
let cosyVoice = TTS.cosyVoice2()
try await cosyVoice.load()
let speaker = try await cosyVoice.prepareSpeaker(from: audioFileURL)
try await cosyVoice.say("Speaking with your voice.", speaker: speaker)

// With style instructions
try await cosyVoice.say("This is exciting news!", speaker: speaker, instruction: "Speak with enthusiasm")

// Voice conversion - transform audio to sound like the speaker
let converted = try await cosyVoice.convertVoice(from: sourceAudioURL, to: speaker)

// Chatterbox - custom voices from reference audio and emotion control
let chatterbox = TTS.chatterbox()
try await chatterbox.load()
let referenceAudio = try await chatterbox.prepareReferenceAudio(from: audioFileURL)
try await chatterbox.say("Speaking with your reference audio.", referenceAudio: referenceAudio)

// OuteTTS - custom voices from reference audio
let outetts = TTS.outetts()
try await outetts.load()
let speaker = try await OuteTTSSpeakerProfile.load(from: "speaker.json")
try await outetts.say("Using reference audio.", speaker: speaker)

// Orpheus - emotional expressions
let orpheus = TTS.orpheus()
try await orpheus.load()
try await orpheus.say("Ha! <laugh> That's funny.", voice: .tara)

// Marvis - streaming audio
let marvis = TTS.marvis()
try await marvis.load()
try await marvis.sayStreaming("This plays as it generates.", voice: .conversationalA)

// For more control over playback
let orpheus = TTS.orpheus()
try await orpheus.load()
let audio = try await orpheus.generate("Hello!", voice: .tara)
await audio.play()

```

### Speech to text

```swift
import MLXAudio

// Whisper - multilingual speech recognition
let whisper = STT.whisper(model: .largeTurbo)
try await whisper.load()

// Transcribe audio file (language auto-detected)
let result = try await whisper.transcribe(audioFileURL)
print(result.text)

// Transcribe with specific language
let result = try await whisper.transcribe(audioFileURL, language: .spanish)

// Translate to English
let translation = try await whisper.translate(audioFileURL)

// Detect language only
let (language, confidence) = try await whisper.detectLanguage(audioFileURL)
print("\(language.displayName) (\(confidence))")
```

## Building

Build the library:

```sh
xcodebuild -scheme mlx-audio -destination 'platform=macOS' build
```

Build the example app:

```sh
xcodebuild -project 'examples/TTS App/TTS App.xcodeproj' -scheme 'TTS App' -destination 'platform=macOS' build
```

## Legal and ethical considerations

Voice synthesis technology should be used responsibly. Obtain consent before using voice recordings, respect intellectual property and personality rights, never use synthetic voices for deception or fraud, and comply with applicable laws in your jurisdiction.

## License

This project is licensed under the MIT License.

The main MLXAudio library includes all TTS engines except Kokoro. The separate Kokoro library imports [espeak-ng-spm](https://github.com/espeak-ng/espeak-ng-spm) as a Swift package, which is licensed under GPLv3. To use Kokoro, explicitly import the separate Kokoro library.

## History

Commit [22b498c](https://github.com/DePasqualeOrg/mlx-swift-audio/commit/22b498ceaf01fa2ee138bb36c62799172efbd6ab) in this repository corresponds to commit [0ee931b](https://github.com/DePasqualeOrg/mlx-audio/commit/0ee931b6971a338f7c48176a86db217a434a0036) ([PR #279](https://github.com/Blaizzy/mlx-audio/pull/279)) in [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio), in which the Swift library and example app were completely rewritten. The commit history of files from mlx-audio has been preserved.
