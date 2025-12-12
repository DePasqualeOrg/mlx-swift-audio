import Foundation
import Hub
import TiktokenSwift

/// Whisper tokenizer using TiktokenSwift for BPE tokenization
///
/// Provides quick access to special tokens and language-specific encoding
class WhisperTokenizer {
  private let encoding: CoreBpe
  private let specialTokens: [String: Int]

  // Special token IDs (from Whisper spec)
  // CRITICAL FIX: These values now match Python mlx-audio tokenizer exactly
  // Verified against Python tiktoken output (see dump_whisper_tokens.py)
  let eot: Int = 50257 // <|endoftext|>
  let sot: Int = 50258 // <|startoftranscript|>
  let sotPrev: Int = 50362 // <|startofprev|>
  let sotLm: Int = 50361 // <|startoflm|>
  let noSpeech: Int = 50363 // <|nospeech|>
  let noTimestamps: Int = 50364 // <|notimestamps|>
  let timestampBegin: Int = 50365 // <|0.00|>

  // Task tokens
  let translate: Int = 50359 // <|translate|>
  let transcribe: Int = 50360 // <|transcribe|>

  private init(encoding: CoreBpe, specialTokens: [String: Int]) {
    self.encoding = encoding
    self.specialTokens = specialTokens
  }

  /// Load tokenizer for Whisper
  ///
  /// Downloads and caches the appropriate Whisper tokenizer from HuggingFace
  ///
  /// - Parameter isMultilingual: Whether to load multilingual or English-only vocabulary
  /// - Returns: Initialized tokenizer
  static func load(isMultilingual: Bool) async throws -> WhisperTokenizer {
    // Whisper has two vocabulary files:
    // 1. multilingual.tiktoken - Used by multilingual Whisper models (tiny, base, small, medium, large-v3, large-v3-turbo)
    //    Contains 50,257 base vocabulary tokens optimized for multilingual speech recognition
    //    and translation across 99 languages
    //
    // 2. gpt2.tiktoken - Used by English-only Whisper models (tiny.en, base.en, small.en, medium.en)
    //    Contains the standard GPT-2 vocabulary (50,256 tokens) for English-only transcription
    //
    // Download from HuggingFace (same approach as model weights)

    // Download tokenizer file from HuggingFace using Hub API
    // This repo contains the official OpenAI Whisper tokenizer files
    let tokenizerRepo = "JosefAlbers/whisper"
    let vocabularyFile = isMultilingual ? "multilingual.tiktoken" : "gpt2.tiktoken"

    let tokenizerDir = try await Hub.snapshot(
      from: tokenizerRepo,
      matching: [vocabularyFile]
    )

    let tiktokenFile = tokenizerDir.appending(path: vocabularyFile)

    guard FileManager.default.fileExists(atPath: tiktokenFile.path) else {
      throw STTError.tokenizationFailed("\(vocabularyFile) not found in downloaded snapshot")
    }

    // Load and parse the tiktoken format (base64-encoded tokens with ranks)
    let data = try Data(contentsOf: tiktokenFile)
    let mergeableRanks = try parseTiktokenBpe(data)

    // Build Whisper-specific special tokens (these are added AFTER the base vocab)
    let specialTokens = buildSpecialTokens()

    // Convert to UInt32 for TiktokenSwift
    let specialTokensUInt32 = specialTokens.mapValues { UInt32($0) }

    // Whisper uses this pattern for tokenization
    let pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

    // Create CoreBpe with Whisper multilingual vocabulary + special tokens
    let whisperEncoding = try newCoreBpe(
      encoder: mergeableRanks,
      specialTokensEncoder: specialTokensUInt32,
      pattern: pattern
    )

    return WhisperTokenizer(encoding: whisperEncoding, specialTokens: specialTokens)
  }

  /// Parse tiktoken BPE format
  ///
  /// Format: base64-encoded token followed by space and rank
  private static func parseTiktokenBpe(_ data: Data) throws -> [[UInt8]: UInt32] {
    guard let content = String(data: data, encoding: .utf8) else {
      throw STTError.tokenizationFailed("Invalid tiktoken data encoding")
    }

    var encoder: [[UInt8]: UInt32] = [:]

    // Split by lines and parse each line
    let lines = content.split(separator: "\n")
    for line in lines {
      let trimmed = line.trimmingCharacters(in: .whitespaces)
      if trimmed.isEmpty { continue }

      // Each line has format: "base64_token rank"
      let parts = trimmed.split(separator: " ", maxSplits: 1)
      guard parts.count == 2,
            let rank = UInt32(parts[1])
      else {
        continue
      }

      // Decode the base64 token
      guard let tokenData = Data(base64Encoded: String(parts[0])) else {
        continue
      }

      // Store as byte array
      encoder[Array(tokenData)] = rank
    }

    return encoder
  }

  /// Build all Whisper special tokens
  private static func buildSpecialTokens() -> [String: Int] {
    var tokens: [String: Int] = [:]

    // CRITICAL FIX: Special tokens now match Python mlx-audio exactly
    // Verified against Python tiktoken output (see dump_whisper_tokens.py)

    // Core tokens
    tokens["<|endoftext|>"] = 50257
    tokens["<|startoftranscript|>"] = 50258

    // Language tokens: <|en|>, <|zh|>, etc. (99 languages, tokens 50259-50357)
    let languageCodes = [
      "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
      "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
      "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
      "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
      "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
      "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
      "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
      "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
      "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
      "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    ]

    for (i, lang) in languageCodes.enumerated() {
      tokens["<|\(lang)|>"] = 50259 + i
    }

    // Task tokens (translate=50359, transcribe=50360)
    tokens["<|translate|>"] = 50359
    tokens["<|transcribe|>"] = 50360

    // Other special tokens
    tokens["<|startoflm|>"] = 50361
    tokens["<|startofprev|>"] = 50362
    tokens["<|nospeech|>"] = 50363
    tokens["<|notimestamps|>"] = 50364

    // Timestamp tokens: <|0.00|> through <|30.00|> (1501 tokens, 0.02s increments)
    // Tokens 50365-51865 for vocab size 51866
    for i in 0 ... 1500 {
      let seconds = Float(i) * 0.02
      tokens["<|\(String(format: "%.2f", seconds))|>"] = 50365 + i
    }

    return tokens
  }

  /// Encode text to token IDs
  ///
  /// - Parameter text: Input text
  /// - Returns: Token IDs
  func encode(_ text: String) -> [Int] {
    // Use encodeOrdinary to avoid encoding special tokens in regular text
    encoding.encodeOrdinary(text: text).map { Int($0) }
  }

  /// Encode text with special tokens
  ///
  /// - Parameter text: Input text that may contain special tokens
  /// - Returns: Token IDs
  func encodeWithSpecialTokens(_ text: String) -> [Int] {
    encoding.encodeWithSpecialTokens(text: text).map { Int($0) }
  }

  /// Decode token IDs to text
  ///
  /// Filters out special tokens and timestamps
  ///
  /// - Parameter tokens: Token IDs
  /// - Returns: Decoded text
  func decode(_ tokens: [Int]) -> String {
    // Filter out special tokens (50256-50362) and timestamp tokens (>= 50363)
    // Keep only regular vocabulary tokens (0-50255)
    let textTokens = tokens.filter { $0 < 50256 }

    // Decode using CoreBpe
    let tokensUInt32 = textTokens.map { UInt32($0) }
    return (try? encoding.decode(tokens: tokensUInt32)) ?? ""
  }

  /// Decode token IDs including timestamps
  ///
  /// - Parameter tokens: Token IDs
  /// - Returns: Decoded text with timestamp annotations
  func decodeWithTimestamps(_ tokens: [Int]) -> String {
    var result = ""
    var currentText = ""
    var currentTokens: [UInt32] = []

    for token in tokens {
      if isTimestampToken(token) {
        // Flush any accumulated text
        if !currentTokens.isEmpty {
          if let decoded = try? encoding.decode(tokens: currentTokens) {
            currentText += decoded
          }
          currentTokens.removeAll()
        }

        // Add timestamp annotation
        let time = decodeTimestamp(token)
        result += currentText
        result += "<|\(String(format: "%.2f", time))|>"
        currentText = ""
      } else if token < eot {
        // Regular vocabulary token (0-50255)
        currentTokens.append(UInt32(token))
      }
      // else: skip special tokens (50256-50362)
    }

    // Flush remaining text
    if !currentTokens.isEmpty {
      if let decoded = try? encoding.decode(tokens: currentTokens) {
        result += decoded
      }
    }

    return result
  }

  /// Build SOT (Start-Of-Transcript) sequence
  ///
  /// - Parameters:
  ///   - language: Optional language code (e.g., "en", "zh")
  ///   - task: Transcription task
  /// - Returns: SOT token sequence
  func sotSequence(language: String?, task: TranscriptionTask) -> [Int] {
    var sequence = [sot]

    // Add language token if specified
    if let language, let langToken = specialTokens["<|\(language)|>"] {
      sequence.append(langToken)
    }

    // Add task token
    let taskToken = task == .transcribe ? transcribe : translate
    sequence.append(taskToken)

    return sequence
  }

  /// Build SOT sequence including no-timestamps token
  ///
  /// - Parameters:
  ///   - language: Optional language code
  ///   - task: Transcription task
  /// - Returns: SOT token sequence with no-timestamps
  func sotSequenceIncludingNoTimestamps(language: String?, task: TranscriptionTask) -> [Int] {
    var sequence = sotSequence(language: language, task: task)
    sequence.append(noTimestamps)
    return sequence
  }

  /// Get language token for a language code
  ///
  /// - Parameter language: Language code (e.g., "en")
  /// - Returns: Language token ID, or nil if not found
  func languageToken(for language: String) -> Int? {
    specialTokens["<|\(language)|>"]
  }

  /// Check if a token is a language token
  ///
  /// - Parameter token: Token ID
  /// - Returns: True if token is a language token
  func isLanguageToken(_ token: Int) -> Bool {
    token >= 50259 && token < 50358 // Language tokens are 50259-50357 (was 50258-50356)
  }

  /// Check if a token is a timestamp token
  ///
  /// - Parameter token: Token ID
  /// - Returns: True if token is a timestamp token
  func isTimestampToken(_ token: Int) -> Bool {
    token >= timestampBegin
  }

  /// Decode timestamp token to seconds
  ///
  /// - Parameter token: Timestamp token ID
  /// - Returns: Time in seconds
  func decodeTimestamp(_ token: Int) -> Float {
    guard token >= timestampBegin else { return 0.0 }
    let index = token - timestampBegin
    return Float(index) * 0.02
  }

  /// Get list of non-speech tokens to suppress
  ///
  /// Returns tokens for speaker tags and non-speech annotations like:
  /// - ♪♪♪ (music)
  /// - ( SPEAKING FOREIGN LANGUAGE )
  /// - [DAVID] (speaker names)
  ///
  /// - Returns: Array of token IDs to suppress
  func nonSpeechTokens() -> [Int] {
    // Symbols to suppress (split into individual characters and strings)
    var symbols: [String] = []

    // Individual characters
    for char in "\"#()*+/:;<=>@[\\]^_`{|}~「」『』" {
      symbols.append(String(char))
    }

    // Multi-character sequences
    symbols += ["<<", ">>", "<<<", ">>>", "--", "---", "-(", "-[", "('", "(\"", "((", "))", "(((", ")))", "[[", "]]", "{{", "}}", "♪♪", "♪♪♪"]

    // Musical symbols (U+2640 to U+267F)
    let miscellaneous = ["♩", "♪", "♫", "♬", "♭", "♮", "♯"]

    var result = Set<Int>()

    // Allow hyphens and quotes between words, but not at the beginning
    let dashTokens = encoding.encodeOrdinary(text: " -").map { Int($0) }
    if !dashTokens.isEmpty {
      result.insert(dashTokens[0])
    }
    let quoteTokens = encoding.encodeOrdinary(text: " '").map { Int($0) }
    if !quoteTokens.isEmpty {
      result.insert(quoteTokens[0])
    }

    // Encode symbols and add their tokens
    for symbol in symbols + miscellaneous {
      // Try encoding the symbol alone
      let tokens = encoding.encodeOrdinary(text: symbol).map { Int($0) }
      if !tokens.isEmpty {
        if tokens.count == 1 || miscellaneous.contains(symbol) {
          result.insert(tokens[0])
        }
      }

      // Try encoding with leading space
      let spacedTokens = encoding.encodeOrdinary(text: " " + symbol).map { Int($0) }
      if !spacedTokens.isEmpty {
        if spacedTokens.count == 1 || miscellaneous.contains(symbol) {
          result.insert(spacedTokens[0])
        }
      }
    }

    return Array(result).sorted()
  }
}
