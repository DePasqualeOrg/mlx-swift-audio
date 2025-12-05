//
//  EnTokenizer.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/tokenizer.py
//  English text tokenizer for Chatterbox TTS
//

import Foundation
import MLX

// MARK: - Special Tokens

/// Start of text token
let SOT = "[START]"

/// End of text token
let EOT = "[STOP]"

/// Unknown token
let UNK = "[UNK]"

/// Space token (replaces actual spaces)
let SPACE = "[SPACE]"

/// All special tokens
let SpecialTokens = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

// MARK: - EnTokenizer

/// English text tokenizer for Chatterbox TTS
///
/// Uses a vocabulary file (tokenizer.json) to tokenize text into token IDs.
/// This is a simplified implementation that loads a vocabulary from JSON.
class EnTokenizer {
  /// Token to ID mapping
  private let vocabToId: [String: Int]

  /// ID to token mapping
  private let idToVocab: [Int: String]

  /// Vocabulary size
  var vocabSize: Int {
    vocabToId.count
  }

  /// Initialize tokenizer from vocabulary file
  init(vocabFilePath: String) throws {
    let fileURL = URL(fileURLWithPath: vocabFilePath)
    let data = try Data(contentsOf: fileURL)

    // Parse tokenizer.json format
    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
      // Try to get vocab from different locations in the JSON
      var vocab: [String: Int]?

      // Check for "model" -> "vocab" structure (HF tokenizers format)
      if let model = json["model"] as? [String: Any],
         let modelVocab = model["vocab"] as? [String: Int]
      {
        vocab = modelVocab
      }
      // Check for direct "vocab" key
      else if let directVocab = json["vocab"] as? [String: Int] {
        vocab = directVocab
      }
      // Check for added_tokens which is also in HF format
      else if let addedTokens = json["added_tokens"] as? [[String: Any]] {
        var tempVocab: [String: Int] = [:]
        for tokenInfo in addedTokens {
          if let content = tokenInfo["content"] as? String,
             let id = tokenInfo["id"] as? Int
          {
            tempVocab[content] = id
          }
        }
        if !tempVocab.isEmpty {
          vocab = tempVocab
        }
      }

      guard let finalVocab = vocab else {
        throw TokenizerError.invalidFormat("Could not find vocabulary in tokenizer file")
      }

      vocabToId = finalVocab
      idToVocab = Dictionary(uniqueKeysWithValues: finalVocab.map { ($1, $0) })
    } else {
      throw TokenizerError.invalidFormat("Could not parse tokenizer JSON")
    }

    // Verify required special tokens exist
    try checkVocab()
  }

  /// Initialize tokenizer from vocabulary dictionary
  init(vocab: [String: Int]) throws {
    vocabToId = vocab
    idToVocab = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })
    try checkVocab()
  }

  /// Verify required special tokens exist in vocabulary
  private func checkVocab() throws {
    guard vocabToId[SOT] != nil else {
      throw TokenizerError.missingToken(SOT)
    }
    guard vocabToId[EOT] != nil else {
      throw TokenizerError.missingToken(EOT)
    }
  }

  /// Convert text to token IDs
  func textToTokens(_ text: String) -> MLXArray {
    let tokenIds = encode(text)
    let int32Tokens = tokenIds.map { Int32($0) }
    return MLXArray(int32Tokens).reshaped([1, -1])
  }

  /// Encode text to token IDs
  /// Replaces spaces with SPACE token before encoding
  func encode(_ text: String) -> [Int] {
    // Replace spaces with SPACE token
    let processedText = text.replacingOccurrences(of: " ", with: SPACE)

    // Character-level tokenization (simplified)
    // In production, this would use the full tokenizer algorithm
    var tokens: [Int] = []

    var i = processedText.startIndex
    while i < processedText.endIndex {
      var matched = false

      // Try to match special tokens first
      for specialToken in SpecialTokens {
        if processedText[i...].hasPrefix(specialToken) {
          if let id = vocabToId[specialToken] {
            tokens.append(id)
            i = processedText.index(i, offsetBy: specialToken.count)
            matched = true
            break
          }
        }
      }

      if !matched {
        // Try to match longest substring in vocabulary
        var longestMatch: (String, Int)?
        var endIdx = processedText.index(after: i)

        while endIdx <= processedText.endIndex {
          let substr = String(processedText[i ..< endIdx])
          if let id = vocabToId[substr] {
            longestMatch = (substr, id)
          }
          if endIdx < processedText.endIndex {
            endIdx = processedText.index(after: endIdx)
          } else {
            break
          }
        }

        if let (matchedStr, id) = longestMatch {
          tokens.append(id)
          i = processedText.index(i, offsetBy: matchedStr.count)
        } else {
          // Unknown character, use UNK token or skip
          if let unkId = vocabToId[UNK] {
            tokens.append(unkId)
          }
          i = processedText.index(after: i)
        }
      }
    }

    return tokens
  }

  /// Decode token IDs back to text
  func decode(_ tokenIds: MLXArray) -> String {
    var ids = tokenIds
    if ids.ndim == 2 {
      ids = ids[0]
    }

    let count = ids.shape[0]
    var tokens: [String] = []

    for i in 0 ..< count {
      let id = Int(ids[i].item(Int32.self))
      if let token = idToVocab[id] {
        tokens.append(token)
      }
    }

    // Join and clean up
    var text = tokens.joined()
    text = text.replacingOccurrences(of: SPACE, with: " ")
    text = text.replacingOccurrences(of: EOT, with: "")
    text = text.replacingOccurrences(of: UNK, with: "")
    text = text.replacingOccurrences(of: SOT, with: "")

    return text
  }

  /// Decode token IDs back to text (from array)
  func decode(_ tokenIds: [Int]) -> String {
    decode(MLXArray(tokenIds.map { Int32($0) }))
  }

  /// Get start-of-text token ID
  func getSotTokenId() -> Int {
    vocabToId[SOT] ?? 0
  }

  /// Get end-of-text token ID
  func getEotTokenId() -> Int {
    vocabToId[EOT] ?? 0
  }
}

// MARK: - Errors

/// Tokenizer errors
enum TokenizerError: Error, LocalizedError {
  case invalidFormat(String)
  case missingToken(String)

  var errorDescription: String? {
    switch self {
      case let .invalidFormat(message):
        "Invalid tokenizer format: \(message)"
      case let .missingToken(token):
        "Tokenizer missing required token: \(token)"
    }
  }
}
