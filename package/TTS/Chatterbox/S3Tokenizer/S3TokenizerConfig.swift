//
//  S3TokenizerConfig.swift
//  MLXAudio
//
//  Ported from mlx_audio/codec/models/s3/
//

import Foundation

/// S3Tokenizer constants
enum S3TokenizerConstants {
  static let s3Sr: Int = 16000 // Sample rate for S3Tokenizer
  static let s3Hop: Int = 160 // 100 frames/sec
  static let s3TokenHop: Int = 640 // 25 tokens/sec
  static let s3TokenRate: Int = 25
  static let speechVocabSize: Int = 6561 // 3^8
}

/// Configuration for S3Tokenizer V2
struct S3TokenizerModelConfig: Codable, Sendable {
  var nMels: Int = 128
  var nAudioCtx: Int = 1500
  var nAudioState: Int = 1280
  var nAudioHead: Int = 20
  var nAudioLayer: Int = 6
  var nCodebookSize: Int = 6561 // 3^8

  enum CodingKeys: String, CodingKey {
    case nMels = "n_mels"
    case nAudioCtx = "n_audio_ctx"
    case nAudioState = "n_audio_state"
    case nAudioHead = "n_audio_head"
    case nAudioLayer = "n_audio_layer"
    case nCodebookSize = "n_codebook_size"
  }

  init() {}

  init(
    nMels: Int = 128,
    nAudioCtx: Int = 1500,
    nAudioState: Int = 1280,
    nAudioHead: Int = 20,
    nAudioLayer: Int = 6,
    nCodebookSize: Int = 6561,
  ) {
    self.nMels = nMels
    self.nAudioCtx = nAudioCtx
    self.nAudioState = nAudioState
    self.nAudioHead = nAudioHead
    self.nAudioLayer = nAudioLayer
    self.nCodebookSize = nCodebookSize
  }
}
