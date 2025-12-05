//
//  SNACConfig.swift
//  MLXAudio
//
//  Created by Ben Harraway on 14/05/2025.
//
import Foundation

struct SNACConfig: Codable {
  let samplingRate: Int
  let encoderDim: Int
  let encoderRates: [Int]
  let decoderDim: Int
  let decoderRates: [Int]
  let attnWindowSize: Int?
  let codebookSize: Int
  let codebookDim: Int
  let vqStrides: [Int]
  let noise: Bool
  let depthwise: Bool
  let latentDim: Int

  private enum CodingKeys: String, CodingKey {
    case samplingRate = "sampling_rate"
    case encoderDim = "encoder_dim"
    case encoderRates = "encoder_rates"
    case decoderDim = "decoder_dim"
    case decoderRates = "decoder_rates"
    case attnWindowSize = "attn_window_size"
    case codebookSize = "codebook_size"
    case codebookDim = "codebook_dim"
    case vqStrides = "vq_strides"
    case noise
    case depthwise
    case latentDim = "latent_dim"
  }

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    samplingRate = try container.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
    encoderDim = try container.decodeIfPresent(Int.self, forKey: .encoderDim) ?? 48
    encoderRates = try container.decodeIfPresent([Int].self, forKey: .encoderRates) ?? [2, 4, 8, 8]
    decoderDim = try container.decodeIfPresent(Int.self, forKey: .decoderDim) ?? 1024
    decoderRates = try container.decodeIfPresent([Int].self, forKey: .decoderRates) ?? [8, 8, 4, 2]
    attnWindowSize = try container.decodeIfPresent(Int.self, forKey: .attnWindowSize)
    codebookSize = try container.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 4096
    codebookDim = try container.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 8
    vqStrides = try container.decodeIfPresent([Int].self, forKey: .vqStrides) ?? [4, 2, 1]
    noise = try container.decodeIfPresent(Bool.self, forKey: .noise) ?? true
    depthwise = try container.decodeIfPresent(Bool.self, forKey: .depthwise) ?? true

    // Calculate latentDim if not provided
    if let decoded = try container.decodeIfPresent(Int.self, forKey: .latentDim) {
      latentDim = decoded
    } else {
      latentDim = encoderDim * Int(pow(2.0, Double(encoderRates.count)))
    }
  }

  init(
    samplingRate: Int = 24000,
    encoderDim: Int = 48,
    encoderRates: [Int] = [2, 4, 8, 8],
    decoderDim: Int = 1024,
    decoderRates: [Int] = [8, 8, 4, 2],
    attnWindowSize: Int? = nil,
    codebookSize: Int = 4096,
    codebookDim: Int = 8,
    vqStrides: [Int] = [4, 2, 1],
    noise: Bool = true,
    depthwise: Bool = true,
    latentDim: Int? = nil,
  ) {
    self.samplingRate = samplingRate
    self.encoderDim = encoderDim
    self.encoderRates = encoderRates
    self.decoderDim = decoderDim
    self.decoderRates = decoderRates
    self.attnWindowSize = attnWindowSize
    self.codebookSize = codebookSize
    self.codebookDim = codebookDim
    self.vqStrides = vqStrides
    self.noise = noise
    self.depthwise = depthwise

    // Calculate latentDim if not provided
    if let latentDim {
      self.latentDim = latentDim
    } else {
      self.latentDim = encoderDim * Int(pow(2.0, Double(encoderRates.count)))
    }
  }
}
