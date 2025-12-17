// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

import Foundation
import MLX
import MLXNN

// MARK: - Rotary Position Embeddings

/// Precompute frequency tensor for rotary embeddings
func precomputeFreqsCis(
  dim: Int,
  end: Int,
  theta: Float = 10000.0,
  scaling: Float? = nil,
) -> (MLXArray, MLXArray) {
  let halfDim = dim / 2
  let freqsExponent = MLXArray(0 ..< halfDim).asType(.float32) / Float(dim)
  let freqs = 1.0 / MLX.pow(MLXArray(theta), freqsExponent)

  var t = MLXArray(0 ..< end).asType(.float32)
  if let scaling {
    t = t * scaling
  }

  let freqsOuter = MLX.outer(t, freqs).asType(.float32)
  let cosFreqs = MLX.cos(freqsOuter)
  let sinFreqs = MLX.sin(freqsOuter)

  // Concatenate to double the dimension
  let cosFreqsDouble = MLX.concatenated([cosFreqs, cosFreqs], axis: -1)
  let sinFreqsDouble = MLX.concatenated([sinFreqs, sinFreqs], axis: -1)

  return (cosFreqsDouble, sinFreqsDouble)
}

/// Apply rotary embeddings to query and key tensors
func applyRotaryEmb(
  xq: MLXArray,
  xk: MLXArray,
  cos: MLXArray,
  sin: MLXArray,
) -> (MLXArray, MLXArray) {
  // Expand dimensions for broadcasting: (1, T, 1, D)
  let cosExpanded = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
  let sinExpanded = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 2)

  let D = xq.shape[xq.ndim - 1]
  let halfD = D / 2

  // Split and rotate query
  let xqHalfL = xq[0..., 0..., 0..., 0 ..< halfD]
  let xqHalfR = xq[0..., 0..., 0..., halfD...]
  let xqRotated = MLX.concatenated([-xqHalfR, xqHalfL], axis: -1)

  // Split and rotate key
  let xkHalfL = xk[0..., 0..., 0..., 0 ..< halfD]
  let xkHalfR = xk[0..., 0..., 0..., halfD...]
  let xkRotated = MLX.concatenated([-xkHalfR, xkHalfL], axis: -1)

  // Apply rotation
  let xqOut = xq * cosExpanded + xqRotated * sinExpanded
  let xkOut = xk * cosExpanded + xkRotated * sinExpanded

  return (xqOut, xkOut)
}

// MARK: - Multi-Head Attention

/// Basic multi-head attention
class MultiHeadAttention: Module {
  let nHead: Int

  @ModuleInfo(key: "query") var query: Linear
  @ModuleInfo(key: "key") var key: Linear
  @ModuleInfo(key: "value") var value: Linear
  @ModuleInfo(key: "out") var out: Linear

  init(nState: Int, nHead: Int) {
    self.nHead = nHead
    _query.wrappedValue = Linear(nState, nState)
    _key.wrappedValue = Linear(nState, nState, bias: false)
    _value.wrappedValue = Linear(nState, nState)
    _out.wrappedValue = Linear(nState, nState)
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXArray? = nil,
  ) -> (MLXArray, MLXArray?) {
    let q = query(x)
    let k = key(x)
    let v = value(x)

    let (wv, qk) = qkvAttention(q: q, k: k, v: v, mask: mask)
    return (out(wv), qk)
  }

  func qkvAttention(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    mask: MLXArray? = nil,
  ) -> (MLXArray, MLXArray?) {
    let B = q.shape[0]
    let T = q.shape[1]
    let D = q.shape[2]
    let scale = pow(Float(D / nHead), -0.25)

    let qReshaped = q.reshaped([B, T, nHead, -1]).transposed(0, 2, 1, 3) * scale
    let kReshaped = k.reshaped([B, T, nHead, -1]).transposed(0, 2, 1, 3) * scale
    let vReshaped = v.reshaped([B, T, nHead, -1]).transposed(0, 2, 1, 3)

    let output = scaledDotProductAttention(
      queries: qReshaped,
      keys: kReshaped,
      values: vReshaped,
      scale: 1.0,
      mask: mask,
    )
    let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([B, T, D])

    return (outputReshaped, nil)
  }
}

// MARK: - FSQ Codebook

/// Finite Scalar Quantization Codebook
class FSQCodebook: Module {
  let level: Int

  @ModuleInfo(key: "project_down") var projectDown: Linear

  init(dim: Int, level: Int = 3) {
    self.level = level
    _projectDown.wrappedValue = Linear(dim, 8)
  }

  func preprocess(_ x: MLXArray) -> MLXArray {
    // Rearrange: ... d -> (...) d
    let lastDim = x.shape[x.ndim - 1]
    let totalElements = x.shape.dropLast().reduce(1, *)
    return x.reshaped([totalElements, lastDim])
  }

  func encode(_ x: MLXArray) -> MLXArray {
    let xShape = x.shape
    // Pre-process
    let xFlat = preprocess(x)
    // Quantize
    var h = projectDown(xFlat).asType(.float32)
    h = MLX.tanh(h)
    h = h * 0.9990000128746033
    h = MLX.round(h) + 1

    // Create powers for base conversion: [1, 3, 9, 27, 81, 243, 729, 2187]
    let powers = MLX.pow(
      MLXArray(Float(level)),
      MLXArray(0 ..< 8).asType(.float32),
    )
    let mu = MLX.sum(h * powers.expandedDimensions(axis: 0), axis: -1)
    let ind = mu.reshaped([xShape[0], xShape[1]]).asType(.int32)

    return ind
  }
}

// MARK: - FSQ Vector Quantization

/// Finite Scalar Quantization Vector Quantization
class FSQVectorQuantization: Module {
  let codebookSize: Int

  @ModuleInfo(key: "fsq_codebook") var fsqCodebook: FSQCodebook

  init(dim: Int, codebookSize: Int) {
    precondition(codebookSize == 6561, "FSQ codebook size must be 3^8 = 6561")
    self.codebookSize = codebookSize
    _fsqCodebook.wrappedValue = FSQCodebook(dim: dim, level: 3)
  }

  func encode(_ x: MLXArray) -> MLXArray {
    fsqCodebook.encode(x)
  }
}

// MARK: - FSMN Multi-Head Attention

/// Multi-head attention with FSMN (Feedforward Sequential Memory Network)
class FSMNMultiHeadAttention: Module {
  let nHead: Int
  let leftPadding: Int
  let rightPadding: Int

  @ModuleInfo(key: "query") var query: Linear
  @ModuleInfo(key: "key") var key: Linear
  @ModuleInfo(key: "value") var value: Linear
  @ModuleInfo(key: "out") var out: Linear
  @ModuleInfo(key: "fsmn_block") var fsmnBlock: Conv1d

  init(nState: Int, nHead: Int, kernelSize: Int = 31) {
    self.nHead = nHead
    leftPadding = (kernelSize - 1) / 2
    rightPadding = kernelSize - 1 - leftPadding

    _query.wrappedValue = Linear(nState, nState)
    _key.wrappedValue = Linear(nState, nState, bias: false)
    _value.wrappedValue = Linear(nState, nState)
    _out.wrappedValue = Linear(nState, nState)
    // FSMN uses depthwise convolution (groups=nState) without bias
    _fsmnBlock.wrappedValue = Conv1d(
      inputChannels: nState,
      outputChannels: nState,
      kernelSize: kernelSize,
      stride: 1,
      padding: 0,
      groups: nState,
      bias: false,
    )
  }

  func forwardFsmn(_ inputs: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let b = inputs.shape[0]
    let t = inputs.shape[1]
    let n = inputs.shape[2]
    let d = inputs.shape[3]

    var inputsReshaped = inputs.reshaped([b, t, n * d])

    if let mask, mask.shape[2] > 0 {
      inputsReshaped = inputsReshaped * mask
    }

    // Pad left and right
    let padLeft = MLXArray.zeros([b, leftPadding, inputsReshaped.shape[2]], dtype: inputsReshaped.dtype)
    let padRight = MLXArray.zeros([b, rightPadding, inputsReshaped.shape[2]], dtype: inputsReshaped.dtype)
    let xPadded = MLX.concatenated([padLeft, inputsReshaped, padRight], axis: 1)

    var x = fsmnBlock(xPadded)
    x = x + inputsReshaped

    if let mask {
      x = x * mask
    }

    return x
  }

  func qkvAttention(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    mask: MLXArray? = nil,
    maskPad: MLXArray? = nil,
    freqsCis: (MLXArray, MLXArray)? = nil,
  ) -> (MLXArray, MLXArray?, MLXArray) {
    let B = q.shape[0]
    let T = q.shape[1]
    let D = q.shape[2]
    let scale = pow(Float(D / nHead), -0.25)

    var qReshaped = q.reshaped([B, T, nHead, -1])
    var kReshaped = k.reshaped([B, T, nHead, -1])
    let vReshaped = v.reshaped([B, T, nHead, -1])

    if let (cos, sin) = freqsCis {
      let cosSlice = cos[0 ..< T]
      let sinSlice = sin[0 ..< T]
      (qReshaped, kReshaped) = applyRotaryEmb(
        xq: qReshaped,
        xk: kReshaped,
        cos: cosSlice,
        sin: sinSlice,
      )
    }

    let fsmMemory = forwardFsmn(vReshaped, mask: maskPad)

    let qTransposed = qReshaped.transposed(0, 2, 1, 3) * scale
    let kTransposed = kReshaped.transposed(0, 2, 1, 3) * scale
    let vTransposed = vReshaped.transposed(0, 2, 1, 3)

    let output = scaledDotProductAttention(
      queries: qTransposed,
      keys: kTransposed,
      values: vTransposed,
      scale: 1.0,
      mask: mask,
    )
    let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([B, T, D])

    return (outputReshaped, nil, fsmMemory)
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXArray? = nil,
    maskPad: MLXArray? = nil,
    freqsCis: (MLXArray, MLXArray)? = nil,
  ) -> (MLXArray, MLXArray?) {
    let q = query(x)
    let k = key(x)
    let v = value(x)

    let (wv, _, fsmMemory) = qkvAttention(
      q: q, k: k, v: v,
      mask: mask, maskPad: maskPad, freqsCis: freqsCis,
    )

    return (out(wv) + fsmMemory, nil)
  }
}

// MARK: - Residual Attention Block

/// Residual attention block with FSMN
class S3ResidualAttentionBlock: Module {
  @ModuleInfo(key: "attn") var attn: FSMNMultiHeadAttention
  @ModuleInfo(key: "attn_ln") var attnLn: LayerNorm
  @ModuleInfo(key: "mlp") var mlp: Sequential
  @ModuleInfo(key: "mlp_ln") var mlpLn: LayerNorm

  init(nState: Int, nHead: Int, kernelSize: Int = 31) {
    _attn.wrappedValue = FSMNMultiHeadAttention(
      nState: nState,
      nHead: nHead,
      kernelSize: kernelSize,
    )
    _attnLn.wrappedValue = LayerNorm(dimensions: nState, eps: 1e-6)

    let nMlp = nState * 4
    _mlp.wrappedValue = Sequential {
      Linear(nState, nMlp)
      GELU()
      Linear(nMlp, nState)
    }
    _mlpLn.wrappedValue = LayerNorm(dimensions: nState)
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXArray? = nil,
    maskPad: MLXArray? = nil,
    freqsCis: (MLXArray, MLXArray)? = nil,
  ) -> MLXArray {
    var result = x + attn(attnLn(x), mask: mask, maskPad: maskPad, freqsCis: freqsCis).0
    result = result + mlp(mlpLn(result))
    return result
  }
}

// MARK: - Audio Encoder V2

/// Audio encoder for S3TokenizerV2
class AudioEncoderV2: Module {
  let stride: Int
  /// Precomputed RoPE frequencies - underscore prefix excludes from parameter validation
  var _freqsCisCos: MLXArray
  var _freqsCisSin: MLXArray

  @ModuleInfo(key: "conv1") var conv1: Conv1d
  @ModuleInfo(key: "conv2") var conv2: Conv1d
  @ModuleInfo(key: "blocks") var blocks: [S3ResidualAttentionBlock]

  init(nMels: Int, nState: Int, nHead: Int, nLayer: Int, stride: Int) {
    self.stride = stride

    _conv1.wrappedValue = Conv1d(
      inputChannels: nMels,
      outputChannels: nState,
      kernelSize: 3,
      stride: stride,
      padding: 1,
    )
    _conv2.wrappedValue = Conv1d(
      inputChannels: nState,
      outputChannels: nState,
      kernelSize: 3,
      stride: 2,
      padding: 1,
    )

    let (cos, sin) = precomputeFreqsCis(dim: 64, end: 1024 * 2)
    _freqsCisCos = cos
    _freqsCisSin = sin

    _blocks.wrappedValue = (0 ..< nLayer).map { _ in
      S3ResidualAttentionBlock(nState: nState, nHead: nHead)
    }
  }

  func callAsFunction(_ x: MLXArray, xLen: MLXArray) -> (MLXArray, MLXArray) {
    // x: (batch_size, n_mels, T)
    // xLen: (batch_size,)

    var mask = makeNonPadMask(lengths: xLen)
    mask = mask.expandedDimensions(axis: 1) // (B, 1, T)

    var xTransposed = x.transposed(0, 2, 1) // (B, T, n_mels)
    var maskTransposed = mask.transposed(0, 2, 1) // (B, T, 1)

    xTransposed = conv1(xTransposed * maskTransposed)
    xTransposed = gelu(xTransposed)
    var xLenUpdated = (xLen + 2 - 1 * (3 - 1) - 1) / stride + 1

    mask = makeNonPadMask(lengths: xLenUpdated)
    maskTransposed = mask.expandedDimensions(axis: -1) // (B, T, 1)

    xTransposed = conv2(xTransposed * maskTransposed)
    xTransposed = gelu(xTransposed)
    // Break up complex expression for compiler
    let kernelTerm = 1 * (3 - 1)
    xLenUpdated = (xLenUpdated + 2 - kernelTerm - 1) / 2 + 1

    mask = makeNonPadMask(lengths: xLenUpdated)
    let maskPad = mask.expandedDimensions(axis: -1) // (B, T, 1)
    var maskBias = maskToBias(mask, dtype: xTransposed.dtype)
    maskBias = maskBias.expandedDimensions(axis: 1) // (B, 1, T)

    for block in blocks {
      xTransposed = block(xTransposed, mask: maskBias, maskPad: maskPad, freqsCis: (_freqsCisCos, _freqsCisSin))
    }

    return (xTransposed, xLenUpdated)
  }
}

// MARK: - S3TokenizerV2

/// S3 tokenizer v2 implementation
class S3TokenizerV2: Module {
  let config: S3TokenizerModelConfig

  @ModuleInfo(key: "encoder") var encoder: AudioEncoderV2
  @ModuleInfo(key: "quantizer") var quantizer: FSQVectorQuantization

  init(name: String = "speech_tokenizer_v2_25hz", config: S3TokenizerModelConfig = S3TokenizerModelConfig()) {
    var configUpdated = config
    if !name.contains("v1") {
      precondition(name.contains("v2") || name.isEmpty, "S3TokenizerV2 requires v2 in name or empty name")
      configUpdated.nCodebookSize = 6561 // 3^8
    }
    self.config = configUpdated

    _encoder.wrappedValue = AudioEncoderV2(
      nMels: configUpdated.nMels,
      nState: configUpdated.nAudioState,
      nHead: configUpdated.nAudioHead,
      nLayer: configUpdated.nAudioLayer,
      stride: 2,
    )
    _quantizer.wrappedValue = FSQVectorQuantization(
      dim: configUpdated.nAudioState,
      codebookSize: configUpdated.nCodebookSize,
    )
  }

  func callAsFunction(_ mel: MLXArray, melLen: MLXArray) -> (MLXArray, MLXArray) {
    quantize(mel: mel, melLen: melLen)
  }

  /// Quantize mel spectrogram to tokens
  func quantize(mel: MLXArray, melLen: MLXArray) -> (MLXArray, MLXArray) {
    // Check if any audio exceeds 30 seconds
    // At 16kHz with hop_length=160: 30s = 3000 frames
    let maxFrames = 3000
    let longAudioMask = melLen .> maxFrames

    if MLX.any(longAudioMask).item(Bool.self) {
      // Has long audio - need special processing
      return quantizeMixedBatch(
        mel: mel,
        melLen: melLen,
        longAudioMask: longAudioMask,
        maxFrames: maxFrames,
      )
    } else {
      // All short audio - use simple path
      let (hidden, codeLen) = encoder(mel, xLen: melLen)
      let code = quantizer.encode(hidden)
      return (code, codeLen)
    }
  }

  /// Handle mixed batch with both short and long audio
  private func quantizeMixedBatch(
    mel: MLXArray,
    melLen: MLXArray,
    longAudioMask: MLXArray,
    maxFrames _: Int,
  ) -> (MLXArray, MLXArray) {
    let batchSize = mel.shape[0]

    // Sliding window parameters
    let sampleRate = 16000
    let hopLength = 160
    let windowSize = 30 // seconds
    let overlap = 4 // seconds

    let framesPerWindow = windowSize * sampleRate / hopLength // 3000
    let framesPerOverlap = overlap * sampleRate / hopLength // 400
    let framesPerStride = framesPerWindow - framesPerOverlap // 2600

    // Collect all segments
    var allSegments: [MLXArray] = []
    var allSegmentsLen: [Int] = []
    var segmentInfo: [[String: Any]] = []

    for batchIdx in 0 ..< batchSize {
      let audioMel = mel[batchIdx]
      let audioMelLen = Int(melLen[batchIdx].item(Int.self))
      let isLongAudio = longAudioMask[batchIdx].item(Bool.self)

      if !isLongAudio {
        // Short audio: process as single segment
        var segment = audioMel[0..., 0 ..< audioMelLen]
        let segLen = audioMelLen

        if segLen < framesPerWindow {
          let padSize = framesPerWindow - segLen
          segment = MLX.padded(segment, widths: [IntOrPair(0), IntOrPair((0, padSize))])
        }

        allSegments.append(segment)
        allSegmentsLen.append(segLen)
        segmentInfo.append([
          "batch_idx": batchIdx,
          "is_long_audio": false,
          "segment_idx": 0,
          "total_segments": 1,
        ])
      } else {
        // Long audio: split into segments
        var start = 0
        var segmentIdx = 0
        var segmentCount = 0

        while start < audioMelLen {
          let end = min(start + framesPerWindow, audioMelLen)
          var segment = audioMel[0..., start ..< end]
          let segLen = segment.shape[1]

          if segLen < framesPerWindow {
            let padSize = framesPerWindow - segLen
            segment = MLX.padded(segment, widths: [IntOrPair(0), IntOrPair((0, padSize))])
          }

          allSegments.append(segment)
          allSegmentsLen.append(segLen)
          segmentInfo.append([
            "batch_idx": batchIdx,
            "is_long_audio": true,
            "segment_idx": segmentIdx,
            "total_segments": -1, // Will update later
          ])

          segmentIdx += 1
          segmentCount += 1
          start += framesPerStride
        }

        // Update total_segments
        for i in (segmentInfo.count - segmentCount) ..< segmentInfo.count {
          segmentInfo[i]["total_segments"] = segmentCount
        }
      }
    }

    if allSegments.isEmpty {
      return (
        MLXArray.zeros([batchSize, 0], dtype: .int32),
        MLXArray.zeros([batchSize], dtype: .int32),
      )
    }

    // Process all segments
    let unifiedBatchMel = MLX.stacked(allSegments)
    let unifiedBatchLens = MLXArray(allSegmentsLen.map { Int32($0) })

    let (hidden, codeLen) = encoder(unifiedBatchMel, xLen: unifiedBatchLens)
    let codes = quantizer.encode(hidden)

    // Reorganize results
    var results: [Int: Any] = [:]

    for (segIdx, info) in segmentInfo.enumerated() {
      let batchIdx = info["batch_idx"] as! Int
      let isLongAudio = info["is_long_audio"] as! Bool

      let segCodeLen = Int(codeLen[segIdx].item(Int.self))
      let segmentCode = codes[segIdx, 0 ..< segCodeLen]
      let segmentCodeList = (0 ..< segCodeLen).map { Int(segmentCode[$0].item(Int32.self)) }

      if !isLongAudio {
        results[batchIdx] = (MLXArray(segmentCodeList.map { Int32($0) }), segmentCodeList.count)
      } else {
        if results[batchIdx] == nil {
          results[batchIdx] = [[Int]]()
        }
        var existing = results[batchIdx] as! [[Int]]
        existing.append(segmentCodeList)
        results[batchIdx] = existing
      }
    }

    // Merge long audio segments
    for batchIdx in 0 ..< batchSize {
      if longAudioMask[batchIdx].item(Bool.self) {
        let audioCodes = results[batchIdx] as! [[Int]]
        let tokenRate = 25 // V2 uses 25Hz
        let mergedCodes = mergeTokenizedSegments(audioCodes, overlap: overlap, tokenRate: tokenRate)
        results[batchIdx] = (MLXArray(mergedCodes.map { Int32($0) }), mergedCodes.count)
      }
    }

    // Build output
    var maxCodeLen = 0
    for batchIdx in 0 ..< batchSize {
      let (_, len) = results[batchIdx] as! (MLXArray, Int)
      maxCodeLen = max(maxCodeLen, len)
    }

    var outputList: [MLXArray] = []
    var lenList: [Int32] = []

    for batchIdx in 0 ..< batchSize {
      var (codeTensor, codeLenVal) = results[batchIdx] as! (MLXArray, Int)
      if codeTensor.shape[0] < maxCodeLen {
        codeTensor = MLX.padded(codeTensor, widths: [IntOrPair((0, maxCodeLen - codeTensor.shape[0]))])
      }
      outputList.append(codeTensor)
      lenList.append(Int32(codeLenVal))
    }

    let outputCodes = MLX.stacked(outputList)
    let outputCodesLen = MLXArray(lenList)

    return (outputCodes, outputCodesLen)
  }

  /// Simple quantization without long audio handling
  func quantizeSimple(mel: MLXArray, melLen: MLXArray) -> (MLXArray, MLXArray) {
    let (hidden, codeLen) = encoder(mel, xLen: melLen)
    let code = quantizer.encode(hidden)
    return (code, codeLen)
  }
}

// MARK: - S3TokenizerV3

/// S3 tokenizer V3 implementation
/// V3 is architecturally identical to V2, but with 12 transformer blocks instead of 6.
/// Used by CosyVoice3 for speech tokenization.
class S3TokenizerV3: Module {
  let config: S3TokenizerV3ModelConfig

  @ModuleInfo(key: "encoder") var encoder: AudioEncoderV2
  @ModuleInfo(key: "quantizer") var quantizer: FSQVectorQuantization

  init(name _: String = "speech_tokenizer_v3", config: S3TokenizerV3ModelConfig = S3TokenizerV3ModelConfig()) {
    self.config = config

    _encoder.wrappedValue = AudioEncoderV2(
      nMels: config.nMels,
      nState: config.nAudioState,
      nHead: config.nAudioHead,
      nLayer: config.nAudioLayer, // 12 layers for V3
      stride: 2
    )
    _quantizer.wrappedValue = FSQVectorQuantization(
      dim: config.nAudioState,
      codebookSize: config.nCodebookSize
    )
  }

  func callAsFunction(_ mel: MLXArray, melLen: MLXArray) -> (MLXArray, MLXArray) {
    quantize(mel: mel, melLen: melLen)
  }

  /// Quantize mel spectrogram to tokens
  func quantize(mel: MLXArray, melLen: MLXArray) -> (MLXArray, MLXArray) {
    // Check if any audio exceeds 30 seconds
    // At 16kHz with hop_length=160: 30s = 3000 frames
    let maxFrames = 3000
    let longAudioMask = melLen .> maxFrames

    if MLX.any(longAudioMask).item(Bool.self) {
      // Has long audio - need special processing
      return quantizeMixedBatch(
        mel: mel,
        melLen: melLen,
        longAudioMask: longAudioMask,
        maxFrames: maxFrames
      )
    } else {
      // All short audio - use simple path
      let (hidden, codeLen) = encoder(mel, xLen: melLen)
      let code = quantizer.encode(hidden)
      return (code, codeLen)
    }
  }

  /// Handle mixed batch with both short and long audio
  private func quantizeMixedBatch(
    mel: MLXArray,
    melLen: MLXArray,
    longAudioMask: MLXArray,
    maxFrames _: Int
  ) -> (MLXArray, MLXArray) {
    let batchSize = mel.shape[0]

    // Sliding window parameters
    let sampleRate = 16000
    let hopLength = 160
    let windowSize = 30 // seconds
    let overlap = 4 // seconds

    let framesPerWindow = windowSize * sampleRate / hopLength // 3000
    let framesPerOverlap = overlap * sampleRate / hopLength // 400
    let framesPerStride = framesPerWindow - framesPerOverlap // 2600

    // Collect all segments
    var allSegments: [MLXArray] = []
    var allSegmentsLen: [Int] = []
    var segmentInfo: [[String: Any]] = []

    for batchIdx in 0 ..< batchSize {
      let audioMel = mel[batchIdx]
      let audioMelLen = Int(melLen[batchIdx].item(Int.self))
      let isLongAudio = longAudioMask[batchIdx].item(Bool.self)

      if !isLongAudio {
        // Short audio: process as single segment
        var segment = audioMel[0..., 0 ..< audioMelLen]
        let segLen = audioMelLen

        if segLen < framesPerWindow {
          let padSize = framesPerWindow - segLen
          segment = MLX.padded(segment, widths: [IntOrPair(0), IntOrPair((0, padSize))])
        }

        allSegments.append(segment)
        allSegmentsLen.append(segLen)
        segmentInfo.append([
          "batch_idx": batchIdx,
          "is_long_audio": false,
          "segment_idx": 0,
          "total_segments": 1,
        ])
      } else {
        // Long audio: split into segments
        var start = 0
        var segmentIdx = 0
        var segmentCount = 0

        while start < audioMelLen {
          let end = min(start + framesPerWindow, audioMelLen)
          var segment = audioMel[0..., start ..< end]
          let segLen = segment.shape[1]

          if segLen < framesPerWindow {
            let padSize = framesPerWindow - segLen
            segment = MLX.padded(segment, widths: [IntOrPair(0), IntOrPair((0, padSize))])
          }

          allSegments.append(segment)
          allSegmentsLen.append(segLen)
          segmentInfo.append([
            "batch_idx": batchIdx,
            "is_long_audio": true,
            "segment_idx": segmentIdx,
            "total_segments": -1,
          ])

          segmentIdx += 1
          segmentCount += 1
          start += framesPerStride
        }

        // Update total_segments
        for i in (segmentInfo.count - segmentCount) ..< segmentInfo.count {
          segmentInfo[i]["total_segments"] = segmentCount
        }
      }
    }

    if allSegments.isEmpty {
      return (
        MLXArray.zeros([batchSize, 0]).asType(.int32),
        MLXArray.zeros([batchSize]).asType(.int32)
      )
    }

    // Process all segments
    let unifiedBatchMel = MLX.stacked(allSegments)
    let unifiedBatchLens = MLXArray(allSegmentsLen.map { Int32($0) })

    let (hidden, codeLen) = encoder(unifiedBatchMel, xLen: unifiedBatchLens)
    let codes = quantizer.encode(hidden)

    // Reorganize results
    var results: [Int: Any] = [:]

    for (segIdx, info) in segmentInfo.enumerated() {
      let batchIdx = info["batch_idx"] as! Int
      let isLongAudio = info["is_long_audio"] as! Bool

      let segCodeLen = Int(codeLen[segIdx].item(Int.self))
      let segmentCode = codes[segIdx, 0 ..< segCodeLen]
      let segmentCodeList = (0 ..< segCodeLen).map { Int(segmentCode[$0].item(Int32.self)) }

      if !isLongAudio {
        results[batchIdx] = (MLXArray(segmentCodeList.map { Int32($0) }), segmentCodeList.count)
      } else {
        var existing = (results[batchIdx] as? [[Int]]) ?? []
        existing.append(segmentCodeList)
        results[batchIdx] = existing
      }
    }

    // Merge long audio segments
    for batchIdx in 0 ..< batchSize {
      if longAudioMask[batchIdx].item(Bool.self) {
        let audioCodes = results[batchIdx] as! [[Int]]
        let tokenRate = 25 // V3 uses 25Hz like V2
        let mergedCodes = mergeTokenizedSegments(audioCodes, overlap: overlap, tokenRate: tokenRate)
        results[batchIdx] = (MLXArray(mergedCodes.map { Int32($0) }), mergedCodes.count)
      }
    }

    // Build output
    var maxCodeLen = 0
    for batchIdx in 0 ..< batchSize {
      let (_, len) = results[batchIdx] as! (MLXArray, Int)
      maxCodeLen = max(maxCodeLen, len)
    }

    var outputList: [MLXArray] = []
    var lenList: [Int32] = []

    for batchIdx in 0 ..< batchSize {
      var (codeTensor, codeLenVal) = results[batchIdx] as! (MLXArray, Int)
      if codeTensor.shape[0] < maxCodeLen {
        codeTensor = MLX.padded(codeTensor, widths: [IntOrPair((0, maxCodeLen - codeTensor.shape[0]))])
      }
      outputList.append(codeTensor)
      lenList.append(Int32(codeLenVal))
    }

    let outputCodes = MLX.stacked(outputList)
    let outputCodesLen = MLXArray(lenList)

    return (outputCodes, outputCodesLen)
  }

  /// Simple quantization without long audio handling
  func quantizeSimple(mel: MLXArray, melLen: MLXArray) -> (MLXArray, MLXArray) {
    let (hidden, codeLen) = encoder(mel, xLen: melLen)
    let code = quantizer.encode(hidden)
    return (code, codeLen)
  }
}
