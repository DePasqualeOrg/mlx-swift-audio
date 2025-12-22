// Copyright © Microsoft (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/microsoft/VibeVoice
// License: licenses/vibevoice.txt

import Foundation
import MLX
import MLXNN

// MARK: - Voice Cache

/// Pre-computed voice conditioning for VibeVoice generation
public struct VibeVoiceVoiceCache {
  let lmHidden: MLXArray
  let ttsHidden: MLXArray
  let negTtsHidden: MLXArray
  let lmCache: [(MLXArray, MLXArray)]
  let ttsCache: [(MLXArray, MLXArray)]
  let negTtsCache: [(MLXArray, MLXArray)]
  let negLmCache: [(MLXArray, MLXArray)]?
}

// MARK: - VibeVoice Model

/// VibeVoice streaming TTS model
///
/// This model generates speech from text using a Qwen2-based language model
/// backbone with a diffusion-based prediction head.
///
/// Architecture:
/// - languageModel: Lower transformer layers for text encoding
/// - ttsLanguageModel: Upper transformer layers for TTS
/// - acousticTokenizer: VAE decoder for latents -> audio
/// - predictionHead: Diffusion model for speech latent prediction
/// - ttsEosClassifier: Binary classifier for end-of-speech detection
class VibeVoiceModel: Module {
  let config: VibeVoiceConfig

  @ModuleInfo(key: "language_model") var languageModel: VibeVoiceQwen2Model
  @ModuleInfo(key: "tts_language_model") var ttsLanguageModel: VibeVoiceQwen2Model
  @ModuleInfo(key: "tts_input_types") var ttsInputTypes: Embedding
  @ModuleInfo(key: "acoustic_tokenizer") var acousticTokenizer: AcousticTokenizer
  @ModuleInfo(key: "acoustic_connector") var acousticConnector: SpeechConnector
  @ModuleInfo(key: "prediction_head") var predictionHead: DiffusionHead
  @ModuleInfo(key: "tts_eos_classifier") var ttsEosClassifier: BinaryClassifier

  @ParameterInfo(key: "speech_scaling_factor") var speechScalingFactor: MLXArray
  @ParameterInfo(key: "speech_bias_factor") var speechBiasFactor: MLXArray

  let noiseScheduler: DPMSolverMultistepScheduler
  let ddpmInferenceSteps: Int

  // Voice cache state
  var voiceCache: VibeVoiceVoiceCache?

  init(config: VibeVoiceConfig) {
    self.config = config

    // Calculate layer split
    let decoderConfig = config.decoderConfig
    let ttsLayers = config.ttsBackboneNumHiddenLayers
    let lmLayers = decoderConfig.numHiddenLayers - ttsLayers

    // Create configs for split models
    let lmConfig = Qwen2DecoderConfig(
      modelType: decoderConfig.modelType,
      hiddenSize: decoderConfig.hiddenSize,
      intermediateSize: decoderConfig.intermediateSize,
      numAttentionHeads: decoderConfig.numAttentionHeads,
      numKeyValueHeads: decoderConfig.numKeyValueHeads,
      numHiddenLayers: lmLayers,
      rmsNormEps: decoderConfig.rmsNormEps,
      vocabSize: decoderConfig.vocabSize,
      maxPositionEmbeddings: decoderConfig.maxPositionEmbeddings,
      ropeTheta: decoderConfig.ropeTheta,
      headDim: decoderConfig.headDim
    )

    let ttsLmConfig = Qwen2DecoderConfig(
      modelType: decoderConfig.modelType,
      hiddenSize: decoderConfig.hiddenSize,
      intermediateSize: decoderConfig.intermediateSize,
      numAttentionHeads: decoderConfig.numAttentionHeads,
      numKeyValueHeads: decoderConfig.numKeyValueHeads,
      numHiddenLayers: ttsLayers,
      rmsNormEps: decoderConfig.rmsNormEps,
      vocabSize: decoderConfig.vocabSize,
      maxPositionEmbeddings: decoderConfig.maxPositionEmbeddings,
      ropeTheta: decoderConfig.ropeTheta,
      headDim: decoderConfig.headDim
    )

    // Language models
    // Base LM doesn't have final norm (it continues into ttsLanguageModel)
    _languageModel.wrappedValue = VibeVoiceQwen2Model(config: lmConfig, useNorm: false)
    _ttsLanguageModel.wrappedValue = VibeVoiceQwen2Model(config: ttsLmConfig, useNorm: true)

    // TTS input type embeddings (0=speech, 1=text)
    _ttsInputTypes.wrappedValue = Embedding(embeddingCount: 2, dimensions: decoderConfig.hiddenSize)

    // Acoustic tokenizer (VAE decoder)
    _acousticTokenizer.wrappedValue = AcousticTokenizer(config: config.acousticTokenizerConfig)

    // Speech connector
    _acousticConnector.wrappedValue = SpeechConnector(
      inputDim: config.acousticVaeDim,
      outputDim: decoderConfig.hiddenSize
    )

    // Diffusion head
    _predictionHead.wrappedValue = DiffusionHead(config: config.diffusionHeadConfig)

    // TTS EOS classifier
    _ttsEosClassifier.wrappedValue = BinaryClassifier(hiddenSize: decoderConfig.hiddenSize)

    // Noise scheduler
    noiseScheduler = DPMSolverMultistepScheduler(
      numTrainTimesteps: config.diffusionHeadConfig.ddpmNumSteps,
      betaSchedule: config.diffusionHeadConfig.ddpmBetaSchedule,
      predictionType: config.diffusionHeadConfig.predictionType
    )

    // Scaling factors (will be loaded from weights)
    _speechScalingFactor.wrappedValue = MLXArray(1.0)
    _speechBiasFactor.wrappedValue = MLXArray(0.0)

    // Inference settings
    ddpmInferenceSteps = config.diffusionHeadConfig.ddpmNumInferenceSteps
  }

  /// Audio sample rate
  var sampleRate: Int {
    config.sampleRate
  }

  /// Get the token embedding layer
  func getInputEmbeddings() -> Embedding? {
    languageModel.embedTokens
  }

  /// Load a VibeVoice voice-cache (.safetensors) for conditioning
  ///
  /// Expected keys (per layer):
  /// - lm_hidden
  /// - lm_key_{i}, lm_value_{i}
  /// - tts_lm_hidden
  /// - tts_lm_key_{i}, tts_lm_value_{i}
  /// - neg_tts_lm_hidden
  /// - neg_lm_key_{i}, neg_lm_value_{i} (optional)
  /// - neg_tts_lm_key_{i}, neg_tts_lm_value_{i}
  func loadVoice(voicePath: URL) throws {
    let tensors = try MLX.loadArrays(url: voicePath)

    let lmLayers = languageModel.config.numHiddenLayers
    let ttsLayers = ttsLanguageModel.config.numHiddenLayers

    func loadTensor(_ name: String) throws -> MLXArray {
      guard let tensor = tensors[name] else {
        throw VibeVoiceError.voiceCacheMissingKey(name)
      }
      return tensor
    }

    func loadKV(_ prefix: String, _ i: Int) throws -> (MLXArray, MLXArray) {
      var k = try loadTensor("\(prefix)_key_\(i)")
      var v = try loadTensor("\(prefix)_value_\(i)")
      // Swift caches are stored as (B, kv_heads, seq, head_dim)
      // Our attention cache expects (B, seq, kv_heads, head_dim)
      if k.ndim == 4 {
        k = k.transposed(0, 2, 1, 3)
      }
      if v.ndim == 4 {
        v = v.transposed(0, 2, 1, 3)
      }
      return (k, v)
    }

    // Load caches and hidden states
    let lmHidden = try loadTensor("lm_hidden")
    let ttsHidden = try loadTensor("tts_lm_hidden")
    let negTtsHidden = try loadTensor("neg_tts_lm_hidden")

    var lmCache: [(MLXArray, MLXArray)] = []
    for i in 0 ..< lmLayers {
      lmCache.append(try loadKV("lm", i))
    }

    var ttsCache: [(MLXArray, MLXArray)] = []
    for i in 0 ..< ttsLayers {
      ttsCache.append(try loadKV("tts_lm", i))
    }

    var negTtsCache: [(MLXArray, MLXArray)] = []
    for i in 0 ..< ttsLayers {
      negTtsCache.append(try loadKV("neg_tts_lm", i))
    }

    // Optional negative LM cache
    var negLmCache: [(MLXArray, MLXArray)]?
    let hasNegLm = (0 ..< lmLayers).allSatisfy { i in
      tensors["neg_lm_key_\(i)"] != nil && tensors["neg_lm_value_\(i)"] != nil
    }
    if hasNegLm {
      var cache: [(MLXArray, MLXArray)] = []
      for i in 0 ..< lmLayers {
        cache.append(try loadKV("neg_lm", i))
      }
      negLmCache = cache
    }

    voiceCache = VibeVoiceVoiceCache(
      lmHidden: lmHidden,
      ttsHidden: ttsHidden,
      negTtsHidden: negTtsHidden,
      lmCache: lmCache,
      ttsCache: ttsCache,
      negTtsCache: negTtsCache,
      negLmCache: negLmCache
    )
  }

  /// Sample speech latents using diffusion with classifier-free guidance
  /// - Parameters:
  ///   - condition: Positive conditioning, shape (B, hiddenSize)
  ///   - negCondition: Negative conditioning, shape (B, hiddenSize)
  ///   - cfgScale: Classifier-free guidance scale
  ///   - ddpmSteps: Number of diffusion steps
  /// - Returns: Sampled speech latents, shape (B, acousticVaeDim)
  func sampleSpeechTokens(
    condition: MLXArray,
    negCondition: MLXArray,
    cfgScale: Float = 3.0,
    ddpmSteps: Int? = nil
  ) -> MLXArray {
    // Use float32 for diffusion math to reduce artifacts
    let condFloat = condition.asType(.float32)
    let negCondFloat = negCondition.asType(.float32)

    // Reset scheduler for new generation
    noiseScheduler.reset()
    noiseScheduler.setTimesteps(ddpmSteps ?? ddpmInferenceSteps)

    // Concatenate conditions for batched prediction
    let conditionCombined = MLX.concatenated([condFloat, negCondFloat], axis: 0)

    // Initialize noise
    let batchSize = condition.shape[0]
    let latentDim = config.acousticVaeDim
    var speech = MLX.random.normal([batchSize, latentDim]).asType(.float32)

    var prevX0: MLXArray?

    // Get timesteps as list
    guard let timesteps = noiseScheduler.timesteps else {
      return speech
    }
    let timestepsList = timesteps.asArray(Int32.self).map { Int($0) }

    for tVal in timestepsList {
      // Create timestep array for both positive and negative
      let timestepArray = MLXArray([Float(tVal), Float(tVal)])

      // Duplicate speech for batched CFG prediction
      let combinedSpeech = MLX.concatenated([speech, speech], axis: 0)

      // Predict v/epsilon
      let eps = predictionHead(
        noisyImages: combinedSpeech,
        timesteps: timestepArray,
        condition: conditionCombined
      )

      // Apply CFG
      let condEps = eps[0 ..< batchSize]
      let uncondEps = eps[batchSize...]
      let guidedEps = uncondEps + cfgScale * (condEps - uncondEps)

      // Duplicate for scheduler (it expects same batch size as input)
      let fullEps = MLX.concatenated([guidedEps, guidedEps], axis: 0)
      let fullSpeech = MLX.concatenated([speech, speech], axis: 0)

      // Scheduler step with multi-order support
      let output = noiseScheduler.step(
        modelOutput: fullEps,
        timestep: tVal,
        sample: fullSpeech,
        prevX0: prevX0
      )

      // Extract just the first half (positive conditioning result)
      speech = output.prevSample[0 ..< batchSize]
      prevX0 = output.x0Pred?[0 ..< batchSize]
    }

    return speech
  }

  /// Generate speech from text
  /// - Parameters:
  ///   - inputIds: Token IDs, shape (B, L)
  ///   - maxTokens: Maximum number of speech tokens to generate
  ///   - cfgScale: Classifier-free guidance scale
  ///   - ddpmSteps: Override diffusion inference steps
  /// - Returns: Generated audio waveform
  func generate(
    inputIds: MLXArray,
    maxTokens: Int = 512,
    cfgScale: Float = 1.5,
    ddpmSteps: Int? = nil
  ) -> MLXArray {
    guard let _ = voiceCache else {
      fatalError("Voice cache not loaded. Call loadVoice first.")
    }

    let batchSize = 1
    let seqLen = inputIds.shape[1]

    // Use voice cache
    var lmCache = voiceCache!.lmCache
    var ttsCache = voiceCache!.ttsCache
    var ttsHidden: MLXArray? = voiceCache!.ttsHidden
    var negHidden: MLXArray? = voiceCache!.negTtsHidden
    var negCache = voiceCache!.negTtsCache

    var speechLatents: [MLXArray] = []
    var finished = false
    var step = 0
    var textPos = 0

    while !finished, step < maxTokens {
      if textPos < seqLen {
        let curTextIds = inputIds[0..., textPos ..< min(seqLen, textPos + VibeVoiceConstants.ttsTextWindowSize)]
        let curWindow = curTextIds.shape[1]
        textPos += curWindow

        let textEmbeds = languageModel.embedTokens!(curTextIds)
        let (lmOut, newLmCache) = languageModel(inputsEmbeds: textEmbeds, cache: lmCache)
        lmCache = newLmCache

        let textType = MLXArray.ones([batchSize, curWindow], type: Int32.self)
        let typeEmbed = ttsInputTypes(textType)
        let ttsIn = lmOut + typeEmbed
        let (ttsOut, newTtsCache) = ttsLanguageModel(inputsEmbeds: ttsIn, cache: ttsCache)
        ttsCache = newTtsCache

        if ttsHidden == nil {
          ttsHidden = ttsOut
        } else {
          ttsHidden = MLX.concatenated([ttsHidden!, ttsOut], axis: 1)
        }

        // Process negative conditioning
        let negEmbed = MLXArray.zeros([batchSize, curWindow, config.decoderConfig.hiddenSize])
        let negTypeEmbed = ttsInputTypes(MLXArray.ones([batchSize, curWindow], type: Int32.self))
        let negIn = negEmbed + negTypeEmbed
        let (negOut, newNegCache) = ttsLanguageModel(inputsEmbeds: negIn, cache: negCache)
        negCache = newNegCache

        if negHidden == nil {
          negHidden = negOut
        } else {
          negHidden = MLX.concatenated([negHidden!, negOut], axis: 1)
        }
      }

      guard let currentTtsHidden = ttsHidden, let currentNegHidden = negHidden else {
        break
      }

      for _ in 0 ..< VibeVoiceConstants.ttsSpeechWindowSize {
        let positiveCondition = currentTtsHidden[0..., -1, 0...]
        let negativeCondition = currentNegHidden[0..., -1, 0...]

        var speechLatent = sampleSpeechTokens(
          condition: positiveCondition,
          negCondition: negativeCondition,
          cfgScale: cfgScale,
          ddpmSteps: ddpmSteps
        )
        speechLatent = speechLatent.expandedDimensions(axis: 1)

        speechLatents.append(speechLatent)

        let acousticEmbed = acousticConnector(speechLatent)

        let typeEmbed = ttsInputTypes(MLXArray.zeros([batchSize, 1], type: Int32.self))
        let ttsInput = acousticEmbed + typeEmbed

        let (ttsOut, newTtsCache) = ttsLanguageModel(inputsEmbeds: ttsInput, cache: ttsCache)
        ttsCache = newTtsCache
        ttsHidden = MLX.concatenated([ttsHidden!, ttsOut], axis: 1)

        let negTypeEmbed = ttsInputTypes(MLXArray.zeros([batchSize, 1], type: Int32.self))
        let negInput = acousticEmbed + negTypeEmbed
        let (negOut, newNegCache) = ttsLanguageModel(inputsEmbeds: negInput, cache: negCache)
        negCache = newNegCache
        negHidden = MLX.concatenated([negHidden!, negOut], axis: 1)

        let eosLogits = MLX.sigmoid(ttsEosClassifier(ttsOut[0..., -1, 0...]))
        if eosLogits[0].item(Float.self) > 0.5 {
          finished = true
          break
        }

        step += 1
        if step >= maxTokens {
          finished = true
          break
        }
      }
    }

    if speechLatents.isEmpty {
      return MLXArray([Float]())
    }

    let speechLatentSeq = MLX.concatenated(speechLatents, axis: 1)
    let scaledLatents = speechLatentSeq / speechScalingFactor - speechBiasFactor
    let audio = acousticTokenizer.decode(scaledLatents)
    return audio.squeezed(axes: [0, 1])
  }
}

// MARK: - Errors

public enum VibeVoiceError: Error, LocalizedError {
  case voiceCacheMissingKey(String)
  case modelNotLoaded
  case tokenizerNotLoaded
  case invalidInput(String)

  public var errorDescription: String? {
    switch self {
      case let .voiceCacheMissingKey(key):
        "Voice cache missing key: \(key)"
      case .modelNotLoaded:
        "Model not loaded"
      case .tokenizerNotLoaded:
        "Tokenizer not loaded"
      case let .invalidInput(message):
        "Invalid input: \(message)"
    }
  }
}
