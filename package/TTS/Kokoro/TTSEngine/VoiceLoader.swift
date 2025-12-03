//
//  Kokoro-tts-lib
//
import Foundation
import Hub
import MLX

// Utility class for loading voices from Hugging Face Hub.
// Voice files are downloaded as safetensors and cached on disk automatically.
class VoiceLoader {
  private init() {}

  // Hugging Face repo configuration
  static let defaultRepoId = "mlx-community/Kokoro-82M-bf16"

  static var availableVoices: [KokoroTTS.Voice] {
    KokoroTTS.Voice.allCases
  }

  /// Load a voice from Hugging Face Hub (safetensors).
  /// Files are cached locally by Hub.snapshot() to avoid re-downloading.
  static func loadVoice(
    _ voice: KokoroTTS.Voice,
    repoId: String = defaultRepoId,
    progressHandler: @escaping (Progress) -> Void = { _ in },
  ) async throws -> MLXArray {
    let voiceId = voice.identifier
    let filename = "voices/\(voiceId).safetensors"

    let modelDirectoryURL = try await Hub.snapshot(
      from: repoId,
      matching: [filename],
      progressHandler: progressHandler,
    )

    let voiceFileURL = modelDirectoryURL.appending(path: filename)
    return try loadVoiceFromFile(voiceFileURL)
  }

  /// Load voice array from a local safetensors file
  private static func loadVoiceFromFile(_ url: URL) throws -> MLXArray {
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw VoiceLoaderError.voiceFileNotFound(url.lastPathComponent)
    }
    let weights = try MLX.loadArrays(url: url)
    guard let voiceArray = weights["voice"] else {
      throw VoiceLoaderError.invalidVoiceFile("Missing 'voice' key in safetensors file")
    }
    return voiceArray
  }

  enum VoiceLoaderError: LocalizedError {
    case voiceFileNotFound(String)
    case invalidVoiceFile(String)

    var errorDescription: String? {
      switch self {
        case let .voiceFileNotFound(filename):
          "Voice file not found: \(filename). Check your internet connection and try again."
        case let .invalidVoiceFile(message):
          "Invalid voice file: \(message)"
      }
    }
  }
}

// Extension to add utility methods to KokoroTTS.Voice
extension KokoroTTS.Voice {
  /// The voice identifier used for file names (e.g., "af_heart")
  var identifier: String {
    switch self {
      case .afAlloy: "af_alloy"
      case .afAoede: "af_aoede"
      case .afBella: "af_bella"
      case .afHeart: "af_heart"
      case .afJessica: "af_jessica"
      case .afKore: "af_kore"
      case .afNicole: "af_nicole"
      case .afNova: "af_nova"
      case .afRiver: "af_river"
      case .afSarah: "af_sarah"
      case .afSky: "af_sky"
      case .amAdam: "am_adam"
      case .amEcho: "am_echo"
      case .amEric: "am_eric"
      case .amFenrir: "am_fenrir"
      case .amLiam: "am_liam"
      case .amMichael: "am_michael"
      case .amOnyx: "am_onyx"
      case .amPuck: "am_puck"
      case .amSanta: "am_santa"
      case .bfAlice: "bf_alice"
      case .bfEmma: "bf_emma"
      case .bfIsabella: "bf_isabella"
      case .bfLily: "bf_lily"
      case .bmDaniel: "bm_daniel"
      case .bmFable: "bm_fable"
      case .bmGeorge: "bm_george"
      case .bmLewis: "bm_lewis"
      case .efDora: "ef_dora"
      case .emAlex: "em_alex"
      case .ffSiwis: "ff_siwis"
      case .hfAlpha: "hf_alpha"
      case .hfBeta: "hf_beta"
      case .hfOmega: "hm_omega"
      case .hmPsi: "hm_psi"
      case .ifSara: "if_sara"
      case .imNicola: "im_nicola"
      case .jfAlpha: "jf_alpha"
      case .jfGongitsune: "jf_gongitsune"
      case .jfNezumi: "jf_nezumi"
      case .jfTebukuro: "jf_tebukuro"
      case .jmKumo: "jm_kumo"
      case .pfDora: "pf_dora"
      case .pmSanta: "pm_santa"
      case .zfXiaobei: "zf_xiaobei"
      case .zfXiaoni: "zf_xiaoni"
      case .zfXiaoxiao: "zf_xiaoxiao"
      case .zfXiaoyi: "zf_xiaoyi"
      case .zmYunjian: "zm_yunjian"
      case .zmYunxi: "zm_yunxi"
      case .zmYunxia: "zm_yunxia"
      case .zmYunyang: "zm_yunyang"
    }
  }

  static func fromIdentifier(_ identifier: String) -> KokoroTTS.Voice? {
    KokoroTTS.Voice.allCases.first { $0.identifier == identifier }
  }
}
