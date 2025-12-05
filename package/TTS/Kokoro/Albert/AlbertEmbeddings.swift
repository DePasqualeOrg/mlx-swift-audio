//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertEmbeddings: Module {
  @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
  @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding
  @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding
  @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
  let dropout: Dropout

  init(config: AlbertConfig) {
    dropout = Dropout(p: config.hiddenDropoutProb)

    _wordEmbeddings.wrappedValue = Embedding(
      embeddingCount: config.vocabSize,
      dimensions: config.embeddingSize,
    )
    _positionEmbeddings.wrappedValue = Embedding(
      embeddingCount: config.maxPositionEmbeddings,
      dimensions: config.embeddingSize,
    )
    _tokenTypeEmbeddings.wrappedValue = Embedding(
      embeddingCount: config.typeVocabSize,
      dimensions: config.embeddingSize,
    )
    _layerNorm.wrappedValue = LayerNorm(
      dimensions: config.embeddingSize,
      eps: config.layerNormEps,
    )
  }

  func callAsFunction(
    _ inputIds: MLXArray,
    tokenTypeIds: MLXArray? = nil,
    positionIds: MLXArray? = nil,
  ) -> MLXArray {
    let seqLength = inputIds.shape[1]

    let positionIdsUsed: MLXArray = if let positionIds {
      positionIds
    } else {
      MLX.expandedDimensions(MLXArray(0 ..< seqLength), axes: [0])
    }

    let tokenTypeIdsUsed: MLXArray = if let tokenTypeIds {
      tokenTypeIds
    } else {
      MLXArray.zeros(like: inputIds)
    }

    let wordsEmbeddings = wordEmbeddings(inputIds)
    let positionEmbeddingsResult = positionEmbeddings(positionIdsUsed)
    let tokenTypeEmbeddingsResult = tokenTypeEmbeddings(tokenTypeIdsUsed)
    var embeddings = wordsEmbeddings + positionEmbeddingsResult + tokenTypeEmbeddingsResult
    embeddings = layerNorm(embeddings)
    embeddings = dropout(embeddings)
    return embeddings
  }
}
