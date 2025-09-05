namespace py thrift_gen.mlengine

include "entities.thrift"
include "exceptions.thrift"

struct ModelTrainingRequest {
  1: entities.ModelCard modelCard,
  2: string trainingDataLocation,
  3: optional map<string, string> parameters
}

struct ModelTrainingResult {
  1: entities.ModelCard modelCard,
  2: entities.TrainingStatus status,
  3: optional string errorMessage
}

typedef list<entities.PrimitiveValue> PrimitiveRow
typedef map<i32, PrimitiveRow> PrimitiveBatch

union PredictionInput {
  1: PrimitiveBatch primitiveBatchInput,
  // Extendable: Add other batch types, e.g., imageBatchInput, textBatchInput
}

struct CategoricalPredictionResult {
  1: string predictedCategory,
  2: double confidence
}

typedef list<CategoricalPredictionResult> CategoricalPredictionResultRow
typedef map<i32, CategoricalPredictionResultRow> CategoricalPredictionBatchResult

union PredictionBatchResult {
  1: CategoricalPredictionBatchResult categoricalPredictionResults,
  // Extendable: regressionResults, etc.
}

struct ModelPredictionBatchRequest {
  1: entities.ModelCard modelCard,
  2: PredictionInput input
}

struct ModelPredictionResult {
  1: entities.ModelCard modelCard,
  2: PredictionBatchResult result,
  3: optional string errorMessage
}

struct DatasetPreparationRequest {
  1: entities.Transactions transactions,
  2: double testSplitRatio = 0.2,
  3: i32 minSamplesPerCategory = 1,
  4: optional string datasetName
}

struct DatasetInfo {
  1: string datasetId,
  2: string datasetName,
  3: string trainingDataLocation,
  4: string testDataLocation,
  5: i32 trainingSamples,
  6: i32 testSamples,
  7: i32 categories,
  8: map<string, i32> categoryBreakdown,
  9: optional string dateFrom,
  10: optional string dateTo
}

struct DatasetPreparationResult {
  1: DatasetInfo datasetInfo,
  2: optional string errorMessage
}

service MLEngine {
  list<entities.ModelCard> getModels(1: optional entities.ModelType modelType)
    throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.NotFoundException notFound
    )

  bool deleteModels(1: list<entities.ModelCard> modelCards)
    throws (
      1: exceptions.NotFoundException notFound,
      2: exceptions.ConflictException conflict,
      3: exceptions.InternalException internalError,
      4: exceptions.UnauthorizedException unauthorized
    )

  DatasetPreparationResult prepareDatasets(1: DatasetPreparationRequest request)
    throws (
      1: exceptions.ValidationException validationError,
      2: exceptions.InternalException internalError,
      3: exceptions.UnauthorizedException unauthorized
    )

  list<ModelTrainingResult> trainModels(1: list<ModelTrainingRequest> trainingRequests)
    throws (
      1: exceptions.ValidationException validationError,
      2: exceptions.InternalException internalError,
      3: exceptions.UnauthorizedException unauthorized
    )

  list<ModelPredictionResult> getPredictions(1: list<ModelPredictionBatchRequest> predictionRequests)
    throws (
      1: exceptions.ValidationException validationError,
      2: exceptions.InternalException internalError,
      3: exceptions.UnauthorizedException unauthorized,
      4: exceptions.RemoteServiceException remoteError
    )

  list<DatasetInfo> getDatasets(1: optional list<string> datasetIds, 2: optional string budgetId)
    throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.NotFoundException notFound
    )

  list<bool> deleteDatasets(1: list<string> datasetIds)
    throws (
      1: exceptions.NotFoundException notFound,
      2: exceptions.InternalException internalError,
      3: exceptions.UnauthorizedException unauthorized
    )

  bool cancelTraining(1: entities.ModelCard modelCard)
    throws (
      1: exceptions.NotFoundException notFound,
      2: exceptions.ConflictException conflict,
      3: exceptions.InternalException internalError,
      4: exceptions.UnauthorizedException unauthorized
    )
}