namespace py thrift_gen.predictionmanager

include "entities.thrift"
include "ml_engine.thrift"
include "exceptions.thrift"

struct PredictionRequest {
  1: entities.Transactions transactions,
  2: optional entities.ModelCard modelCard
}

struct PredictionResult {
  1: ml_engine.CategoricalPredictionBatchResult results,
  2: optional string errorMessage
}

struct TrainingDataPreparationRequest {
  1: string budgetId,
  2: i32 monthsBack = 6,
  3: double testSplitRatio = 0.2,
  4: i32 minSamplesPerCategory = 1
}

service PredictionManager {
  list<entities.ModelCard> getModels(1: optional entities.ModelType modelType)
    throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.NotFoundException notFound
    )

  ml_engine.DatasetPreparationResult prepareTrainingData(1: TrainingDataPreparationRequest request)
    throws (
      1: exceptions.ValidationException validationError,
      2: exceptions.InternalException internalError,
      3: exceptions.UnauthorizedException unauthorized,
      4: exceptions.NotFoundException notFound
    )

  ml_engine.ModelTrainingResult trainModel(1: ml_engine.ModelTrainingRequest params)
    throws (
      1: exceptions.ValidationException validationError,
      2: exceptions.InternalException internalError,
      3: exceptions.UnauthorizedException unauthorized
    )

  PredictionResult getPredictions(1: PredictionRequest request)
    throws (
      1: exceptions.ValidationException validationError,
      2: exceptions.InternalException internalError,
      3: exceptions.UnauthorizedException unauthorized,
      4: exceptions.RemoteServiceException remoteError
    )

  bool deleteModel(1: entities.ModelCard modelCard)
    throws (
      1: exceptions.NotFoundException notFound,
      2: exceptions.ConflictException conflict,
      3: exceptions.InternalException internalError,
      4: exceptions.UnauthorizedException unauthorized
    )

  bool cancelTraining(1: entities.ModelCard modelCard)
    throws (
      1: exceptions.NotFoundException notFound,
      2: exceptions.ConflictException conflict,
      3: exceptions.InternalException internalError,
      4: exceptions.UnauthorizedException unauthorized
    )
}