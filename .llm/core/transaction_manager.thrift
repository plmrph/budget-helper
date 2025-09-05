namespace py thrift_gen.transactionmanager

include "entities.thrift"
include "exceptions.thrift"

struct TransactionEdit {
  1: string transactionId,
  2: optional string categoryId,
  3: optional bool approved,
  4: optional string memo,
  5: optional list<entities.Metadata> metadata
}


struct SyncResultItem {
  1: string transactionId,
  2: entities.SyncStatus status,
  3: optional string errorMessage
}

struct SyncResult {
  1: list<SyncResultItem> results,
  2: entities.SyncStatus batchStatus
}

/**
  * Get information about budgets and associated entities.
  *
  * If no parameters are provided, returns info for the default budget.
  * If budgetIds are provided, returns info for those budgets.
  * If entityTypes is provided, also returns data for specified entity types (limited to Account, Payee, Category).
  */
struct BudgetsInfoResult {
  1: list<entities.Budget> budgets,
  2: optional list<entities.Account> accounts,
  3: optional list<entities.Payee> payees,
  4: optional list<entities.Category> categories
}

service TransactionManager {
  BudgetsInfoResult getBudgetsInfo(
    1: optional list<string> budgetIds,
    2: optional list<entities.EntityType> entityTypes
  ) throws (
    1: exceptions.NotFoundException notFound,
    2: exceptions.ValidationException validationError,
    3: exceptions.InternalException internalError,
    4: exceptions.UnauthorizedException unauthorized
  )

  entities.Transaction getTransaction(1: string transactionId)
    throws (
      1: exceptions.NotFoundException notFound,
      2: exceptions.InternalException internalError,
      3: exceptions.UnauthorizedException unauthorized
    )

  list<entities.Transaction> getTransactions(1: list<string> transactionIds)
    throws (
      1: exceptions.NotFoundException notFound,
      2: exceptions.InternalException internalError,
      3: exceptions.UnauthorizedException unauthorized
    )

  list<entities.Transaction> updateTransactions(1: list<TransactionEdit> transactionEdits)
    throws (
      1: exceptions.NotFoundException notFound,
      2: exceptions.ValidationException validationError,
      3: exceptions.ConflictException conflict,
      4: exceptions.InternalException internalError,
      5: exceptions.UnauthorizedException unauthorized
    )

  entities.Transaction attachTransactionMetadata(1: string transactionId, 2: entities.Metadata metadata)
    throws (
      1: exceptions.NotFoundException notFound,
      2: exceptions.ValidationException validationError,
      3: exceptions.InternalException internalError,
      4: exceptions.UnauthorizedException unauthorized
    )

  entities.Transaction detachTransactionMetadata(1: string transactionId, 2: string metadataId)
    throws (
      1: exceptions.NotFoundException notFound,
      2: exceptions.ValidationException validationError,
      3: exceptions.InternalException internalError,
      4: exceptions.UnauthorizedException unauthorized
    )

  list<entities.Metadata> findTransactionMetadata(1: string transactionId)
    throws (
      1: exceptions.NotFoundException notFound,
      2: exceptions.ValidationException validationError,
      3: exceptions.InternalException internalError,
      4: exceptions.UnauthorizedException unauthorized
    )

  list<entities.Transaction> getPendingTransactions()
    throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized
    )

  list<entities.Transaction> getAllTransactions()
    throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized
    )

  SyncResult syncTransactionsIn(
    1: optional entities.BudgetingPlatformType budgetPlatform
  ) throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.RemoteServiceException remoteError
    )

  SyncResult syncTransactionsOut(
    1: optional entities.BudgetingPlatformType budgetPlatform
  ) throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.RemoteServiceException remoteError
    )
}