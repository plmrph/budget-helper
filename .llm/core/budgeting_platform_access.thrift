namespace py thrift_gen.budgetingplatformaccess

include "entities.thrift"
include "exceptions.thrift"



service BudgetingPlatformAccess {
  /**
   * Authenticate to the external platform (e.g., BudgetingPlatform, EmailPlatform)
   * If already authenticated, returns status. Otherwise, attempts authentication and persists result.
   * Fetches information from Configs singleton, and populates auth data back to Configs
   */
  bool authenticate()
    throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.RemoteServiceException remoteError
    )

  list<entities.Account> getAccounts(1: optional entities.BudgetingPlatformType platform)
    throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.NotFoundException notFound,
      4: exceptions.RemoteServiceException remoteError
    )

  list<entities.Category> getCategories(1: optional entities.BudgetingPlatformType platform)
    throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.NotFoundException notFound,
      4: exceptions.RemoteServiceException remoteError
    )

  list<entities.Payee> getPayees(1: optional entities.BudgetingPlatformType platform)
    throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.NotFoundException notFound,
      4: exceptions.RemoteServiceException remoteError
    )

  list<entities.Budget> getBudgets(1: optional entities.BudgetingPlatformType platform)
    throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.NotFoundException notFound,
      4: exceptions.RemoteServiceException remoteError
    )

  list<entities.Transaction> getTransactions(
    1: optional entities.BudgetingPlatformType platform,
    2: optional bool isPending,
    3: optional string startDate,
    4: optional string endDate
  ) throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.NotFoundException notFound,
      4: exceptions.RemoteServiceException remoteError
  )

  bool updateTransactions(
    1: list<entities.Transaction> transactions,
    2: optional entities.BudgetingPlatformType platform
  ) throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.ConflictException conflict,
      4: exceptions.RemoteServiceException remoteError
  )

}