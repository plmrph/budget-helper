namespace py thrift_gen.metadatasourceaccess

include "entities.thrift"
include "exceptions.thrift"

enum MetadataSourceType {
  Email = 1
  // Extendable
}

struct MetadataSourceQuery {
  1: MetadataSourceType sourceType,
  2: optional string searchPhrase,
  3: optional i64 startTime,
  4: optional i64 endTime,
  5: optional i32 limit,
  6: optional i32 offset
}

service MetadataSourceAccess {
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
    
  list<entities.Metadata> getMetadata(
    1: list<MetadataSourceQuery> queries
  ) throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized,
      3: exceptions.NotFoundException notFound,
      4: exceptions.RemoteServiceException remoteError
  )
}