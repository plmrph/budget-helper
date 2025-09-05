namespace py thrift_gen.databasestoreaccess

include "entities.thrift"
include "exceptions.thrift"

enum FilterOperator {
  EQ = 1,
  NEQ = 2,
  GT = 3,
  GTE = 4,
  LT = 5,
  LTE = 6,
  LIKE = 7,
  IN = 8
  // Extendable
}

enum SortDirection {
  ASC = 1,
  DESC = 2
}

typedef string Timestamp

union FilterValue {
  1: string stringValue,
  2: i64 intValue,
  3: double doubleValue,
  4: bool boolValue,
  5: Timestamp timestampValue
}

struct Filter {
  1: string fieldName,
  2: FilterOperator operator,
  3: FilterValue value
}

struct Sort {
  1: string fieldName,
  2: SortDirection direction
}

struct Query {
  1: entities.EntityType entityType, 
  2: optional list<Filter> filters,
  3: optional list<Sort> sort,
  4: optional i32 limit,
  5: optional i32 offset
}

struct QueryResult {
  1: list<entities.Entity> entities,
  2: optional i32 totalCount,
  3: optional i32 pageNumber,
  4: optional i32 pageSize
}

service DatabaseStoreAccess {
  list<entities.Entity> upsertEntities(
    1: list<entities.Entity> entities
  ) throws (
    1: exceptions.ValidationException validationError,
    2: exceptions.InternalException internalError,
    3: exceptions.UnauthorizedException unauthorized
  )

  list<string> deleteEntities(
    1: entities.EntityType entityType,
    2: list<string> entityIds
  ) throws (
    1: exceptions.ValidationException validationError,
    2: exceptions.InternalException internalError,
    3: exceptions.UnauthorizedException unauthorized
  )

  list<entities.Entity> getEntitiesById(
    1: entities.EntityType entityType,
    2: list<string> entityIds
  ) throws (
    1: exceptions.NotFoundException notFound,
    2: exceptions.InternalException internalError,
    3: exceptions.UnauthorizedException unauthorized
  )

  QueryResult getEntities(
    1: Query query
  ) throws (
    1: exceptions.ValidationException validationError,
    2: exceptions.InternalException internalError,
    3: exceptions.UnauthorizedException unauthorized
  )

  list<entities.Entity> getEntitySummaries(
    1: entities.EntityType entityType,
    2: optional list<Filter> filters
  ) throws (
    1: exceptions.ValidationException validationError,
    2: exceptions.InternalException internalError,
    3: exceptions.UnauthorizedException unauthorized
  )

  list<string> getEntityIds(
    1: entities.EntityType entityType,
    2: optional list<Filter> filters
  ) throws (
    1: exceptions.ValidationException validationError,
    2: exceptions.InternalException internalError,
    3: exceptions.UnauthorizedException unauthorized
  )
}