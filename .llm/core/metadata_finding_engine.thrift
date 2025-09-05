namespace py thrift_gen.metadatafindingengine

include "entities.thrift"
include "exceptions.thrift"

struct MetadataFilter {
  1: optional list<entities.MetadataType> sourceTypes,
  2: optional i64 startTime,
  3: optional i64 endTime,
  4: optional string searchPhrase
}

struct MetadataCandidate {
  1: entities.Metadata metadata
}

service MetadataFindingEngine {
  list<MetadataCandidate> getMetadataCandidates(
    1: entities.Transactions transactions,
    2: optional MetadataFilter filter
  ) throws (
    1: exceptions.NotFoundException notFound,
    2: exceptions.ValidationException validationError,
    3: exceptions.InternalException internalError,
    4: exceptions.UnauthorizedException unauthorized,
    5: exceptions.RemoteServiceException remoteError
  )

}