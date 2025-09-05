namespace py thrift_gen.filestoreaccess

include "entities.thrift"
include "exceptions.thrift"

enum FileFormat {
  JSON = 1,
  CSV = 2,
  XML = 3,
  BINARY = 4
  // Extendable
}

struct FileMetadata {
  1: string path,
  2: FileFormat format,
  3: optional i64 size,
  4: optional string checksum,
  5: optional i64 lastModified
}

struct FileQuery {
  1: string pathPattern,
  2: optional FileFormat format,
  3: optional i64 minSize,
  4: optional i64 maxSize,
  5: optional i64 modifiedAfter,
  6: optional i64 modifiedBefore
}

service FileStoreAccess {
  bool writeFile(
    1: string path,
    2: binary data,
    3: optional FileFormat format
  ) throws (
    1: exceptions.ValidationException validationError,
    2: exceptions.InternalException internalError,
    3: exceptions.UnauthorizedException unauthorized
  )

  binary readFile(
    1: string path
  ) throws (
    1: exceptions.NotFoundException notFound,
    2: exceptions.InternalException internalError,
    3: exceptions.UnauthorizedException unauthorized
  )

  bool deleteFile(
    1: string path
  ) throws (
    1: exceptions.NotFoundException notFound,
    2: exceptions.InternalException internalError,
    3: exceptions.UnauthorizedException unauthorized
  )

  list<FileMetadata> listFiles(
    1: FileQuery query
  ) throws (
    1: exceptions.ValidationException validationError,
    2: exceptions.InternalException internalError,
    3: exceptions.UnauthorizedException unauthorized
  )

  FileMetadata getFileMetadata(
    1: string path
  ) throws (
    1: exceptions.NotFoundException notFound,
    2: exceptions.InternalException internalError,
    3: exceptions.UnauthorizedException unauthorized
  )

  bool fileExists(
    1: string path
  ) throws (
    1: exceptions.InternalException internalError,
    2: exceptions.UnauthorizedException unauthorized
  )
}