namespace py thrift_gen.exceptions

exception NotFoundException {
  1: string message
}

exception ValidationException {
  1: string message
}

exception InternalException {
  1: string message
}

exception ConflictException {
  1: string message
}

exception UnauthorizedException {
  1: string message
}

// For remote service errors
exception RemoteServiceException {
  1: string message
}