namespace py thrift_gen.configs

include "entities.thrift"
include "exceptions.thrift"

service Configs {
  list<entities.ConfigItem> updateConfigs(1: list<entities.ConfigItem> configs)
    throws (
      1: exceptions.ValidationException validationError,
      2: exceptions.NotFoundException notFound,
      3: exceptions.InternalException internalError,
      4: exceptions.UnauthorizedException unauthorized
    )

  list<entities.ConfigItem> getConfigs(
    1: optional entities.ConfigType type,
    2: optional string key
  ) throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized
  )

  bool resetConfigs(1: optional entities.ConfigType type)
    throws (
      1: exceptions.InternalException internalError,
      2: exceptions.UnauthorizedException unauthorized
    )

}