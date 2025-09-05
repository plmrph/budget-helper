namespace py thrift_gen.entities


enum MetadataType {
  Email = 1,
  Prediction = 2
  // Extendable
}

enum EmailPlatformType {
  Gmail = 1
  // Extendable
}

enum BudgetingPlatformType {
  YNAB = 1
  // Extendable
}

// ML Model registry
enum ModelType {
  PXBlendSC = 1
  // Extendable
}

union PrimitiveValue {
  1: string stringValue,
  2: double doubleValue,
  3: i64 intValue,
  4: bool boolValue
  // Extendable
}

union ExternalSystem {
  1: BudgetingPlatformType budgetingPlatformType,
  2: EmailPlatformType emailPlatformType,
  3: ModelType modelType
  // Extendable
}

struct Metadata {
  1: string id,
  2: MetadataType type,
  3: MetadataValue value,
  4: ExternalSystem sourceSystem,
  5: optional string description
  6: optional map<string, PrimitiveValue> properties
}

union MetadataValue {
  1: string stringValue,
  2: PredictionResult predictionResult
  // Extendable
}

struct Transaction {
  1: string id,
  2: string date,
  3: i64 amount,
  4: bool approved,
  5: BudgetingPlatformType platformType,
  6: optional string payeeId,
  7: optional string categoryId,
  8: optional string accountId,
  9: optional string budgetId,
  10: optional string memo,
  11: optional list<Metadata> metadata
}

enum ConfigType {
  System = 1,
  Email = 2,
  AI = 3,
  Display = 4,
  ExternalSystem = 5
  // Extendable
}

// Sync state for a budgeting platform
struct SyncState {
  1: BudgetingPlatformType platformType,
  2: optional string lastSyncInTime,   // ISO8601 string
  3: optional string lastSyncOutTime   // ISO8601 string
}

union ConfigValue {
  1: string stringValue,
  2: i64 intValue,
  3: double doubleValue,
  4: bool boolValue,
  5: list<string> stringList,
  6: map<string, string> stringMap,
  7: SyncState syncState
  // Extendable
}

struct ConfigItem {
  1: string key,  
  2: ConfigType type,
  3: ConfigValue value,
  4: optional string description
}

enum SyncStatus {
  Success = 1,
  Pending = 2,
  Partial = 3,
  Fail = 4
  // Extendable
}

enum TrainingStatus {
  Scheduled = 1,
  Pending = 2,
  Success = 3,
  Fail = 4
  // Extendable
}

struct PredictionResult {
  1: string predictedCategory,
  2: double confidence,
  3: optional string modelVersion
}

struct ModelCard {
  1: ModelType modelType,
  2: string name, 
  3: string version,
  4: optional string description,
  5: TrainingStatus status,
  6: optional string trainedDate,
  7: optional map<string, string> performanceMetrics
}

// Account, Category, Payee, Budget entities for budgeting platform
struct Account {
  1: string id,
  2: string name,
  3: string type,
  4: BudgetingPlatformType platformType,
  5: optional string institution,
  6: optional string currency,
  7: optional double balance,
  8: optional string status,
  9: optional string budgetId
}

struct Category {
  1: string id,
  2: string name,
  3: BudgetingPlatformType platformType,
  4: optional string description,
  5: optional bool isIncomeCategory,
  6: optional string budgetId
}

struct Payee {
  1: string id,
  2: string name,
  3: BudgetingPlatformType platformType,
  4: optional string description,
  5: optional string budgetId
}

struct Budget {
  1: string id,
  2: string name,
  3: string currency,
  4: BudgetingPlatformType platformType,
  5: optional double totalAmount,
  6: optional string startDate,
  7: optional string endDate
}

// Transaction reference union for cross-service communication
union Transactions {
  1: list<Transaction> transactions,
  2: list<string> transactionIds
}

// File entity for file storage
struct FileEntity {
  1: string path,        // Unique identifier for the file
  2: string name,
  3: optional string contentType,
  4: optional i64 size,
  5: optional string checksum,
  6: optional i64 lastModified,
  7: optional map<string, string> metadata
}

enum EntityType {
  Transaction = 1,
  Metadata = 2,
  ConfigItem = 3,
  ModelCard = 4,
  Account = 5,
  Category = 6,
  Payee = 7,
  Budget = 8,
  FileEntity = 9
  // Extendable
}

// Union type for all entities
union Entity {
  1: Transaction transaction,
  2: Metadata metadata,
  3: ConfigItem configItem,
  4: ModelCard modelCard,
  5: Account account,
  6: Category category,
  7: Payee payee,
  8: Budget budget,
  9: FileEntity file
}