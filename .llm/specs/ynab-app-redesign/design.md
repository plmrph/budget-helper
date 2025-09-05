# Design Document

## Overview

This document outlines the design for a redesigned Budget Helper application following The Method architecture principles. The system uses volatility-based decomposition to create a maintainable, extensible architecture with Apache Thrift service contracts, Svelte frontend, FastAPI backend, and Docker Compose orchestration.

## Architecture

### High-Level Architecture (The Method)

The architecture follows The Method's volatility-based decomposition with strict layer separation:

```mermaid
flowchart TD
  %% Layers
  %% Client Layer
  subgraph ClientLayer["Client Layer"]
    UIClient["UI Client(s)"]
  end

  %% Business Layer
  subgraph BusinessLayer["Business Layer"]
    direction TB
    %% Managers (top row) - Encapsulate use case sequence volatility
    TransactionManager["TransactionManager<br/>// Encapsulates transaction workflow volatility"]
    PredictionManager["PredictionManager<br/>// Encapsulates ML prediction workflow volatility"]
    %% Engines (bottom row) - Encapsulate business activity volatility
    MetadataFindingEngine["MetadataFindingEngine<br/>// Encapsulates metadata search algorithm volatility"]
    MLEngine["MLEngine<br/>// Encapsulates ML model training/prediction volatility"]
  end

  %% API Layer
  APILayer["API Layer<br/>// Encapsulates HTTP protocol and client interface volatility"]

  %% Resource Layer
  subgraph ResourceLayer["Resource Layer"]
    direction TB
    %% ResourceAccess (top row) - Encapsulate resource access volatility
    BudgetingPlatformAccess["BudgetingPlatformAccess<br/>// Encapsulates YNAB API integration volatility"]
    MetadataSourceAccess["MetadataSourceAccess<br/>// Encapsulates email/SMS source volatility"]
    DatabaseStoreAccess["DatabaseStoreAccess<br/>// Encapsulates database technology volatility"]
    FileStoreAccess["FileStoreAccess<br/>// Encapsulates file storage volatility"]
    %% Resources (bottom row) - Least volatile components
    PostgresDB[("Postgres DB")]
    %% External Resources (inside system boundary, dotted)
    subgraph ExternalResources["External Resources"]
      direction TB
      extYNABAPI["YNAB API"]
      extMetadataSources["Gmail, SMS, etc."]
    end
  end

  %% Utilities Bar
  subgraph UtilitiesBar["Utilities (Cross-cutting concerns)"]
      Security["Security (OAuth, etc.)"]
      Logging["Logging/Diagnostics"]
      Config["Configuration/Settings"]
  end

  %% Connections
  UIClient --> APILayer
  APILayer --> TransactionManager
  APILayer --> PredictionManager

  TransactionManager --> MetadataFindingEngine
  TransactionManager --> MLEngine
  TransactionManager --> BudgetingPlatformAccess
  TransactionManager --> MetadataSourceAccess
  TransactionManager --> DatabaseStoreAccess

  PredictionManager --> MLEngine
  PredictionManager --> DatabaseStoreAccess

  MetadataFindingEngine --> MetadataSourceAccess
  MetadataFindingEngine --> DatabaseStoreAccess

  MLEngine --> DatabaseStoreAccess
  MLEngine --> FileStoreAccess

  BudgetingPlatformAccess --> extYNABAPI
  MetadataSourceAccess --> extMetadataSources
  DatabaseStoreAccess --> PostgresDB

  %% Styling for external resources
  classDef extResource fill:#fff,stroke-dasharray: 5 5;
  class extYNABAPI,extMetadataSources extResource;
```

### Technology Stack

**Frontend:**
- **Framework**: Svelte 5 with Vite 7
- **UI Components**: shadcn-svelte (use out-of-the-box components whenever possible)
- **Build Tool**: Vite for development and production builds
- **Package Manager**: npm

**Backend:**
- **Framework**: FastAPI with Python 3.x
- **Database**: PostgreSQL 15 with asyncpg driver
- **Service Interfaces**: Apache Thrift for service contracts and data models
- **Data Types**: Generated Thrift types (`backend/thrift_gen/gen-py/*/ttypes.py`) - **MUST BE USED**
- **Legacy Data Validation**: Pydantic v2 models (transitioning to Thrift types)
- **ASGI Server**: Uvicorn with standard extras

**Infrastructure:**
- **Containerization**: Docker with Docker Compose
- **Reverse Proxy**: nginx (Alpine)
- **Database Migrations**: dbmate
- **Environment**: Separate .env files for frontend/backend

### CRITICAL: Core Architecture
- **NEVER modify files in `.llm/core/` directory** without explicit approval
- All service interfaces defined in `.llm/core/*.thrift` files
- Generated Python types in `backend/thrift_gen/gen-py/*/ttypes.py` MUST be used for all internal data structures
- Always refer to `.llm/core/system_architecture.md` for service communication patterns

## Service Interfaces and Volatility Encapsulation

All service interfaces are defined using Apache Thrift IDL files in `.llm/core/` directory. Each service encapsulates specific volatilities according to The Method principles:

### Business Layer Services

#### 1. TransactionManager (Manager - "What")
**File**: `.llm/core/transaction_manager.thrift`  
**Volatility Encapsulated**: Transaction workflow sequence volatility - how transaction use cases are orchestrated  
**Layer**: Business Layer (Manager)  
**Reusability**: Reusable across multiple clients

```thrift
service TransactionManager {
  entities.Transaction getTransaction(1: string transactionId)
  list<entities.Transaction> getTransactions(1: list<string> transactionIds)
  list<entities.Transaction> updateTransactions(1: list<TransactionEdit> transactionEdits)
  entities.Transaction attachTransactionMetadata(1: string transactionId, 2: entities.Metadata metadata)
  list<entities.Transaction> getPendingTransactions()
  list<entities.Transaction> getAllTransactions()
  SyncResult syncTransactionsIn(1: optional entities.BudgetingPlatformType budgetPlatform)
  SyncResult syncTransactionsOut(1: optional entities.BudgetingPlatformType budgetPlatform)
}
```

#### 2. PredictionManager (Manager - "What")
**File**: `.llm/core/prediction_manager.thrift`  
**Volatility Encapsulated**: ML prediction workflow sequence volatility - how prediction use cases are orchestrated  
**Layer**: Business Layer (Manager)  
**Reusability**: Reusable across multiple clients

```thrift
service PredictionManager {
  list<entities.ModelCard> getModels(1: optional entities.ModelType modelType)
  ml_engine.ModelTrainingResult trainModel(1: ml_engine.ModelTrainingRequest params)
  PredictionResult getPredictions(1: PredictionRequest request)
  bool deleteModel(1: entities.ModelCard modelCard)
}
```

#### 3. MetadataFindingEngine (Engine - "How")
**File**: `.llm/core/metadata_finding_engine.thrift`  
**Volatility Encapsulated**: Metadata search algorithm volatility - how metadata is found and matched  
**Layer**: Business Layer (Engine)  
**Reusability**: Highly reusable across managers

```thrift
service MetadataFindingEngine {
  list<MetadataCandidate> getMetadataCandidates(
    1: entities.Transactions transactions,
    2: optional MetadataFilter filter
  )
}
```

#### 4. MLEngine (Engine - "How")
**File**: `.llm/core/ml_engine.thrift`  
**Volatility Encapsulated**: ML model training and prediction algorithm volatility  
**Layer**: Business Layer (Engine)  
**Reusability**: Highly reusable across managers

```thrift
service MLEngine {
  list<entities.ModelCard> getModels(1: optional entities.ModelType modelType)
  bool deleteModels(1: list<entities.ModelCard> modelCards)
  list<ModelTrainingResult> trainModels(1: list<ModelTrainingRequest> trainingRequests)
  list<ModelPredictionResult> getPredictions(1: list<ModelPredictionBatchRequest> predictionRequests)
}
```

### Resource Access Layer Services

#### 5. BudgetingPlatformAccess (ResourceAccess - "Where")
**File**: `.llm/core/budgeting_platform_access.thrift`  
**Volatility Encapsulated**: YNAB API integration volatility - changes in external budgeting platform APIs  
**Layer**: Resource Access Layer  
**Reusability**: Most reusable - exposes atomic business verbs

```thrift
service BudgetingPlatformAccess {
  bool authenticate()
  list<entities.Account> getAccounts(1: optional entities.BudgetingPlatformType platform)
  list<entities.Category> getCategories(1: optional entities.BudgetingPlatformType platform)
  list<entities.Payee> getPayees(1: optional entities.BudgetingPlatformType platform)
  list<entities.Budget> getBudgets(1: optional entities.BudgetingPlatformType platform)
  list<entities.Transaction> getTransactions(...)
  bool updateTransactions(1: list<entities.Transaction> transactions, ...)
}
```

#### 6. MetadataSourceAccess (ResourceAccess - "Where")
**File**: `.llm/core/metadata_source_access.thrift`  
**Volatility Encapsulated**: Email/SMS source integration volatility - changes in metadata source APIs  
**Layer**: Resource Access Layer  
**Reusability**: Most reusable - exposes atomic business verbs

```thrift
service MetadataSourceAccess {
  bool authenticate()
  list<entities.Metadata> getMetadata(1: list<MetadataSourceQuery> queries)
}
```

#### 7. DatabaseStoreAccess (ResourceAccess - "Where")
**File**: `.llm/core/database_store_access.thrift`  
**Volatility Encapsulated**: Database technology volatility - changes in storage technology or schema  
**Layer**: Resource Access Layer  
**Reusability**: Most reusable - exposes atomic business verbs, not CRUD

```thrift
service DatabaseStoreAccess {
  list<entities.Entity> upsertEntities(1: list<entities.Entity> entities)
  list<string> deleteEntities(1: entities.EntityType entityType, 2: list<string> entityIds)
  list<entities.Entity> getEntitiesById(1: entities.EntityType entityType, 2: list<string> entityIds)
  QueryResult getEntities(1: Query query)
  list<entities.Entity> getEntitySummaries(1: entities.EntityType entityType, ...)
  list<string> getEntityIds(1: entities.EntityType entityType, ...)
}
```

#### 8. FileStoreAccess (ResourceAccess - "Where")
**File**: `.llm/core/file_store_access.thrift`  
**Volatility Encapsulated**: File storage technology volatility - changes in file storage systems  
**Layer**: Resource Access Layer  
**Reusability**: Most reusable - exposes atomic business verbs

```thrift
service FileStoreAccess {
  bool writeFile(1: string path, 2: binary data, 3: optional FileFormat format)
  binary readFile(1: string path)
  bool deleteFile(1: string path)
  list<FileMetadata> listFiles(1: FileQuery query)
  FileMetadata getFileMetadata(1: string path)
  bool fileExists(1: string path)
}
```

### Core Data Models

All data models are defined in `.llm/core/entities.thrift` and include:

- **Transaction**: Core business entity with metadata support
- **Metadata**: Extensible metadata system (Email, Prediction, etc.)
- **ModelCard**: ML model registry and versioning
- **Account, Category, Payee, Budget**: Budgeting platform entities
- **ConfigItem**: Configuration management with typed values
- **FileEntity**: File storage metadata

### Exception Handling

Standardized exceptions defined in `.llm/core/exceptions.thrift`:

```thrift
exception NotFoundException { 1: string message }
exception ValidationException { 1: string message }
exception InternalException { 1: string message }
exception ConflictException { 1: string message }
exception UnauthorizedException { 1: string message }
exception RemoteServiceException { 1: string message }
```

### Layer Interaction Rules (The Method)

The architecture enforces strict layer interaction rules to maintain volatility encapsulation:

#### ALLOWED Interactions:
- **API Layer → Managers**: FastAPI routes call TransactionManager and PredictionManager
- **Managers → Engines + ResourceAccess**: Managers orchestrate Engines and ResourceAccess services
- **Engines → ResourceAccess**: Engines use ResourceAccess services for data operations
- **ResourceAccess → Resources**: ResourceAccess services interact with databases and external APIs
- **Any component → Utilities**: All components can use Security, Logging, Config services

#### FORBIDDEN Interactions:
- **Calling Up Layers**: Engine → Manager, ResourceAccess → Engine
- **Calling Sideways**: Manager → Manager, Engine → Engine, ResourceAccess → ResourceAccess
- **Clients calling Engines directly**: API Layer must go through Managers
- **Clients calling multiple Managers in same use case**: Indicates functional decomposition

### Service Implementation Mapping

Each Thrift service maps to exactly one Python implementation:

#### Managers (Business Orchestration)
- `TransactionManager` service → `backend/business_layer/transaction_manager.py`
- `PredictionManager` service → `backend/business_layer/prediction_manager.py`

#### Engines (Business Activities)
- `MLEngine` service → `backend/business_layer/ml_engine.py`
- `MetadataFindingEngine` service → `backend/business_layer/metadata_finding_engine.py`

#### Resource Access (Data Operations)
- `DatabaseStoreAccess` service → `backend/resource_layer/database_store_access/database_resource_strategy.py`
- `BudgetingPlatformAccess` service → `backend/resource_layer/budgeting_platform_access/budgeting_platform_access.py`
- `MetadataSourceAccess` service → `backend/resource_layer/metadata_source_access/metadata_source_access.py`
- `FileStoreAccess` service → `backend/resource_layer/file_store_access/file_store_access.py`

#### Utility Services
- `Configs` service → `backend/configs.py`

### Frontend Component Architecture

Frontend uses Svelte 5 with shadcn-svelte components and follows modern patterns:

#### Core Principles:
- **Prioritize shadcn-svelte components** over custom implementations
- **Use shadcn-svelte for**: buttons, forms, tables, dialogs, dropdowns, inputs
- **Create custom components only** when shadcn-svelte doesn't provide needed functionality
- **Follow Svelte 5 modern syntax** and patterns
- **Communicate with backend** through well-defined API endpoints
- **Handle authentication and state** through proper stores

## Data Models and Type System

### Generated Thrift Types (PRIMARY)

All internal data structures MUST use generated Thrift types from `backend/thrift_gen/gen-py/*/ttypes.py`:

```python
# Import generated types
from entities.ttypes import (
    Transaction, Metadata, ModelCard, Account, Category, 
    Payee, Budget, ConfigItem, FileEntity, EntityType,
    MetadataType, BudgetingPlatformType, ModelType
)
from exceptions.ttypes import (
    NotFoundException, ValidationException, InternalException,
    ConflictException, UnauthorizedException, RemoteServiceException
)
```

### Key Data Models

#### Transaction Entity
```thrift
struct Transaction {
  1: string id,
  2: string date,
  3: i64 amount,  // in milliunits
  4: bool approved,
  5: BudgetingPlatformType platformType,
  6: optional string payeeId,
  7: optional string categoryId,
  8: optional string accountId,
  9: optional string budgetId,
  10: optional string memo,
  11: optional list<Metadata> metadata
}
```

#### Metadata System
```thrift
struct Metadata {
  1: MetadataType type,
  2: MetadataValue value,
  3: ExternalSystem sourceSystem,
  4: optional string description
}

union MetadataValue {
  1: string stringValue,
  2: PredictionResult predictionResult
}
```

#### ML Model Registry
```thrift
struct ModelCard {
  1: ModelType modelType,
  2: string name, 
  3: string version,
  4: optional string description,
  5: TrainingStatus status,
  6: optional string trainedDate,
  7: optional map<string, string> performanceMetrics
}
```

### Pydantic Models

Current `backend/models/` directory contains Pydantic models that are being gradually replaced with Thrift-generated types. Use Pydantic only for FastAPI request/response serialization when absolutely necessary.

### API Response Format

FastAPI responses use standard HTTP status codes with consistent JSON format:

```json
{
  "success": true,
  "data": { /* Thrift-serialized data */ },
  "message": "Optional success message"
}
```

Error responses:
```json
{
  "success": false,
  "message": "Error description",
  "error_code": "VALIDATION_ERROR",
  "details": { /* Additional error context */ }
}
```

## Error Handling

### Thrift Exception System

All services use standardized Thrift exceptions defined in `.llm/core/exceptions.thrift`:

```python
# Generated exception types
from exceptions.ttypes import (
    NotFoundException,      # Entity not found
    ValidationException,    # Input validation failed
    InternalException,      # Internal system error
    ConflictException,      # Resource conflict (e.g., concurrent modification)
    UnauthorizedException,  # Authentication/authorization failed
    RemoteServiceException  # External service error (YNAB API, Gmail API)
)
```

### Backend Error Handling

```python
# FastAPI exception handlers
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from exceptions.ttypes import *

async def thrift_exception_handler(request: Request, exc: Exception):
    """Handle Thrift exceptions with appropriate HTTP status codes"""
    
    if isinstance(exc, NotFoundException):
        status_code = 404
    elif isinstance(exc, ValidationException):
        status_code = 400
    elif isinstance(exc, UnauthorizedException):
        status_code = 401
    elif isinstance(exc, ConflictException):
        status_code = 409
    elif isinstance(exc, RemoteServiceException):
        status_code = 502
    elif isinstance(exc, InternalException):
        status_code = 500
    else:
        status_code = 500
    
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": exc.message,
            "error_type": exc.__class__.__name__
        }
    )
```

### Frontend Error Handling

Frontend uses Svelte stores for centralized error management:

```javascript
// stores/error.js
import { writable } from 'svelte/store';

export const errors = writable([]);

export function handleApiError(error) {
  const errorMessage = error.response?.data?.message || 'An unexpected error occurred';
  const errorType = error.response?.data?.error_type || 'UnknownError';
  
  errors.update(errs => [...errs, {
    id: crypto.randomUUID(),
    message: errorMessage,
    type: errorType,
    timestamp: new Date(),
    dismissible: true
  }]);
}
```

## Testing Strategy

### Container-First Testing Philosophy

All testing follows container-first principles to ensure consistency:

- **Run all tests through Docker containers** for consistency
- Use `docker compose exec backend python -m pytest` for running tests
- **Never run pip, python, or pytest directly on host system**
- When testing APIs with curl commands, create scripts that make all requests instead of one-by-one

### Testing Approach

#### Focus Areas:
- **Test public interfaces** of services, not internal implementation details
- **Prioritize integration tests** and API endpoint testing over extensive unit tests
- **Test service interfaces** to ensure contract compliance with Thrift definitions
- **Test layer interactions** follow The Method architecture rules
- Use mocking for external dependencies (YNAB API, Gmail API)

#### What NOT to Test:
- Don't create extensive unit tests for every line of code
- Don't test internal implementation details
- Stop wasting time writing tons of tests - instead, write cleaner simpler code that has less chance of bugs

### Thrift Contract Testing

```python
# Test service contract compliance
def test_transaction_manager_contract():
    """Ensure TransactionManager implements Thrift interface correctly"""
    from transactionmanager.ttypes import TransactionEdit
    from entities.ttypes import Transaction
    
    # Test that service methods match Thrift contract
    manager = TransactionManager()
    transaction = manager.getTransaction("test-id")
    assert isinstance(transaction, Transaction)
```

### Integration Testing

```python
# Test layer interactions
def test_manager_to_engine_interaction():
    """Test that Managers properly call Engines following The Method rules"""
    # Verify TransactionManager → MetadataFindingEngine interaction
    # Verify PredictionManager → MLEngine interaction
```

## Deployment Configuration

### Docker Compose Architecture

The system uses a multi-container architecture with proper service dependencies:

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - frontend
      - backend
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    env_file:
      - .env.frontend
    volumes:
      - ./frontend:/app
      - /app/node_modules
    depends_on:
      - backend
    restart: unless-stopped

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    env_file:
      - .env.backend
    environment:
      - GMAIL_CLIENT_ID=${GMAIL_CLIENT_ID:-}
      - GMAIL_CLIENT_SECRET=${GMAIL_CLIENT_SECRET:-}
      - GMAIL_REDIRECT_URI=${GMAIL_REDIRECT_URI:-http://localhost:8000/api/email/callback}
    volumes:
      - ./backend:/app
      - ./.llm/core:/app/.llm/core  # Mount Thrift definitions
    ports:
      - "8000:8000"
    depends_on:
      dbmate:
        condition: service_completed_successfully
      postgres:
        condition: service_healthy
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-ynab_db}
      POSTGRES_USER: ${POSTGRES_USER:-ynab_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-ynab_password}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backend/db:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-ynab_user} -d ${POSTGRES_DB:-ynab_db}"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 30s
    restart: unless-stopped

  dbmate:
    image: amacneil/dbmate:latest
    environment:
      DATABASE_URL: postgresql://ynab_user:ynab_password@postgres:5432/ynab_db?sslmode=disable
    volumes:
      - ./backend/resource_layer/db:/db  # Updated path for migrations
    depends_on:
      postgres:
        condition: service_healthy
    command: ["up"]
    restart: "no"

volumes:
  postgres_data:
```

### Container-First Development Rules

#### Python Package Management:
- **NEVER install Python packages via pip directly on host**
- Use `docker compose exec backend uv add <package>` for adding dependencies
- Dependencies are managed in `backend/pyproject.toml` and automatically synced
- Run all Python commands through container: `docker compose exec backend python script.py`

#### Development Commands:

```bash
# Start all services
docker compose up --build

# Stop services
docker compose down

# Stop and remove volumes (reset database)
docker compose down -v

# Install Python packages into container
docker compose exec backend uv add <package>

# Run tests through container
docker compose exec backend uv run python -m pytest

# Run Python commands through container
docker compose exec backend python script.py
```

### Environment Configuration

```bash
# .env.frontend
VITE_API_URL=http://localhost:8000
VITE_APP_NAME=Budget Helper
VITE_DEBUG=true
```

```bash
# .env.backend
# Database
DATABASE_URL=postgresql://ynab_user:ynab_password@postgres:5432/ynab_db
POSTGRES_DB=ynab_db
POSTGRES_USER=ynab_user
POSTGRES_PASSWORD=ynab_password

# External APIs
YNAB_API_URL=https://api.youneedabudget.com/v1
GMAIL_API_URL=https://gmail.googleapis.com/gmail/v1

# OAuth Configuration
GMAIL_CLIENT_ID=your-gmail-client-id
GMAIL_CLIENT_SECRET=your-gmail-client-secret
GMAIL_REDIRECT_URI=http://localhost:8000/api/email/callback

# Application
DEBUG=true
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here
```

### Service Ports
- **nginx (main)**: 80
- **Frontend**: 3000 (internal)
- **Backend**: 8000 (direct access)
- **PostgreSQL**: 5432

## Architecture Benefits

This design provides:

1. **Volatility Encapsulation**: Each service encapsulates specific types of change, making the system resilient to requirements evolution
2. **Clean Layer Separation**: Strict interaction rules prevent architectural degradation
3. **High Reusability**: Lower layers (ResourceAccess, Resources) are highly reusable across use cases
4. **Testability**: Well-defined interfaces enable comprehensive testing strategies
5. **Extensibility**: New features can be added by composing existing services or adding new vertical slices
6. **Maintainability**: Changes are contained to appropriate layers, minimizing system-wide impact

The architecture follows The Method principles to create a robust, maintainable system that can evolve with changing business requirements while preserving the significant investment in lower-layer components.