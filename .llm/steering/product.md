# Product Overview

Budget Helper is a redesigned application that helps users reconcile their YNAB (You Need A Budget) transactions with email receipts and AI-powered categorization.

## Core Features

### Transaction Management
- Import transactions from YNAB budgeting platform
- View and manage transaction details (amount, category, payee, memo)
- Update transaction approval status and metadata
- Sync transactions bidirectionally with YNAB

### Metadata Integration
- Find and attach email receipts to transactions
- Support for Gmail integration with OAuth authentication
- Extensible metadata system for future sources (SMS, etc.)
- Search and filter metadata by date, content, and source

### AI-Powered Categorization
- Train machine learning models on transaction history
- Predict transaction categories with confidence scores
- Support for AutoGluon ML framework
- Model versioning and performance tracking

### Account and Budget Management
- Handle multiple financial accounts
- Category and payee management
- Budget tracking integration with YNAB
- Configuration management for external systems

## Architecture

### CRITICAL: Core Files (DO NOT MODIFY)
The files in `.llm/core/` directory contain the authoritative system design and MUST NOT be modified without explicit approval. These files define all service contracts and data models using Apache Thrift.

### The Method Architecture
The system follows The Method architecture with strict layer separation based on volatility-based decomposition:

- **Client Layer**: Svelte frontend and external API consumers
- **Business Layer**: TransactionManager and PredictionManager orchestrate workflows
- **Engine Layer**: MLEngine and MetadataFindingEngine perform business activities
- **Resource Access Layer**: DatabaseStoreAccess, BudgetingPlatformAccess, MetadataSourceAccess
- **Resource Layer**: PostgreSQL database, YNAB API, Gmail API

### Service Interface Contracts
- All service interfaces defined in `.llm/core/*.thrift` files
- Generated Python types in `.llm/core/gen-py/*/ttypes.py` MUST be used for all internal data structures
- Service contracts define exact method signatures, parameters, return types, and exceptions
- Each Thrift service maps to exactly one Python implementation class

### Layer Interaction Rules (STRICTLY ENFORCED)
**ALLOWED:** API Layer → Managers → Engines/ResourceAccess → Resources
**FORBIDDEN:** Calling up layers, calling sideways within layers, clients calling engines directly

## Key Business Logic

- Transactions stored in milliunits (e.g., $1.00 = 1000)
- Transaction states: approved/unapproved with metadata attachments
- Email-transaction matching for receipt reconciliation
- AI-powered category prediction with confidence scoring
- Bidirectional sync with YNAB platform
- Extensible metadata system supporting multiple sources
- Model training and prediction workflows

## Development Principles

### Container-First Development
- **Install Python packages into containers, not directly on host**
- Use `docker compose exec backend uv add <package>` for adding dependencies
- Run all tests and Python commands through containers for consistency
- Dependencies are managed in `backend/pyproject.toml` and automatically synced

### Component Responsibilities
#### Managers ("What" - Use Case Orchestration)
- Implement complete business workflows from start to finish
- Handle sequencing of activities and error recovery
- Coordinate between Engines and ResourceAccess services
- Should be "almost expendable" - easy to modify when requirements change

#### Engines ("How" - Business Activities)
- Perform atomic business activities that can be reused
- Encapsulate volatile business rules and calculations
- Designed for reuse across multiple Managers
- No direct resource access - must use ResourceAccess services

#### ResourceAccess ("Where" - Data Operations)
- Expose atomic business verbs, NOT CRUD operations
- Convert business operations to I/O operations against resources
- Handle resource-specific concerns (connection pooling, retries, caching)
- Most reusable components in the system

### Data Type Usage
- **PRIMARY:** Use generated Thrift types from `.llm/core/gen-py/*/ttypes.py`
- Use Pydantic only for FastAPI request/response serialization when absolutely necessary

### Testing Philosophy
- Focus on testing public interfaces, not internal implementation details
- Prioritize integration tests and API endpoint testing
- Stop wasting time writing tons of tests. Instead, write cleaner simpler code that has less chance of bugs
- Test service interfaces to ensure contract compliance with Thrift definitions
- Run all tests through Docker containers for consistency
- If there's multiple of the same thing to test or run, write a script, and then run and clean it up
- 