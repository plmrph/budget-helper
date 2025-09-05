# Project Structure

## CRITICAL: Core Files (DO NOT MODIFY)

The files in `.llm/core/` directory contain the authoritative system design and MUST NOT be modified without explicit approval. These files define:

- **Thrift Service Interfaces**: All service contracts and data models
- **System Architecture**: Component relationships and communication patterns  
- **Generated Types**: Python types in `backend/thrift_gen/` that should be used throughout the application
- **Method Guidance**: Architectural principles and layer responsibilities

**IMMUTABLE DESIGN PRINCIPLES:**
1. **NEVER modify files in `.llm/core/` directory** without explicit approval
2. If there's an absolute show stopper requiring changes, confirm with the user first
3. The design has been carefully crafted using The Method principles

**Always refer to `.llm/core/system_architecture.md` for service communication patterns and architectural decisions.**

## Root Level
- `docker-compose.yml` - Multi-service container orchestration
- `nginx.conf` - Reverse proxy configuration
- `.env.frontend` / `.env.backend` - Environment configuration files
- `README.md` - Project documentation and setup instructions
- `.llm/core/` - **IMMUTABLE** system design and interfaces

## Backend Architecture (Following The Method)

The backend follows a strict 4-layer architecture based on volatility-based decomposition:

### Layer 1: API Layer (`backend/api/`)
- FastAPI routes and HTTP handling
- Request/response transformation
- Authentication middleware
- CORS configuration

### Layer 2: Business Layer (`backend/`)
#### Managers (Orchestration - "What")
- `transaction_manager.py` - TransactionManager service implementation
- `prediction_manager.py` - PredictionManager service implementation
- Handle use case orchestration and workflow sequencing
- Call Engines and ResourceAccess services

#### Engines (Activities - "How") 
- `ml_engine.py` - MLEngine service implementation
- `metadata_finding_engine.py` - MetadataFindingEngine service implementation
- Perform atomic business activities
- Designed for reuse across Managers

### Layer 3: Resource Access Layer (`backend/resource/`)
- `database_store_access/database_resource_access.py` - DatabaseStoreAccess service implementation
- `budgeting_platform_access.py` - BudgetingPlatformAccess service implementation  
- `metadata_source_access.py` - MetadataSourceAccess service implementation
- Expose atomic business verbs, not CRUD operations
- Convert business operations to I/O operations

### Layer 4: Resource Layer
- PostgreSQL database
- External APIs (YNAB, Gmail)
- File systems and caches

## Data Models and Types

### Use Generated Thrift Types
- **ALWAYS** use types from `backend/thrift_gen/gen-py/*/ttypes.py` for internal data structures
- Import from: `entities.ttypes`, `exceptions.ttypes`, etc.
- These provide consistent, validated data structures across all services

### Pydantic Models
- Current `backend/models/` directory contains Pydantic models
- Gradually replace with Thrift-generated types
- Use Pydantic only for FastAPI request/response serialization when needed

## Frontend (`frontend/`)
```
frontend/
├── package.json       # Node.js dependencies and scripts
├── vite.config.js     # Vite build configuration
├── svelte.config.js   # Svelte compiler configuration
├── Dockerfile         # Frontend container configuration
├── index.html         # Main HTML template
├── src/
│   ├── main.js        # Application entry point
│   ├── App.svelte     # Root Svelte component
│   ├── app.css        # Global styles
│   ├── lib/           # Reusable Svelte components
│   └── assets/        # Static assets
└── public/            # Public static files
```

## The Method Architecture Implementation

### Layer Interaction Rules (STRICTLY ENFORCED)

**Allowed Interactions:**
- API Layer → Managers
- Managers → Engines, ResourceAccess services  
- Engines → ResourceAccess services
- ResourceAccess → Resources (DB, External APIs)
- Any component → Utility services (Security, Logging, Config)

**FORBIDDEN Interactions:**
- Calling up layers (Engine → Manager, ResourceAccess → Engine)
- Calling sideways within layers (Manager → Manager, Engine → Engine)
- Clients calling multiple Managers in same use case
- Clients calling Engines directly

### Service Implementation Mapping

Each Thrift service interface maps to exactly one Python class:

#### Managers (Business Orchestration)
- `TransactionManager` → `backend/transaction_manager.py`
- `PredictionManager` → `backend/prediction_manager.py`

#### Engines (Business Activities)  
- `MLEngine` → `backend/ml_engine.py`
- `MetadataFindingEngine` → `backend/metadata_finding_engine.py`

#### Resource Access (Data Operations)
- `DatabaseStoreAccess` → `backend/resource_layer/database_store_access/database_resource_access.py`
- `BudgetingPlatformAccess` → `backend/resource/budgeting_platform_access.py`
- `MetadataSourceAccess` → `backend/resource/metadata_source_access.py`

#### Utility Services
- `Configs` → `backend/configs.py`
- Authentication, Logging, Security services

### Component Responsibilities

#### Managers ("What" - Use Case Orchestration)
- Implement complete business workflows
- Handle sequencing of activities
- Coordinate between Engines and ResourceAccess
- Absorb volatility in use case changes
- Should be "almost expendable" - easy to modify

#### Engines ("How" - Business Activities)
- Perform atomic business activities
- Designed for reuse across Managers
- Encapsulate volatile business rules
- No direct resource access - use ResourceAccess services

#### ResourceAccess ("Where" - Data Operations)
- Expose atomic business verbs (not CRUD)
- Convert business operations to I/O operations
- Handle resource-specific concerns (connection pooling, retries)
- Most reusable components in the system

### Development Decision Framework

Before making any architectural change, ask these critical questions:

1. **"What layer am I in?"** - Identify the current architectural layer
2. **"Is this change appropriate for this layer?"** - Verify the change aligns with layer responsibilities
3. **"Does this violate The Method principles?"** - Check against volatility-based decomposition rules
4. **"Does this violate interaction rules?"** - Verify no forbidden layer interactions
5. **"Can I reuse existing components?"** - Prefer composition over new components

### When Implementing Services
1. Follow the exact Thrift interface contract
2. Use generated types for all data structures
3. Implement proper exception handling as defined in Thrift
4. Maintain layer separation and interaction rules

### When Adding New Functionality
1. Determine if it's a new use case (Manager), activity (Engine), or data operation (ResourceAccess)
2. Check if existing components can handle the functionality through composition
3. If new components are needed, follow The Method naming conventions
4. Ensure the new component fits within the volatility-based decomposition

## Frontend Component Guidelines

- Prioritize shadcn-svelte components over custom implementations
- Use shadcn-svelte for: buttons, forms, tables, dialogs, dropdowns, inputs
- Only create custom components when shadcn-svelte doesn't provide the needed functionality
- Follow Svelte 5 modern syntax and patterns
- Communicate with backend through well-defined API endpoints
- Handle authentication and state management through proper stores

## Container-First Development

### Docker Development Rules
1. **Stop trying to install Python packages via pip directly. Install these into the container. Run things through the container not raw.**
   - Use `docker compose exec backend uv add <package>` for adding dependencies
   - Dependencies are managed in `backend/pyproject.toml` and automatically synced
   - Run tests and Python commands through the container: `docker compose exec backend uv run python -m pytest`
   - Avoid running pip, python, or pytest directly on the host system

### Testing Best Practices
- Focus on testing public interfaces of files, not internal implementation details
- Don't create extensive unit tests for every line of code
- Prioritize integration tests and API endpoint testing
- Test service interfaces to ensure contract compliance with Thrift definitions
- Use mocking for external dependencies (YNAB API, Gmail API)
- Run all tests through Docker containers for consistency
- Stop wasting time writing tons of tests. Instead, write cleaner simpler code that has less chance of bugs. This application isn't that complicated.
- When trying to test inside the container with curl commands, for example when testing the API, instead of making the requests one by one create a script that makes all the requests

### Code Quality Tools
- **Ruff**: Fast Python linter and formatter configured in `backend/pyproject.toml`
- **Format code**: `docker compose exec backend uv run ruff format .` inside the "backend" folder 
- **Check linting**: `docker compose exec backend uv run ruff check .` inside the "backend" folder 
- **Auto-fix issues**: `docker compose exec backend uv run ruff check --fix .` inside the "backend" folder 
- Always run ruff formatting before completing development tasks