# Technology Stack

## Backend
- **Framework**: FastAPI with Python 3.x
- **Database**: PostgreSQL 15 with asyncpg driver
- **Service Interfaces**: Apache Thrift for service contracts and data models
- **Data Types**: Generated Thrift types (`backend/thrift_gen/gen-py/*/ttypes.py`) - **MUST BE USED**
- **Legacy Data Validation**: Pydantic v2 models (transitioning to Thrift types)
- **ASGI Server**: Uvicorn with standard extras

### CRITICAL: Core Architecture
- **NEVER modify files in `.llm/core/` directory** without explicit approval
- All service interfaces defined in `.llm/core/*.thrift` files
- Generated Python types in `backend/thrift_gen/gen-py/*/ttypes.py` MUST be used for all internal data structures
- Always refer to `.llm/core/system_architecture.md` for service communication patterns

## Frontend
- **Framework**: Svelte 5 with Vite 7 -> make sure to use Svelte 5 patterns. So onclick=, not on:click=. Check the existing codebase for guidance
- **UI Components**: shadcn-svelte (use out-of-the-box components whenever possible)
- **Build Tool**: Vite for development and production builds
- **Package Manager**: npm

## Infrastructure
- **Containerization**: Docker with Docker Compose
- **Reverse Proxy**: nginx (Alpine)
- **Database Migrations**: dbmate
- **Environment**: Separate .env files for frontend/backend

## Common Commands

### Development Setup
```bash
# Start all services
docker compose up --build

# Stop services
docker compose down

# Stop and remove volumes (reset database)
docker compose down -v
```

### Frontend Development
```bash
cd frontend
npm run dev      # Development server
npm run build    # Production build
npm run preview  # Preview production build
```

### Backend Development
```bash
cd backend
# Backend runs automatically in Docker with hot reload
# Direct access at http://localhost:8000

# Code formatting and linting
docker compose exec backend uv run ruff format .    # Format all Python files inside the "backend" folder 
docker compose exec backend uv run ruff check .     # Check for linting issues inside the "backend" folder  
docker compose exec backend uv run ruff check --fix .  # Auto-fix linting issuesinside the "backend" folder 
```

### Database Operations
```bash
# Migrations are handled automatically by dbmate service
# Manual migration files in backend/db/migrations/
```

### Container-First Development
```bash
# Install Python packages into container (NOT directly on host)
docker compose exec backend uv add <package>

# Run tests through container
docker compose exec backend uv run python -m pytest

# Run Python commands through container
docker compose exec backend uv run python script.py
```

## Service Ports
- nginx (main): 80
- Frontend: 3000 (internal)
- Backend: 8000 (direct access)
- PostgreSQL: 5432
## Service Architecture

### Thrift Service Definitions
- All service interfaces defined in `.llm/core/*.thrift` files
- Generated Python implementations in `backend/thrift_gen/gen-py/`
- Use generated types for all internal data structures
- Service contracts define exact method signatures and data models
- Only the methods in Thrift structs should be public methods of classes. Every other method should be private.

### Service Implementation Mapping
Each Thrift service maps to exactly one Python implementation:

#### Managers (Business Orchestration)
- `TransactionManager` service → `backend/transaction_manager.py`
- `PredictionManager` service → `backend/prediction_manager.py`
- `Configs` service → `backend/configs.py`

#### Engines (Business Activities)
- `MLEngine` service → `backend/ml_engine.py`
- `MetadataFindingEngine` service → `backend/metadata_finding_engine.py`

#### Resource Access (Data Operations)
- `DatabaseStoreAccess` service → `backend/resource_layer/database_store_access/database_resource_access.py`
- `BudgetingPlatformAccess` service → `backend/resource/budgeting_platform_access.py`
- `MetadataSourceAccess` service → `backend/resource/metadata_source_access.py`

### Layer Communication
- API Layer uses FastAPI for HTTP endpoints
- Business Layer implements Thrift service interfaces
- Resource Access Layer provides atomic business operations
- All layers use generated Thrift types for data consistency

### Layer Interaction Rules (STRICTLY ENFORCED)

**ALLOWED Interactions:**
- API Layer → Managers only
- Managers → Engines and ResourceAccess services
- Engines → ResourceAccess services only
- ResourceAccess → Resources (DB, External APIs)
- Any component → Utility services (Security, Logging, Config)

**FORBIDDEN Interactions:**
- Calling up layers (Engine → Manager, ResourceAccess → Engine)
- Calling sideways within layers (Manager → Manager direct calls, Engine → Engine)
- Clients calling multiple Managers in same use case
- Clients calling Engines directly

## Testing Guidelines

### Container-First Testing
- **Run all tests through Docker containers for consistency**
- Use `docker compose exec backend python -m pytest` for running tests
- Avoid running pip, python, or pytest directly on the host system
- When testing APIs with curl commands, create scripts that make all requests instead of one-by-one
- When needing to test APIs, avoid trying to create python scripts and just create the bash scripts right away.

### Testing Best Practices
- Focus on testing public interfaces of files, not internal implementation details
- Don't create extensive unit tests for every line of code
- Prioritize integration tests and API endpoint testing
- Test service interfaces to ensure contract compliance with Thrift definitions
- Use mocking for external dependencies (YNAB API, Gmail API)
- Test layer interactions follow The Method architecture rules
- Stop wasting time writing tons of tests. Instead, write cleaner simpler code that has less chance of bugs. This application isn't that complicated.
- Anytime you create a 1-off script for testing, make sure to clean it up after yourself
- Anytime you're trying to run 1-off commands to test something, think "will I do this multiple times, and should I create a small script instead?"


### Data Type Usage in Tests
- Use generated Thrift types from `backend/thrift_gen/gen-py/*/ttypes.py` for all test data
- Test exception handling using Thrift-defined exceptions
- Verify service contracts match Thrift interface definitions

### Python Style & Code Quality
- DO NOT add from ... import statements in the middles of files. Only add them at the top of a file. Follow PEP 8 guidelines
- DO NOT add useless comments like "# moved to..." or "# PUBLIC METHODS - These correspond exactly..." or anything that is self-evident by the naming of the methods or variables. Non-obvious comments about the "why" something was implemented a certain way are good, and sometimes comments to separate logical sections are OK
- **ALWAYS run ruff formatting before completing tasks**: `docker compose exec backend uv run ruff format .` inside the "backend" folder 
- **Check for linting issues**: `docker compose exec backend uv run ruff check .` and fix any issues found inside the "backend" folder 
- Ruff is configured in `backend/pyproject.toml` with sensible defaults for line length (88), import sorting, and code style