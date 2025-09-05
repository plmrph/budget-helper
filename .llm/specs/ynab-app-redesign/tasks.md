# Implementation Plan

## Core Infrastructure and Service Implementation

- [x] 1. Set up project structure and development environment
  - Docker Compose configuration with frontend, backend, nginx, postgres, and dbmate containers
  - Environment files (.env.frontend and .env.backend) 
  - Nginx reverse proxy configuration
  - Database migrations and schema setup
  - _Requirements: 1.4, 1.5, 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 2. Implement Thrift service contracts and generated types
  - All service interfaces defined in `.llm/core/*.thrift` files
  - Generated Python types in `backend/thrift_gen/*/ttypes.py`
  - Core data models (Transaction, Metadata, ModelCard, etc.) from entities.thrift
  - Exception handling system from exceptions.thrift
  - _Requirements: 2.2, 2.3, 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 3. Implement Resource Access Layer services
  - DatabaseStoreAccess service with atomic business verbs
  - BudgetingPlatformAccess service for YNAB API integration
  - MetadataSourceAccess service for email/metadata sources
  - Connection pooling, error handling, and resource management
  - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4. Implement Business Layer services (partial)
  - TransactionManager service for transaction workflow orchestration
  - PredictionManager service for ML prediction workflows
  - MetadataFindingEngine service for metadata search algorithms
  - MLEngine service for ML model training and prediction
  - _Requirements: 2.2, 2.3, 2.4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 5. Implement API Layer with FastAPI
  - FastAPI application structure with route organization
  - Authentication, transaction, email, ML, and settings endpoints
  - Dependency injection for service access
  - Middleware for CORS, error handling, and logging
  - _Requirements: 1.1, 1.2, 1.3, 8.1, 8.2_

- [x] 6. Implement frontend structure with Svelte 5
  - Svelte 5 application with Vite build system
  - shadcn-svelte component library integration
  - Basic page components (Dashboard, Transactions, Settings)
  - Authentication components and API client
  - _Requirements: 1.1, 8.1, 8.2, 8.3, 8.4, 8.5_

## Missing Service Implementations and Integrations

- [x] 7. Wire through Configs singleton to be used in components where required
  - Update all service implementations to use ConfigService for configuration management
  - Integrate ConfigService with DatabaseStoreAccess for persistent configuration storage
  - Add configuration endpoints to API layer
  - Update frontend to use configuration API for settings management
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 8. Complete TransactionManager service implementation
  - Implement missing methods: attachTransactionMetadata, getPendingTransactions, getTransactions
  - Add proper integration with MetadataFindingEngine for metadata attachment
  - Implement transaction synchronization with YNAB (syncTransactionsIn/Out)
  - Add transaction history tracking integration
  - _Requirements: 2.2, 2.3, 2.4, 6.1, 6.2, 6.3_

- [x] 9. Complete PredictionManager service implementation
  - Implement model management methods (getModels, deleteModel)
  - Add proper integration with MLEngine for training and predictions
  - Implement model versioning and performance tracking
  - Add training data preparation and validation
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 10. Complete MetadataFindingEngine service implementation
  - ✅ Implement getMetadataCandidates method with transaction matching algorithms
  - ✅ Add email-transaction matching logic using date, amount, and payee patterns in the strategy
  - ✅ Integrate with MetadataSourceAccess for email search
  - ✅ Add metadata filtering and ranking capabilities
  - _Requirements: 3.2, 3.3, 3.4_

- SKIP FOR NOW [ ] 11. Complete MLEngine service implementation
  - Implement model training workflow with data preparation
  - Add prediction methods for single and batch transactions
  - Implement model backup, restore, and versioning
  - Add model performance metrics and evaluation
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- SKIP FOR NOW [ ] 12. Implement FileStoreAccess service
  - Create FileStoreAccess service implementation following Thrift interface
  - Add file storage operations for ML models and backups
  - Implement file metadata management and querying
  - Add integration with MLEngine for model persistence
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

## Database and Data Layer Completion

- [ ] 13. Complete DatabaseStoreAccess implementation
  - Implement missing entity operations for all entity types
  - Add proper query support with filtering, sorting, and pagination
  - Implement entity relationship management
  - Add transaction support for atomic operations
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2_

- [ ] 14. Implement configuration persistence in database
  - Add configuration tables to database schema
  - Implement ConfigItem storage and retrieval in DatabaseStoreAccess
  - Add configuration change tracking and history
  - Implement configuration validation and type safety
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 15. Add metadata storage and indexing
  - Implement metadata storage tables for emails and predictions
  - Add full-text search capabilities for email content
  - Implement metadata relationship tracking with transactions
  - Add metadata caching and performance optimization
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

## API Layer Completion

- [ ] 16. Complete transaction API endpoints
  - Implement missing transaction operations (batch updates, metadata attachment)
  - Add transaction filtering with advanced search capabilities
  - Implement transaction history and undo functionality
  - Add transaction synchronization endpoints
  - _Requirements: 2.2, 2.3, 2.4, 6.1, 6.2, 6.3_

- [ ] 17. Complete ML/AI API endpoints
  - Implement model training endpoints with progress tracking
  - Add batch prediction endpoints for multiple transactions
  - Implement model management endpoints (backup, restore, delete)
  - Add training data preparation and validation endpoints
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 18. Complete email/metadata API endpoints
  - Implement metadata candidate search endpoints
  - Add email search with transaction context
  - Implement metadata attachment workflow endpoints
  - Add email search configuration and history endpoints
  - _Requirements: 3.2, 3.3, 3.4_

- [ ] 19. Complete configuration API endpoints
  - Implement configuration CRUD endpoints using ConfigService
  - Add configuration validation and type checking
  - Implement configuration import/export functionality
  - Add configuration change history tracking
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

## Frontend Component Implementation

- [x] 20. Create advanced form and input components
  - Implement FormField component with validation and error display (will be used for editing Memo in-line in the grid)
  - Create DropdownSelect component with search functionality (will be used for selecting Categories, and later on Metadata matches)
  - The component should work in-line per row inside the Transaction grid, so like clicking on Category will use this component
  - Build AutocompleteSelect component with async data loading
  - Add form state management and submission handling
  - _Requirements: 8.3, 8.4, 8.5_

- [x] 21. Create data table component with advanced features
  - Implement DataTable component with sorting, filtering, and pagination
  - USE shadcn primitives to build this
  - Add column configuration and customization (displayed columns should be configured in Settings)
  - Create row selection and bulk action capabilities
  ---- Implement virtual scrolling for large datasets
  - _Requirements: 8.3, 8.4, 8.5_

- [x] 22. Create settings management UI
  - Implement SettingsLayout with tabbed interface for the different types of configs (see defaults)
  - Add configuration form components with validation
  - Make sure changes are saved to the DB and fetched from the DB
  - Create settings import/export functionality
  - Implement settings reset button
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.2, 8.3, 8.4_

- [x] 23. Implement transaction management UI
  - Create transaction list view with filtering, sorting, and search on the columns
  - Add transaction detail view with inline editing
  - Implement bulk transaction operations UI
  ---- Create transaction history and undo functionality
  - _Requirements: 2.2, 2.3, 2.4, 6.2, 6.3, 8.3, 8.4_

- [ ] 24. Implement email search and metadata UI
  - Add a column to the Transaction Grid to the right of Approved with the name being an icon that represents "metadata"
  - in this column will be different metadata icons like email, and in the future other metadata
  - the email icon should be green if only 1 matching email is found given the criteria, and yellow if multiple are found
  - click it will bring up the Create email search interface with transaction context that has the other results and the option to change the search then and there
  - Create email search interface with transaction context
  - Add email preview and selection components
  - Implement metadata attachment workflow UI
  - Create email search history and saved searches but make it simple
  - _Requirements: 3.2, 3.3, 3.4, 8.3, 8.4_

- [x] 25. Implement ML training and prediction UI
  - Create model training interface with progress tracking as a separate page like Settings or Transactions
  - On the page maybe it makes sense to use the https://ui.shadcn.com/docs/components/carousel for the next Visualization, Training, validation, and set as default/manage models step, and we can just not set the <Carousel.Previous /> <Carousel.Next /> so we change pages programmatically
  - this should be the first default card. The UI should contain all model management needs (set model as default, delete model, show datasets that were pulled, delete datasets). Remember to use shadcn components. 
  - this should be the next card - Implement pulling of transactions training data from the local database. It should be split 80/20 into training and test sets and stored locally as .csv files. The test set should contain at least 1 sample of each category from the overall set.
  - this should be the next card -  Add training data visualization and statistics like total count, count per category etc. On this view, the user should have a button "start training" which should lead to the next steps
  - - Implement passing the training dataset location to the backend PredictionManager for training
  - - Implement passing the test dataset location to the backend PredictionManager for validation
  - this should be the next card - Implement an interface showing the validation dataset and the predictions for each row so a user can visually confirm correctness
  - Keep in mind, the ML functionality the PredictionManager calls should be abstracted in the ML Engine
  - Update the ML Engine and get rid of any hard coded or test methods / values
  - Implement a strategy pattern so there is a strategy implementation per-model. The ML Engine should only interface with the strategy
  - Only implment a dummy strategy for now, I will add actual Model strategies later
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 8.3, 8.4_


## Integration and Testing

- [ ] 26. Implement service integration testing
  - Create integration tests for service layer interactions
  - Test Manager → Engine → ResourceAccess call patterns
  - Verify Thrift contract compliance across all services
  - Test error handling and exception propagation
  - _Requirements: 9.1, 9.2, 9.3_

- [ ] 27. Implement API integration testing
  - Create end-to-end API tests for all endpoints
  - Test authentication flows and authorization
  - Verify request/response serialization with Thrift types
  - Test error handling and HTTP status codes
  - _Requirements: 9.1, 9.2, 9.3_

- [ ] 28. Implement frontend integration testing
  - Create component integration tests with API mocking
  - Test user workflows and navigation
  - Verify error handling and loading states
  - Test responsive design and accessibility
  - _Requirements: 9.1, 9.2, 9.3, 8.5_

## Final System Integration

- [ ] 29. Complete YNAB API integration
  - Implement full bidirectional transaction synchronization
  - Add proper error handling for YNAB API rate limits
  - Implement OAuth token refresh and management
  - Test with real YNAB data and edge cases
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 30. Complete Gmail API integration
  - Implement full email search with advanced filters
  - Add proper OAuth flow and token management
  - Implement email content parsing and normalization
  - Test with real Gmail data and various email formats
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 31. Implement ML model training pipeline
  - Create complete training data preparation workflow
  - Implement model training with hyperparameter tuning
  - Add model evaluation and performance metrics
  - Implement model deployment and versioning
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 32. Final system integration and deployment
  - Test complete Docker Compose setup with all services
  - Verify all service integrations work correctly
  - Test database migrations and data consistency
  - Validate environment configuration and security
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_