# Implementation Plan

- [x] 1. Create editable cell components for inline editing
  - Search codebase for existing editable input components, ComboBox usage patterns, and cell component examples
  - Create EditableTextCell component for memo field editing
  - Create EditableCategoryCell component with ComboBox for category selection
  - Create EditableApprovalCell component with ComboBox for approval status
  - Implement visual feedback states (editing, success, error) in each component
  - Add keyboard navigation support (Enter to save, Escape to cancel)
  - _Requirements: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 5.1_

- [x] 2. Enhance transaction store with non-blocking updates
  - Search codebase for existing store patterns, async update methods, and optimistic update implementations
  - Add updateTransactionField method with optimistic updates
  - Implement _performAsyncUpdate private method for background processing
  - Add _showTransientSuccess and _showTransientError methods for visual feedback
  - Implement automatic error reversion when async updates fail
  - Add transaction state tracking (_saveStatus, _saveError, _saveTimestamp)
  - _Requirements: 4.1, 4.2, 5.2, 5.3, 5.4_

- [x] 3. Update AdvancedDataTable component to use editable cells
  - Search codebase for existing data table implementations, cell rendering patterns, and event handling examples
  - Replace static createRawSnippet cells with interactive components for memo, category, approval
  - Implement onSave handlers that call store.updateTransactionField
  - Add category data fetching and management for category dropdown
  - Implement row-level focus management for save triggers
  - Add visual indicators for save status using transaction state
  - _Requirements: 1.1, 1.3, 2.3, 3.3, 5.1, 5.2_

- [x] 4. Enhance TransactionManager with automatic YNAB sync
  - Search codebase for existing sync-out implementations, BudgetingPlatformAccess usage, and transaction update patterns
  - Add _syncUpdatedTransactions private method to trigger sync-out after updates
  - Modify existing updateTransactions method to call sync after successful database updates
  - Implement error handling that doesn't fail local updates if YNAB sync fails
  - Add logging for sync operations and failures
  - Ensure sync operations use existing BudgetingPlatformAccess code paths
  - _Requirements: 4.1, 4.3, 4.4, 6.1, 6.2, 6.3, 6.4_

- [x] 4.1. Implement TransactionManager.getBudgetsInfo method
  - Search codebase for existing BudgetingPlatformAccess usage patterns and budget data handling
  - Implement getBudgetsInfo method in TransactionManager following Thrift contract
  - Add logic to get default budget when no budgetIds provided
  - Add logic to fetch categories, payees, accounts based on entityTypes parameter
  - Implement proper error handling with Thrift-defined exceptions
  - Add logging for budget info operations
  - _Requirements: 6.6, 7.1, 7.2_

- [x] 4.2. Create budget info API endpoint
  - Search codebase for existing API endpoint patterns and route implementations
  - Create GET /api/budgets/info endpoint in budgets.py
  - Wire endpoint to call TransactionManager.getBudgetsInfo (not direct BudgetingPlatformAccess)
  - Add query parameter handling for budgetIds and entityTypes
  - Implement proper error handling and response formatting
  - Remove existing architectural violations in budget routes
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 4.3. Create budget store for frontend data caching
  - Search codebase for existing store patterns and data caching implementations
  - Create budget.js store with loadBudgetInfo method
  - Add helper methods for ID-to-name lookups (getCategoryName, getPayeeName, getAccountName)
  - Implement loading states and error handling
  - Add cache invalidation and refresh capabilities
  - Integrate with existing API client patterns
  - _Requirements: 2.7, 7.4, 7.5_

- [x] 4.4. Update EditableCategoryCell to use budget store and category IDs
  - Search codebase for existing ComboBox usage and category handling patterns
  - Modify EditableCategoryCell to use budgetStore for category options
  - Change category display to show "Uncategorized" when no category is set
  - Update onSave to send categoryId instead of category name
  - Add proper loading states while budget data is being fetched
  - Implement category filtering and search functionality
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.7_

- [x] 4.5. Remove architectural violations and cleanup
  - Search codebase for _enrichTransactionsWithNames usage and remove it
  - Remove direct BudgetingPlatformAccess calls from API layer budget routes
  - Update transaction display to rely on frontend ID-to-name mapping
  - Remove getUniqueCategories usage in favor of budget store data
  - Clean up any remaining category name-based operations
  - Ensure all category operations use IDs consistently
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 5. Add comprehensive error handling and user feedback
  - Search codebase for existing error handling patterns, notification systems, and user feedback components
  - Implement error boundary components for graceful error handling
  - Add toast notifications or subtle indicators for sync status
  - Implement retry mechanisms for failed operations
  - Add loading states during initial data fetch and budget info loading
  - Ensure all async operations handle network failures gracefully
  - _Requirements: 1.5, 2.5, 3.5, 4.3, 4.4, 5.3, 5.4, 5.5_


-------------
- [ ] 6. Create unit tests for editable components
  - Search codebase for existing component test patterns, testing utilities, and mock implementations
  - Write tests for EditableTextCell component behavior and state management
  - Write tests for EditableCategoryCell dropdown functionality and selection
  - Write tests for EditableApprovalCell approval status changes
  - Test keyboard navigation and accessibility features
  - Test error states and visual feedback
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 7. Create integration tests for store and API interactions
  - Search codebase for existing integration test patterns, API mocking, and store testing examples
  - Test updateTransactionField method with successful updates
  - Test optimistic updates and error reversion scenarios
  - Test async background processing and status updates
  - Test API error handling and user feedback
  - Mock TransactionManager and test sync integration
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.2, 5.3, 5.4_

- [ ] 8. Add backend tests for enhanced TransactionManager
  - Search codebase for existing backend test patterns, service mocking, and Thrift testing examples
  - Test updateTransactions method with sync-out integration
  - Test _syncUpdatedTransactions method behavior
  - Test error handling when YNAB sync fails but local update succeeds
  - Test that existing sync-out code paths are properly reused
  - Verify Thrift exception handling follows defined contracts
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 10. Performance optimization and final polish
  - Search codebase for existing performance optimization patterns, debouncing implementations, and cleanup examples
  - Implement debouncing for rapid successive edits
  - Add batch update capabilities for multiple simultaneous changes
  - Optimize re-rendering during optimistic updates
  - Add proper cleanup for timeouts and async operations
  - Ensure memory leaks are prevented in long-running sessions
  - _Requirements: 4.5, 5.2, 5.5_