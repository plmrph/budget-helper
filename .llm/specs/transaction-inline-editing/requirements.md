# Requirements Document

## Introduction

This feature restores and enhances the inline editing functionality for the TransactionDataTable component. Users should be able to click on memo, category, and approval fields within the transaction table to edit them directly, with changes automatically saved to both the local database and synced to YNAB.

## Requirements

### Requirement 1

**User Story:** As a user, I want to edit transaction memo fields directly in the table, so that I can quickly update transaction descriptions without opening separate dialogs.

#### Acceptance Criteria

1. WHEN I click on a memo cell in the transaction table THEN the cell SHALL become editable with a text input field
2. WHEN I type in the memo field THEN the input SHALL accept text up to reasonable limits (e.g., 00 characters)
3. WHEN the row loses focus OR I press Enter THEN the memo change SHALL be saved automatically
5. IF the save fails THEN an error message SHALL be displayed and the field SHALL revert to the original value

### Requirement 2

**User Story:** As a user, I want to edit transaction categories directly in the table, so that I can quickly recategorize transactions using a dropdown selection.

#### Acceptance Criteria

1. WHEN I click on a category cell in the transaction table THEN a ComboBox dropdown SHALL appear with ALL available categories from my budget
2. WHEN I select a category from the dropdown THEN the selection SHALL be applied immediately using the category ID
3. WHEN the row loses focus THEN the category change SHALL be saved automatically with the correct category ID
4. WHEN a category is not set THEN the cell SHALL display "Uncategorized" instead of "Select Category"
5. IF the save fails THEN an error message SHALL be displayed and the field SHALL revert to the original value
6. WHEN I type in the category ComboBox THEN it SHALL filter available categories based on my input
7. WHEN the page is loading THEN the system SHALL fetch all available categories, accounts, and payees from the TransactionManager (who in turn will pull them from the database or from the budgeting platform)

### Requirement 3

**User Story:** As a user, I want to edit transaction approval status directly in the table, so that I can quickly approve or unapprove transactions.

#### Acceptance Criteria

1. WHEN I click on an approval status cell THEN a ComboBox dropdown SHALL appear with "Approved" and "Unapproved" options
2. WHEN I select an approval status THEN the selection SHALL be applied immediately
3. WHEN the row loses focus OR a status is selected THEN the approval change SHALL be saved automatically
5. IF the save fails THEN an error message SHALL be displayed and the field SHALL revert to the original value

### Requirement 4

**User Story:** As a user, I want my transaction edits to be automatically synced to YNAB, so that my changes are reflected in my budgeting platform without manual sync operations.

#### Acceptance Criteria

1. WHEN a transaction field is successfully saved to the local database THEN the system SHALL automatically trigger a sync to YNAB
2. WHEN the YNAB sync is in progress THEN a subtle indicator SHALL show the sync status somewhere in a corner or something
3. IF the YNAB sync fails THEN a warning message SHALL be displayed but the local change SHALL remain
4. WHEN the YNAB sync succeeds THEN any sync indicators SHALL be cleared
5. WHEN multiple edits happen quickly THEN the system SHALL batch sync operations to avoid excessive API calls

### Requirement 5

**User Story:** As a user, I want visual feedback during editing operations, so that I understand the current state of my changes.

#### Acceptance Criteria

1. WHEN a field is in edit mode THEN it SHALL have a distinct visual appearance (e.g., border highlight)
3. WHEN a save operation succeeds THEN a brief success indicator SHALL be shown
4. WHEN a save operation fails THEN an error state SHALL be visually indicated
5. WHEN a field reverts due to error THEN the reversion SHALL be visually smooth

### Requirement 6

**User Story:** As a developer, I want the inline editing to follow The Method architecture principles, so that the code is maintainable and follows established patterns.

#### Acceptance Criteria

1. WHEN implementing the update functionality THEN it SHALL use the TransactionManager service for orchestration
2. WHEN saving to the database THEN it SHALL use the DatabaseStoreAccess service
3. WHEN syncing to YNAB THEN it SHALL use the BudgetingPlatformAccess service
4. WHEN the sync operation is triggered THEN it SHALL reuse existing sync-out code paths
5. WHEN handling errors THEN it SHALL use Thrift-defined exception types
6. WHEN fetching budget information THEN it SHALL use the TransactionManager.getBudgetsInfo method
7. WHEN the API layer needs budget data THEN it SHALL call TransactionManager, not directly access BudgetingPlatformAccess

### Requirement 7

**User Story:** As a developer, I want to eliminate architectural violations and improve data consistency, so that the system follows proper layering and has reliable category/payee/account data.

#### Acceptance Criteria

1. WHEN the frontend needs category/payee/account data THEN it SHALL fetch this via a proper API endpoint that calls TransactionManager.getBudgetsInfo
2. WHEN the API layer needs budget information THEN it SHALL NOT directly call BudgetingPlatformAccess but SHALL use TransactionManager
3. WHEN transactions are displayed THEN the system SHALL NOT use _enrichTransactionsWithNames but SHALL rely on frontend ID-to-name mapping
4. WHEN category selection occurs THEN the frontend SHALL send category IDs, not category names
5. WHEN the system starts up THEN budget information SHALL be cached on the frontend for efficient lookups