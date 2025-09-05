/**
 * @fileoverview Transaction store for managing YNAB transactions
 * Provides reactive state management for transaction data
 */

import { writable, derived, get } from 'svelte/store';
import { transactionApi } from '../api/client.js';
import { settingsStore } from '../stores/settings.js';
import { budgetStore } from './budget.js';

/**
 * Transaction store state
 * @typedef {Object} TransactionStoreState
 * @property {Array} transactions - Array of transaction objects
 * @property {boolean} loading - Loading state
 * @property {string|null} error - Error message if any
 * @property {Object|null} pagination - Pagination information
 * @property {number} maxTransactionsToLoad - Maximum number of transactions to load
 * @property {boolean} hasLoadedForCurrentBudget - Whether transactions have been loaded for current budget
 */


/**
 * Create the main transaction store
 */
function createTransactionStore() {
  const { subscribe, set, update } = writable({
    transactions: [],
    loading: false,
    error: null,
    pagination: null,
    hasLoadedForCurrentBudget: false,
    currentBudgetId: null
  });

  return {
    subscribe,
    
    /**
     * Load transactions with optional parameters
     * @param {Object} params - Query parameters for filtering/pagination
     */
    async load(params = {}) {
      update(state => ({ ...state, loading: true, error: null }));
      
      try {
        // Get current max transactions setting
        const maxTransactions = await settingsStore.getSettingValue("system.max_transactions_to_load");
        
        // Merge max transactions with other params
        const loadParams = {
          maxTransactionsToLoad: maxTransactions,
          ...params
        };
        
        const result = await transactionApi.getAll(loadParams);
        
        if (result.success !== false) {
          update(state => ({
            ...state,
            transactions: result.data || result.transactions || [],
            pagination: result.pagination || null,
            loading: false,
            error: null,
            hasLoadedForCurrentBudget: true
          }));
        } else {
          update(state => ({
            ...state,
            loading: false,
            error: result.error || 'Failed to load transactions'
          }));
        }
      } catch (error) {
        console.error('Error loading transactions:', error);
        update(state => ({
          ...state,
          loading: false,
          error: error.message || 'Failed to load transactions'
        }));
      }
    },

    /**
     * Refresh transactions (reload with current or provided parameters)
     * @param {Object} params - Optional query parameters for filtering/pagination
     */
    async refresh(params = {}) {
      await this.load(params);
    },

    /**
     * Refresh a single transaction by ID
     * @param {string} transactionId - Transaction ID to refresh
     */
    async refreshTransaction(transactionId) {
      try {
        const result = await transactionApi.getById(transactionId);
        
        if (result.success !== false && result.data) {
          // Merge server data with local data to preserve optimistic updates
          update(state => ({
            ...state,
            transactions: state.transactions.map(t => {
              if (t.id === transactionId) {
                // Preserve any fields that might have pending optimistic updates
                // by checking if they're different from what we expect
                const serverData = result.data;
                const localData = t;
                
                // Merge server data first, then preserve any local changes that seem newer
                const merged = { ...serverData, ...localData };
                
                // Only update core fields from server, preserve UI state and optimistic changes
                return {
                  ...merged,
                  // Always use server data for metadata (emails) and core transaction fields
                  metadata: serverData.metadata || [],
                  date: serverData.date,
                  amount: serverData.amount,
                  payeeId: serverData.payeeId,
                  accountId: serverData.accountId,
                  memo: serverData.memo,
                  // Preserve optimistic updates for category and approval if they seem recent
                  categoryId: localData._saveStatus === 'saving' ? localData.categoryId : serverData.categoryId,
                  approved: localData._saveStatus === 'saving' ? localData.approved : serverData.approved,
                };
              }
              return t;
            })
          }));
        }
      } catch (error) {
        console.error('Error refreshing transaction:', error);
        // Don't update error state for single transaction refresh failures
      }
    },

    /**
     * Reset the store to initial state
     */
    reset() {
      set({
        transactions: [],
        loading: false,
        error: null,
        pagination: null,
        hasLoadedForCurrentBudget: false,
        currentBudgetId: null
      });
    },

    /**
     * Mark that transactions should be reloaded for the current budget
     * Called when budget changes or when forced reload is needed
     */
    markForReload() {
      update(state => ({
        ...state,
        hasLoadedForCurrentBudget: false
      }));
    },

    /**
     * Set the current budget ID and mark for reload if it changed
     * @param {string|null} budgetId - Current budget ID
     */
    setBudgetId(budgetId) {
      update(state => {
        const budgetChanged = state.currentBudgetId !== budgetId;
        return {
          ...state,
          currentBudgetId: budgetId,
          hasLoadedForCurrentBudget: budgetChanged ? false : state.hasLoadedForCurrentBudget
        };
      });
    },

    /**
     * Update a single transaction in the store
     * @param {Object} updatedTransaction - Updated transaction object
     */
    updateTransaction(updatedTransaction) {
      update(state => ({
        ...state,
        transactions: state.transactions.map(t => 
          t.id === updatedTransaction.id ? updatedTransaction : t
        )
      }));
    },

    /**
     * Add a new transaction to the store
     * @param {Object} newTransaction - New transaction object
     */
    addTransaction(newTransaction) {
      update(state => ({
        ...state,
        transactions: [newTransaction, ...state.transactions]
      }));
    },

    /**
     * Remove a transaction from the store
     * @param {string} transactionId - Transaction ID to remove
     */
    removeTransaction(transactionId) {
      update(state => ({
        ...state,
        transactions: state.transactions.filter(t => t.id !== transactionId)
      }));
    },

    /**
     * Update a single transaction field with optimistic updates and non-blocking UI
     * @param {string} transactionId - Transaction ID
     * @param {Object} changes - Fields to update
     * @returns {Promise<boolean>} Success status
     */
    async updateTransactionField(transactionId, changes) {
      // Find the existing transaction to preserve all its data
      let existingTransaction = null;
      update(state => {
        existingTransaction = state.transactions.find(t => t.id === transactionId);
        return state;
      });
      
      if (!existingTransaction) {
        console.warn(`Transaction ${transactionId} not found for update`);
        return false;
      }
      
      // Optimistic update - merge changes with existing transaction data
      const optimisticTransaction = { ...existingTransaction, ...changes };
      this.updateTransaction(optimisticTransaction);
      
      // Fire-and-forget async update - don't block UI
      // Use setTimeout to ensure the optimistic update renders first
      setTimeout(() => {
        this._performAsyncUpdate(transactionId, changes);
      }, 0);
      
      return true; // Return immediately for responsive UI
    },

    /**
     * Perform async update without blocking UI
     * @private
     */
    async _performAsyncUpdate(transactionId, changes) {
      try {
        const result = await transactionApi.update(transactionId, changes);
        if (result.success !== false) {
          // Update with server response, but preserve any newer local changes
          const serverTransaction = result.data || result;
          update(state => ({
            ...state,
            transactions: state.transactions.map(t => {
              if (t.id === transactionId) {
                // Only update fields that were part of the original changes
                // This prevents overwriting newer optimistic updates
                const updatedFields = {};
                Object.keys(changes).forEach(key => {
                  if (serverTransaction.hasOwnProperty(key)) {
                    updatedFields[key] = serverTransaction[key];
                  }
                });
                return { ...t, ...updatedFields };
              }
              return t;
            })
          }));
          // Show subtle success indicator
          this._showTransientSuccess(transactionId);
        } else {
          // Revert optimistic update and show error
          await this.refreshTransaction(transactionId);
          this._showTransientError(transactionId, result.error || 'Update failed');
        }
      } catch (error) {
        // Revert optimistic update and show error
        await this.refreshTransaction(transactionId);
        this._showTransientError(transactionId, error.message);
      }
    },

    /**
     * Show transient success indicator
     * @private
     */
    _showTransientSuccess(transactionId) {
      // Update transaction with temporary success state
      update(state => ({
        ...state,
        transactions: state.transactions.map(t => 
          t.id === transactionId 
            ? { ...t, _saveStatus: 'success', _saveTimestamp: Date.now() }
            : t
        )
      }));
      
      // Clear success indicator after 2 seconds
      setTimeout(() => {
        update(state => ({
          ...state,
          transactions: state.transactions.map(t => 
            t.id === transactionId 
              ? { ...t, _saveStatus: null, _saveTimestamp: null }
              : t
          )
        }));
      }, 2000);
    },

    /**
     * Show transient error indicator
     * @private
     */
    _showTransientError(transactionId, errorMessage) {
      // Update transaction with temporary error state
      update(state => ({
        ...state,
        transactions: state.transactions.map(t => 
          t.id === transactionId 
            ? { ...t, _saveStatus: 'error', _saveError: errorMessage, _saveTimestamp: Date.now() }
            : t
        )
      }));
      
      // Clear error indicator after 5 seconds
      setTimeout(() => {
        update(state => ({
          ...state,
          transactions: state.transactions.map(t => 
            t.id === transactionId 
              ? { ...t, _saveStatus: null, _saveError: null, _saveTimestamp: null }
              : t
          )
        }));
      }, 5000);
    }
  };
}

/**
 * Main transaction store instance
 */
export const transactionStore = createTransactionStore();

/**
 * Enhanced transactions derived store
 * Adds computed properties and formatting to transactions
 */
export const enhancedTransactions = derived(
  [transactionStore, budgetStore],
  ([$transactionStore, $budgetStore]) => {
    if (!$transactionStore.transactions) return [];
    
    return $transactionStore.transactions.map(transaction => {
      // Get resolved names from budget store
      const payeeName = getPayeeNameFromBudget(transaction, $budgetStore);
      const categoryName = getCategoryNameFromBudget(transaction, $budgetStore);
      const accountName = getAccountNameFromBudget(transaction, $budgetStore);
      
      return {
        ...transaction,
  // Signal whether budget metadata has finished hydrating
  budgetInitialized: Boolean($budgetStore?.isInitialized),
        
        // Format amount for display
        formattedAmount: new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD'
        }).format((transaction.amount || 0) / 1000),
        
        // Format date for display
        formattedDate: transaction.date ? 
          new Date(transaction.date).toLocaleDateString() : '',
        
        // Determine if transaction is income or expense
        isIncome: (transaction.amount || 0) > 0,
        isExpense: (transaction.amount || 0) < 0,
        
        // Get payee name (with fallback)
        payeeName: payeeName,
        
        // Get category name (with fallback)
        categoryName: categoryName,
        
        // Get account name (with fallback)
        accountName: accountName,
        
        // String versions for filtering
        approvedString: transaction.approved ? 'true' : 'false',
        
        // Check if transaction has email attachments
        hasEmailAttachments: hasEmailMetadata(transaction),
        
        // Count of email attachments
        emailAttachmentCount: getEmailAttachmentCount(transaction),
        
        // Check if transaction has AI predictions
        hasAIPredictions: hasAIMetadata(transaction),
        
        // Create searchable text combining all relevant fields
        searchableText: createSearchableText(transaction, payeeName, categoryName, accountName)
      };
    });
  }
);

/**
 * Helper function to get payee name from budget store
 * @param {Object} transaction - Transaction object
 * @param {Object} budgetData - Budget store data
 * @returns {string} Payee name
 */
export function getPayeeNameFromBudget(transaction, budgetData) {
  // If budget isn't initialized yet, avoid showing fallback labels to reduce flicker
  const budgetReady = Boolean(budgetData?.isInitialized);
  // Try to get payee name from metadata first
  if (transaction.metadata && Array.isArray(transaction.metadata)) {
    for (const meta of transaction.metadata) {
      if (meta.type === 2 && meta.value && meta.value.stringValue) {
        try {
          const displayData = JSON.parse(meta.value.stringValue);
          if (displayData.payeeName) {
            return displayData.payeeName;
          }
        } catch (e) {
          // Continue to next metadata item
        }
      }
    }
  }
  
  // Try to get from budget store
  if (transaction.payeeId && budgetData.payees) {
    const payee = budgetData.payees.find(p => p.id === transaction.payeeId);
    if (payee) {
      return payee.name;
    }
  }
  
  // If budget not ready, return empty string to avoid 'Unknown' flicker
  if (!budgetReady) return '';

  // Fallback to payeeId or Unknown once initialized
  return transaction.payeeId || 'Unknown';
}

/**
 * Helper function to get category name from budget store
 * @param {Object} transaction - Transaction object
 * @param {Object} budgetData - Budget store data
 * @returns {string} Category name
 */
function getCategoryNameFromBudget(transaction, budgetData) {
  const budgetReady = Boolean(budgetData?.isInitialized);
  // Try to get category name from metadata first
  if (transaction.metadata && Array.isArray(transaction.metadata)) {
    for (const meta of transaction.metadata) {
      if (meta.type === 2 && meta.value && meta.value.stringValue) {
        try {
          const displayData = JSON.parse(meta.value.stringValue);
          if (displayData.categoryName) {
            return displayData.categoryName;
          }
        } catch (e) {
          // Continue to next metadata item
        }
      }
    }
  }
  
  // Try to get from budget store
  if (transaction.categoryId && budgetData.categories) {
    const category = budgetData.categories.find(c => c.id === transaction.categoryId);
    if (category) {
      return category.name;
    }
  }
  
  // If budget not ready, return empty string to avoid 'Uncategorized' flicker
  if (!budgetReady) return '';

  // Fallback to categoryId or Uncategorized once initialized
  return transaction.categoryId || 'Uncategorized';
}

/**
 * Helper function to get account name from budget store
 * @param {Object} transaction - Transaction object
 * @param {Object} budgetData - Budget store data
 * @returns {string} Account name
 */
function getAccountNameFromBudget(transaction, budgetData) {
  const budgetReady = Boolean(budgetData?.isInitialized);
  // Try to get account name from metadata first
  if (transaction.metadata && Array.isArray(transaction.metadata)) {
    for (const meta of transaction.metadata) {
      if (meta.type === 2 && meta.value && meta.value.stringValue) {
        try {
          const displayData = JSON.parse(meta.value.stringValue);
          if (displayData.accountName) {
            return displayData.accountName;
          }
        } catch (e) {
          // Continue to next metadata item
        }
      }
    }
  }
  
  // Try to get from budget store
  if (transaction.accountId && budgetData.accounts) {
    const account = budgetData.accounts.find(a => a.id === transaction.accountId);
    if (account) {
      return account.name;
    }
  }
  
  if (!budgetReady) return '';

  // Fallback to accountId or Unknown Account once initialized
  return transaction.accountId || 'Unknown Account';
}

/**
 * Helper function to check if transaction has email metadata
 * @param {Object} transaction - Transaction object
 * @returns {boolean} True if has email attachments
 */
function hasEmailMetadata(transaction) {
  if (!transaction.metadata || !Array.isArray(transaction.metadata)) {
    return false;
  }
  
  return transaction.metadata.some(meta => 
    meta.type === 'Email' || meta.type === 1
  );
}

/**
 * Helper function to count email attachments
 * @param {Object} transaction - Transaction object
 * @returns {number} Number of email attachments
 */
function getEmailAttachmentCount(transaction) {
  if (!transaction.metadata || !Array.isArray(transaction.metadata)) {
    return 0;
  }
  
  return transaction.metadata.filter(meta => 
    meta.type === 'Email' || meta.type === 1
  ).length;
}

/**
 * Helper function to check if transaction has AI predictions
 * @param {Object} transaction - Transaction object
 * @returns {boolean} True if has AI predictions
 */
function hasAIMetadata(transaction) {
  if (!transaction.metadata || !Array.isArray(transaction.metadata)) {
    return false;
  }
  
  return transaction.metadata.some(meta => 
    meta.type === 'Prediction' || meta.type === 2
  );
}

/**
 * Helper function to create searchable text from transaction fields
 * @param {Object} transaction - Transaction object
 * @param {string} payeeName - Resolved payee name
 * @param {string} categoryName - Resolved category name
 * @param {string} accountName - Resolved account name
 * @returns {string} Combined searchable text
 */
function createSearchableText(transaction, payeeName, categoryName, accountName) {
  const searchFields = [];
  
  // Add memo/description
  if (transaction.memo) {
    searchFields.push(transaction.memo);
  }
  
  // Add resolved names (these are the most important for search)
  if (payeeName && payeeName !== 'Unknown') {
    searchFields.push(payeeName);
  }
  
  if (categoryName && categoryName !== 'Uncategorized') {
    searchFields.push(categoryName);
  }
  
  if (accountName && accountName !== 'Unknown Account') {
    searchFields.push(accountName);
  }
  
  // Add IDs as fallback (in case names aren't available)
  if (transaction.payeeId) {
    searchFields.push(transaction.payeeId);
  }
  
  if (transaction.categoryId) {
    searchFields.push(transaction.categoryId);
  }
  
  if (transaction.accountId) {
    searchFields.push(transaction.accountId);
  }
  
  // Add budget ID (which might contain meaningful text)
  if (transaction.budgetId) {
    searchFields.push(transaction.budgetId);
  }
  
  // Add formatted amount for searching by amount
  if (transaction.amount !== undefined && transaction.amount !== null) {
    const formattedAmount = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(Math.abs(transaction.amount) / 1000);
    searchFields.push(formattedAmount);
    
    // Also add the raw amount as a string for exact searches
    searchFields.push((Math.abs(transaction.amount) / 1000).toString());
  }
  
  // Add date in various formats for date searching
  if (transaction.date) {
    const date = new Date(transaction.date);
    searchFields.push(date.toLocaleDateString()); // MM/DD/YYYY
    searchFields.push(date.toISOString().split('T')[0]); // YYYY-MM-DD
    searchFields.push(date.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })); // Month DD, YYYY
  }
  
  // Add approval status
  if (transaction.approved !== undefined) {
    searchFields.push(transaction.approved ? 'approved' : 'unapproved');
  }
  
  // Add cleared status
  if (transaction.cleared) {
    searchFields.push(transaction.cleared);
  }
  
  // Add platform type
  if (transaction.platformType !== undefined) {
    // Convert platform type number to readable text
    const platformNames = {
      1: 'YNAB'
    };
    const platformName = platformNames[transaction.platformType] || `Platform${transaction.platformType}`;
    searchFields.push(platformName);
  }
  
  // Add metadata content if available
  if (transaction.metadata && Array.isArray(transaction.metadata)) {
    transaction.metadata.forEach(meta => {
      // Add email metadata
      if (meta.type === 'Email' || meta.type === 1) {
        if (meta.value && meta.value.stringValue) {
          try {
            const emailData = JSON.parse(meta.value.stringValue);
            if (emailData.subject) {
              searchFields.push(emailData.subject);
            }
            if (emailData.sender) {
              searchFields.push(emailData.sender);
            }
            if (emailData.snippet) {
              searchFields.push(emailData.snippet);
            }
          } catch (e) {
            // If content isn't JSON, add it as-is
            searchFields.push(meta.value.stringValue);
          }
        }
      }
      
      // Add display names metadata (type 2)
      if (meta.type === 2) {
        if (meta.value && meta.value.stringValue) {
          try {
            const displayData = JSON.parse(meta.value.stringValue);
            if (displayData.payeeName) {
              searchFields.push(displayData.payeeName);
            }
            if (displayData.categoryName) {
              searchFields.push(displayData.categoryName);
            }
            if (displayData.accountName) {
              searchFields.push(displayData.accountName);
            }
          } catch (e) {
            // If content isn't JSON, add it as-is
            searchFields.push(meta.value.stringValue);
          }
        }
        
        // Also check for AI prediction results
        if (meta.value && meta.value.predictionResult) {
          const prediction = meta.value.predictionResult;
          if (prediction.predictedCategory) {
            searchFields.push(prediction.predictedCategory);
          }
        }
      }
    });
  }
  
  // Join all fields with spaces and convert to lowercase for case-insensitive search
  return searchFields.join(' ').toLowerCase();
}