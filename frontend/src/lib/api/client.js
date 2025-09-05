/**
 * @fileoverview API client for backend communication
 * Handles all HTTP requests to the FastAPI backend
 */

/**
 * Base API configuration
 */
const API_BASE_URL = typeof window !== 'undefined' ? '/api' : 'http://localhost:8000/api';

/**
 * HTTP request options
 * @typedef {Object} RequestOptions
 * @property {string} method - HTTP method
 * @property {Object} [headers] - Request headers
 * @property {string|FormData} [body] - Request body
 */

/**
 * API response wrapper
 * @template T
 * @typedef {Object} ApiResponse
 * @property {boolean} success - Request success status
 * @property {T} [data] - Response data
 * @property {string} [error] - Error message
 * @property {number} [status] - HTTP status code
 */

/**
 * Loading state store
 */
import { writable } from 'svelte/store';
export const isLoading = writable(false);
export const apiErrors = writable([]);

/**
 * Add error to error store
 * @param {string} error - Error message
 */
function addError(error) {
  apiErrors.update(errors => [...errors, { id: Date.now(), message: error, timestamp: new Date() }]);
}

/**
 * Make HTTP request to API
 * @param {string} endpoint - API endpoint
 * @param {RequestOptions} [options] - Request options
 * @returns {Promise<ApiResponse>}
 */
async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;

  const defaultHeaders = {
    'Content-Type': 'application/json',
  };

  const config = {
    method: options.method || 'GET',
    headers: { ...defaultHeaders, ...options.headers },
    ...options
  };

  isLoading.set(true);

  try {
    const response = await fetch(url, config);
    const status = response.status;

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = errorData.detail || errorData.message || `HTTP ${response.status}: ${response.statusText}`;
      addError(errorMessage);
      return { success: false, error: errorMessage, status };
    }

    const data = await response.json();
    return { ...data, status };
  } catch (error) {
    console.error(`API request failed: ${endpoint}`, error);
    const errorMessage = error.message || 'Network error occurred';
    addError(errorMessage);
    return { success: false, error: errorMessage };
  } finally {
    isLoading.set(false);
  }
}

/**
 * Authentication API methods
 */
export const authApi = {
  /**
   * Get overall authentication status
   * @returns {Promise<ApiResponse<{ynab: boolean, gmail: boolean}>>}
   */
  async getStatus() {
    return apiRequest('/auth/status');
  },

  /**
   * YNAB authentication methods
   */
  ynab: {
    /**
     * Connect to YNAB with personal access token
     * @param {string} token - YNAB personal access token
     * @returns {Promise<ApiResponse<{success: boolean}>>}
     */
    async connect(token) {
      return apiRequest('/auth/ynab/connect', {
        method: 'POST',
        body: JSON.stringify({ personal_access_token: token })
      });
    },

    /**
     * Disconnect from YNAB
     * @returns {Promise<ApiResponse<{success: boolean}>>}
     */
    async disconnect() {
      return apiRequest('/auth/ynab/disconnect', { method: 'POST' });
    },

    /**
     * Get YNAB authentication status
     * @returns {Promise<ApiResponse<{authenticated: boolean}>>}
     */
    async getStatus() {
      return apiRequest('/auth/ynab/status');
    }
  },

  /**
   * Gmail authentication methods
   */
  gmail: {
    /**
     * Connect to Gmail with OAuth configuration
     * @param {Object} [config] - OAuth configuration
     * @returns {Promise<ApiResponse<{auth_url: string}>>}
     */
    async connect(config = {}) {
      const defaultConfig = {
        client_id: "your_client_id",
        client_secret: "your_client_secret",
        redirect_uri: `${window.location.origin}/api/auth/gmail/callback`,
        scopes: ["https://www.googleapis.com/auth/gmail.readonly"]
      };

      return apiRequest('/auth/gmail/connect', {
        method: 'POST',
        body: JSON.stringify({ ...defaultConfig, ...config })
      });
    },

    /**
     * Disconnect from Gmail
     * @returns {Promise<ApiResponse<{success: boolean}>>}
     */
    async disconnect() {
      return apiRequest('/auth/gmail/disconnect', { method: 'POST' });
    },

    /**
     * Get Gmail authentication status
     * @returns {Promise<ApiResponse<{authenticated: boolean}>>}
     */
    async getStatus() {
      return apiRequest('/auth/gmail/status');
    }
  }
};

/**
 * Transaction API methods
 */
export const transactionApi = {
  /**
   * Get all transactions with server-side filtering, pagination, and sorting
   * @param {Object} [params] - Query parameters
   * @param {string} [params.account_id] - Filter by account ID
   * @param {string} [params.category_id] - Filter by category ID
   * @param {string} [params.payee_name] - Filter by payee name (partial match)
   * @param {boolean} [params.approved] - Filter by approval status
   * @param {string} [params.cleared] - Filter by cleared status
   * @param {boolean} [params.deleted] - Include deleted transactions
   * @param {boolean} [params.has_email] - Filter by email attachment status
   * @param {boolean} [params.has_ai_category] - Filter by AI category prediction status
   * @param {string} [params.date_from] - Filter transactions from this date (ISO string)
   * @param {string} [params.date_to] - Filter transactions to this date (ISO string)
   * @param {number} [params.amount_min] - Minimum amount in milliunits
   * @param {number} [params.amount_max] - Maximum amount in milliunits
   * @param {number} [params.limit] - Maximum number of transactions to return (1-1000)
   * @param {number} [params.offset] - Number of transactions to skip
   * @param {string} [params.sort_by] - Field to sort by (date, amount, payee_name, approved)
   * @param {string} [params.sort_order] - Sort order (asc, desc)
   * @returns {Promise<ApiResponse<{data: Transaction[], pagination: Object}>>}
   */
  async getAll(params = {}) {
    // Clean up params - remove null/undefined values and map frontend params to backend params
    const cleanParams = {};
    Object.keys(params).forEach(key => {
      if (params[key] !== null && params[key] !== undefined && params[key] !== '') {
        // Map frontend parameter names to backend parameter names
        if (key === 'maxTransactionsToLoad') {
          cleanParams['limit'] = params[key];
        } else if (key === 'sorting' && Array.isArray(params[key]) && params[key].length > 0) {
          // Convert frontend sorting array to backend sort_by and sort_order
          const firstSort = params[key][0];
          if (firstSort.column) {
            cleanParams['sort_by'] = firstSort.column;
            cleanParams['sort_order'] = firstSort.direction || 'desc';
          }
        } else {
          cleanParams[key] = params[key];
        }
      }
    });

    const queryString = new URLSearchParams(cleanParams).toString();
    const endpoint = queryString ? `/transactions/?${queryString}` : '/transactions/';
    return apiRequest(endpoint);
  },

  /**
   * Get transaction by ID
   * @param {string} id - Transaction ID
   * @returns {Promise<ApiResponse<Transaction>>}
   */
  async getById(id) {
    return apiRequest(`/transactions/${id}`);
  },

  /**
   * Update transaction
   * @param {string} id - Transaction ID
   * @param {Partial<Transaction>} updates - Transaction updates
   * @returns {Promise<ApiResponse<Transaction>>}
   */
  async update(id, updates) {
    // Map frontend camelCase keys to backend snake_case where required
    const payload = {
      // pass through approved & memo as-is
      approved: updates.approved,
      memo: updates.memo,
      // backend expects category_id, not categoryId
      category_id: updates.category_id ?? updates.categoryId,
      // optionally map payee_name if ever sent here
      payee_name: updates.payee_name ?? updates.payeeName,
      // ignore other fields for now
    };

    // Remove undefined keys to avoid sending nulls
    Object.keys(payload).forEach((k) => payload[k] === undefined && delete payload[k]);

    return apiRequest(`/transactions/${id}`, {
      method: 'PUT',
      body: JSON.stringify(payload)
    });
  },

  /**
   * Update multiple transactions
   * @param {Array<{id: string, updates: Partial<Transaction>}>} transactions - Transactions to update
   * @returns {Promise<ApiResponse<Transaction[]>>}
   */
  async updateBatch(transactions) {
    return apiRequest('/transactions/batch', {
      method: 'PUT',
      body: JSON.stringify(transactions)
    });
  },

  /**
   * Get transaction history
   * @param {string} id - Transaction ID
   * @returns {Promise<ApiResponse<HistoryEntry[]>>}
   */
  async getHistory(id) {
    return apiRequest(`/transactions/${id}/history`);
  },

  /**
   * Sync transactions from YNAB (import)
   * @param {string} [budgetPlatform] - Budget platform type (defaults to YNAB)
   * @returns {Promise<ApiResponse<{results: Array, batchStatus: string}>>}
   */
  async syncFromYnab(budgetPlatform = 'YNAB') {
    return apiRequest('/transactions/sync/in', {
      method: 'POST',
      body: JSON.stringify({ budgetPlatform })
    });
  },

  /**
   * Preview import from YNAB without applying
   */
  async previewImport(budgetPlatform = 'YNAB') {
    return apiRequest('/transactions/sync/in/preview', {
      method: 'POST',
      body: JSON.stringify({ budgetPlatform })
    });
  },

  /**
   * Apply selected import diffs
   * @param {{add:string[], update:string[], delete:string[]}} selection
   */
  async applyImport(selection) {
    return apiRequest('/transactions/sync/in/apply', {
      method: 'POST',
      body: JSON.stringify(selection)
    });
  },

  /**
   * Sync transactions to YNAB (export)
   * @param {string} [budgetPlatform] - Budget platform type (defaults to YNAB)
   * @returns {Promise<ApiResponse<{results: Array, batchStatus: string}>>}
   */
  async syncToYnab(budgetPlatform = 'YNAB') {
    return apiRequest('/transactions/sync/out', {
      method: 'POST',
      body: JSON.stringify({ budgetPlatform })
    });
  },

  /**
   * Preview export to YNAB without applying
   */
  async previewExport(budgetPlatform = 'YNAB') {
    return apiRequest('/transactions/sync/out/preview', {
      method: 'POST',
      body: JSON.stringify({ budgetPlatform })
    });
  },

  /**
   * Apply selected export diffs
   * @param {{add:string[], update:string[], delete:string[]}} selection
   */
  async applyExport(selection) {
    return apiRequest('/transactions/sync/out/apply', {
      method: 'POST',
      body: JSON.stringify(selection)
    });
  },

  /**
   * Unified preview for Sync Budget dialog (lightweight, paginated later)
   */
  async previewUnified(payload = {}) {
    return apiRequest('/transactions/sync/preview', { method: 'POST', body: JSON.stringify(payload || {}) });
  },

  /**
   * Apply unified sync plan: { plan: [{ id, action: 'left'|'right' }] }
   */
  async applyUnified(plan) {
    return apiRequest('/transactions/sync/apply', {
      method: 'POST',
      body: JSON.stringify(plan)
    });
  }
  ,
  /** Reset sync tracking on the server to start from now */
  async resetSyncTracking() {
    return apiRequest('/transactions/sync/reset-tracking', { method: 'POST', body: JSON.stringify({}) });
  }
};

/**
 * Email API methods
 */
export const emailApi = {
  /**
   * Search emails for a specific transaction
   * @param {string} transactionId - Transaction ID to search emails for
   * @param {string} [customQuery] - Optional custom search query
   * @returns {Promise<ApiResponse<{emails: Email[], total_found: number}>>}
   */
  async searchForTransaction(transactionId, customQuery = null) {
    const queryParams = customQuery ? `?q=${encodeURIComponent(customQuery)}` : '';
    return apiRequest(`/email-search/${transactionId}/search${queryParams}`);
  },

  /**
   * Attach email to transaction
   * @param {string} transactionId - Transaction ID
   * @param {Object} emailData - Email data to attach
   * @param {string} emailData.email_id - Email ID
   * @param {string} emailData.email_subject - Email subject
   * @param {string} emailData.email_sender - Email sender
   * @param {string} emailData.email_date - Email date
   * @param {string} [emailData.email_snippet] - Email snippet
   * @param {string} [emailData.email_body_text] - Email body text
   * @param {string} [emailData.email_body_html] - Email body HTML
   * @param {string} [emailData.email_url] - Email URL
   * @returns {Promise<ApiResponse<{success: boolean}>>}
   */
  async attachToTransaction(transactionId, emailData) {
    return apiRequest(`/email-search/${transactionId}/attach-email`, {
      method: 'POST',
      body: JSON.stringify(emailData)
    });
  },

  /**
   * Detach email from transaction
   * @param {string} transactionId - Transaction ID
   * @param {string} emailId - Email ID
   * @returns {Promise<ApiResponse<{success: boolean}>>}
   */
  async detachFromTransaction(transactionId, emailId) {
    return apiRequest(`/email-search/${transactionId}/detach-email`, {
      method: 'POST',
      body: JSON.stringify({ email_id: emailId })
    });
  },

  /**
   * Bulk search emails for multiple transactions
   * @param {Array<string>} transactionIds - Array of transaction IDs to search emails for
   * @returns {Promise<ApiResponse<{results: Object, total_processed: number, total_attached: number}>>}
   */
  async bulkSearchForTransactions(transactionIds) {
    return apiRequest('/email-search/bulk-search', {
      method: 'POST',
      body: JSON.stringify({ transaction_ids: transactionIds })
    });
  }
};

/**
 * AI/ML API methods
 */
export const mlApi = {
  /**
   * Get category prediction for transactions
   * @param {Object} requestData - Prediction request data
   * @param {Array<string>} requestData.transaction_ids - Array of transaction IDs to predict
   * @param {string} [requestData.ml_model_name] - Optional specific model to use
   * @returns {Promise<ApiResponse<{predictions: Array}>>}
   */
  async predict(requestData) {
    return apiRequest('/ml/predict', {
      method: 'POST',
      body: JSON.stringify(requestData)
    });
  },

  /**
   * Get category predictions for multiple transactions
   * @param {Array} transactions - Array of transaction data
   * @returns {Promise<ApiResponse<{predictions: Array}>>}
   */
  async predictBatch(transactions) {
    return apiRequest('/ml/predict/batch', {
      method: 'POST',
      body: JSON.stringify({ transactions })
    });
  },

  /**
   * Train ML model asynchronously
   * @param {Object} trainingParams - Training parameters
   * @param {string} trainingParams.ml_model_name - Name for the model
   * @param {string} trainingParams.ml_model_type - Type of model to train
   * @param {string} trainingParams.training_data_location - Path to training data
   * @param {Object} [trainingParams.training_params] - Additional training parameters
   * @returns {Promise<ApiResponse<{status: string, model_name: string}>>}
   */
  async train(trainingParams) {
    return apiRequest('/ml/train', {
      method: 'POST',
      body: JSON.stringify(trainingParams)
    });
  },

  /**
   * Get current training status (if any training is in progress)
   * @returns {Promise<ApiResponse<{current_training: string, training_active: boolean, status: Object}>>}
   */
  async getCurrentTrainingStatus() {
    return apiRequest('/ml/train/status');
  },

  /**
   * Get training status for a specific model
   * @param {string} modelName - Model name to check status for
   * @returns {Promise<ApiResponse<{status: string, progress: number, message: string}>>}
   */
  async getTrainingStatus(modelName) {
    const encodedModelName = encodeURIComponent(modelName);
    return apiRequest(`/ml/train/status/${encodedModelName}`);
  },

  /**
   * Get available models
   * @returns {Promise<ApiResponse<Array>>}
   */
  async getModels() {
    return apiRequest('/ml/models');
  },

  /**
   * Load a specific model
   * @param {string} modelName - Model name to load
   * @returns {Promise<ApiResponse<{success: boolean}>>}
   */
  async loadModel(modelName) {
    return apiRequest(`/ml/models/${modelName}/load`, { method: 'POST' });
  },

  /**
   * Delete a model
   * @param {string} modelName - Model name to delete
   * @returns {Promise<ApiResponse<{success: boolean}>>}
   */
  async deleteModel(modelName) {
    const encodedModelName = encodeURIComponent(modelName);
    return apiRequest(`/ml/models/${encodedModelName}`, { method: 'DELETE' });
  },

  /**
   * Set a model as the default for future predictions
   * @param {string} modelName - Model name to set as default
   * @returns {Promise<ApiResponse<{success: boolean}>>}
   */
  async setDefaultModel(modelName) {
    return apiRequest(`/ml/models/${modelName}/set-default`, { method: 'POST' });
  },

  /**
   * Cancel training for a model and delete the model record
   * @param {string} modelName - Model name to cancel training for
   * @returns {Promise<ApiResponse<{success: boolean}>>}
   */
  async cancelTraining(modelName) {
    const encodedModelName = encodeURIComponent(modelName);
    return apiRequest(`/ml/train/cancel/${encodedModelName}`, { method: 'POST' });
  },

  /**
   * Backup a model
   * @param {string} modelName - Model name to backup
   * @returns {Promise<ApiResponse<Blob>>}
   */
  async backupModel(modelName) {
    return apiRequest(`/ml/models/${modelName}/backup`);
  },

  /**
   * Restore a model
   * @param {File} modelFile - Model file to restore
   * @returns {Promise<ApiResponse<{success: boolean}>>}
   */
  async restoreModel(modelFile) {
    const formData = new FormData();
    formData.append('file', modelFile);

    return apiRequest('/ml/models/restore', {
      method: 'POST',
      headers: {}, // Remove Content-Type to let browser set it for FormData
      body: formData
    });
  },

  /**
   * Get training data preview
   * @param {string} budgetId - Budget ID
   * @param {number} [monthsBack] - Months of data to preview
   * @returns {Promise<ApiResponse<{data: Array, stats: Object}>>}
   */
  async getTrainingData(budgetId, monthsBack = 6) {
    return apiRequest(`/ml/training-data?budget_id=${budgetId}&months_back=${monthsBack}`);
  },

  /**
   * Get training data statistics
   * @param {string} [budgetId] - Budget ID to filter transactions
   * @param {number} [monthsBack] - Months of data to analyze
   * @returns {Promise<ApiResponse<Object>>}
   */
  async getTrainingDataStats(budgetId = null, monthsBack = 12) {
    const params = new URLSearchParams();
    if (budgetId) params.append('budget_id', budgetId);
    if (monthsBack) params.append('months_back', monthsBack.toString());
    
    const queryString = params.toString();
    const endpoint = queryString ? `/ml/training-data/stats?${queryString}` : '/ml/training-data/stats';
    return apiRequest(endpoint);
  },

  /**
   * Prepare training data from transactions
   * @param {Object} params - Training data preparation parameters
   * @param {string} [params.budget_id] - Budget ID to filter transactions
   * @param {number} [params.months_back] - Number of months of data to include
   * @param {number} [params.test_split_ratio] - Ratio of data to use for testing
   * @param {number} [params.min_samples_per_category] - Minimum samples per category in test set
   * @returns {Promise<ApiResponse<Object>>}
   */
  async prepareTrainingData(params) {
    return apiRequest('/ml/training-data/prepare', {
      method: 'POST',
      body: JSON.stringify(params)
    });
  },

  /**
   * Get model metrics
   * @param {string} [modelName] - Specific model name (optional)
   * @returns {Promise<ApiResponse<Object>>}
   */
  async getMetrics(modelName) {
    const endpoint = modelName ? `/ml/metrics?model_name=${modelName}` : '/ml/metrics';
    return apiRequest(endpoint);
  },

  /**
   * List available training datasets
   * @param {string} [budgetId] - Filter by budget ID
   * @returns {Promise<ApiResponse<Array>>}
   */
  async getDatasets(budgetId = null) {
    const params = new URLSearchParams();
    if (budgetId) params.append('budget_id', budgetId);
    
    const queryString = params.toString();
    const endpoint = queryString ? `/ml/datasets?${queryString}` : '/ml/datasets';
    return apiRequest(endpoint);
  },

  /**
   * Get details of a specific dataset
   * @param {string} datasetId - Dataset ID
   * @returns {Promise<ApiResponse<Object>>}
   */
  async getDataset(datasetId) {
    return apiRequest(`/ml/datasets/${datasetId}`);
  },

  /**
   * Delete a training dataset
   * @param {string} datasetId - Dataset ID to delete
   * @returns {Promise<ApiResponse<{success: boolean}>>}
   */
  async deleteDataset(datasetId) {
    return apiRequest(`/ml/datasets/${datasetId}`, { method: 'DELETE' });
  }
};

/**
 * Budgets API methods
 */
export const budgetsApi = {
  /**
   * Get all available budgets
   * @returns {Promise<ApiResponse<{budgets: Array}>>}
   */
  async getAll() {
    return apiRequest('/budgets');
  },

  /**
   * Get currently selected budget
   * @returns {Promise<ApiResponse<{selected_budget_id: string}>>}
   */
  async getSelected() {
    return apiRequest('/budgets/selected');
  },

  /**
   * Select a budget
   * @param {string} budgetId - Budget ID to select
   * @returns {Promise<ApiResponse<{selected_budget_id: string}>>}
   */
  async select(budgetId) {
    return apiRequest('/budgets/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ budget_id: budgetId })
    });
  },

  /**
   * Get budget information including categories, payees, and accounts
   * @param {Array<string>|null} budgetIds - List of budget IDs (null for default)
   * @param {Array<string>|null} entityTypes - List of entity types to include
   * @param {boolean} refreshData - Whether to refresh data from YNAB before returning
   * @returns {Promise<ApiResponse<{budgets: Array, categories: Array, payees: Array, accounts: Array}>>}
   */
  async getBudgetInfo(budgetIds = null, entityTypes = ['Category', 'Payee', 'Account'], refreshData = false) {
    // Build query parameters
    const params = new URLSearchParams();
    if (budgetIds && budgetIds.length > 0) {
      budgetIds.forEach(id => params.append('budget_ids', id));
    }
    if (entityTypes && entityTypes.length > 0) {
      entityTypes.forEach(type => params.append('entity_types', type));
    }
    if (refreshData) {
      params.append('refresh_data', 'true');
    }
    
    const queryString = params.toString();
    const endpoint = queryString ? `/budgets/info?${queryString}` : '/budgets/info';
    
    return apiRequest(endpoint);
  }
};

// Export budgetApi as alias for budgetsApi for consistency
export const budgetApi = budgetsApi;

/**
 * Settings API methods
 */
export const settingsApi = {
  /**
   * Get all application settings
   * @returns {Promise<ApiResponse<Settings>>}
   */
  async getAll() {
    return apiRequest('/settings/');
  },

  /**
   * Update all application settings
   * @param {Object} settings - Complete settings object
   * @returns {Promise<ApiResponse<Settings>>}
   */
  async updateAll(settings) {
    return apiRequest('/settings/', {
      method: 'PUT',
      body: JSON.stringify(settings)
    });
  },

  /**
   * Get email search settings
   * @returns {Promise<ApiResponse<Object>>}
   */
  async getEmailSearch() {
    return apiRequest('/settings/email-search');
  },

  /**
   * Update email search settings
   * @param {Object} settings - Email search settings
   * @returns {Promise<ApiResponse<Object>>}
   */
  async updateEmailSearch(settings) {
    return apiRequest('/settings/email-search', {
      method: 'PUT',
      body: JSON.stringify(settings)
    });
  },

  /**
   * Get display settings
   * @returns {Promise<ApiResponse<Object>>}
   */
  async getDisplay() {
    return apiRequest('/settings/display');
  },

  /**
   * Update display settings
   * @param {Object} settings - Display settings
   * @returns {Promise<ApiResponse<Object>>}
   */
  async updateDisplay(settings) {
    return apiRequest('/settings/display', {
      method: 'PUT',
      body: JSON.stringify(settings)
    });
  },

  /**
   * Export all settings
   * @returns {Promise<ApiResponse<Object>>}
   */
  async export() {
    return apiRequest('/settings/export');
  },

  /**
   * Import settings
   * @param {Object} settings - Settings to import
   * @param {boolean} [overwriteExisting=true] - Whether to overwrite existing settings
   * @returns {Promise<ApiResponse<{success: boolean}>>}
   */
  async import(settings, overwriteExisting = true) {
    return apiRequest('/settings/import', {
      method: 'POST',
      body: JSON.stringify({
        settings: settings,
        overwrite_existing: overwriteExisting
      })
    });
  },

  /**
   * Reset settings to defaults
   * @param {Object} options - Reset options
   * @param {string} [options.config_type] - Type of configs to reset (if null, resets all)
   * @returns {Promise<ApiResponse<{success: boolean}>>}
   */
  async reset(options = {}) {
    return apiRequest('/settings/reset', { 
      method: 'POST',
      body: JSON.stringify(options)
    });
  },

  /**
   * Get settings change history
   * @returns {Promise<ApiResponse<Array>>}
   */
  async getHistory() {
    return apiRequest('/settings/history');
  },

  /**
   * Undo a settings change
   * @param {string} historyId - History entry ID
   * @returns {Promise<ApiResponse<{success: boolean}>>}
   */
  async undoChange(historyId) {
    return apiRequest(`/settings/history/${historyId}/undo`, { method: 'POST' });
  }
};

/**
 * Health check API
 */
export const healthApi = {
  /**
   * Check overall API health
   * @returns {Promise<ApiResponse<{status: string}>>}
   */
  async check() {
    return apiRequest('/health/');
  },

  /**
   * Check database health
   * @returns {Promise<ApiResponse<{status: string, details: Object}>>}
   */
  async checkDatabase() {
    return apiRequest('/health/database');
  },

  /**
   * Check external services health
   * @returns {Promise<ApiResponse<{ynab: Object, gmail: Object}>>}
   */
  async checkServices() {
    return apiRequest('/health/services');
  }
};