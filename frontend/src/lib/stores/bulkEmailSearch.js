/**
 * @fileoverview Bulk email search store for managing bulk search state and results
 */

import { writable, derived } from 'svelte/store';
import { emailApi } from '../api/client.js';

/**
 * Bulk email search store state
 * @typedef {Object} BulkEmailSearchState
 * @property {boolean} isSearching - Whether bulk search is in progress
 * @property {Object} results - Search results per transaction ID
 * @property {number} totalProcessed - Total transactions processed
 * @property {number} totalAttached - Total emails auto-attached
 * @property {string|null} error - Error message if any
 * @property {Array<string>} processingTransactionIds - Transaction IDs currently being processed
 */

/**
 * Create the bulk email search store
 */
function createBulkEmailSearchStore() {
  const { subscribe, set, update } = writable({
    isSearching: false,
    results: {},
    totalProcessed: 0,
    totalAttached: 0,
    error: null,
    processingTransactionIds: []
  });

  return {
    subscribe,
    
    /**
     * Start bulk email search for visible transactions in batches
     * @param {Array<Object>} transactions - Array of transaction objects
     */
    async searchForTransactions(transactions) {
      if (!transactions || transactions.length === 0) {
        return;
      }

      const transactionIds = transactions.map(t => t.id);
      const BATCH_SIZE = 5; // Process 5 transactions at a time
      
      update(state => ({
        ...state,
        isSearching: true,
        error: null,
        processingTransactionIds: transactionIds,
        results: {},
        totalProcessed: 0,
        totalAttached: 0
      }));

      try {
        let allResults = {};
        let totalProcessed = 0;
        let totalAttached = 0;

        // Split transactions into batches
        for (let i = 0; i < transactionIds.length; i += BATCH_SIZE) {
          const batch = transactionIds.slice(i, i + BATCH_SIZE);
          
          console.log(`Processing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(transactionIds.length / BATCH_SIZE)}: ${batch.length} transactions`);
          
          try {
            const result = await emailApi.bulkSearchForTransactions(batch);
            
            if (result.success !== false) {
              // Merge results from this batch
              allResults = { ...allResults, ...(result.results || {}) };
              totalProcessed += result.total_processed || 0;
              totalAttached += result.total_attached || 0;
              
              // Update state with partial results for real-time feedback
              update(state => ({
                ...state,
                results: allResults,
                totalProcessed,
                totalAttached
              }));
              
            } else {
              console.error(`Batch ${Math.floor(i / BATCH_SIZE) + 1} failed:`, result.error);
              // Continue with other batches even if one fails
            }
          } catch (batchError) {
            console.error(`Batch ${Math.floor(i / BATCH_SIZE) + 1} error:`, batchError);
            // Continue with other batches even if one fails
          }
          
          // Small delay between batches to avoid overwhelming the server
          if (i + BATCH_SIZE < transactionIds.length) {
            await new Promise(resolve => setTimeout(resolve, 100));
          }
        }

        // Final update
        update(state => ({
          ...state,
          isSearching: false,
          results: allResults,
          totalProcessed,
          totalAttached,
          processingTransactionIds: []
        }));

      } catch (error) {
        console.error('Bulk email search error:', error);
        update(state => ({
          ...state,
          isSearching: false,
          error: error.message || 'Bulk email search failed',
          processingTransactionIds: []
        }));
      }
    },

    /**
     * Get search result for a specific transaction
     * @param {string} transactionId - Transaction ID
     * @returns {Object|null} Search result for the transaction
     */
    getResultForTransaction(transactionId) {
      let result = null;
      update(state => {
        result = state.results[transactionId] || null;
        return state;
      });
      return result;
    },

    /**
     * Clear all search results
     */
    clearResults() {
      update(state => ({
        ...state,
        results: {},
        totalProcessed: 0,
        totalAttached: 0,
        error: null,
        processingTransactionIds: []
      }));
    },

    /**
     * Reset the store to initial state
     */
    reset() {
      set({
        isSearching: false,
        results: {},
        totalProcessed: 0,
        totalAttached: 0,
        error: null,
        processingTransactionIds: []
      });
    }
  };
}

/**
 * Main bulk email search store instance
 */
export const bulkEmailSearchStore = createBulkEmailSearchStore();

/**
 * Derived store for getting email status per transaction
 */
export const transactionEmailStatus = derived(
  bulkEmailSearchStore,
  ($bulkEmailSearchStore) => {
    const statusMap = {};
    
    Object.entries($bulkEmailSearchStore.results).forEach(([transactionId, result]) => {
      if (result.status === 'success') {
        const emailCount = result.email_count || 0;
        const attached = result.attached || false;
        
        if (attached) {
          statusMap[transactionId] = { status: 'attached', count: 1 };
        } else if (emailCount > 1) {
          statusMap[transactionId] = { status: 'multiple', count: emailCount };
        } else if (emailCount === 1) {
          statusMap[transactionId] = { status: 'single', count: 1 };
        } else {
          statusMap[transactionId] = { status: 'none', count: 0 };
        }
      } else if (result.status === 'error') {
        statusMap[transactionId] = { status: 'error', count: 0 };
      }
    });
    
    return statusMap;
  }
);