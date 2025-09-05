/**
 * @fileoverview Bulk ML prediction store for managing batched predictions and results
 */

import { writable } from 'svelte/store';
import { mlApi } from '../api/client.js';
import { settingsStore } from './settings.js';

/**
 * Bulk ML prediction store state
 * @typedef {Object} BulkMLPredictState
 * @property {boolean} isPredicting - Whether bulk prediction is in progress
 * @property {Object<string, Array<{categoryId: string, confidence: number}>>} predictions - Predictions per transaction ID
 * @property {number} totalProcessed - Total transactions processed
 * @property {number} totalWithPredictions - Total with at least one prediction
 * @property {string|null} error - Error message if any
 * @property {Array<string>} processingTransactionIds - Transaction IDs currently being processed
 * @property {boolean} hasDefaultModel - Whether a default ML model is available
 */

function createBulkMLPredictStore() {
  const { subscribe, set, update } = writable({
    isPredicting: false,
    predictions: {},
    totalProcessed: 0,
    totalWithPredictions: 0,
    error: null,
    processingTransactionIds: [],
    hasDefaultModel: false // Track if a default ML model is available
  });

  async function checkDefaultModel(force = false) {
    if (force && typeof settingsStore.refresh === 'function') {
      try {
        await settingsStore.refresh();
      } catch (err) {
        console.warn('settingsStore.refresh failed in checkDefaultModel:', err);
      }
    }

    const defaultModelName = await settingsStore.getSettingValue('system.default_model_name');
    update(state => ({ ...state, hasDefaultModel: !!defaultModelName }));
    return !!defaultModelName;
  }

  // Call checkDefaultModel on store creation
  checkDefaultModel();

  return {
    subscribe,
    checkDefaultModel,

    /**
     * Run predictions in batches for the provided transactions.
     * Only uses transaction IDs and leaves attaching/updating to the user.
     * @param {Array<{id: string}>} transactions - Visible transactions to predict for
     */
    async predictForTransactions(transactions) {
      if (!transactions || transactions.length === 0) return;

      const transactionIds = transactions.map(t => t.id);
      const BATCH_SIZE = 25; // keep requests efficient for slower models later

      update(state => ({
        ...state,
        isPredicting: true,
        error: null,
        processingTransactionIds: transactionIds,
        // keep existing predictions so previously predicted rows persist
        totalProcessed: 0,
        totalWithPredictions: 0
      }));

      try {
        let allPredictions = {};
        let totalProcessed = 0;
        let totalWithPredictions = 0;

        // Process in batches
        for (let i = 0; i < transactionIds.length; i += BATCH_SIZE) {
          const batch = transactionIds.slice(i, i + BATCH_SIZE);

          try {
            const resp = await mlApi.predict({ transaction_ids: batch });

            if (resp && resp.success && resp.data && Array.isArray(resp.data.predictions)) {
              // resp.data.predictions: [{ transactionId, predictions: [{categoryId, confidence}] }]
              for (const p of resp.data.predictions) {
                const preds = Array.isArray(p.predictions) ? p.predictions : [];
                if (preds.length > 0) totalWithPredictions += 1;
                allPredictions[p.transactionId] = preds;
              }
            } else if (resp && resp.success === false) {
              // Log but continue other batches
              console.error('Bulk ML predict batch failed:', resp.message || resp.error);
            }
          } catch (batchErr) {
            console.error('Bulk ML predict batch error:', batchErr);
            // Continue remaining batches
          }

          totalProcessed += batch.length;

          // Update partial results so UI can react without waiting for all batches
          update(state => ({
            ...state,
            predictions: { ...state.predictions, ...allPredictions },
            totalProcessed,
            totalWithPredictions
          }));

          // brief pause between batches
          if (i + BATCH_SIZE < transactionIds.length) {
            await new Promise(r => setTimeout(r, 75));
          }
        }

        update(state => ({
          ...state,
          isPredicting: false,
          predictions: { ...state.predictions, ...allPredictions },
          totalProcessed,
          totalWithPredictions,
          processingTransactionIds: []
        }));
      } catch (err) {
        console.error('Bulk ML prediction error:', err);
        update(state => ({
          ...state,
          isPredicting: false,
          error: err?.message || 'Bulk ML prediction failed',
          processingTransactionIds: []
        }));
      }
    },

    /**
     * Get predictions for a specific transaction ID (if computed)
     * @param {string} transactionId
     * @returns {Array<{categoryId: string, confidence: number}>|null}
     */
    getPredictionsForTransaction(transactionId) {
      let result = null;
      update(state => {
        result = state.predictions[transactionId] || null;
        return state;
      });
      return result;
    },

    /** Reset store to initial state */
    reset() {
      set({
        isPredicting: false,
        predictions: {},
        totalProcessed: 0,
        totalWithPredictions: 0,
        error: null,
        processingTransactionIds: [],
        hasDefaultModel: false
      });
      // Optionally re-check model after reset
      checkDefaultModel();
    }
  };
}

export const bulkMLPredictStore = createBulkMLPredictStore();
