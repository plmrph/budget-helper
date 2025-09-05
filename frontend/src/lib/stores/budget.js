/**
 * @fileoverview Budget store for managing budget information and caching
 * Provides reactive state management for categories, payees, accounts, and budgets
 */

import { writable, derived, get } from 'svelte/store';
import { budgetsApi } from '../api/client.js';
import { authStore } from './auth.js';
import { settingsStore } from "../stores/settings.js";

/**
 * Budget store state
 * @typedef {Object} BudgetStoreState
 * @property {Array} budgets - Array of budget objects
 * @property {Array} categories - Array of category objects
 * @property {Array} payees - Array of payee objects
 * @property {Array} accounts - Array of account objects
 * @property {boolean} loading - Loading state
 * @property {string|null} error - Error message if any
 * @property {Date|null} lastUpdated - Last update timestamp
 * @property {boolean} isInitialized - Whether initial load has completed
 */

/**
 * Create the main budget store
 */
function createBudgetStore() {
  const { subscribe, set, update } = writable({
    budgets: [],
    categories: [],
    payees: [],
    accounts: [],
    selectedBudgetId: null,
    loading: false,
    error: null,
    lastUpdated: null,
    isInitialized: false
  });
  return {
    subscribe,
    /**
     * Load all available budgets from the API
     * @returns {Promise<Array>} List of available budgets
     */
    async loadAllBudgets() {
      update(state => ({ ...state, loading: true, error: null }));
      const result = await budgetsApi.getAll();
      // Prefer returned data if present regardless of success flag.
      const budgets = Array.isArray(result.data) ? result.data : [];

      if (!budgets.length && result.error) {
        // No budgets and backend reported an error — debug log and treat as empty.
        console.debug('[Budget Store] loadAllBudgets debug:', result.error);
      }

      update(state => ({
        ...state,
        budgets: budgets,
        loading: false,
        error: null,
        lastUpdated: new Date(),
        isInitialized: true
      }));

      return budgets;
    },

    /**
     * Load budget information from the API
     * @param {Array<string>|null} budgetIds - List of budget IDs (null for default)
     * @param {Array<string>|null} entityTypes - List of entity types to load
     * @param {boolean} refreshData - Whether to refresh data from YNAB before returning
     * @returns {Promise<Object>} Budget info result
     */
    async loadBudgetInfo(budgetIds = null, entityTypes = ['Category', 'Payee', 'Account'], refreshData = false) {
        const infoResult = await budgetsApi.getBudgetInfo(budgetIds, entityTypes, refreshData);
        const data = infoResult.data || infoResult;
        const current = get({ subscribe });
        const budgetsFromResponse = Array.isArray(data.budgets) ? data.budgets : null;
        const budgets = (budgetIds == null && budgetsFromResponse) ? budgetsFromResponse : (current?.budgets || []);

        update(s => ({
          ...s,
          budgets: budgets,
          categories: data.categories || [],
          payees: data.payees || [],
          accounts: data.accounts || [],
          loading: false,
          error: null,
          lastUpdated: new Date(),
          isInitialized: true
        }));

      return data;
    },

        /**
     * Refresh metadata from YNAB (sync from external platform)
     * @returns {Promise<Object>} Budget info result
     */
    async refreshFromYNAB() {
      // If no selected budget is set, only refresh the budgets list.
      const state = get(this);
      if (!state.selectedBudgetId) {
        return await this.loadAllBudgets();
      }

      return await this.loadBudgetInfo([state.selectedBudgetId], ['Category', 'Payee', 'Account'], true);
    },

    /**
     * Select a budget by ID
     * @param {string} budgetId - Budget ID to select
     * @returns {Promise<boolean>} Success status
     */
    async selectBudget(budgetId) {
      try {
        // Get the current state to access selectedBudgetId
        const currentState = get({ subscribe });
        if (budgetId == currentState.selectedBudgetId) return true; // No change
        const result = await budgetsApi.select(budgetId);
        if (result.success) {
          update(state => {
            const selected = state.budgets.find(b => b.id === budgetId) || null;
            return { ...state, selectedBudgetId: budgetId};
          });
          return true;
        } else {
          console.error('Failed to select budget:', result.error);
          return false;
        }
      } catch (error) {
        console.error('Error selecting budget:', error);
        return false;
      }
    },

    /**
     * Reset the store to initial state
     */
    reset() {
      set({
        budgets: [],
        categories: [],
        payees: [],
        accounts: [],
        selectedBudgetId: null,
        loading: false,
        error: null,
        lastUpdated: null,
        isInitialized: false
      });
    },

    /**
     * Load selected budget id from backend and select it
     * @returns {Promise<string|null>} selected budget id
     */
    async loadSelectedBudgetId() {
      try {
        const budgetId = await settingsStore.getSettingValue("system.selected_budget_id");
        if (budgetId) {
          // use selectBudget to update selectedBudgetId and load details
          await this.selectBudget(budgetId);
        }
        return budgetId;
      } catch (e) {
        console.debug('[Budget Store] loadSelectedBudgetId failed:', e);
        return null;
      }
    },
  };
}

export const budgetStore = createBudgetStore();

const initializeBudgets = async () => {
  await budgetStore.loadAllBudgets().then(() => {
    // After loading budgets, load the selected budget and its metadata
    budgetStore.loadSelectedBudgetId().then((selectedBudgetId) => {
      if (selectedBudgetId) {
        // Load metadata for the selected budget
        budgetStore.loadBudgetInfo([selectedBudgetId], ['Category', 'Payee', 'Account'], false);
      }
    });
  });
};

// Subscribe to auth store changes and initialize budgets when YNAB becomes authenticated
const unsubscribe = authStore.subscribe((authState) => {
  if (!authState.isInitialized) return;
  if (authState?.ynab?.isAuthenticated) {
    initializeBudgets();
  } else if (!authState?.ynab?.isAuthenticated) {
    budgetStore.reset();
  }
});