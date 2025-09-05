/**
 * @fileoverview Clean authentication store with 2-way binding
 * Simple reactive state that reflects backend status
 */

import { writable, derived } from 'svelte/store';
import { authApi } from '../api/client.js';

/**
 * @typedef {Object} ServiceState
 * @property {boolean} isAuthenticated
 * @property {boolean} isLoading
 * @property {string|null} error
 */

/**
 * @typedef {Object} AuthState
 * @property {ServiceState} ynab
 * @property {ServiceState} gmail
 * @property {boolean} isInitialized
 */

/**
 * Initial service state
 */
function createInitialServiceState() {
  return {
    isAuthenticated: false,
    isLoading: false,
    error: null
  };
}

/**
 * Initial auth state
 */
const initialState = {
  ynab: createInitialServiceState(),
  gmail: createInitialServiceState(),
  isInitialized: false
};

// Create the main auth store - this is the single source of truth
export const authStore = writable(initialState);

// Derived stores for individual services
export const ynabAuth = derived(authStore, $authStore => $authStore.ynab);
export const gmailAuth = derived(authStore, $authStore => $authStore.gmail);

// Derived store for overall authentication status
export const authStatus = derived(authStore, $authStore => ({
  allConnected: $authStore.ynab.isAuthenticated && $authStore.gmail.isAuthenticated,
  anyConnected: $authStore.ynab.isAuthenticated || $authStore.gmail.isAuthenticated,
  isLoading: $authStore.ynab.isLoading || $authStore.gmail.isLoading,
  isInitialized: $authStore.isInitialized
}));

/**
 * Clean auth actions - just call backend APIs and update reactive state
 */
export const authActions = {
  /**
   * Initialize by loading status from backend on page load
   */
  async initialize() {
    console.debug('🔄 Loading auth status from backend...');

    try {
      // Load both statuses in parallel to reduce requests
      const [ynabResult, gmailResult] = await Promise.all([
        authApi.ynab.getStatus(),
        authApi.gmail.getStatus()
      ]);

      const ynabAuth = ynabResult.success && ynabResult.data?.status?.is_authenticated;
      const gmailAuth = gmailResult.success && gmailResult.data?.status?.is_authenticated;

      // Update reactive state
      authStore.update(state => ({
        ...state,
        ynab: {
          isAuthenticated: ynabAuth || false,
          isLoading: false,
          error: ynabResult.success ? null : ynabResult.error
        },
        gmail: {
          isAuthenticated: gmailAuth || false,
          isLoading: false,
          error: gmailResult.success ? null : gmailResult.error
        },
        isInitialized: true
      }));

      console.debug('✅ Auth status loaded:', { ynab: ynabAuth, gmail: gmailAuth });

    } catch (error) {
      console.error('❌ Failed to load auth status:', error);

      authStore.update(state => ({
        ...state,
        ynab: { ...state.ynab, isLoading: false, error: error.message },
        gmail: { ...state.gmail, isLoading: false, error: error.message },
        isInitialized: true
      }));
    }
  },

  /**
   * Connect to service - calls backend API and updates reactive state
   */
  async connect(service, credentials) {
    console.log(`🔐 Connecting to ${service}...`);

    // Set loading state
    authStore.update(state => ({
      ...state,
      [service]: { ...state[service], isLoading: true, error: null }
    }));

    try {
      // Call backend API
      const result = await authApi[service].connect(credentials);

      if (result.success) {
        // Small delay to ensure backend has processed the token storage
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Get updated status from backend
        const statusResult = await authApi[service].getStatus();
        const isAuth = statusResult.success && statusResult.data?.status?.is_authenticated;

        // Update reactive state
        authStore.update(state => ({
          ...state,
          [service]: {
            isAuthenticated: isAuth || false,
            isLoading: false,
            error: null
          }
        }));

        console.log(`✅ ${service} connected, authenticated: ${isAuth}`);
        return { success: true, data: result.data };
      } else {
        // Update error state
        authStore.update(state => ({
          ...state,
          [service]: {
            ...state[service],
            isLoading: false,
            error: result.error || 'Connection failed'
          }
        }));

        return { success: false, error: result.error };
      }
    } catch (error) {
      console.error(`❌ ${service} connection error:`, error);

      authStore.update(state => ({
        ...state,
        [service]: {
          ...state[service],
          isLoading: false,
          error: error.message
        }
      }));

      return { success: false, error: error.message };
    }
  },

  /**
   * Disconnect from service - calls backend API and updates reactive state
   */
  async disconnect(service) {
    console.log(`🔓 Disconnecting from ${service}...`);

    authStore.update(state => ({
      ...state,
      [service]: { ...state[service], isLoading: true }
    }));

    try {
      // Call backend API
      const result = await authApi[service].disconnect();

      // Get updated status from backend
      const statusResult = await authApi[service].getStatus();
      const isAuth = statusResult.success && statusResult.data?.status?.is_authenticated;

      // Update reactive state
      authStore.update(state => ({
        ...state,
        [service]: {
          isAuthenticated: isAuth || false,
          isLoading: false,
          error: result.success ? null : result.error
        }
      }));

      console.log(`✅ ${service} disconnected`);
      return result;
    } catch (error) {
      console.error(`❌ ${service} disconnect error:`, error);

      authStore.update(state => ({
        ...state,
        [service]: {
          ...state[service],
          isLoading: false,
          error: error.message
        }
      }));

      return { success: false, error: error.message };
    }
  },

  /**
   * Refresh status for a specific service
   */
  async refreshService(service) {
    console.log(`🔄 Refreshing ${service} status...`);

    authStore.update(state => ({
      ...state,
      [service]: { ...state[service], isLoading: true }
    }));

    try {
      const result = await authApi[service].getStatus();
      const isAuth = result.success && result.data?.status?.is_authenticated;

      authStore.update(state => ({
        ...state,
        [service]: {
          isAuthenticated: isAuth || false,
          isLoading: false,
          error: result.success ? null : result.error
        }
      }));

      console.log(`✅ ${service} status refreshed:`, isAuth);
      return { success: true, authenticated: isAuth };
    } catch (error) {
      console.error(`❌ ${service} status refresh error:`, error);

      authStore.update(state => ({
        ...state,
        [service]: {
          ...state[service],
          isLoading: false,
          error: error.message
        }
      }));

      return { success: false, error: error.message };
    }
  }
};

authActions.initialize();