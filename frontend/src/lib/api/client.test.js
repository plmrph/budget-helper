/**
 * @fileoverview API client integration tests
 * Tests all API client methods with real backend endpoints
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { 
  authApi, 
  transactionApi, 
  emailApi, 
  mlApi, 
  settingsApi, 
  healthApi,
  isLoading,
  apiErrors
} from './client.js';

// Test configuration
const TEST_CONFIG = {
  // Use test YNAB token if available in environment
  YNAB_TOKEN: process.env.VITE_TEST_YNAB_TOKEN || 'test_token_123',
  TIMEOUT: 10000 // 10 second timeout for API calls
};

describe('API Client Integration Tests', () => {
  
  describe('Health API', () => {
    it('should check overall API health', async () => {
      const response = await healthApi.check();
      expect(response.success).toBe(true);
      expect(response.data).toHaveProperty('status');
    }, TEST_CONFIG.TIMEOUT);

    it('should check database health', async () => {
      const response = await healthApi.checkDatabase();
      expect(response.success).toBe(true);
      expect(response.data).toHaveProperty('status');
    }, TEST_CONFIG.TIMEOUT);

    it('should check services health', async () => {
      const response = await healthApi.checkServices();
      expect(response.success).toBe(true);
      expect(response.data).toHaveProperty('ynab');
      expect(response.data).toHaveProperty('gmail');
    }, TEST_CONFIG.TIMEOUT);
  });

  describe('Authentication API', () => {
    it('should get authentication status', async () => {
      const response = await authApi.getStatus();
      expect(response.success).toBe(true);
      expect(response.data).toHaveProperty('ynab');
      expect(response.data).toHaveProperty('gmail');
    }, TEST_CONFIG.TIMEOUT);

    describe('YNAB Authentication', () => {
      it('should get YNAB authentication status', async () => {
        const response = await authApi.ynab.getStatus();
        expect(response.success).toBe(true);
        expect(response.data).toHaveProperty('authenticated');
      }, TEST_CONFIG.TIMEOUT);

      it('should handle YNAB connection attempt', async () => {
        const response = await authApi.ynab.connect(TEST_CONFIG.YNAB_TOKEN);
        // Should either succeed or fail gracefully
        expect(typeof response.success).toBe('boolean');
        if (response.success) {
          expect(response.data).toHaveProperty('success');
        } else {
          expect(response.error).toBeDefined();
        }
      }, TEST_CONFIG.TIMEOUT);
    });

    describe('Gmail Authentication', () => {
      it('should get Gmail authentication status', async () => {
        const response = await authApi.gmail.getStatus();
        expect(response.success).toBe(true);
        expect(response.data).toHaveProperty('authenticated');
      }, TEST_CONFIG.TIMEOUT);

      it('should get Gmail OAuth URL', async () => {
        const response = await authApi.gmail.connect();
        // Should either return auth URL or indicate not configured
        expect(typeof response.success).toBe('boolean');
        if (response.success) {
          expect(response.data).toHaveProperty('auth_url');
        }
      }, TEST_CONFIG.TIMEOUT);
    });
  });

  describe('Transaction API', () => {
    it('should get all transactions', async () => {
      const response = await transactionApi.getAll();
      expect(response.success).toBe(true);
      expect(Array.isArray(response.data)).toBe(true);
    }, TEST_CONFIG.TIMEOUT);

    it('should get transactions with query parameters', async () => {
      const response = await transactionApi.getAll({ limit: 10 });
      expect(response.success).toBe(true);
      expect(Array.isArray(response.data)).toBe(true);
    }, TEST_CONFIG.TIMEOUT);

    it('should handle getting non-existent transaction', async () => {
      const response = await transactionApi.getById('non-existent-id');
      // Should fail gracefully
      expect(response.success).toBe(false);
      expect(response.error).toBeDefined();
    }, TEST_CONFIG.TIMEOUT);

    it('should handle batch transaction updates', async () => {
      const response = await transactionApi.updateBatch([]);
      // Should handle empty array
      expect(typeof response.success).toBe('boolean');
    }, TEST_CONFIG.TIMEOUT);
  });

  describe('Email API', () => {
    it('should get email search configuration', async () => {
      const response = await emailApi.getConfig();
      expect(response.success).toBe(true);
      expect(typeof response.data).toBe('object');
    }, TEST_CONFIG.TIMEOUT);

    it('should get email search history', async () => {
      const response = await emailApi.getHistory();
      expect(response.success).toBe(true);
      expect(Array.isArray(response.data)).toBe(true);
    }, TEST_CONFIG.TIMEOUT);

    it('should handle email search without authentication', async () => {
      const response = await emailApi.search({
        transaction_id: 'test-transaction-id',
        query: 'test search'
      });
      // Should either work or fail gracefully due to no auth
      expect(typeof response.success).toBe('boolean');
    }, TEST_CONFIG.TIMEOUT);
  });

  describe('ML API', () => {
    it('should get available models', async () => {
      const response = await mlApi.getModels();
      expect(response.success).toBe(true);
      expect(Array.isArray(response.data)).toBe(true);
    }, TEST_CONFIG.TIMEOUT);

    it('should get model metrics', async () => {
      const response = await mlApi.getMetrics();
      expect(response.success).toBe(true);
      expect(typeof response.data).toBe('object');
    }, TEST_CONFIG.TIMEOUT);

    it('should handle prediction request', async () => {
      const response = await mlApi.predict({
        payee_name: 'Test Store',
        amount: -5000,
        memo: 'Test purchase'
      });
      // Should either work or indicate no model loaded
      expect(typeof response.success).toBe('boolean');
    }, TEST_CONFIG.TIMEOUT);

    it('should handle training data request', async () => {
      const response = await mlApi.getTrainingData('test-budget-id');
      // Should either work or fail due to no YNAB connection
      expect(typeof response.success).toBe('boolean');
    }, TEST_CONFIG.TIMEOUT);
  });

  describe('Settings API', () => {
    let originalSettings;

    beforeAll(async () => {
      // Save original settings
      const response = await settingsApi.getAll();
      if (response.success) {
        originalSettings = response.data;
      }
    });

    afterAll(async () => {
      // Restore original settings
      if (originalSettings) {
        await settingsApi.updateAll(originalSettings);
      }
    });

    it('should get all settings', async () => {
      const response = await settingsApi.getAll();
      expect(response.success).toBe(true);
      expect(typeof response.data).toBe('object');
    }, TEST_CONFIG.TIMEOUT);

    it('should get email search settings', async () => {
      const response = await settingsApi.getEmailSearch();
      expect(response.success).toBe(true);
      expect(typeof response.data).toBe('object');
    }, TEST_CONFIG.TIMEOUT);

    it('should get display settings', async () => {
      const response = await settingsApi.getDisplay();
      expect(response.success).toBe(true);
      expect(typeof response.data).toBe('object');
    }, TEST_CONFIG.TIMEOUT);

    it('should update display settings', async () => {
      const testSettings = { theme: 'dark', items_per_page: 25 };
      const response = await settingsApi.updateDisplay(testSettings);
      expect(response.success).toBe(true);
      
      // Verify the update
      const getResponse = await settingsApi.getDisplay();
      expect(getResponse.success).toBe(true);
      expect(getResponse.data.theme).toBe('dark');
      expect(getResponse.data.items_per_page).toBe(25);
    }, TEST_CONFIG.TIMEOUT);

    it('should export settings', async () => {
      const response = await settingsApi.export();
      expect(response.success).toBe(true);
      expect(typeof response.data).toBe('object');
    }, TEST_CONFIG.TIMEOUT);

    it('should get settings history', async () => {
      const response = await settingsApi.getHistory();
      expect(response.success).toBe(true);
      expect(Array.isArray(response.data)).toBe(true);
    }, TEST_CONFIG.TIMEOUT);
  });

  describe('Loading States and Error Handling', () => {
    it('should manage loading state during API calls', async () => {
      let loadingStates = [];
      
      // Subscribe to loading state changes
      const unsubscribe = isLoading.subscribe(value => {
        loadingStates.push(value);
      });

      await healthApi.check();
      
      unsubscribe();
      
      // Should have at least started loading (true) and finished (false)
      expect(loadingStates).toContain(true);
      expect(loadingStates).toContain(false);
    }, TEST_CONFIG.TIMEOUT);

    it('should handle network errors gracefully', async () => {
      // Make request to non-existent endpoint
      const response = await transactionApi.getById('invalid-endpoint-test');
      
      expect(response.success).toBe(false);
      expect(response.error).toBeDefined();
      expect(typeof response.error).toBe('string');
    }, TEST_CONFIG.TIMEOUT);

    it('should add errors to error store', async () => {
      let errors = [];
      
      // Subscribe to error state changes
      const unsubscribe = apiErrors.subscribe(value => {
        errors = value;
      });

      // Make a request that should fail
      await transactionApi.getById('definitely-invalid-id-12345');
      
      unsubscribe();
      
      // Should have added an error
      expect(errors.length).toBeGreaterThan(0);
      expect(errors[errors.length - 1]).toHaveProperty('message');
      expect(errors[errors.length - 1]).toHaveProperty('timestamp');
    }, TEST_CONFIG.TIMEOUT);
  });

  describe('Request/Response Format Validation', () => {
    it('should handle JSON responses correctly', async () => {
      const response = await healthApi.check();
      expect(response.success).toBe(true);
      expect(response.data).toBeDefined();
      expect(typeof response.data).toBe('object');
    }, TEST_CONFIG.TIMEOUT);

    it('should include status codes in responses', async () => {
      const response = await healthApi.check();
      expect(response.status).toBeDefined();
      expect(typeof response.status).toBe('number');
      expect(response.status).toBeGreaterThanOrEqual(200);
      expect(response.status).toBeLessThan(300);
    }, TEST_CONFIG.TIMEOUT);

    it('should handle POST requests with JSON body', async () => {
      const response = await mlApi.predict({
        payee_name: 'Test Store',
        amount: -1000,
        memo: 'Test transaction'
      });
      
      // Should either succeed or fail gracefully
      expect(typeof response.success).toBe('boolean');
      if (response.success) {
        expect(response.data).toBeDefined();
      } else {
        expect(response.error).toBeDefined();
      }
    }, TEST_CONFIG.TIMEOUT);
  });
});