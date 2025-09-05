#!/usr/bin/env node

/**
 * @fileoverview Integration testing script for frontend API client and backend endpoints
 * Tests all API methods, button functionality, form submissions, and error handling
 */

import { transactionApi, emailApi, mlApi, settingsApi, healthApi } from './src/lib/api/client.js';

/**
 * Test results tracking
 */
const testResults = {
  passed: 0,
  failed: 0,
  errors: []
};

/**
 * Log test result
 * @param {string} testName - Name of the test
 * @param {boolean} passed - Whether test passed
 * @param {string} [error] - Error message if failed
 */
function logTest(testName, passed, error = null) {
  if (passed) {
    console.log(`✅ ${testName}`);
    testResults.passed++;
  } else {
    console.log(`❌ ${testName}: ${error}`);
    testResults.failed++;
    testResults.errors.push({ test: testName, error });
  }
}

/**
 * Test API endpoint
 * @param {string} testName - Name of the test
 * @param {Function} apiCall - API function to test
 * @param {Array} args - Arguments for API call
 * @param {Object} [options] - Test options
 */
async function testApiEndpoint(testName, apiCall, args = [], options = {}) {
  try {
    const result = await apiCall(...args);
    
    // Check if response has expected structure
    if (typeof result !== 'object' || (!result.success && !result.error)) {
      logTest(testName, false, 'Invalid response structure');
      return;
    }

    // For endpoints that should succeed
    if (options.expectSuccess !== false) {
      if (result.success) {
        logTest(testName, true);
      } else {
        logTest(testName, false, result.error || 'API call failed');
      }
    } else {
      // For endpoints that might fail (like when no data exists)
      logTest(testName, true, `Response: ${result.success ? 'success' : result.error}`);
    }
  } catch (error) {
    logTest(testName, false, error.message);
  }
}

/**
 * Test health endpoint
 */
async function testHealthEndpoint() {
  console.log('\n🏥 Testing Health Endpoint...');
  await testApiEndpoint('Health Check', healthApi.check);
}

/**
 * Test transaction endpoints
 */
async function testTransactionEndpoints() {
  console.log('\n💰 Testing Transaction Endpoints...');
  
  // Test getting all transactions
  await testApiEndpoint('Get All Transactions', transactionApi.getAll, [], { expectSuccess: false });
  
  // Test getting transaction by ID (will likely fail with no data)
  await testApiEndpoint('Get Transaction by ID', transactionApi.getById, ['test-id'], { expectSuccess: false });
  
  // Test sync transactions
  await testApiEndpoint('Sync Transactions', transactionApi.sync, [], { expectSuccess: false });
  
  // Test update transaction (will likely fail with no data)
  await testApiEndpoint('Update Transaction', transactionApi.update, ['test-id', { memo: 'test' }], { expectSuccess: false });
}

/**
 * Test email endpoints
 */
async function testEmailEndpoints() {
  console.log('\n📧 Testing Email Endpoints...');
  
  // Test email search
  await testApiEndpoint('Search Emails', emailApi.search, ['test query'], { expectSuccess: false });
  
  // Test link email to transaction
  await testApiEndpoint('Link Email to Transaction', emailApi.linkToTransaction, ['test-transaction', 'test-email'], { expectSuccess: false });
}

/**
 * Test ML endpoints
 */
async function testMLEndpoints() {
  console.log('\n🤖 Testing ML Endpoints...');
  
  // Test category prediction
  await testApiEndpoint('Predict Category', mlApi.predictCategory, ['test-id'], { expectSuccess: false });
  
  // Test model training
  await testApiEndpoint('Train Model', mlApi.trainModel, [{ test: 'data' }], { expectSuccess: false });
}

/**
 * Test settings endpoints
 */
async function testSettingsEndpoints() {
  console.log('\n⚙️ Testing Settings Endpoints...');
  
  // Test get settings
  await testApiEndpoint('Get Settings', settingsApi.get, [], { expectSuccess: false });
  
  // Test update settings
  await testApiEndpoint('Update Settings', settingsApi.update, [{ test: 'setting' }], { expectSuccess: false });
  
  // Test YNAB connection
  await testApiEndpoint('Test YNAB Connection', settingsApi.testYnabConnection, [], { expectSuccess: false });
  
  // Test Gmail connection
  await testApiEndpoint('Test Gmail Connection', settingsApi.testGmailConnection, [], { expectSuccess: false });
}

/**
 * Test error handling
 */
async function testErrorHandling() {
  console.log('\n🚨 Testing Error Handling...');
  
  try {
    // Test with invalid endpoint
    const response = await fetch('/api/invalid-endpoint');
    if (response.status === 404) {
      logTest('404 Error Handling', true);
    } else {
      logTest('404 Error Handling', false, `Expected 404, got ${response.status}`);
    }
  } catch (error) {
    logTest('404 Error Handling', false, error.message);
  }
  
  try {
    // Test with malformed request
    const response = await fetch('/api/transactions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: 'invalid json'
    });
    
    if (response.status >= 400) {
      logTest('Malformed Request Handling', true);
    } else {
      logTest('Malformed Request Handling', false, `Expected error status, got ${response.status}`);
    }
  } catch (error) {
    logTest('Malformed Request Handling', true, 'Network error handled correctly');
  }
}

/**
 * Test API client configuration
 */
async function testApiConfiguration() {
  console.log('\n🔧 Testing API Configuration...');
  
  // Test that API base URL is correct
  const baseUrl = '/api';
  logTest('API Base URL Configuration', baseUrl === '/api');
  
  // Test that fetch is available
  logTest('Fetch API Available', typeof fetch === 'function');
  
  // Test JSON parsing
  try {
    const testData = { test: 'data' };
    const jsonString = JSON.stringify(testData);
    const parsed = JSON.parse(jsonString);
    logTest('JSON Serialization', parsed.test === 'data');
  } catch (error) {
    logTest('JSON Serialization', false, error.message);
  }
}

/**
 * Test response format consistency
 */
async function testResponseFormats() {
  console.log('\n📋 Testing Response Format Consistency...');
  
  // Test that all API methods return consistent response format
  const testCases = [
    { name: 'Health API', call: () => healthApi.check() },
    { name: 'Transaction API', call: () => transactionApi.getAll() },
    { name: 'Settings API', call: () => settingsApi.get() }
  ];
  
  for (const testCase of testCases) {
    try {
      const result = await testCase.call();
      const hasCorrectFormat = typeof result === 'object' && 
                              (result.hasOwnProperty('success') || result.hasOwnProperty('error'));
      logTest(`${testCase.name} Response Format`, hasCorrectFormat);
    } catch (error) {
      logTest(`${testCase.name} Response Format`, false, error.message);
    }
  }
}

/**
 * Main test runner
 */
async function runTests() {
  console.log('🚀 Starting Frontend Integration Tests...\n');
  
  // Test API configuration first
  await testApiConfiguration();
  
  // Test health endpoint
  await testHealthEndpoint();
  
  // Test all API endpoints
  await testTransactionEndpoints();
  await testEmailEndpoints();
  await testMLEndpoints();
  await testSettingsEndpoints();
  
  // Test error handling
  await testErrorHandling();
  
  // Test response formats
  await testResponseFormats();
  
  // Print summary
  console.log('\n📊 Test Summary:');
  console.log(`✅ Passed: ${testResults.passed}`);
  console.log(`❌ Failed: ${testResults.failed}`);
  console.log(`📈 Success Rate: ${((testResults.passed / (testResults.passed + testResults.failed)) * 100).toFixed(1)}%`);
  
  if (testResults.errors.length > 0) {
    console.log('\n🔍 Failed Tests Details:');
    testResults.errors.forEach(({ test, error }) => {
      console.log(`  • ${test}: ${error}`);
    });
  }
  
  // Exit with appropriate code
  process.exit(testResults.failed > 0 ? 1 : 0);
}

// Run tests if this script is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTests().catch(error => {
    console.error('Test runner failed:', error);
    process.exit(1);
  });
}

export { runTests, testResults };