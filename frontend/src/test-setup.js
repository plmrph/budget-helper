/**
 * @fileoverview Test setup configuration
 * Sets up global test environment and mocks
 */

// Mock fetch for API tests
global.fetch = async (url, options = {}) => {
  console.log('Mock fetch:', url, options?.method || 'GET');
  
  // Mock successful responses for health checks
  if (url.includes('/health/')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ status: 'healthy' })
    };
  }
  
  if (url.includes('/health/database')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ status: 'healthy', details: {} })
    };
  }
  
  if (url.includes('/health/services')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ ynab: { status: 'healthy' }, gmail: { status: 'healthy' } })
    };
  }
  
  // Mock auth status
  if (url.includes('/auth/status')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ ynab: false, gmail: false })
    };
  }
  
  if (url.includes('/auth/ynab/status')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ authenticated: false })
    };
  }
  
  if (url.includes('/auth/gmail/status')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ authenticated: false })
    };
  }
  
  if (url.includes('/auth/ynab/connect')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ success: true })
    };
  }
  
  if (url.includes('/auth/gmail/connect')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ auth_url: 'https://accounts.google.com/oauth/authorize' })
    };
  }
  
  // Mock transactions
  if (url.includes('/transactions/') && !url.includes('/transactions/batch')) {
    if (url.includes('non-existent-id') || url.includes('invalid-endpoint-test')) {
      return {
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Transaction not found' })
      };
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        id: 'test-id',
        payee_name: 'Test Payee',
        amount: -1000,
        date: '2024-01-01'
      })
    };
  }
  
  if (url.includes('/transactions/batch')) {
    return {
      ok: true,
      status: 200,
      json: async () => ([])
    };
  }
  
  if (url.includes('/transactions')) {
    return {
      ok: true,
      status: 200,
      json: async () => ([])
    };
  }
  
  // Mock email API
  if (url.includes('/email-search/config')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ days_before: 3, days_after: 3 })
    };
  }
  
  if (url.includes('/email-search/history')) {
    return {
      ok: true,
      status: 200,
      json: async () => ([])
    };
  }
  
  if (url.includes('/email-search/search')) {
    return {
      ok: true,
      status: 200,
      json: async () => ([])
    };
  }
  
  // Mock ML API
  if (url.includes('/ml/models')) {
    return {
      ok: true,
      status: 200,
      json: async () => ([])
    };
  }
  
  if (url.includes('/ml/metrics')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ accuracy: 0.85 })
    };
  }
  
  if (url.includes('/ml/predict')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ predictions: [] })
    };
  }
  
  if (url.includes('/ml/training-data')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ data: [], stats: {} })
    };
  }
  
  // Mock settings
  if (url.includes('/settings/display') && options?.method === 'PUT') {
    const body = JSON.parse(options.body);
    return {
      ok: true,
      status: 200,
      json: async () => body
    };
  }
  
  if (url.includes('/settings/display')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ theme: 'light', items_per_page: 50 })
    };
  }
  
  if (url.includes('/settings/email-search')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({ days_before: 3, days_after: 3 })
    };
  }
  
  if (url.includes('/settings/export')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        email_search: { days_before: 3, days_after: 3 },
        display: { theme: 'light' },
        ai: { auto_categorize: false }
      })
    };
  }
  
  if (url.includes('/settings/history')) {
    return {
      ok: true,
      status: 200,
      json: async () => ([])
    };
  }
  
  if (url.includes('/settings/')) {
    return {
      ok: true,
      status: 200,
      json: async () => ({
        email_search: { days_before: 3, days_after: 3 },
        display: { theme: 'light' },
        ai: { auto_categorize: false }
      })
    };
  }
  
  // Default mock response for unhandled endpoints
  console.log('Unhandled endpoint:', url);
  return {
    ok: false,
    status: 404,
    json: async () => ({ detail: 'Not found' })
  };
};

// Mock window.open for OAuth flows
global.window = global.window || {};
global.window.open = () => {};

// Mock URL.createObjectURL for file downloads
global.URL = global.URL || {};
global.URL.createObjectURL = () => 'mock-url';
global.URL.revokeObjectURL = () => {};

// Mock document methods for file operations
global.document = global.document || {};
global.document.createElement = (tag) => {
  if (tag === 'a') {
    return {
      href: '',
      download: '',
      click: () => {}
    };
  }
  return {};
};
global.document.body = global.document.body || {};
global.document.body.appendChild = () => {};
global.document.body.removeChild = () => {};