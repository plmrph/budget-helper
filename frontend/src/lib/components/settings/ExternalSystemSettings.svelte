<!--
  @fileoverview External System Settings component for third-party integrations
-->
<script>
  import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card/index.js';
  import { Label } from '../ui/label/index.js';
  import { Checkbox } from '../ui/checkbox/index.js';
  import { Button } from '../ui/button/index.js';

  export let settings = {};
  export let onchange;

  // External system setting keys from backend ConfigKeys
  const EXTERNAL_KEYS = {
    AUTO_SYNC_ON_STARTUP: 'external.auto_sync_on_startup'
  };

  // Get setting value helper
  function getSettingValue(key, defaultValue = '') {
    const setting = settings[key];
    if (!setting || !setting.value) return defaultValue;
    
    // Extract value from ConfigValue union
    const value = setting.value;
    return value.stringValue ?? value.intValue ?? value.doubleValue ?? value.boolValue ?? defaultValue;
  }

  // Update setting helper
  function updateSetting(key, value, description = '') {
    const currentValue = getSettingValue(key);
    
    // Only update if the value has actually changed
    if (currentValue !== value) {
      onchange?.({ detail: { key, value, type: 'ExternalSystem', description } });
    }
  }

  // Sync status tracking
  let isSyncing = false;
  let lastSyncTime = null;
  let syncMessage = '';

  // Manual sync function
  async function performManualSync() {
    try {
      isSyncing = true;
      syncMessage = 'Syncing with external systems...';
      
      // This would call the actual sync API
      // For now, simulate the sync process
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      lastSyncTime = new Date();
      syncMessage = 'Sync completed successfully!';
      setTimeout(() => syncMessage = '', 3000);
    } catch (error) {
      syncMessage = 'Sync failed. Please try again.';
      setTimeout(() => syncMessage = '', 5000);
    } finally {
      isSyncing = false;
    }
  }
</script>

<div class="external-system-settings columns-1 lg:columns-2 gap-6 space-y-6" data-testid="external-system-settings">
  <!-- Synchronization Settings -->
  <Card class="break-inside-avoid mb-6">
    <CardHeader>
      <CardTitle>Synchronization Settings</CardTitle>
      <CardDescription>Configure how the application syncs with external systems</CardDescription>
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="flex items-center space-x-2">
        <Checkbox
          id="auto-sync-startup"
          checked={getSettingValue(EXTERNAL_KEYS.AUTO_SYNC_ON_STARTUP, true)}
          onCheckedChange={(checked) => updateSetting(EXTERNAL_KEYS.AUTO_SYNC_ON_STARTUP, checked, 'Automatically sync external systems on startup')}
          data-testid="auto-sync-startup-checkbox"
        />
        <Label for="auto-sync-startup" class="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
          Automatically sync on application startup
        </Label>
      </div>
      <p class="text-sm text-muted-foreground ml-6">
        When enabled, the application will automatically sync with YNAB and other connected services when it starts up.
      </p>
    </CardContent>
  </Card>

  <!-- Manual Sync Control -->
  <Card class="break-inside-avoid mb-6">
    <CardHeader>
      <CardTitle>Manual Synchronization</CardTitle>
      <CardDescription>Manually trigger synchronization with external systems</CardDescription>
    </CardHeader>
    <CardContent class="space-y-4">
      {#if syncMessage}
        <div class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-3">
          <div class="text-blue-800 dark:text-blue-200 text-sm">{syncMessage}</div>
        </div>
      {/if}

      <div class="flex items-center justify-between">
        <div>
          <p class="text-sm font-medium">Sync with External Systems</p>
          <p class="text-sm text-muted-foreground">
            Pull latest data from YNAB and other connected services
          </p>
          {#if lastSyncTime}
            <p class="text-xs text-muted-foreground mt-1">
              Last sync: {lastSyncTime.toLocaleString()}
            </p>
          {/if}
        </div>
        <Button 
          onclick={performManualSync} 
          disabled={isSyncing}
          data-testid="manual-sync-button"
        >
          {isSyncing ? 'Syncing...' : 'Sync Now'}
        </Button>
      </div>
    </CardContent>
  </Card>

  <!-- YNAB Integration Status -->
  <Card class="break-inside-avoid mb-6">
    <CardHeader>
      <CardTitle>YNAB Integration</CardTitle>
      <CardDescription>Status and configuration for YNAB synchronization</CardDescription>
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">Connection Status</span>
          <span class="text-sm text-green-600 dark:text-green-400">Connected</span>
        </div>
        
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">API Rate Limit</span>
          <span class="text-sm text-muted-foreground">200 requests/hour</span>
        </div>
        
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">Last Sync</span>
          <span class="text-sm text-muted-foreground">
            {lastSyncTime ? lastSyncTime.toLocaleString() : 'Never'}
          </span>
        </div>
      </div>

      <div class="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-3">
        <p class="text-sm text-yellow-800 dark:text-yellow-200">
          <strong>Note:</strong> YNAB has API rate limits. Frequent syncing may temporarily block requests.
        </p>
      </div>
    </CardContent>
  </Card>

  <!-- Gmail Integration Status -->
  <Card class="break-inside-avoid mb-6">
    <CardHeader>
      <CardTitle>Gmail Integration</CardTitle>
      <CardDescription>Status and configuration for Gmail email search</CardDescription>
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">Connection Status</span>
          <span class="text-sm text-muted-foreground">Not Connected</span>
        </div>
        
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">OAuth Status</span>
          <span class="text-sm text-muted-foreground">Not Configured</span>
        </div>
        
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium">Search Quota</span>
          <span class="text-sm text-muted-foreground">1,000,000,000 units/day</span>
        </div>
      </div>

      <div class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-3">
        <p class="text-sm text-blue-800 dark:text-blue-200">
          Configure Gmail OAuth credentials in the System settings to enable email search functionality.
        </p>
      </div>
    </CardContent>
  </Card>

  <!-- Sync Behavior Configuration -->
  <Card class="break-inside-avoid mb-6">
    <CardHeader>
      <CardTitle>Sync Behavior</CardTitle>
      <CardDescription>Advanced synchronization options and troubleshooting</CardDescription>
    </CardHeader>
    <CardContent>
      <div class="space-y-4">
        <div class="bg-gray-50 dark:bg-gray-900/20 border border-gray-200 dark:border-gray-800 rounded-md p-4">
          <h4 class="font-medium text-gray-900 dark:text-gray-100 mb-2">Sync Process</h4>
          <ol class="text-sm text-gray-700 dark:text-gray-300 space-y-1 list-decimal list-inside">
            <li>Authenticate with external services</li>
            <li>Fetch latest transaction data from YNAB</li>
            <li>Update local database with new/changed transactions</li>
            <li>Sync transaction approvals and metadata back to YNAB</li>
            <li>Update sync timestamps and status</li>
          </ol>
        </div>

        <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
          <h4 class="font-medium text-red-900 dark:text-red-100 mb-2">Troubleshooting</h4>
          <ul class="text-sm text-red-800 dark:text-red-200 space-y-1">
            <li>• If sync fails, check your internet connection and API credentials</li>
            <li>• Rate limit errors will resolve automatically after waiting</li>
            <li>• Large datasets may take several minutes to sync completely</li>

          </ul>
        </div>
      </div>
    </CardContent>
  </Card>
</div>