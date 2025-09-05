<!--
  @fileoverview Unified authentication status display component
  Shows overall authentication status for all services
-->
<script>
  import { onMount, createEventDispatcher } from "svelte";
  import { authStore, authStatus, authActions } from "../stores/auth.js";
  import { budgetStore } from "../stores/budget.js";
    import { settingsStore } from "$lib/stores/settings.js";

  const dispatch = createEventDispatcher();

  /**
   * @typedef {Object} ServiceStatus
   * @property {boolean} authenticated
   * @property {boolean} loading
   * @property {string|null} error
   * @property {Date|null} lastChecked
   */

  /**
   * @typedef {Object} OverallStatus
   * @property {Object<string, ServiceStatus>} services
   * @property {boolean} allConnected
   * @property {boolean} anyConnected
   * @property {boolean} loading
   * @property {Date|null} lastChecked
   */

   const services = $derived({
    ynab: {
      authenticated: $authStore.ynab.isAuthenticated,
      loading: $authStore.ynab.isLoading,
      error: $authStore.ynab.error,
      lastChecked: $authStore.ynab.lastChecked,
    },
    gmail: {
      authenticated: $authStore.gmail.isAuthenticated,
      loading: $authStore.gmail.isLoading,
      error: $authStore.gmail.error,
      lastChecked: $authStore.gmail.lastChecked,
    },
  });

  /** @type {boolean} */
  let { compact = false, showRefresh = true, autoRefresh = false, refreshInterval = 30000 } = $props();
  let refreshTimer;
  let refreshingMetadata = $state(false);

  /**
   * Refresh YNAB metadata from external platform
   */
  async function refreshYNABMetadata() {
    try {
      refreshingMetadata = true;
      await budgetStore.refreshFromYNAB();
      
      dispatch("metadataRefreshed", {
        service: 'ynab',
        success: true
      });
    } catch (error) {
      console.error('Failed to refresh YNAB metadata:', error);
      
      dispatch("metadataRefreshed", {
        service: 'ynab',
        success: false,
        error: error.message
      });
    } finally {
      refreshingMetadata = false;
    }
  }

  /**
   * Check authentication status for all services
   */
  async function checkAllStatus() {
    dispatch("statusUpdate", {
      services,
      allConnected: $authStatus.allConnected,
      anyConnected: $authStatus.anyConnected,
    });
  }

  /**
   * Update status for a specific service
   * @param {string} serviceName
   * @param {boolean} authenticated
   */
  export function updateServiceStatus(serviceName, authenticated) {
    // With the simplified store, we just refresh the service status from backend
    authActions.refreshService(serviceName);

    dispatch("statusUpdate", {
      services,
      allConnected: $authStatus.allConnected,
      anyConnected: $authStatus.anyConnected,
    });
  }

  /**
   * Get display name for service
   * @param {string} serviceName
   * @returns {string}
   */
  function getServiceDisplayName(serviceName) {
    const names = {
      ynab: "YNAB",
      gmail: "Gmail",
    };
    return names[serviceName] || serviceName.toUpperCase();
  }

  /**
   * Get status color class
   * @param {ServiceStatus} serviceStatus
   * @returns {string}
   */
  function getStatusColor(serviceStatus) {
    if (serviceStatus.loading) return "bg-yellow-500";
    if (serviceStatus.error) return "bg-red-500";
    return serviceStatus.authenticated ? "bg-green-500" : "bg-gray-400";
  }

  /**
   * Get status text
   * @param {ServiceStatus} serviceStatus
   * @returns {string}
   */
  function getStatusText(serviceStatus) {
    if (serviceStatus.loading) return "Checking...";
    if (serviceStatus.error) return "Error";
    return serviceStatus.authenticated ? "Connected" : "Disconnected";
  }

  let refresh = false;

    /**
   * Wait for auth store to be fully initialized
   * @returns {Promise<void>}
   */
  function waitForAuthInitialized() {
    return new Promise((resolve) => {
      // If already initialized, resolve immediately
      if ($authStore.isInitialized) {
        resolve();
        return;
      }
      // Otherwise, subscribe and wait
      const unsubscribe = authStore.subscribe((state) => {
        if (state.isInitialized) {
          unsubscribe();
          resolve();
        }
      });
    });
  }

  onMount(async () => {
    await waitForAuthInitialized();
    if ($authStore.ynab.isAuthenticated && await settingsStore.getSettingValue("external.auto_sync_on_startup")) {
      // YNAB just became authenticated, refresh metadata
      refreshYNABMetadata();
    } else if (!$authStore.ynab.isAuthenticated) {
      console.log('YNAB not authenticated on mount, skipping metadata refresh.');
      // YNAB just became disconnected, reset budget store
      budgetStore.reset();
    }
  });
</script>

<div class="auth-status {compact ? 'compact' : ''}" data-testid="auth-status">
  {#if compact}
    <!-- Compact view -->
    <div class="flex items-center space-x-2">
      {#each Object.entries(services) as [serviceName, serviceStatus]}
        <div
          class="flex items-center space-x-1"
          title="{getServiceDisplayName(serviceName)}: {getStatusText(
            serviceStatus,
          )}"
        >
          <div
            class="w-2 h-2 rounded-full {getStatusColor(serviceStatus)}"
            data-testid="status-dot-{serviceName}"
          ></div>
          <span class="text-xs text-muted-foreground">
            {getServiceDisplayName(serviceName)}
          </span>
        </div>
      {/each}

      {#if showRefresh}
        <button
          onclick={checkAllStatus}
          disabled={$authStatus.isLoading}
          data-testid="refresh-compact"
          aria-label="Refresh authentication status"
          class="p-1 h-6 w-6 text-muted-foreground hover:text-foreground disabled:opacity-50"
        >
          <svg
            class="w-3 h-3 {$authStatus.isLoading ? 'animate-spin' : ''}"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
        </button>
      {/if}
    </div>
  {:else}
    <!-- Full view -->
    <div class="bg-card text-card-foreground rounded-lg border p-4">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-medium text-foreground">
          Authentication Status
        </h3>

        {#if showRefresh}
          <button
            onclick={checkAllStatus}
            disabled={$authStatus.isLoading}
            data-testid="refresh-full"
            class="inline-flex items-center px-3 py-1 border border-border text-sm font-medium rounded-md shadow-sm text-foreground bg-background hover:bg-accent focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {#if $authStatus.isLoading}
              <svg
                class="w-4 h-4 mr-2 animate-spin"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Checking...
            {:else}
              <svg
                class="w-4 h-4 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Refresh
            {/if}
          </button>
        {/if}
      </div>

      <!-- Overall status summary -->
      <div
        class="mb-4 p-3 rounded-md {$authStatus.allConnected
          ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
          : $authStatus.anyConnected
            ? 'bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800'
            : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'}"
      >
        <div class="flex items-center">
          <div
            class="w-4 h-4 rounded-full mr-3 {$authStatus.allConnected
              ? 'bg-green-500'
              : $authStatus.anyConnected
                ? 'bg-yellow-500'
                : 'bg-red-500'}"
          ></div>
          <div>
            <div
              class="text-sm font-medium {$authStatus.allConnected
                ? 'text-green-800 dark:text-green-200'
                : $authStatus.anyConnected
                  ? 'text-gray-900 dark:text-yellow-200'
                  : 'text-red-800 dark:text-red-200'}"
            >
              {#if $authStatus.allConnected}
                All services connected
              {:else if $authStatus.anyConnected}
                Some services connected
              {:else}
                No services connected
              {/if}
            </div>
            {#if $authStatus.lastGlobalCheck}
              <div class="text-xs text-muted-foreground mt-1">
                Last checked: {$authStatus.lastGlobalCheck.toLocaleString()}
              </div>
            {/if}
          </div>
        </div>
      </div>

      <!-- Individual service status -->
      <div class="space-y-3">
        {#each Object.entries(services) as [serviceName, serviceStatus]}
          <div
            class="flex items-center justify-between p-3 border rounded-md"
            data-testid="service-status-{serviceName}"
          >
            <div class="flex items-center">
              <div
                class="w-3 h-3 rounded-full mr-3 {getStatusColor(
                  serviceStatus,
                )}"
              ></div>
              <div>
                <div class="text-sm font-medium text-foreground">
                  {getServiceDisplayName(serviceName)}
                </div>
                <div class="text-xs text-muted-foreground">
                  {getStatusText(serviceStatus)}
                  {#if serviceStatus.lastChecked}
                    • {serviceStatus.lastChecked.toLocaleTimeString()}
                  {/if}
                </div>
              </div>
            </div>

            <div class="flex items-center gap-2">
              {#if serviceStatus.error}
                <div
                  class="text-xs text-red-600 dark:text-red-400 max-w-xs truncate"
                  title={serviceStatus.error}
                >
                  {serviceStatus.error}
                </div>
              {/if}

              {#if serviceName === 'ynab' && serviceStatus.authenticated}
                <button
                  onclick={refreshYNABMetadata}
                  disabled={refreshingMetadata}
                  class="inline-flex items-center px-2 py-1 border border-border text-xs font-medium rounded text-foreground bg-background hover:bg-accent focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Refresh budget metadata from YNAB"
                >
                  {#if refreshingMetadata}
                    <svg class="w-3 h-3 mr-1 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                    </svg>
                    Syncing...
                  {:else}
                    <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                    </svg>
                    Refresh metadata from YNAB
                  {/if}
                </button>
              {/if}
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .auth-status.compact {
    @apply inline-flex;
  }
</style>
