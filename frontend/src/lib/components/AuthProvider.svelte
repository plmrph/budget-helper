<!--
  @fileoverview Reusable authentication provider component
  Handles authentication for different services (YNAB, Gmail) with unified interface
-->
<script>
  import { onMount } from "svelte";
  import { authActions, authStore } from "../stores/auth.js";
  import { settingsStore } from "../stores/settings.js";
  import { Card, CardHeader, CardTitle, CardContent } from "./ui/card/index.js";

  /**
   * @typedef {Object} AuthConfig
   * @property {string} serviceName - Internal service name (e.g., 'ynab', 'gmail')
   * @property {string} displayName - Human-readable service name
   * @property {string} description - Service description
   * @property {string} [inputType] - Input type for credentials ('password', 'text')
   * @property {string} [inputPlaceholder] - Placeholder text for input
   * @property {string} [inputLabel] - Label for input field
   * @property {boolean} [requiresInput] - Whether service requires manual input
   * @property {boolean} [isOAuth] - Whether service uses OAuth flow
   */

  /**
   * @typedef {Object} AuthStatus
   * @property {boolean} isAuthenticated
   * @property {boolean} isLoading
   * @property {string|null} error
   * @property {Date|null} lastChecked
   * @property {boolean} canRetry
   */

  /** @type {AuthConfig} */
  export let config;
  export const onstatuschange = undefined; // For external reference only
  export let onconnected;
  export let ondisconnected;

  // Get service-specific auth state from store
  $: serviceState = $authStore[config?.serviceName] || {
    isAuthenticated: false,
    isLoading: false,
    error: null,
    lastChecked: null,
  };

  /** @type {string} */
  let inputValue = "";

  /** @type {Object<string, string>} */
  let inputValues = {};

  /** @type {boolean} */
  let isButtonDisabled = true;

  // Initialize inputValues when config changes
  $: if (config?.inputFields && Object.keys(inputValues).length === 0) {
    config.inputFields.forEach((field) => {
      inputValues[field.name] = "";
    });
    inputValues = { ...inputValues }; // Trigger reactivity

    // Load existing OAuth credentials from settings if this is Gmail
    if (config.serviceName === "gmail") {
      loadGmailCredentials();
    }
  }

  // Reactive statement to update button disabled state
  $: {
    if (serviceState.isLoading) {
      isButtonDisabled = true;
    } else if (!config?.requiresInput) {
      isButtonDisabled = false;
    } else if (config.inputFields) {
      // OAuth with multiple fields
      isButtonDisabled = config.inputFields.some(
        (field) => field.required && !inputValues[field.name]?.trim(),
      );
    } else {
      // Single input field
      isButtonDisabled = !inputValue.trim();
    }
  }

  /** @type {number} */
  let retryCount = 0;
  const MAX_RETRIES = 3;
  const RETRY_DELAY = 2000; // 2 seconds

  /**
   * Check authentication status for the service
   */
  async function checkStatus() {
    if (!config?.serviceName) return;

    // Call backend API - the store handles all state updates
    await authActions.refreshService(config.serviceName);
  }

  /**
   * Connect to the service - clean and simple
   */
  async function connect() {
    if (!config?.serviceName) return;

    let credentials;

    if (config.isOAuth) {
      // OAuth flow (like Gmail)
      if (config.inputFields) {
        // OAuth with configuration
        const oauthConfig = {};

        for (const field of config.inputFields) {
          const value = inputValues[field.name]?.trim();
          if (field.required && !value) {
            return; // Validation handled by reactive button state
          }
          if (value) {
            oauthConfig[field.name] = value;
          }
        }

        // Add default redirect URI and scopes
        oauthConfig.redirect_uri = `${window.location.origin}/api/auth/gmail/callback`;
        oauthConfig.scopes = ["https://www.googleapis.com/auth/gmail.readonly"];

        credentials = oauthConfig;
      } else {
        // Simple OAuth without configuration
        credentials = {};
      }
    } else {
      // Token-based authentication (like YNAB)
      if (!inputValue.trim()) {
        return; // Validation handled by reactive button state
      }
      credentials = inputValue.trim();
    }

    // Call backend API - the store handles all state updates
    const result = await authActions.connect(config.serviceName, credentials);

    if (result.success) {
      // Check for OAuth URL
      const oauthUrl = result.data?.oauth_url || result.data?.auth_url;

      if (oauthUrl) {
        // Open OAuth URL in popup window
        const popup = window.open(
          oauthUrl,
          "oauth_popup",
          "width=600,height=600,scrollbars=yes,resizable=yes",
        );

        // Listen for popup messages
        const messageHandler = async (event) => {
          if (event.origin !== window.location.origin) return;

          if (event.data?.type === "gmail_auth_success") {
            // Clean up
            window.removeEventListener("message", messageHandler);
            try {
              if (popup && !popup.closed) {
                popup.close();
              }
            } catch (e) {
              // Ignore COOP errors when closing popup
            }

            // Refresh auth status
            await authActions.refreshService(config.serviceName);
            onconnected?.({ detail: { service: config.serviceName } });
          }
        };

        window.addEventListener("message", messageHandler);

        // Fallback: poll for authentication completion if popup message fails
        const pollInterval = setInterval(async () => {
          // Check if popup was closed manually (safely handle COOP errors)
          try {
            if (popup && popup.closed) {
              clearInterval(pollInterval);
              window.removeEventListener("message", messageHandler);
              await authActions.refreshService(config.serviceName);
              return;
            }
          } catch (e) {
            // Ignore COOP errors when checking popup.closed
          }

          await authActions.refreshService(config.serviceName);
          if (serviceState.isAuthenticated) {
            clearInterval(pollInterval);
            window.removeEventListener("message", messageHandler);
            try {
              if (popup && !popup.closed) {
                popup.close();
              }
            } catch (e) {
              // Ignore COOP errors when closing popup
            }
            onconnected?.({ detail: { service: config.serviceName } });
          }
        }, 2000);

        // Stop polling after 5 minutes
        setTimeout(() => {
          clearInterval(pollInterval);
          window.removeEventListener("message", messageHandler);
          try {
            if (popup && !popup.closed) {
              popup.close();
            }
          } catch (e) {
            // Ignore COOP errors when closing popup
          }
        }, 300000);
      } else {
        // Token-based auth succeeded - clear inputs
        inputValue = "";
        inputValues = {};
        retryCount = 0;

        onconnected?.({ detail: { service: config.serviceName } });
      }
    }
  }

  /**
   * Disconnect from the service - clean and simple
   */
  async function disconnect() {
    if (!config?.serviceName) return;

    // Call backend API - the store handles all state updates
    const result = await authActions.disconnect(config.serviceName);

    if (result.success) {
      retryCount = 0;
      ondisconnected?.({ detail: { service: config.serviceName } });
    }
  }

  /**
   * Retry the last failed operation
   */
  async function retry() {
    if (retryCount >= MAX_RETRIES) return;

    retryCount++;

    // Wait before retrying
    setTimeout(async () => {
      await authActions.refreshService(config.serviceName);
    }, RETRY_DELAY);
  }

  /**
   * Clear error state
   */
  function clearError() {
    authStore.update((state) => ({
      ...state,
      [config.serviceName]: {
        ...state[config.serviceName],
        error: null,
      },
    }));
  }

  /**
   * Load Gmail OAuth credentials from settings
   */
  async function loadGmailCredentials() {
      const authConfig = JSON.parse(await settingsStore.getSettingValue("email.gmail.auth_config"));
      if (authConfig?.client_id) inputValues.client_id = authConfig.client_id;
      if (authConfig?.client_secret) inputValues.client_secret = authConfig.client_secret;
      inputValues = { ...inputValues }; // Trigger reactivity
  }

  // Load Gmail credentials when component mounts or when disconnected
  onMount(() => {
    if (config?.serviceName === "gmail" && config?.inputFields) {
      loadGmailCredentials();
    }
  });

  // Load credentials when Gmail becomes disconnected (token expired)
  $: if (
    config?.serviceName === "gmail" &&
    !serviceState.isAuthenticated &&
    config?.inputFields
  ) {
    loadGmailCredentials();
  }

  // Reactive statement to validate config
  $: if (config && !config.serviceName) {
    console.error("AuthProvider: serviceName is required in config");
  }
</script>

<Card data-testid="auth-provider-{config?.serviceName}">
  <CardHeader>
    <CardTitle class="flex items-center justify-between">
      <span>{config?.displayName || "Authentication"}</span>
      <div class="flex items-center space-x-2">
        <!-- Status indicator -->
        <div
          class="w-3 h-3 rounded-full {serviceState.isAuthenticated
            ? 'bg-green-500'
            : 'bg-red-500'}"
          title={serviceState.isAuthenticated ? "Connected" : "Disconnected"}
          data-testid="status-indicator"
        ></div>
        <span class="text-sm text-muted-foreground">
          {serviceState.isAuthenticated ? "Connected" : "Disconnected"}
        </span>
      </div>
    </CardTitle>
  </CardHeader>

  <CardContent class="space-y-4">
    {#if config?.description}
      <p class="text-sm text-muted-foreground">{config.description}</p>
    {/if}

    {#if config?.helpText}
      <div
        class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-3 mt-2"
      >
        {#if Array.isArray(config.helpText)}
          <p class="text-sm text-blue-800 dark:text-blue-200 mb-2">
            To get OAuth credentials:
          </p>
          <ol
            class="text-sm text-blue-800 dark:text-blue-200 list-decimal list-inside space-y-1"
          >
            {#each config.helpText as step}
              <li>{step}</li>
            {/each}
          </ol>
        {:else}
          <p class="text-sm text-blue-800 dark:text-blue-200">
            {config.helpText}
          </p>
        {/if}
      </div>
    {/if}

    <!-- Error/Warning display -->
    {#if serviceState.error}
      <div
        class="{serviceState.error.includes('Backend sync issue')
          ? 'bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800'
          : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'} rounded-md p-3"
        data-testid="error-display"
      >
        <div class="flex items-start justify-between">
          <div
            class="text-sm {serviceState.error.includes('Backend sync issue')
              ? 'text-gray-900 dark:text-yellow-200'
              : 'text-red-800 dark:text-red-200'}"
          >
            {#if serviceState.error.includes("Backend sync issue")}
              ⚠️ {serviceState.error}
            {:else}
              {serviceState.error}
            {/if}
          </div>
          <button
            onclick={clearError}
            class="{serviceState.error.includes('Backend sync issue')
              ? 'text-gray-700 hover:text-gray-900 dark:text-yellow-400 dark:hover:text-yellow-200'
              : 'text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-200'} ml-2"
            title="Clear {serviceState.error.includes('Backend sync issue')
              ? 'warning'
              : 'error'}"
          >
            ×
          </button>
        </div>

        {#if serviceState.error.includes("Backend sync issue")}
          <div class="mt-2 text-xs text-gray-800 dark:text-yellow-300">
            Your authentication is preserved from recent login. This warning
            will clear automatically.
          </div>
        {:else if retryCount < MAX_RETRIES}
          <button
            onclick={retry}
            class="mt-2 text-sm text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-200 underline"
            data-testid="retry-button"
          >
            Retry ({MAX_RETRIES - retryCount} attempts left)
          </button>
        {/if}
      </div>
    {/if}

    <!-- Authentication controls -->
    <div class="space-y-3">
      {#if !serviceState.isAuthenticated}
        <!-- Connection form -->
        {#if config?.requiresInput}
          {#if config?.inputFields}
            <!-- Multiple input fields (OAuth with configuration) -->
            {#each config.inputFields as field}
              <div class="mb-4">
                <label
                  for="auth-input-{config.serviceName}-{field.name}"
                  class="block text-sm font-medium text-foreground mb-2"
                >
                  {field.label}
                  {#if field.required}<span
                      class="text-red-500 dark:text-red-400">*</span
                    >{/if}
                </label>
                {#if field.type === "password"}
                  <input
                    id="auth-input-{config.serviceName}-{field.name}"
                    type="password"
                    bind:value={inputValues[field.name]}
                    placeholder={field.placeholder}
                    disabled={serviceState.isLoading}
                    data-testid="auth-input-{field.name}"
                    class="w-full px-3 py-2 border border-input bg-background text-foreground rounded-md shadow-sm placeholder:text-muted-foreground focus:outline-none focus:ring-primary focus:border-primary"
                  />
                {:else}
                  <input
                    id="auth-input-{config.serviceName}-{field.name}"
                    type="text"
                    bind:value={inputValues[field.name]}
                    placeholder={field.placeholder}
                    disabled={serviceState.isLoading}
                    data-testid="auth-input-{field.name}"
                    class="w-full px-3 py-2 border border-input bg-background text-foreground rounded-md shadow-sm placeholder:text-muted-foreground focus:outline-none focus:ring-primary focus:border-primary"
                  />
                {/if}
              </div>
            {/each}
          {:else}
            <!-- Single input field (token-based) -->
            <div>
              <label
                for="auth-input-{config.serviceName}"
                class="block text-sm font-medium text-foreground mb-2"
              >
                {config.inputLabel || "Token"}
              </label>
              {#if config.inputType === "text"}
                <input
                  id="auth-input-{config.serviceName}"
                  type="text"
                  bind:value={inputValue}
                  placeholder={config.inputPlaceholder || "Enter credentials"}
                  disabled={serviceState.isLoading}
                  data-testid="auth-input"
                  class="w-full px-3 py-2 border border-input bg-background text-foreground rounded-md shadow-sm placeholder:text-muted-foreground focus:outline-none focus:ring-primary focus:border-primary"
                />
              {:else}
                <input
                  id="auth-input-{config.serviceName}"
                  type="password"
                  bind:value={inputValue}
                  placeholder={config.inputPlaceholder || "Enter credentials"}
                  disabled={serviceState.isLoading}
                  data-testid="auth-input"
                  class="w-full px-3 py-2 border border-input bg-background text-foreground rounded-md shadow-sm placeholder:text-muted-foreground focus:outline-none focus:ring-primary focus:border-primary"
                />
              {/if}
            </div>
          {/if}
        {/if}

        <button
          onclick={connect}
          disabled={isButtonDisabled}
          data-testid="connect-button"
          class="w-full inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-primary-foreground bg-primary hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {#if serviceState.isLoading}
            Connecting...
          {:else if config?.isOAuth}
            Connect with {config.displayName}
          {:else}
            Connect
          {/if}
        </button>
      {:else}
        <!-- Connected state -->
        <div class="flex items-center justify-between">
          <div class="text-sm text-green-700 dark:text-green-400">
            ✓ Connected successfully
          </div>

          <div class="flex space-x-2">
            <button
              onclick={checkStatus}
              disabled={serviceState.isLoading}
              data-testid="refresh-button"
              class="inline-flex items-center px-3 py-1 border border-border text-sm font-medium rounded-md shadow-sm text-foreground bg-background hover:bg-accent focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {serviceState.isLoading ? "Checking..." : "Refresh"}
            </button>

            <button
              onclick={disconnect}
              disabled={serviceState.isLoading}
              data-testid="disconnect-button"
              class="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded-md shadow-sm text-destructive-foreground bg-destructive hover:bg-destructive/90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-destructive disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {serviceState.isLoading ? "Disconnecting..." : "Disconnect"}
            </button>
          </div>
        </div>
      {/if}
    </div>
  </CardContent>
</Card>
