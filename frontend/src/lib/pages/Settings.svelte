<!--
  @fileoverview Settings Layout component with tabbed interface for different configuration types
  Implements comprehensive settings management with validation, import/export, and reset functionality
-->
<script>
  import { onMount } from "svelte";
  import { settingsApi, budgetsApi } from "../api/client.js";
  import { settingsStore } from "../stores/settings.js";
  import { Button } from "../components/ui/button/index.js";
  import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
  } from "../components/ui/card/index.js";
  import { Separator } from "../components/ui/separator/index.js";
  import SystemSettings from "../components/settings/SystemSettings.svelte";
  import EmailSettings from "../components/settings/EmailSettings.svelte";
  import AISettings from "../components/settings/AISettings.svelte";

  import ExternalSystemSettings from "../components/settings/ExternalSystemSettings.svelte";
  import AuthProvider from "../components/AuthProvider.svelte";
  import AuthStatus from "../components/AuthStatus.svelte";
  import { budgetStore } from "../stores/budget.js";
  import { authStore } from "../stores/auth.js";

  // Tab configuration
  const tabs = [
    { id: "system", label: "System", icon: "⚙️", component: SystemSettings },
    { id: "email", label: "Email", icon: "📧", component: EmailSettings },
    { id: "ai", label: "AI/ML", icon: "🤖", component: AISettings },
    {
      id: "external",
      label: "External",
      icon: "🔗",
      component: ExternalSystemSettings,
    },
  ];

  // State management
  let activeTab = "system";
  let settings = {};
  let isLoading = false;
  let saveMessage = "";
  let error = null;

  // Authentication configurations
  const authConfigs = {
    ynab: {
      serviceName: "ynab",
      displayName: "YNAB",
      description: "Connect to your YNAB account using a Personal Access Token",
      inputType: "password",
      inputLabel: "Personal Access Token",
      inputPlaceholder: "Enter your YNAB Personal Access Token",
      requiresInput: true,
      isOAuth: false,
    },
    gmail: {
      serviceName: "gmail",
      displayName: "Gmail",
      description:
        "Connect to Gmail for email receipt matching using OAuth 2.0",
      requiresInput: true,
      isOAuth: true,
      helpText: [
        "Go to Google Cloud Console https://console.cloud.google.com/projectcreate ",
        "Create a project",
        "Enable Gmail API", 
        "Create OAuth 2.0 credentials",
        "Add http://localhost to authorized origins",
        "Add http://localhost/api/auth/gmail/callback to redirect URIs"
      ],
      inputFields: [
        {
          name: "client_id",
          label: "OAuth Client ID",
          type: "text",
          placeholder:
            "Enter your Gmail OAuth Client ID (ends with .googleusercontent.com)",
          required: true,
        },
        {
          name: "client_secret",
          label: "OAuth Client Secret",
          type: "password",
          placeholder: "Enter your Gmail OAuth Client Secret",
          required: true,
        },
      ],
    },
  };

  // Budget selection state from store
  $: ({
    budgets,
    selectedBudgetId,
    loading: budgetsLoading,
    error: budgetsError,
  } = $budgetStore);

  /** @type {AuthStatus} */
  let authStatusRef;

  /**
   * Load all settings from the API
   */
  async function loadSettings() {
    try {
      isLoading = true;
      error = null;

      const result = await settingsStore.loadAll();
      if (result) {
        settings = result || {};
      } else {
        error = "Failed to load settings";
      }
    } catch (err) {
      error = "Failed to load settings";
      console.error("Failed to load settings:", err);
    } finally {
      isLoading = false;
    }
  }

  /**
   * Reset settings to defaults
   */
  async function resetSettings(configType = null) {
    if (
      !confirm(
        `Are you sure you want to reset ${configType ? configType : "all"} settings to defaults? This cannot be undone.`,
      )
    ) {
      return;
    }

    try {
      isLoading = true;
      error = null;

      const result = await settingsApi.reset({ config_type: configType });
      if (result.success) {
        saveMessage = `Settings reset to defaults successfully!`;
        setTimeout(() => (saveMessage = ""), 3000);
        // Refresh cached settings store and reload local view
        await settingsStore.refresh();
        await loadSettings(); // Reload settings after reset
      } else {
        error = result.message || "Failed to reset settings";
      }
    } catch (err) {
      error = "Failed to reset settings";
      console.error("Failed to reset settings:", err);
    } finally {
      isLoading = false;
    }
  }

  /**
   * Handle export button click
   */
  function handleExportClick() {
    console.log("Export button clicked!");
    console.log("settingsApi:", settingsApi);
    exportSettings();
  }

  /**
   * Handle reset button click
   */
  function handleResetClick() {
    console.log("Reset button clicked!");
    resetSettings();
  }

  /**
   * Export settings to JSON file
   */
  async function exportSettings() {
    console.log("exportSettings function called!");
    try {
      isLoading = true;
      error = null;

      console.log("About to call settingsApi.export()");
      const result = await settingsApi.export();
      console.log("Export result:", result);
      if (result.success && result.data) {
        // Create a clean export format
        const exportData = {
          version: result.data.version || "1.0",
          exported_at: result.data.exported_at || new Date().toISOString(),
          app_name: "Budget Helper",
          settings: result.data.settings || {},
        };

        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: "application/json" });
        const url = URL.createObjectURL(dataBlob);

        const timestamp = new Date().toISOString().split("T")[0];
        const filename = `ynab-settings-${timestamp}.json`;

        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        link.style.display = "none";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        saveMessage = `Settings exported successfully as ${filename}!`;
        setTimeout(() => (saveMessage = ""), 5000);
      } else {
        error =
          result.message || "Failed to export settings - no data received";
      }
    } catch (err) {
      error = "Failed to export settings - please try again";
      console.error("Failed to export settings:", err);
    } finally {
      isLoading = false;
    }
  }

  /**
   * Import settings from JSON file
   */
  async function importSettings(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
      isLoading = true;
      error = null;

      // Validate file type
      if (!file.name.toLowerCase().endsWith(".json")) {
        error = "Please select a valid JSON file";
        return;
      }

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        error = "File is too large. Maximum size is 10MB";
        return;
      }

      const text = await file.text();
      let importedData;

      try {
        importedData = JSON.parse(text);
      } catch (parseErr) {
        error =
          "Invalid JSON file format. Please check the file and try again.";
        return;
      }

      // Validate the imported data structure
      if (!importedData || typeof importedData !== "object") {
        error = "Invalid settings file format";
        return;
      }

      // Check if it's our export format or raw settings
      let settingsToImport;
      if (importedData.settings && typeof importedData.settings === "object") {
        // Our export format
        settingsToImport = importedData;

        // Show info about the import
        const exportDate = importedData.exported_at
          ? new Date(importedData.exported_at).toLocaleDateString()
          : "unknown";
        const settingsCount = Object.keys(importedData.settings).length;

        if (
          !confirm(
            `Import ${settingsCount} settings from ${exportDate}? This will overwrite existing settings.`,
          )
        ) {
          return;
        }
      } else {
        // Assume it's raw settings data
        settingsToImport = { settings: importedData };

        const settingsCount = Object.keys(importedData).length;
        if (
          !confirm(
            `Import ${settingsCount} settings? This will overwrite existing settings.`,
          )
        ) {
          return;
        }
      }

      const result = await settingsApi.import(settingsToImport);
      if (result.success) {
        const importedCount =
          result.data?.imported_configs?.length || "unknown";
        saveMessage = `Settings imported successfully! ${importedCount} settings updated.`;
        setTimeout(() => (saveMessage = ""), 5000);
        await loadSettings(); // Reload settings after import
      } else {
        error = result.message || "Failed to import settings";
      }
    } catch (err) {
      error =
        "Failed to import settings. Please check the file format and try again.";
      console.error("Failed to import settings:", err);
    } finally {
      isLoading = false;
      // Reset file input
      event.target.value = "";
    }
  }

  /**
   * Extract the actual value from a config value object
   */
  function extractConfigValue(configValue) {
    if (!configValue) return null;

    // Check each possible value type
    const valueTypes = [
      "stringValue",
      "intValue",
      "doubleValue",
      "boolValue",
      "stringList",
      "stringMap",
    ];
    for (const type of valueTypes) {
      if (configValue[type] !== undefined && configValue[type] !== null) {
        return configValue[type];
      }
    }
    return null;
  }

  /**
   * Handle settings changes from child components
   */
  async function handleSettingsChange(event) {
    const { key, value, type, description } = event.detail;

    // Create proper config value structure
    let configValue = {};
    if (typeof value === "string") {
      configValue.stringValue = value;
    } else if (typeof value === "number") {
      if (Number.isInteger(value)) {
        configValue.intValue = value;
      } else {
        configValue.doubleValue = value;
      }
    } else if (typeof value === "boolean") {
      configValue.boolValue = value;
    } else if (Array.isArray(value)) {
      configValue.stringList = value;
    } else if (typeof value === "object") {
      configValue.stringMap = value;
    } else {
      configValue.stringValue = String(value);
    }

    settings[key] = {
      key,
      type: type || "System",
      value: configValue,
      description: description || "",
    };

    settings = { ...settings }; // Trigger reactivity

    // Auto-save the setting immediately
    await autoSaveSetting(key, value, type, description);
  }

  /**
   * Auto-save a single setting
   */
  async function autoSaveSetting(key, value, type, description) {
    try {
      const configsToUpdate = [
        {
          key,
          value,
          type: type || "System",
          description: description || "",
        },
      ];

      const result = await settingsApi.updateAll({ configs: configsToUpdate });
      if (result.success) {
        console.log("Setting auto-saved successfully:", key);
        await settingsStore.refresh();
        // Show brief success message
        saveMessage = `${key.split(".").pop()} updated`;
        setTimeout(() => (saveMessage = ""), 2000);
      } else {
        console.error("Auto-save failed:", result.message);
        error = result.message || "Failed to save setting";
        setTimeout(() => (error = ""), 3000);
      }
    } catch (err) {
      console.error("Auto-save error:", err);
      error = "Failed to save setting";
      setTimeout(() => (error = ""), 3000);
    }
  }


  /**
   * Select a budget
   */
  async function selectBudget(budgetId) {
    const success = await budgetStore.selectBudget(budgetId);
    if (success) {
      saveMessage = "Budget selected successfully!";
      setTimeout(() => (saveMessage = ""), 3000);
      // Refresh budget data after selection
      await budgetStore.refreshFromYNAB();
    } else {
      console.error("Failed to select budget");
    }
  }

  /**
   * Handle authentication status changes
   */
  function handleAuthStatusChange(event) {
    const { service, authenticated } = event.detail;
    console.log(`${service} authentication status changed:`, authenticated);

    if (authStatusRef) {
      authStatusRef.updateServiceStatus(service, authenticated);
    }
  }

  /**
   * Handle successful authentication
   */
  function handleAuthConnected(event) {
    const { service } = event.detail;
    saveMessage = `${authConfigs[service].displayName} connected successfully!`;
    setTimeout(() => (saveMessage = ""), 3000);
  }

  /**
   * Handle disconnection
   */
  function handleAuthDisconnected(event) {
    const { service } = event.detail;
    saveMessage = `${authConfigs[service].displayName} disconnected successfully!`;
    setTimeout(() => (saveMessage = ""), 3000);
  }

  // Load settings and budgets on component mount
  onMount(() => {
    loadSettings();
  });
</script>

<div class="settings-layout" data-testid="settings-layout">
  <!-- Header -->
  <div class="mb-8">
    <h1 class="text-3xl font-bold text-foreground mb-2">Settings</h1>
    <p class="text-muted-foreground">
      Configure your application preferences and integrations
    </p>
  </div>

  <!-- Status Messages -->
  {#if saveMessage}
    <div
      class="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md p-4 mb-6"
      data-testid="success-message"
    >
      <div class="text-green-800 dark:text-green-200">{saveMessage}</div>
    </div>
  {/if}

  {#if error}
    <div
      class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4 mb-6"
      data-testid="error-message"
    >
      <div class="text-red-800 dark:text-red-200">{error}</div>
    </div>
  {/if}

  <!-- Authentication Status Overview -->
  <AuthStatus bind:this={authStatusRef} class="mb-6" />

  <!-- Action Buttons -->
  <div class="flex flex-wrap gap-3 mb-6 mt-8">
    <button
      class="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium outline-none transition-all focus-visible:ring-[3px] disabled:pointer-events-none disabled:opacity-50 bg-background shadow-xs hover:bg-accent hover:text-accent-foreground dark:bg-input/30 dark:border-input dark:hover:bg-input/50 border h-9 px-4 py-2"
      onclick={handleExportClick}
      disabled={isLoading}
      data-testid="export-settings-button"
    >
      {#if isLoading}
        Exporting...
      {:else}
        📥 Export Settings
      {/if}
    </button>

    <div class="relative inline-flex">
      <Button
        variant="outline"
        disabled={isLoading}
        data-testid="import-settings-button"
        onclick={() => {
          const fileInput = document.getElementById(
            "settings-import-file-input",
          );
          fileInput?.click();
        }}
      >
        {#if isLoading}
          Importing...
        {:else}
          📤 Import Settings
        {/if}
      </Button>
      <input
        id="settings-import-file-input"
        type="file"
        accept=".json"
        onchange={importSettings}
        disabled={isLoading}
        class="hidden"
        data-testid="import-file-input"
      />
    </div>

    <button
      class="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium outline-none transition-all focus-visible:ring-[3px] disabled:pointer-events-none disabled:opacity-50 bg-destructive shadow-xs hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60 text-white h-9 px-4 py-2"
      onclick={handleResetClick}
      disabled={isLoading}
      data-testid="reset-all-settings-button"
    >
      Reset All
    </button>
  </div>

  <!-- Tab Navigation -->
  <div class="border-b border-border mb-6">
    <nav class="flex space-x-8" data-testid="settings-tabs">
      {#each tabs as tab}
        <button
          class="py-2 px-1 border-b-2 font-medium text-sm transition-colors duration-200 {activeTab ===
          tab.id
            ? 'border-primary text-primary'
            : 'border-transparent text-muted-foreground hover:text-foreground hover:border-border'}"
          onclick={() => (activeTab = tab.id)}
          data-testid="tab-{tab.id}"
        >
          <span class="mr-2">{tab.icon}</span>
          {tab.label}
        </button>
      {/each}
    </nav>
  </div>

  <!-- Tab Content -->
  <div class="tab-content" data-testid="tab-content-{activeTab}">
    {#if isLoading}
      <div class="flex items-center justify-center py-12">
        <div class="text-muted-foreground">Loading settings...</div>
      </div>
    {:else}
      {#each tabs as tab}
        {#if activeTab === tab.id}
          <div class="space-y-6">
            <!-- Tab-specific reset button -->
            <div class="flex justify-between items-center">
              <h2 class="text-xl font-semibold text-foreground">
                {tab.label} Settings
              </h2>
              <Button
                variant="outline"
                size="sm"
                onclick={() => resetSettings(tab.id)}
                data-testid="reset-{tab.id}-settings-button"
              >
                Reset {tab.label}
              </Button>
            </div>

            <Separator />

            <!-- Render the appropriate settings component -->
            {#if tab.id === "system"}
              <SystemSettings
                {settings}
                onchange={handleSettingsChange}
                {budgets}
                {selectedBudgetId}
                {budgetsLoading}
                {budgetsError}
                {selectBudget}
                {authConfigs}
                onauthstatuschange={handleAuthStatusChange}
                onauthconnected={handleAuthConnected}
                onauthdisconnected={handleAuthDisconnected}
              />
            {:else if tab.id === "email"}
              <EmailSettings {settings} onchange={handleSettingsChange} />
            {:else if tab.id === "ai"}
              <AISettings {settings} onchange={handleSettingsChange} />
            {:else if tab.id === "external"}
              <ExternalSystemSettings
                {settings}
                onchange={handleSettingsChange}
              />
            {/if}
          </div>
        {/if}
      {/each}
    {/if}
  </div>
</div>

<style>
  .settings-layout {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
  }
</style>
