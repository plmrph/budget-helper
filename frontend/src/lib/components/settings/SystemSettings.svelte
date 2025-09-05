<!--
  @fileoverview System Settings component for application-wide configuration
-->
<script>

  import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
  } from "../ui/card/index.js";
  import { Label } from "../ui/label/index.js";
  import { Input } from "../ui/input/index.js";
  import DropdownSelect from "../ui/dropdown-select/DropdownSelect.svelte";
  import AuthProvider from "../AuthProvider.svelte";
  import { settingsStore } from "$lib/stores/settings.js";

  export let settings = {};
  export let budgets = [];
  export let selectedBudgetId = null;
  export let budgetsLoading = false;
  export let budgetsError = null;
  export let selectBudget = () => {};
  export let authConfigs = {};
  export let onchange;
  export let onauthstatuschange;
  export let onauthconnected;
  export let onauthdisconnected;

  // System setting keys from backend ConfigKeys
  const SYSTEM_KEYS = {
    APP_NAME: "app.name",
    APP_VERSION: "app.version",
    DEFAULT_BUDGET_PLATFORM: "system.default_budget_platform",
    DEFAULT_EMAIL: "system.default_email",
    DEFAULT_MODEL_TYPE: "system.default_model_type",
    SELECTED_BUDGET_ID: "system.selected_budget_id",
    MAX_TRANSACTIONS_TO_LOAD: "system.max_transactions_to_load",
  };

  // Get setting value helper
  function getSettingValue(key, defaultValue = "") {
    const setting = settings[key];
    if (!setting || !setting.value) return defaultValue;

    // Extract value from ConfigValue union
    const value = setting.value;
    return (
      value.stringValue ??
      value.intValue ??
      value.doubleValue ??
      value.boolValue ??
      defaultValue
    );
  }

  // Update setting helper
  async function updateSetting(key, value, type = "System", description = "") {
    const currentValue = getSettingValue(key);
    
    // Only update if the value has actually changed
    if (currentValue !== value) {
      onchange?.({ detail: { key, value, type, description } });
      await settingsStore.refresh();
    }
  }

  // Handle authentication events
  function handleAuthStatusChange(event) {
    onauthstatuschange?.(event);
  }

  function handleAuthConnected(event) {
    onauthconnected?.(event);
  }

  function handleAuthDisconnected(event) {
    onauthdisconnected?.(event);
  }

  let maxTxError = "";
</script>

<div class="system-settings columns-1 lg:columns-2 gap-6 space-y-6" data-testid="system-settings">
  <!-- Application Information -->
  <Card class="break-inside-avoid mb-6">
    <CardHeader>
      <CardTitle>Application Information</CardTitle>
      <CardDescription
        >Basic application settings and identification</CardDescription
      >
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="space-y-2">
          <Label for="app-name">Application Name</Label>
          <Input
            id="app-name"
            value={getSettingValue(
              SYSTEM_KEYS.APP_NAME,
              "Budget Helper",
            )}
            readonly
            placeholder="Application name"
            data-testid="app-name-input"
            class="bg-muted"
          />
        </div>

        <div class="space-y-2">
          <Label for="app-version">Application Version</Label>
          <Input
            id="app-version"
            value={getSettingValue(SYSTEM_KEYS.APP_VERSION, "1.0.0")}
            readonly
            placeholder="Version number"
            data-testid="app-version-input"
            class="bg-muted"
          />
        </div>
      </div>
    </CardContent>
  </Card>

  <!-- YNAB Authentication -->
  <div class="break-inside-avoid mb-6">
    <AuthProvider
      config={authConfigs.ynab}
      onstatuschange={handleAuthStatusChange}
      onconnected={handleAuthConnected}
      ondisconnected={handleAuthDisconnected}
      data-testid="ynab-auth-provider"
    />
  </div>


  <!-- Budget Selection -->
  <Card class="break-inside-avoid mb-6">
    <CardHeader>
      <CardTitle>Budget Selection</CardTitle>
      <CardDescription>Choose which YNAB budget to sync transactions with</CardDescription>
    </CardHeader>
    <CardContent>
      {#if budgetsError}
        <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-3 mb-4">
          <div class="text-red-800 dark:text-red-200 text-sm">{budgetsError}</div>
        </div>
      {/if}

      {#if budgetsLoading}
        <div class="text-muted-foreground text-sm">Loading budgets...</div>
      {:else if budgets.length === 0}
        <div class="text-muted-foreground text-sm">
          No budgets available. Please connect to YNAB first.
        </div>
      {:else}
        <div class="space-y-3">
          {#each budgets as budget}
            <div class="flex items-center">
              <input
                type="radio"
                id="budget-{budget.id}"
                name="selected-budget"
                value={budget.id}
                checked={selectedBudgetId === budget.id}
                onchange={async () => {
                  selectBudget(budget.id);
                  await updateSetting(SYSTEM_KEYS.SELECTED_BUDGET_ID, budget.id, 'System', 'Selected budget ID');
                }}
                class="h-4 w-4 text-primary focus:ring-primary border-border"
                data-testid="budget-radio-{budget.id}"
              />
              <label for="budget-{budget.id}" class="ml-3 block text-sm">
                <div class="text-foreground font-medium">{budget.name}</div>
                <div class="text-muted-foreground text-xs">{budget.currency}</div>
              </label>
            </div>
          {/each}
        </div>
      {/if}
    </CardContent>
  </Card>


  <!-- Default Platform Settings -->
  <Card class="break-inside-avoid mb-6">
    <CardHeader>
      <CardTitle>Default Platform Settings</CardTitle>
      <CardDescription
        >Configure default platforms for various integrations</CardDescription
      >
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="space-y-4">
        <div class="space-y-2">
          <Label for="default-budget-platform">Default Budget Platform</Label>
          <DropdownSelect
            value={getSettingValue(SYSTEM_KEYS.DEFAULT_BUDGET_PLATFORM, "YNAB")}
            options={["YNAB"]}
            onSelect={async (value) =>
              await updateSetting(
                SYSTEM_KEYS.DEFAULT_BUDGET_PLATFORM,
                value,
                "System",
                "Default budget platform",
              )}
            placeholder="Select budget platform"
            searchable={false}
            data-testid="default-budget-platform-select"
          />
        </div>

        <div class="space-y-2">
          <Label for="max-transactions">Max Transactions to Load</Label>
          <Input
            id="max-transactions"
            type="number"
            min="50"
            max="10000"
            step="50"
            value={getSettingValue(SYSTEM_KEYS.MAX_TRANSACTIONS_TO_LOAD, 500)}
            oninput={(e) => {
              const raw = e.currentTarget.value;
              const num = parseInt(raw, 10);
              if (isNaN(num)) {
                maxTxError = "Please enter a number";
                return;
              }
              if (num < 50) {
                maxTxError = "Must be at least 50";
                return; // do not auto-save invalid values
              }
              if (num > 10000) {
                maxTxError = "Must be 10,000 or less";
                return;
              }
              maxTxError = "";
            }}
            onblur={async (e) => {
              // Persist only when the field loses focus
              const raw = e.currentTarget.value;
              const num = parseInt(raw, 10);
              if (isNaN(num) || num < 50 || num > 10000) {
                // invalid — do nothing
                return;
              }

              // Update backend setting
              await updateSetting(SYSTEM_KEYS.MAX_TRANSACTIONS_TO_LOAD, num, "System", "Max transactions to load");

              // Call global reload function if available (fallback)
              if (typeof window !== 'undefined' && window.reloadTransactions) {
                window.reloadTransactions();
              }
            }}
            placeholder="Max transactions to load"
            data-testid="max-transactions-input"
          />
          {#if maxTxError}
            <div class="text-xs text-destructive mt-1">{maxTxError}</div>
          {/if}
          <span class="text-xs text-muted-foreground">Controls how many transactions are loaded at once in the Transactions page.</span>
        </div>

        <div class="space-y-2">
          <Label for="default-email">Default Email Provider</Label>
          <DropdownSelect
            value={getSettingValue(SYSTEM_KEYS.DEFAULT_EMAIL, "GMAIL")}
            options={["GMAIL"]}
            onSelect={async (value) =>
              await updateSetting(
                SYSTEM_KEYS.DEFAULT_EMAIL,
                value,
                "System",
                "Default email provider",
              )}
            placeholder="Select email provider"
            searchable={false}
            data-testid="default-email-select"
          />
        </div>

        <div class="space-y-2">
          <Label for="default-model-type">Default ML Model</Label>
          <DropdownSelect
            value={getSettingValue(SYSTEM_KEYS.DEFAULT_MODEL_TYPE, "PXBlendSC")}
            options={["PXBlendSC"]}
            onSelect={async (value) =>
              await updateSetting(
                SYSTEM_KEYS.DEFAULT_MODEL_TYPE,
                value,
                "System",
                "Default ML model type",
              )}
            placeholder="Select ML model"
            searchable={false}
            data-testid="default-model-type-select"
          />
        </div>
      </div>
    </CardContent>
  </Card>



  <!-- Gmail Authentication -->
  <div class="break-inside-avoid mb-6">
    <AuthProvider
      config={authConfigs.gmail}
      onstatuschange={handleAuthStatusChange}
      onconnected={handleAuthConnected}
      ondisconnected={handleAuthDisconnected}
      data-testid="gmail-auth-provider"
    />
  </div>
</div>
