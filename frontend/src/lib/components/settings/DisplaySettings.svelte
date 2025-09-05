<!--
  @fileoverview Display Settings component for UI and presentation configuration
-->
<script>
  import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card/index.js';
  import { Label } from '../ui/label/index.js';
  import { Input } from '../ui/input/index.js';
  import DropdownSelect from '../ui/dropdown-select/DropdownSelect.svelte';
  import { Checkbox } from '../ui/checkbox/index.js';

  export let settings = {};
  export let onchange;

  // Display setting keys from backend ConfigKeys
  const DISPLAY_KEYS = {
    DEFAULT_SORT_ORDER: 'display.default_sort_order',
    TRANSACTIONS_PER_PAGE: 'display.transactions_per_page',
    VISIBLE_COLUMNS: 'display.visible_columns',
    THEME: 'display.theme'
  };

  // Available column options
  const COLUMN_OPTIONS = [
    { id: 'date', label: 'Date', description: 'Transaction date' },
    { id: 'payee', label: 'Payee', description: 'Merchant or payee name' },
    { id: 'category', label: 'Category', description: 'Transaction category' },
    { id: 'account', label: 'Account', description: 'Account name' },
    { id: 'amount', label: 'Amount', description: 'Transaction amount' },
    { id: 'memo', label: 'Memo', description: 'Transaction memo/notes' },
    { id: 'approved', label: 'Approved', description: 'Approval status' },
    { id: 'metadata', label: 'Metadata', description: 'Attached metadata count' }
  ];

  // Get setting value helper
  function getSettingValue(key, defaultValue = '') {
    const setting = settings[key];
    if (!setting || !setting.value) return defaultValue;
    
    // Extract value from ConfigValue union
    const value = setting.value;
    return value.stringValue ?? value.intValue ?? value.doubleValue ?? value.boolValue ?? value.stringList ?? defaultValue;
  }

  // Update setting helper
  function updateSetting(key, value, description = '') {
    const currentValue = getSettingValue(key);
    
    // Only update if the value has actually changed
    if (currentValue !== value) {
      onchange?.({ detail: { key, value, type: 'Display', description } });
    }
  }

  // Get visible columns as array
  function getVisibleColumns() {
    const visibleColumns = getSettingValue(DISPLAY_KEYS.VISIBLE_COLUMNS, 'all');
    if (visibleColumns === 'all') {
      return COLUMN_OPTIONS.map(col => col.id);
    }
    if (Array.isArray(visibleColumns)) {
      return visibleColumns;
    }
    if (typeof visibleColumns === 'string') {
      return visibleColumns.split(',').map(col => col.trim()).filter(Boolean);
    }
    return [];
  }

  // Update visible columns
  function updateVisibleColumns(columnId, checked) {
    let visibleColumns = getVisibleColumns();
    
    if (checked) {
      if (!visibleColumns.includes(columnId)) {
        visibleColumns.push(columnId);
      }
    } else {
      visibleColumns = visibleColumns.filter(col => col !== columnId);
    }
    
    // If all columns are selected, use 'all' shorthand
    if (visibleColumns.length === COLUMN_OPTIONS.length) {
      updateSetting(DISPLAY_KEYS.VISIBLE_COLUMNS, 'all', 'Which columns to display in the transaction grid');
    } else {
      updateSetting(DISPLAY_KEYS.VISIBLE_COLUMNS, visibleColumns, 'Which columns to display in the transaction grid');
    }
  }

  // Apply theme change immediately
  function handleThemeChange(theme) {
    updateSetting(DISPLAY_KEYS.THEME, theme, 'UI theme (light, dark, system)');
    
    // Apply theme immediately to document
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else if (theme === 'light') {
      root.classList.remove('dark');
    } else {
      // System theme
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (prefersDark) {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
    }
  }

  $: visibleColumns = getVisibleColumns();
</script>

<div class="display-settings space-y-6" data-testid="display-settings">
  <!-- Theme Configuration -->
  <Card>
    <CardHeader>
      <CardTitle>Theme & Appearance</CardTitle>
      <CardDescription>Configure the visual appearance of the application</CardDescription>
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="space-y-2">
        <Label for="theme-select">Theme</Label>
        <DropdownSelect
          value={getSettingValue(DISPLAY_KEYS.THEME, 'system')}
          options={[
            { value: 'system', label: 'System (Auto)' },
            { value: 'light', label: 'Light' },
            { value: 'dark', label: 'Dark' }
          ]}
          onSelect={handleThemeChange}
          placeholder="Select theme"
          searchable={false}
          data-testid="theme-select"
        />
        <p class="text-sm text-muted-foreground">
          Choose your preferred theme. System will follow your device's theme setting.
        </p>
      </div>
    </CardContent>
  </Card>

  <!-- Transaction Table Configuration -->
  <Card>
    <CardHeader>
      <CardTitle>Transaction Table</CardTitle>
      <CardDescription>Configure how transactions are displayed in the main table</CardDescription>
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="space-y-2">
          <Label for="sort-order-select">Default Sort Order</Label>
          <DropdownSelect
            value={getSettingValue(DISPLAY_KEYS.DEFAULT_SORT_ORDER, 'desc_by_date')}
            options={[
              { value: 'desc_by_date', label: 'Newest First' },
              { value: 'asc_by_date', label: 'Oldest First' },
              { value: 'desc_by_amount', label: 'Highest Amount' },
              { value: 'asc_by_amount', label: 'Lowest Amount' },
              { value: 'payee', label: 'Payee (A-Z)' },
              { value: 'category', label: 'Category (A-Z)' }
            ]}
            onSelect={(value) => updateSetting(DISPLAY_KEYS.DEFAULT_SORT_ORDER, value, 'Default sort order for transactions')}
            placeholder="Select sort order"
            searchable={false}
            data-testid="sort-order-select"
          />
        </div>

        <div class="space-y-2">
          <Label for="transactions-per-page">Transactions Per Page</Label>
          <DropdownSelect
            value={String(getSettingValue(DISPLAY_KEYS.TRANSACTIONS_PER_PAGE, 25))}
            options={['10', '25', '50', '100', '200']}
            onSelect={(value) => updateSetting(DISPLAY_KEYS.TRANSACTIONS_PER_PAGE, parseInt(value), 'Number of transactions to display per page')}
            placeholder="Select page size"
            searchable={false}
            data-testid="transactions-per-page-select"
          />
        </div>
      </div>
    </CardContent>
  </Card>

  <!-- Column Visibility Configuration -->
  <Card>
    <CardHeader>
      <CardTitle>Column Visibility</CardTitle>
      <CardDescription>Choose which columns to display in the transaction table</CardDescription>
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        {#each COLUMN_OPTIONS as column}
          <div class="flex items-center space-x-2">
            <Checkbox
              id="column-{column.id}"
              checked={visibleColumns.includes(column.id)}
              onCheckedChange={(checked) => updateVisibleColumns(column.id, checked)}
              data-testid="column-{column.id}-checkbox"
            />
            <div class="space-y-1">
              <Label for="column-{column.id}" class="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                {column.label}
              </Label>
              <p class="text-xs text-muted-foreground">
                {column.description}
              </p>
            </div>
          </div>
        {/each}
      </div>

      <div class="flex gap-2 pt-4 border-t">
        <button
          class="text-sm text-primary hover:underline"
          onclick={() => {
            COLUMN_OPTIONS.forEach(col => updateVisibleColumns(col.id, true));
          }}
          data-testid="select-all-columns-button"
        >
          Select All
        </button>
        <span class="text-muted-foreground">•</span>
        <button
          class="text-sm text-primary hover:underline"
          onclick={() => {
            COLUMN_OPTIONS.forEach(col => updateVisibleColumns(col.id, false));
          }}
          data-testid="deselect-all-columns-button"
        >
          Deselect All
        </button>
        <span class="text-muted-foreground">•</span>
        <button
          class="text-sm text-primary hover:underline"
          onclick={() => {
            // Reset to essential columns
            const essentialColumns = ['date', 'payee', 'category', 'amount'];
            COLUMN_OPTIONS.forEach(col => {
              updateVisibleColumns(col.id, essentialColumns.includes(col.id));
            });
          }}
          data-testid="reset-columns-button"
        >
          Reset to Essential
        </button>
      </div>
    </CardContent>
  </Card>

  <!-- Display Tips -->
  <Card>
    <CardHeader>
      <CardTitle>Display Tips</CardTitle>
      <CardDescription>Optimize your viewing experience</CardDescription>
    </CardHeader>
    <CardContent>
      <div class="space-y-3 text-sm">
        <div>
          <strong>Theme:</strong> Dark theme can reduce eye strain during extended use, while light theme may be better for detailed work.
        </div>
        <div>
          <strong>Pagination:</strong> Smaller page sizes (10-25) load faster, while larger sizes (50-100) reduce clicking but may be slower.
        </div>
        <div>
          <strong>Column Selection:</strong> Hide unused columns to focus on important data and improve table performance.
        </div>
        <div>
          <strong>Sort Order:</strong> "Newest First" is best for recent activity, while "Oldest First" helps with historical review.
        </div>
      </div>
    </CardContent>
  </Card>
</div>