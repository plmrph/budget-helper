<!--
  @fileoverview Transactions page component - displays and manages YNAB transactions
-->
    {#snippet header({ table })}
  <div class="flex flex-wrap items-center gap-2 w-full mb-2 justify-end">
        <Button
          variant="outline"
          size="sm"
          onclick={() => handleBulkEmailSearch(table)}
          disabled={isBulkEmailSearching || !$gmailAuth.isAuthenticated}
          title="Search emails for all visible transactions on this page"
          class="h-8"
        >
          {#if isBulkEmailSearching}
            <div class="animate-spin h-3 w-3 border-2 border-current border-t-transparent rounded-full mr-2"></div>
            Searching...
          {:else}
            <Mail class="h-3 w-3 mr-2" />
            Email Search
          {/if}
        </Button>
        <Button
          variant="outline"
          size="sm"
          onclick={() => {
            if ($bulkMLPredictStore.hasDefaultModel) {
              handleBulkMLSearch(table);
            }
          }}
          disabled={$bulkMLPredictStore.isPredicting || !$bulkMLPredictStore.hasDefaultModel}
          title="Predict categories for all visible transactions on this page using ML"
          class="h-8"
        >
          {#if $bulkMLPredictStore.isPredicting}
            <div class="animate-spin h-3 w-3 border-2 border-current border-t-transparent rounded-full mr-2"></div>
            Predicting...
          {:else}
            <Brain class="h-3 w-3 mr-2" />
            ML Predict
          {/if}
        </Button>
      </div>
    {/snippet}

<script>
  import { transactionApi, settingsApi } from "../api/client.js";
  import { authStore, gmailAuth } from "../stores/auth.js";
  import { budgetStore } from "../stores/budget.js";
  import { transactionStore, enhancedTransactions } from "../stores/transactions.js";
  import { bulkEmailSearchStore } from "../stores/bulkEmailSearch.js";
  import { bulkMLPredictStore } from "../stores/bulkMLPredict.js";
  import { get } from "svelte/store";
  import AuthProvider from "../components/AuthProvider.svelte";
  import AdvancedDataTable from "../components/AdvancedDataTable.svelte";
  import { createRawSnippet, onMount, onDestroy } from "svelte";
  import { renderComponent, renderSnippet } from "$lib/components/ui/data-table/index.js";
  import DataTableColumnHeader from "../components/data-table/data-table-column-header.svelte";
  import { Checkbox } from "$lib/components/ui/checkbox/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import * as DropdownMenu from "$lib/components/ui/dropdown-menu/index.js";
  import { Ellipsis, Mail, Brain } from "@lucide/svelte";
  import { CheckCircle, CircleHelp } from "@lucide/svelte";
  import MetadataColumn from "../components/MetadataColumn.svelte";
  import SyncBudgetDialog from "../components/SyncBudgetDialog.svelte";
  import { EditableTextCell, EditableCategoryCell, EditableApprovalCell, EditableMLCategoryCell } from "../components/ui/editable-cells/index.js";

      /**
       * Get unique filter options for faceted filters
       */
  function getUniquePayees(transactions) {
    // Only use payees from the currently loaded transactions to keep the list small
    // Expect transactions to have an enhanced `payeeName` property
    if (!Array.isArray(transactions) || transactions.length === 0) return [];
    const unique = new Map();
    for (const t of transactions) {
      const name = (t?.payeeName || "").trim();
      if (!name) continue;
      if (!unique.has(name)) unique.set(name, { label: name, value: name });
    }
    return Array.from(unique.values());
  }

  function getUniqueAccounts(transactions) {
    const budgetData = get(budgetStore);
    if (!budgetData.accounts) return [];

    // Use a Map to deduplicate by name while preserving the first occurrence
    const uniqueAccounts = new Map();
    budgetData.accounts.forEach((account) => {
      if (!uniqueAccounts.has(account.name)) {
        uniqueAccounts.set(account.name, {
          label: account.name,
          value: account.name,
        });
      }
    });

    return Array.from(uniqueAccounts.values());
  }

  function getUniqueCategories(transactions) {
    const budgetData = get(budgetStore);
    if (!budgetData.categories) return [];

    // Use a Map to deduplicate by name while preserving the first occurrence
    const uniqueCategories = new Map();
    budgetData.categories.forEach((category) => {
      if (!uniqueCategories.has(category.name)) {
        uniqueCategories.set(category.name, {
          label: category.name,
          value: category.name,
        });
      }
    });

    return Array.from(uniqueCategories.values());
  }

  function getUniqueMemos(transactions) {
    const memos = new Set();
    transactions.forEach((t) => {
      if (t.memo && t.memo.trim()) {
        memos.add(t.memo.trim());
      }
    });
    return Array.from(memos).map((memo) => ({ label: memo, value: memo }));
  }

  // Handle save functions for editable cells
  function handleMemoSave(transactionId, newMemo) {
    // Non-blocking save for immediate UI response
    transactionStore.updateTransactionField(transactionId, {
      memo: newMemo,
    });
  }

  function handleCategorySave(transactionId, newCategory) {
    transactionStore.updateTransactionField(transactionId, {
      categoryId: newCategory,
    });
  }

  function handleApprovalSave(transactionId, approved) {
    transactionStore.updateTransactionField(transactionId, {
      approved: approved,
    });
  }

  let syncInProgress = false;
  let syncMessage = null;
  let showImportDialog = false;
  let showExportDialog = false;
  // showSyncBudget is declared earlier; remove duplicate declaration to avoid compile error

  // Bulk search state
  let isBulkEmailSearching = false;
  let isBulkMLSearching = false;

  // Client-side table state mirrors for server requests
  let currentSorting = [{ column: "date", direction: "desc" }];
  let currentFilters = {};

  // UI state for table
  let pageSize = 20;
  let searchValue = "";

  // Get reactive state from transaction store
  $: ({ transactions, loading: isLoading, error, hasLoadedForCurrentBudget } = $transactionStore);
  
  let lastTransactionsRef = [];
  let memoizedEnhancedTransactions = [];
  $: {
    if (transactions !== lastTransactionsRef) {
      lastTransactionsRef = transactions;
      memoizedEnhancedTransactions = $enhancedTransactions;
    }
  }
  $: transactionsData = memoizedEnhancedTransactions;
  
  // Get bulk email search state
  $: ({ isSearching: isBulkEmailSearching, totalAttached, error: bulkSearchError } = $bulkEmailSearchStore);

  // Memoize filter options to prevent recalculation on every render
  let cachedPayeeOptions = [];
  let cachedAccountOptions = [];
  let cachedCategoryOptions = [];
  let lastBudgetUpdate = null;
  let lastTransactionsLength = 0;

  $: {
    const budgetData = $budgetStore;
    const currentLength = transactionsData.length;
    
    if (budgetData.lastUpdated !== lastBudgetUpdate || 
        Math.abs(currentLength - lastTransactionsLength) > 10) {
      lastBudgetUpdate = budgetData.lastUpdated;
      lastTransactionsLength = currentLength;
      
      requestAnimationFrame(() => {
        cachedPayeeOptions = getUniquePayees(transactionsData);
        cachedAccountOptions = getUniqueAccounts(transactionsData);
        cachedCategoryOptions = getUniqueCategories(transactionsData);
      });
    }
  }

  // Column definitions for the advanced data table
  // Order: checkbox, date, payee, account, memo, amount, category, status, edit
  const columns = [
    // Selection column
    // {
    //   id: "select",
    //   size: 40,
    //   minSize: 40,
    //   maxSize: 40,
    //   enableResizing: false,
    //   header: ({ table }) =>
    //     renderComponent(Checkbox, {
    //       checked: table.getIsAllPageRowsSelected(),
    //       indeterminate:
    //         table.getIsSomePageRowsSelected() &&
    //         !table.getIsAllPageRowsSelected(),
    //       onCheckedChange: (value) => table.toggleAllPageRowsSelected(!!value),
    //       "aria-label": "Select all",
    //     }),
    //   cell: ({ row }) =>
    //     renderComponent(Checkbox, {
    //       checked: row.getIsSelected(),
    //       onCheckedChange: (value) => row.toggleSelected(!!value),
    //       "aria-label": "Select row",
    //     }),
    // },
    // Date column
    {
      accessorKey: "date",
      header: ({ column }) =>
        renderComponent(DataTableColumnHeader, { column, title: "Date" }),
      size: 100,
      minSize: 75,
      maxSize: 75,
      cell: ({ row }) => {
        const dateSnippet = createRawSnippet((getDate) => {
          const date = new Date(getDate()).toLocaleDateString();
          return {
            render: () => `<div class="text-sm">${date}</div>`,
          };
        });
        return renderSnippet(dateSnippet, row.getValue("date"));
      },
    },
    // Payee column (display name from metadata)
    {
      accessorKey: "payeeId",
      header: ({ column }) =>
        renderComponent(DataTableColumnHeader, { column, title: "Payee" }),
      size: 100,
      minSize: 80,
      maxSize: 500,
      cell: ({ row }) => {
        const payeeSnippet = createRawSnippet((getName) => {
          const name = getName() || "Unknown";
          return {
            render: () => `<div class="font-medium truncate" title="${name}">${name}</div>`,
          };
        });
        return renderSnippet(payeeSnippet, row.original.payeeName);
      },
    },
    // Account column (display name from metadata)
    {
      accessorKey: "accountId",
      header: ({ column }) =>
        renderComponent(DataTableColumnHeader, { column, title: "Account" }),
      size: 120,
      minSize: 100,
      maxSize: 180,
      cell: ({ row }) => {
        const accountSnippet = createRawSnippet((getName) => {
          const name = getName() || "Unknown";
          return {
            render: () => `<div class="text-sm text-muted-foreground truncate" title="${name}">${name}</div>`,
          };
        });
        return renderSnippet(accountSnippet, row.original.accountName);
      },
    },
    // Memo column - flexible width that fills available space (EDITABLE)
    {
      id: "memo",
      accessorKey: "memo",
      header: ({ column }) =>
        renderComponent(DataTableColumnHeader, { column, title: "Memo" }),
      minSize: 50,
      maxSize: 1200,
      cell: ({ row }) => {
        const transaction = row.original;
        return renderComponent(EditableTextCell, {
          value: transaction.memo || "",
          onSave: (newMemo) => handleMemoSave(transaction.id, newMemo),
          placeholder: "Add memo",
          saveStatus: transaction._saveStatus,
          saveError: transaction._saveError,
        });
      },
      filterFn: (row, id, value) => {
        if (!value || !Array.isArray(value)) return true;
        return value.includes(row.getValue(id));
      },
    },
    // Amount column with formatting
    {
      accessorKey: "amount",
      header: ({ column }) =>
        renderComponent(DataTableColumnHeader, { column, title: "Amount" }),
      size: 110,
      minSize: 90,
      maxSize: 140,
      cell: ({ row }) => {
        const amountSnippet = createRawSnippet((getAmount) => {
          const amount = getAmount();
          const formatted = formatAmount(amount);
          const colorClass =
            amount < 0
              ? "text-red-600 dark:text-red-400"
              : "text-green-600 dark:text-green-400";
          return {
            render: () =>
              `<div class="text-right font-medium ${colorClass}">${formatted}</div>`,
          };
        });
        return renderSnippet(amountSnippet, row.getValue("amount"));
      },
    },
    // Category column (display name from budgetStore) (EDITABLE)
    {
      accessorKey: "categoryId",
      header: ({ column }) =>
        renderComponent(DataTableColumnHeader, { column, title: "Category" }),
      size: 130,
      minSize: 100,
      maxSize: 200,
      cell: ({ row }) => {
        const transaction = row.original;
  // Categories list for selection
  const categories = $budgetStore.categories || [];
  const categoryName = row.original.categoryName || "Uncategorized";

        return renderComponent(EditableCategoryCell, {
          value: transaction.categoryId || "",
          categories: categories.map((cat) => ({
            value: cat.id,
            label: cat.name,
          })),
          onSave: (selectedCategoryId) =>
            handleCategorySave(transaction.id, selectedCategoryId),
          placeholder: "Select category",
          saveStatus: transaction._saveStatus,
          saveError: transaction._saveError,
          displayValue: categoryName,
        });
      },
    },
    // Status column with styling (EDITABLE)
    {
      accessorKey: "approved",
      header: ({ column }) => {
        const headerSnippet = createRawSnippet(() => {
          return {
            render: () => `
              <div class="flex items-center justify-center gap-1" title="Approved / Unapproved">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-green-600 dark:text-green-400"><path d="M9 12l2 2 4-4"></path><circle cx="12" cy="12" r="10"></circle></svg>
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-yellow-600 dark:text-yellow-400"><path d="M12 17h.01"></path><path d="M12 7a4 4 0 0 1 4 4c0 1.5-.5 2-2 3l-2 1"></path><circle cx="12" cy="12" r="10"></circle></svg>
              </div>
            `,
          };
        });
        return renderSnippet(headerSnippet);
      },
  size: 56,
  minSize: 44,
  maxSize: 72,
      cell: ({ row }) => {
        const transaction = row.original;
        return renderComponent(EditableApprovalCell, {
          value: transaction.approved || false,
          onSave: (approved) => handleApprovalSave(transaction.id, approved),
          saveStatus: transaction._saveStatus,
          saveError: transaction._saveError,
        });
      },
    },
    // ML Category column - shows predicted categories and allows setting them
    {
      id: "mlCategory",
      header: ({ column }) => {
        const headerSnippet = createRawSnippet(() => {
          return {
            render: () =>
              `<div class="text-center font-medium">ML Category</div>`,
          };
        });
        return renderSnippet(headerSnippet);
      },
      size: 140,
      minSize: 120,
      maxSize: 200,
      enableSorting: false,
      cell: ({ row }) => {
        const transaction = row.original;
        return renderComponent(EditableMLCategoryCell, {
          transactionId: transaction.id,
          onCategorySet: (categoryId) =>
            handleCategorySave(transaction.id, categoryId),
          onApprove: (approved) => handleApprovalSave(transaction.id, approved),
          saveStatus: transaction._saveStatus,
          saveError: transaction._saveError,
        });
      },
    },
    // Metadata column - shows email and other metadata icons
    {
      id: "metadata",
      size: 60,
      minSize: 60,
      maxSize: 60,
      enableResizing: false,
      enableSorting: false,
      header: () => {
        const metadataSnippet = createRawSnippet(() => {
          return {
            render: () => `
              <div class="flex items-center justify-center" title="Metadata">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide-mail">
                  <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                  <polyline points="22,6 12,13 2,6"/>
                </svg>
              </div>
            `,
          };
        });
        return renderSnippet(metadataSnippet);
      },
      cell: ({ row }) => {
        const transaction = row.original;
        return renderComponent(MetadataColumn, {
          transaction,
        });
      },
    },
    // // Actions column - beautiful ellipsis button
    // {
    //   id: "actions",
    //   size: 40,
    //   minSize: 40,
    //   maxSize: 40,
    //   enableResizing: false,
    //   cell: ({ row }) => {
    //     const transaction = row.original;
    //     const actionsSnippet = createRawSnippet(() => {
    //       return {
    //         render: () => `
    //           <button
    //             class="flex h-8 w-8 items-center justify-center rounded-md p-0 text-sm font-medium transition-all hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
    //             onclick="console.log('Edit transaction:', '${transaction.id}')"
    //             aria-label="Open menu"
    //             type="button"
    //           >
    //             <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide-ellipsis">
    //               <circle cx="12" cy="12" r="1"></circle>
    //               <circle cx="19" cy="12" r="1"></circle>
    //               <circle cx="5" cy="12" r="1"></circle>
    //             </svg>
    //             <span class="sr-only">Open Menu</span>
    //           </button>
    //         `,
    //       };
    //     });
    //     return renderSnippet(actionsSnippet);
    //   },
    // },
    // Hidden columns for filtering - these should never be visible
    {
      accessorKey: "searchableText",
      enableColumnFilter: false,
      enableSorting: false,
      enableHiding: false,
    },
    {
      accessorKey: "payeeName",
      enableColumnFilter: true,
      enableSorting: false,
      enableHiding: false,
      filterFn: (row, id, value) => {
        if (!value || !Array.isArray(value)) return true;
        return value.includes(row.getValue(id));
      },
    },
    {
      accessorKey: "accountName",
      enableColumnFilter: true,
      enableSorting: false,
      enableHiding: false,
      filterFn: (row, id, value) => {
        if (!value || !Array.isArray(value)) return true;
        return value.includes(row.getValue(id));
      },
    },
    {
      accessorKey: "categoryName",
      enableColumnFilter: true,
      enableSorting: false,
      enableHiding: false,
      filterFn: (row, id, value) => {
        if (!value || !Array.isArray(value)) return true;
        const transaction = row.original;
  return value.includes(row.getValue(id));
      },
    },
    {
      accessorKey: "approvedString",
      enableColumnFilter: true,
      enableSorting: false,
      enableHiding: false,
      filterFn: (row, id, value) => {
        if (!value || !Array.isArray(value)) return true;
        return value.includes(row.getValue(id));
      },
    },
  ];

  // YNAB authentication configuration
  const ynabAuthConfig = {
    serviceName: "ynab",
    displayName: "YNAB",
    description: "Connect to your YNAB budget to sync transactions",
    requiresInput: true,
    inputType: "password",
    inputLabel: "Personal Access Token",
    inputPlaceholder: "Enter your YNAB Personal Access Token",
    helpText:
      "You can find your Personal Access Token in YNAB Settings > Developer Settings",
  };

  // Get YNAB authentication status from store
  $: ynabAuthenticated = $authStore.ynab?.isAuthenticated || false;

  // Get budget information from store
  $: ({ selectedBudgetId } = $budgetStore);

  /**
   * Load transactions using the transaction store
   * Only loads if no transactions are currently loaded or force is true
   * @param {boolean} force - Force reload even if transactions are already loaded
   */
  async function loadTransactions(force = false) {
    // Only load if we haven't loaded for this budget or if forced
    if (!force && hasLoadedForCurrentBudget && transactions && transactions.length > 0) {
      console.log("Transactions already loaded for current budget, skipping reload");
      return;
    }

    const params = {
      sorting: currentSorting,
      ...currentFilters,
    };

    await transactionStore.load(params);
  }

  /**
   * Force reload transactions - exposed for external use (e.g. from Settings page)
   * This can be called when settings change to refresh the transaction list
   */
  function forceReloadTransactions() {
    console.log("Force reloading transactions due to settings change");
    transactionStore.markForReload();
    loadTransactions(true);
  }

  // Expose the force reload function globally for settings page
  if (typeof window !== 'undefined') {
    window.reloadTransactions = forceReloadTransactions;
  }

  /**
   * Format amount from milliunits to currency
   * @param {number} milliunits - Amount in milliunits
   * @returns {string} Formatted currency string
   */
  function formatAmount(milliunits) {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(milliunits / 1000);
  }

  function handleRowSelectionChange(selectedRows) {
    console.log("Selected transactions:", selectedRows);
  }

  /**
   * Handle sorting changes from the data table - reload data with new sort
   */
  function handleSortingChange(sorting) {
    currentSorting = sorting;
    loadTransactions(true); // Force reload with new sort order
  }

  /**
   * Handle filter changes from the data table - reload data with new filters
   */
  function handleFiltersChange(filters) {
    // Convert client-side filters to server-side parameters for major filters
    const serverFilters = {};

    // Only apply server-side filters for major constraints
    if (filters.approved !== undefined) {
      serverFilters.approved = filters.approved;
    }

    // Handle range filters that need server-side processing
    if (filters.amount) {
      if (Array.isArray(filters.amount) && filters.amount.length === 2) {
        serverFilters.amount_min = filters.amount[0];
        serverFilters.amount_max = filters.amount[1];
      }
    }

    // Only reload if server-side filters changed
    const filtersChanged =
      JSON.stringify(currentFilters) !== JSON.stringify(serverFilters);
    currentFilters = serverFilters;

    if (filtersChanged) {
      loadTransactions(true); // Force reload with new filters
    }
  }

  // Load setting for max transactions on mount once settings are available
  onMount(async () => {
    await budgetStore.loadSelectedBudgetId();
    await loadTransactions();
  });

  // Cleanup global function on component destroy
  onDestroy(() => {
    if (typeof window !== 'undefined') {
      delete window.reloadTransactions;
    }
  });

  /**
   * Sync transactions from YNAB
   */
  async function syncFromYnab() {
    // Open preview dialog instead of immediate import
    showImportDialog = true;
  }

  /**
   * Handle bulk email search for visible transactions on current page
   */
  async function handleBulkEmailSearch(table) {
    if (isBulkEmailSearching) {
      return;
    }

    try {
      // Get only the visible transactions on the current page
      const visibleRows = table.getRowModel().rows;
      const visibleTransactions = visibleRows.map(row => row.original);
      
      if (visibleTransactions.length === 0) {
        return;
      }

      console.log(`Starting bulk email search for ${visibleTransactions.length} visible transactions`);
      
      await bulkEmailSearchStore.searchForTransactions(visibleTransactions);
      
      // Show success message if any emails were attached
      if (totalAttached > 0) {
        syncMessage = `Bulk email search completed! ${totalAttached} emails were automatically attached.`;
        setTimeout(() => {
          syncMessage = null;
        }, 5000);
      }
    } catch (err) {
      error = "Failed to perform bulk email search";
      console.error("Bulk email search failed:", err);
    }
  }

  /**
   * Handle bulk ML category prediction for visible transactions on current page
   * - processes only visible rows
   * - batches requests via store to avoid overloading backend
   * - updates ML column reactively without page refresh
   */
  async function handleBulkMLSearch(table) {
    if ($bulkMLPredictStore?.isPredicting) {
      return;
    }

    try {
      const visibleRows = table.getRowModel().rows;
      const visibleTransactions = visibleRows.map((row) => row.original);
      if (visibleTransactions.length === 0) return;

      console.log(
        `Starting bulk ML prediction for ${visibleTransactions.length} visible transactions`,
      );

      isBulkMLSearching = true;
      await bulkMLPredictStore.predictForTransactions(visibleTransactions);
      isBulkMLSearching = false;

      const processed = $bulkMLPredictStore.totalProcessed || 0;
      const withPreds = $bulkMLPredictStore.totalWithPredictions || 0;
      syncMessage = `ML predictions complete: ${withPreds}/${processed} had predictions.`;
      setTimeout(() => {
        syncMessage = null;
      }, 4000);
    } catch (err) {
      isBulkMLSearching = false;
      error = "Failed to perform bulk ML prediction";
      console.error("Bulk ML prediction failed:", err);
    }
  }
  
  // Track when setting becomes available and force reload if needed
  let hasReloadedForSetting = false;
  
  // Reactive statement to reload when setting becomes available and we already have transactions
  $: if (hasLoadedForCurrentBudget && !hasReloadedForSetting && transactions.length > 0) {
    hasReloadedForSetting = true;
    transactionStore.markForReload();
    loadTransactions(true);
  }

  let showSyncBudget = false;

  function openSyncBudgetDialog() {
    showSyncBudget = true;
  }

  const loadUnifiedPreview = async () => {
    return await transactionApi.previewUnified();
  };
  const applyUnifiedPlan = async (plan) => {
    const res = await transactionApi.applyUnified(plan);
    if (res?.success !== false) {
      await transactionStore.refresh();
    }
    return res;
  };
  const resetSyncTracking = async () => {
    return await transactionApi.resetSyncTracking();
  };
</script>

<div data-testid="transactions-page">
  <style>
    /* Ensure memo column text wraps properly and doesn't overflow */
    :global(.memo-cell) {
      word-wrap: break-word;
      overflow-wrap: break-word;
      white-space: normal;
      max-width: 100%;
    }

    /* Ensure memo column can expand */
    :global(table) {
      table-layout: fixed;
    }

    /* Allow memo column text to wrap properly */
    :global(.memo-cell) {
      white-space: normal;
      word-wrap: break-word;
      overflow-wrap: break-word;
    }

    /* Smooth resize transitions */
    :global(.resize-handle) {
      transition: background-color 0.2s ease;
    }
  </style>
  <div class="mb-6">
    <div class="flex flex-wrap items-center justify-between gap-4">
      <div>
        <h1
          class="text-3xl font-bold text-foreground mb-2"
          data-testid="transactions-title"
        >
          Transactions
        </h1>
        <p class="text-muted-foreground">
          Manage and reconcile your YNAB transactions
          {#if selectedBudgetId}
            {#if $budgetStore.budgets.length > 0}
              {@const selectedBudget = $budgetStore.budgets.find(
                (b) => b.id === selectedBudgetId,
              )}
              {#if selectedBudget}
                <span
                  class="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200"
                >
                  Budget: {selectedBudget.name}
                </span>
              {/if}
            {/if}
          {:else}
            <span
              class="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 dark:bg-yellow-900 text-gray-900 dark:text-yellow-200"
            >
              No budget selected
            </span>
          {/if}
        </p>
      </div>
      {#if ynabAuthenticated && selectedBudgetId}
        <button
          onclick={() => (showSyncBudget = true)}
          class="inline-flex items-center px-4 py-2 border border-input text-sm font-medium rounded-md shadow-sm text-foreground bg-background hover:bg-accent focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ring disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Sync Budget
        </button>
      {/if}
    </div>
  </div>

  {#if error}
    <div
      class="bg-destructive/10 border border-destructive/20 rounded-md p-4 mb-6"
      data-testid="error-message"
    >
      <div class="text-destructive">{error}</div>
    </div>
  {/if}

  {#if bulkSearchError}
    <div
      class="bg-destructive/10 border border-destructive/20 rounded-md p-4 mb-6"
      data-testid="bulk-search-error-message"
    >
      <div class="text-destructive">Bulk search error: {bulkSearchError}</div>
    </div>
  {/if}

  {#if syncMessage}
    <div
      class="bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 rounded-md p-4 mb-6"
      data-testid="sync-message"
    >
      <div class="text-green-800 dark:text-green-200">{syncMessage}</div>
    </div>
  {/if}

  <SyncBudgetDialog
    bind:open={showSyncBudget}
    title="Sync Budget"
  loadPreview={(payload) => transactionApi.previewUnified(payload)}
    applyPlan={(plan) => transactionApi.applyUnified(plan)}
    on:applied={async () => {
      // refresh list after apply
  await transactionStore.refresh();
      showSyncBudget = false;
    }}
  />

  <!-- YNAB Authentication -->
  {#if !ynabAuthenticated}
    <div class="mb-6">
      <AuthProvider
        config={ynabAuthConfig}
        onconnected={() => {
          syncMessage =
            "YNAB connected successfully! You can now sync transactions.";
          setTimeout(() => {
            syncMessage = null;
          }, 5000);
        }}
      />
    </div>
  {:else if !selectedBudgetId}
    <div
      class="bg-yellow-50 dark:bg-yellow-950 border border-yellow-200 dark:border-yellow-800 rounded-md p-4 mb-6"
    >
      <div class="flex">
        <div class="flex-shrink-0">
          <svg
            class="h-5 w-5 text-yellow-500"
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fill-rule="evenodd"
              d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
              clip-rule="evenodd"
            />
          </svg>
        </div>
        <div class="ml-3">
          <h3 class="text-sm font-medium text-gray-900 dark:text-yellow-200">
            No Budget Selected
          </h3>
          <div class="mt-2 text-sm text-gray-800 dark:text-yellow-300">
            <p>
              Please select a budget in <a
                href="#/settings"
                class="font-medium underline hover:text-gray-700 dark:hover:text-yellow-400"
                >Settings</a
              > to view and sync transactions.
            </p>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Transactions Table -->
  {#if isLoading}
    <div
      class="bg-card rounded-lg shadow border border-border p-6 text-center"
      data-testid="loading-message"
    >
      <div class="text-muted-foreground">Loading transactions...</div>
    </div>
  {:else if transactionsData.length === 0}
    <div
      class="bg-card rounded-lg shadow border border-border p-6 text-center"
      data-testid="no-transactions-message"
    >
      <div class="text-muted-foreground">No transactions found.</div>
    </div>
  {:else}



  <AdvancedDataTable
      data={transactionsData}
      {columns}
      searchColumn="searchableText"
      searchPlaceholder="Search transactions..."
      useGlobalFilter={true}
      {pageSize}
      initialColumnVisibility={{
        searchableText: false,
        payeeName: false,
        accountName: false,
        categoryName: false,
        approvedString: false,
      }}
      facetedFilters={[
        {
          column: "payeeName",
          title: "Payee",
          options: cachedPayeeOptions,
        },
        {
          column: "accountName",
          title: "Account",
          options: cachedAccountOptions,
        },
        {
          column: "memo",
          title: "Memo",
          options: getUniqueMemos(transactionsData),
        },
        {
          column: "categoryName",
          title: "Category",
          options: cachedCategoryOptions,
        },
        {
          column: "approvedString",
          title: "Status",
          options: [
            { label: "Approved", value: "true" },
            { label: "Unapproved", value: "false" },
          ],
        },
      ]}
      rangeFilters={[
        {
          column: "amount",
          title: "Amount",
          step: 1000,
          formatValue: (value) => formatAmount(value),
        },
      ]}
      onRowSelectionChange={handleRowSelectionChange}
      searchValue={searchValue}
      {header}
    />
  {/if}
</div>
