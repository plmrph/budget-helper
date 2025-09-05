<!--
  @fileoverview Simple email search history component
  Shows recent email searches for easy access
-->
<script>
  import { onMount } from "svelte";
  import { emailApi } from "../api/client.js";
  import { History, Search, Trash2 } from "@lucide/svelte";
  import { Button } from "$lib/components/ui/button/index.js";
  import * as Card from "$lib/components/ui/card/index.js";

  /**
   * @typedef {Object} SearchHistoryEntry
   * @property {string} id
   * @property {string} query
   * @property {string} transaction_id
   * @property {string} timestamp
   * @property {number} results_count
   */

  /** @type {SearchHistoryEntry[]} */
  let searchHistory = [];
  let isLoading = false;
  let error = null;

  /** @type {function} */
  export let onSearchSelect = (query, transactionId) => {};

  /**
   * Load search history from local storage
   */
  async function loadHistory() {
    try {
      isLoading = true;
      error = null;

      // Try to load from local storage first
      const localHistory = localStorage.getItem('emailSearchHistory');
      if (localHistory) {
        searchHistory = JSON.parse(localHistory);
        error = null;
      } else {
        // Fallback to API if no local history
        const result = await emailApi.getHistory();
        if (result.error) {
          error = result.error;
          searchHistory = [];
        } else {
          searchHistory = result.history || [];
          error = null;
        }
      }
    } catch (err) {
      // If local storage fails, try API
      try {
        const result = await emailApi.getHistory();
        searchHistory = result.history || [];
        error = null;
      } catch (apiErr) {
        error = "Failed to load search history";
        searchHistory = [];
        console.error("Search history error:", err, apiErr);
      }
    } finally {
      isLoading = false;
    }
  }

  /**
   * Clear search history
   */
  async function clearHistory() {
    try {
      // Clear local storage
      localStorage.removeItem('emailSearchHistory');
      searchHistory = [];
    } catch (err) {
      console.error("Clear history error:", err);
    }
  }

  /**
   * Handle search selection
   */
  function handleSearchSelect(entry) {
    onSearchSelect(entry.query, entry.transaction_id);
  }

  /**
   * Format timestamp for display
   */
  function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString();
  }

  /**
   * Truncate query for display
   */
  function truncateQuery(query, maxLength = 50) {
    if (!query) return "";
    return query.length > maxLength ? query.substring(0, maxLength) + "..." : query;
  }

  // Load history on mount
  onMount(() => {
    loadHistory();
  });

  // Export loadHistory for parent components
  export { loadHistory };
</script>

<Card.Header class="pb-3">
  <div class="flex items-center justify-between">
    <Card.Title class="text-sm font-medium flex items-center gap-2">
      <History class="h-4 w-4" />
      Recent Searches
    </Card.Title>
    {#if searchHistory.length > 0}
      <Button
        variant="ghost"
        size="sm"
        onclick={clearHistory}
        class="h-6 px-2 text-xs"
        title="Clear history"
      >
        <Trash2 class="h-3 w-3" />
      </Button>
    {/if}
  </div>
</Card.Header>

<Card.Content class="space-y-2">
  {#if isLoading}
    <div class="flex items-center justify-center py-4">
      <div class="animate-spin h-4 w-4 border-2 border-current border-t-transparent rounded-full"></div>
    </div>
  {:else if error}
    <div class="text-destructive text-xs text-center py-2">
      {error}
    </div>
  {:else if searchHistory.length === 0}
    <div class="text-muted-foreground text-xs text-center py-4">
      No recent searches
    </div>
  {:else}
    <div class="space-y-1 max-h-24 overflow-y-auto">
      {#each searchHistory.slice(0, 5) as entry (entry.id)}
        <button
          onclick={() => handleSearchSelect(entry)}
          class="w-full text-left p-2 rounded-md hover:bg-accent transition-colors text-xs"
        >
          <div class="flex items-center justify-between gap-2">
            <div class="flex items-center gap-2 min-w-0 flex-1">
              <Search class="h-3 w-3 text-muted-foreground flex-shrink-0" />
              <span class="truncate font-medium">
                {truncateQuery(entry.query)}
              </span>
            </div>
            <div class="text-muted-foreground text-xs flex-shrink-0">
              {entry.results_count || 0} results
            </div>
          </div>
          <div class="text-muted-foreground text-xs mt-1 ml-5">
            {formatTimestamp(entry.timestamp)}
          </div>
        </button>
      {/each}
    </div>
  {/if}
</Card.Content>