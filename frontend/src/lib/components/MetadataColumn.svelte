<!--
  @fileoverview Metadata column component for transaction grid
  Shows metadata icons (email, etc.) with status indicators
-->
<script>
  import { emailApi } from "../api/client.js";
  import { transactionStore, getPayeeNameFromBudget } from "../stores/transactions.js";
  import { budgetStore } from "../stores/budget.js";
  import { transactionEmailStatus, bulkEmailSearchStore } from "../stores/bulkEmailSearch.js";
  import { Mail, Search, AlertCircle, ExternalLink } from "@lucide/svelte";
  import { Button } from "$lib/components/ui/button/index.js";
  import * as Dialog from "$lib/components/ui/dialog/index.js";
  import EmailSearchDialog from "./EmailSearchDialog.svelte";

  /**
   * @typedef {Object} Transaction
   * @property {string} id
   * @property {string} date
   * @property {number} amount
   * @property {boolean} approved
   * @property {string} [payeeId]
   * @property {string} [categoryId]
   * @property {string} [accountId]
   * @property {string} [memo]
   * @property {any[]} [metadata]
   */

  /** @type {Transaction} */
  export let transaction;

  /** @type {function} */
  export const onTransactionUpdate = () => {};

  let emailSearchOpen = false;
  let emailCount = 0;
  let emailStatus = "none"; // "none", "single", "multiple", "loading", "error", "attached"
  let isLoading = false;
  let isUserInteraction = false; // flips true if user attaches/detaches via dialog

  // Track last known bulk attached flag for this transaction to trigger a refresh on close
  $: bulkStatus = $transactionEmailStatus[transaction.id];
  $: bulkAttached = !!(bulkStatus && bulkStatus.status === "attached");
  let bulkRefreshedOnce = false;

  // Guarded: When bulk attachment happens for this transaction, refresh once to persist metadata
  // Only refresh if the auto-attached email is not already in transaction metadata
  $: {
    if (bulkAttached && !bulkRefreshedOnce) {
      // Check if the auto-attached email is already in the transaction metadata
      const bulkResult = $transactionEmailStatus[transaction.id];
      const hasAutoAttachedEmail = bulkResult && bulkResult.status === "attached";
      
      if (hasAutoAttachedEmail && transaction.metadata && Array.isArray(transaction.metadata)) {
        // Check if any email metadata exists - if so, the refresh already happened
        const hasEmailMetadata = transaction.metadata.some(meta => 
          meta.type === 1 || meta.type === "Email"
        );
        
        if (!hasEmailMetadata) {
          // Only refresh if no email metadata exists yet (new auto-attach)
          bulkRefreshedOnce = true;
          setTimeout(async () => {
            await transactionStore.refreshTransaction(transaction.id);
            const { count, status } = getEmailMetadataStatusFor(transaction);
            emailCount = count;
            emailStatus = status === "none" && bulkAttached ? "attached" : status;
          }, 50);
        } else {
          // Email metadata already exists, just update local state without refresh
          bulkRefreshedOnce = true;
          const { count, status } = getEmailMetadataStatusFor(transaction);
          emailCount = count;
          emailStatus = status === "none" && bulkAttached ? "attached" : status;
        }
      }
    } else if (!bulkAttached) {
      // Reset guard when bulk status clears (e.g., new search cycle)
      bulkRefreshedOnce = false;
    }
  }

  /**
   * Get payee name with preference to enhanced transaction payload
   */
  function getPayeeName(tx) {
    if (tx?.payeeName) return tx.payeeName;
    return getPayeeNameFromBudget(tx, $budgetStore);
  }

  /**
   * Get email metadata count and status from transaction
   */
  function extractEmailId(meta) {
    try {
      // Standard case: value.stringValue contains JSON string
      const sv = meta?.value?.stringValue ?? meta?.value;
      if (typeof sv === "string") {
        try {
          const obj = JSON.parse(sv);
          return obj?.id || null;
        } catch {
          return null;
        }
      }
      if (sv && typeof sv === "object") {
        return sv?.id || null;
      }
    } catch {
      // ignore
    }
    return null;
  }

  function getEmailMetaEntries(tx) {
    if (!tx?.metadata || !Array.isArray(tx.metadata)) return [];
    return tx.metadata.filter((m) => m?.type === "Email" || m?.type === 1);
  }

  function getDedupedEmailIds(tx) {
    const ids = new Set();
    for (const m of getEmailMetaEntries(tx)) {
      const id = extractEmailId(m);
      if (id) ids.add(id);
    }
    return ids;
  }

  function getEmailMetadataStatusFor(tx) {
    const ids = getDedupedEmailIds(tx);
    const count = ids.size;
    let status = "none";

    if (count === 1) {
      status = "single";
    } else if (count > 1) {
      status = "multiple";
    }

    const firstAttachedEmailId = ids.size ? ids.values().next().value : null;
    return { count, status, firstAttachedEmailId };
  }

  // Single source of truth for current persisted attachment count/status
  $: metaStatusObj = getEmailMetadataStatusFor(transaction);
  $: metaCount = metaStatusObj.count;
  $: metaStatus = metaStatusObj.status;
  $: metaFirstAttachedEmailId = metaStatusObj.firstAttachedEmailId;

  /**
   * Handle metadata icon click
   */
  async function handleMetadataClick() {
    emailSearchOpen = true;
  }

  /**
   * Handle email search dialog close
   */
  function handleEmailSearchClose() {
    emailSearchOpen = false;
    // Only refresh on close if user interaction occurred (attach/detach)
    if (isUserInteraction) {
      setTimeout(async () => {
        await transactionStore.refreshTransaction(transaction.id);
        const { count, status } = getEmailMetadataStatusFor(transaction);
        if (count > 0) {
          emailCount = count;
          emailStatus = status;
        } else {
          emailCount = 0;
          emailStatus = "none";
        }
        isUserInteraction = false; // Reset the flag
      }, 50);
    }
  }

  /**
   * Handle email attachment - update local state and recalculate badge
   */
  async function handleEmailAttach(emailId) {
    // Update local email count immediately for responsive UI
    emailCount = emailCount + 1;
  emailStatus = emailCount === 1 ? "single" : "multiple";
  isUserInteraction = true;

    // Refresh the single transaction from the API to get updated metadata
    // This is done in the background to ensure data consistency
    setTimeout(async () => {
      await transactionStore.refreshTransaction(transaction.id);
    }, 100);
  }

  /**
   * Handle email detachment - update local state and recalculate badge
   */
  async function handleEmailDetach(emailId) {
    // Update local email count immediately for responsive UI
    emailCount = Math.max(0, emailCount - 1);
    emailStatus =
      emailCount === 0 ? "none" : emailCount === 1 ? "single" : "multiple";
  isUserInteraction = true;

  // Avoid mutating the inbound `transaction` prop; rely on store refresh below

    // Refresh the single transaction from the API to get updated metadata
    // This is done in the background to ensure data consistency
    setTimeout(async () => {
      await transactionStore.refreshTransaction(transaction.id);
    }, 100);
  }

  // Initialize and reactively update badge/icon from metadata first, then fall back to bulk
  $: {
  const { count, status } = metaStatusObj;
    const bulkSearchResult = $transactionEmailStatus[transaction.id];

    if (count > 0) {
      emailCount = count;
      emailStatus = status;
    } else if (!isUserInteraction && bulkSearchResult) {
      emailCount = bulkSearchResult.count;
      emailStatus = bulkSearchResult.status;
    } else {
      emailCount = 0;
      emailStatus = "none";
    }
  }

  // Expose raw bulk search result for this transaction (if available)
  $: rawBulkResult = ($bulkEmailSearchStore && $bulkEmailSearchStore.results)
    ? $bulkEmailSearchStore.results[transaction.id]
    : null;

  /**
   * Pick the best email candidate from a bulk search result.
   * Strategy:
   * 1) Prefer highest numeric relevance/score if available.
   * 2) If tie or no score, pick the oldest (earliest) by date.
   */
  function pickBestEmailFromBulk(result) {
    if (!result || !Array.isArray(result.emails) || result.emails.length === 0) return null;

    const emails = result.emails.map((e, idx) => {
      // Prefer numeric relevance fields when present (coerce numeric strings)
      let relevance = null;

      console.debug("Email candidate", e.id, e.properties);
      // Support doubleValue, intValue, or stringValue that can be coerced to number
      if (e?.properties && e.properties.relevance) {
        relevance = e.properties.relevance;
      }

      const date = e.date ? new Date(e.date) : new Date(0);
      return { _orig: e, relevance, date, idx };
    });

    // Sort: relevance desc (higher wins), then date asc (older wins), then stable index
    emails.sort((a, b) => {
      if (a.relevance !== b.relevance) return b.relevance - a.relevance;
      const dateDiff = a.date - b.date;
      if (dateDiff !== 0) return dateDiff;
      return a.idx - b.idx;
    });

    return emails[0]._orig;
  }

  function openTopBulkEmail() {
    let chosen = pickBestEmailFromBulk(rawBulkResult);
    if (!chosen && metaFirstAttachedEmailId) {
      chosen = { url: null, id: metaFirstAttachedEmailId };
    }
    if (!chosen) return;
    const url = chosen.url || `https://mail.google.com/mail/u/0/#inbox/${chosen.id}`;
    window.open(url, "_blank");
  }

  // Get icon color based on status - use local state variables for immediate updates
  $: iconColor = (() => {
    if (isLoading) {
      return "text-blue-600 dark:text-blue-400"; // searching
    }

  // Current attached count from transaction metadata
  const metaCountLocal = metaCount;

    // Auto-attached from bulk search (no user interaction yet) => blue takes precedence
    if (!isUserInteraction && bulkAttached) {
      return "text-blue-600 dark:text-blue-400";
    }

    // If there are persisted attachments, show green (authoritative source)
    if (metaCountLocal > 0) {
      return "text-green-600 dark:text-green-400";
    }

    // If search found items but none attached, show yellow
    if (emailStatus === "single" || emailStatus === "multiple") {
      return "text-yellow-600 dark:text-yellow-400";
    }

    if (emailStatus === "error") {
      return "text-red-600 dark:text-red-400";
    }

    return "text-gray-400 dark:text-gray-600"; // default
  })();

  // Get tooltip text - use local state variables for immediate updates
  $: tooltipText = (() => {
    if (isLoading) {
      return "Searching for emails...";
    } else if (bulkAttached && !isUserInteraction) {
      return `${emailCount || metaCount || 1} email${(emailCount || metaCount || 1) === 1 ? "" : "s"} auto-attached - click to view`;
    } else if (emailStatus === "single" || emailStatus === "multiple") {
      // Check if these are actually attached or just search results
      if (metaCount > 0) {
        return `${emailCount} email${emailCount === 1 ? "" : "s"} attached - click to view`;
      } else {
        return `${emailCount} email${emailCount === 1 ? "" : "s"} found - click to view`;
      }
    } else if (emailStatus === "error") {
      return "Error searching for emails - click to retry";
    } else {
      return "Click to search for emails";
    }
  })();
</script>

<div class="flex items-center justify-center">
  <Button
    variant="ghost"
    size="sm"
    class="h-8 w-8 p-0 hover:bg-accent relative"
    onclick={handleMetadataClick}
    title={tooltipText}
    disabled={isLoading}
  >
    {#if isLoading}
      <div
        class="animate-spin h-4 w-4 border-2 border-current border-t-transparent rounded-full"
      ></div>
    {:else if emailStatus === "error"}
      <AlertCircle class="h-4 w-4 {iconColor}" />
    {:else}
      <Mail class="h-4 w-4 {iconColor}" />
    {/if}

    {#if metaCount > 0 && !isLoading}
      <span class="sr-only">{metaCount} emails</span>
      <!-- Small badge for count -->
      <div
        class="absolute -top-1 -right-1 bg-primary text-primary-foreground text-xs rounded-full h-4 w-4 flex items-center justify-center min-w-4"
      >
        {metaCount > 9 ? "9+" : metaCount}
      </div>
    {/if}
  </Button>

  {#if (metaCount > 0) || (rawBulkResult && Array.isArray(rawBulkResult.emails) && rawBulkResult.emails.length > 0)}
    <Button
      variant="outline"
      size="sm"
      class="h-8 w-8 p-0 ml-2"
      onclick={openTopBulkEmail}
      title="Open most relevant email in Gmail"
    >
      <ExternalLink class="h-4 w-4" />
    </Button>
  {/if}

  <!-- Email Search Dialog -->
  <Dialog.Root
    bind:open={emailSearchOpen}
    onOpenChange={(open) => {
      if (!open) handleEmailSearchClose();
    }}
  >
    <Dialog.Content
      class="!max-w-none !w-[95vw] !max-h-[90vh] !h-[90vh] overflow-hidden flex flex-col"
      style="width: 95vw !important; max-width: 1400px !important; height: 90vh !important; max-height: 90vh !important;"
    >
      <EmailSearchDialog
        {transaction}
        onclose={handleEmailSearchClose}
        onattach={handleEmailAttach}
        ondetach={handleEmailDetach}
      />
    </Dialog.Content>
  </Dialog.Root>
</div>

<style>
  /* Ensure button positioning for badge */
  :global(.relative) {
    position: relative;
  }
</style>
