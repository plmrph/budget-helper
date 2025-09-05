<!--
  @fileoverview Email search dialog component
  Provides email search interface with transaction context
-->
<script>
  import { onMount } from "svelte";
  import { emailApi } from "../api/client.js";
  import { bulkEmailSearchStore } from "../stores/bulkEmailSearch.js";
  import { budgetStore } from "../stores/budget.js";
  import {
    Mail,
    Search,
    Calendar,
    DollarSign,
    User,
    FileText,
    X,
    Check,
    ExternalLink,
    Type,
  } from "@lucide/svelte";
  import { Button } from "$lib/components/ui/button/index.js";
  import * as Dialog from "$lib/components/ui/dialog/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Label } from "$lib/components/ui/label/index.js";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import EmailSearchHistory from "./EmailSearchHistory.svelte";
  import { transactionStore, getPayeeNameFromBudget } from "../stores/transactions.js";

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

  /**
   * @typedef {Object} Email
   * @property {string} id
   * @property {string} subject
   * @property {string} from
   * @property {string} date
   * @property {string} snippet
   * @property {string} [body]
   */

  /** @type {Transaction} */
  export let transaction;

  /** @type {function} */
  export const onclose = () => {};

  /** @type {function} */
  export let onattach = (emailId) => {};

  /** @type {function} */
  export let ondetach = (emailId) => {};

  /** @type {Email[]} */
  let emails = [];
  /** @type {Email[]} */
  let attachedEmails = [];
  let isLoading = false;
  let isLoadingAttached = false;
  let error = null;
  let searchQuery = "";
  let selectedEmailId = null;
  let isAttaching = false;
  let isDetaching = false;
  let historyComponent;
  // Access bulk search results
  $: bulkResults = $bulkEmailSearchStore?.results || {};

  // Search parameters
  let daysBefore = 3;
  let daysAfter = 3;
  let maxResults = 10;
  let showPlainTextOnly = true; // Default to plain text to avoid HTML issues

  /**
   * Format amount for display
   */
  function formatAmount(milliunits) {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(milliunits / 1000);
  }

  /**
   * Format date for display
   */
  function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString();
  }

  /**
   * Get payee name with preference to enhanced transaction payload
   */
  function getPayeeName(tx) {
    if (tx?.payeeName) return tx.payeeName;
    // Fallback to store helper to avoid manual array scans
    return getPayeeNameFromBudget(tx, $budgetStore);
  }

  /**
   * Save search to local history
   */
  function saveSearchToHistory(query, resultsCount) {
    try {
      const history = JSON.parse(
        localStorage.getItem("emailSearchHistory") || "[]",
      );
      const searchEntry = {
        id: Date.now().toString(),
        query:
          query ||
          `${getPayeeName(transaction)} ${formatAmount(transaction.amount)}`,
        transaction_id: transaction.id,
        timestamp: new Date().toISOString(),
        results_count: resultsCount,
      };

      // Add to beginning and keep only last 10
      history.unshift(searchEntry);
      const trimmedHistory = history.slice(0, 10);
      localStorage.setItem(
        "emailSearchHistory",
        JSON.stringify(trimmedHistory),
      );
    } catch (e) {
      console.warn("Failed to save search history:", e);
    }
  }

  /**
   * Search for emails using the new architecture
   */
  async function searchEmails() {
    try {
      isLoading = true;
      error = null;

      // Use the new API that calls TransactionManager.findTransactionMetadata
      // Pass custom search query if provided, otherwise let backend use automatic terms
      const customQuery =
        searchQuery && searchQuery.trim() ? searchQuery.trim() : null;
      const result = await emailApi.searchForTransaction(
        transaction.id,
        customQuery,
      );

      if (result.emails) {
        // Store all search results (deduped); filtering will be handled by filteredSearchResults
        emails = dedupeEmailsById(result.emails || []);

        // Save successful search to history
        saveSearchToHistory(searchQuery, emails.length);
        // Refresh history component to show new search
        if (historyComponent && historyComponent.loadHistory) {
          historyComponent.loadHistory();
        }
      } else {
        error = result.error || "Failed to search emails";
        emails = [];
      }
    } catch (err) {
      error = "Failed to search emails";
      emails = [];
      console.error("Email search error:", err);
    } finally {
      isLoading = false;
    }
  }

  /**
   * Load attached emails from transaction metadata (no API call needed)
   */
  function loadAttachedEmails() {
    try {
      isLoadingAttached = true;
  attachedEmails = [];

      // Extract attached emails from transaction metadata
      if (transaction.metadata && Array.isArray(transaction.metadata)) {
        for (const metadata of transaction.metadata) {
          if (metadata.type === 1 || metadata.type === "Email") {
            // MetadataType.Email
            try {
              if (metadata.value && metadata.value.stringValue) {
                const emailData = JSON.parse(metadata.value.stringValue);
                // Convert to Email object format expected by the UI
                const email = {
                  id: emailData.id || "unknown",
                  thread_id: emailData.thread_id || "unknown",
                  subject: emailData.subject || "No Subject",
                  sender: emailData.sender || "Unknown Sender",
                  date: emailData.date || new Date().toISOString(),
                  snippet: emailData.snippet || "",
                  body_text: emailData.body_text || "",
                  body_html: emailData.body_html || "",
                  url:
                    emailData.url ||
                    `https://mail.google.com/mail/u/0/#inbox/${emailData.id || ""}`,
                  properties:
                    metadata?.properties
                      ? Object.fromEntries(
                          Object.entries(metadata.properties).map(([k, v]) => {
                            const out = {};
                            if (v?.doubleValue !== undefined) out.doubleValue = v.doubleValue;
                            if (v?.intValue !== undefined) out.intValue = v.intValue;
                            if (v?.stringValue !== undefined) out.stringValue = v.stringValue;
                            if (v?.boolValue !== undefined) out.boolValue = v.boolValue;
                            return [k, out];
                          }),
                        )
                      : undefined,
                };
                // Only push valid emails with an id; dedupe later
                if (email.id && email.id !== "unknown") {
                  attachedEmails.push(email);
                }
              }
            } catch (e) {
              console.warn("Error parsing email metadata:", e);
            }
          }
        }
      }
    } catch (err) {
      console.error("Error loading attached emails from metadata:", err);
      attachedEmails = [];
      // Ensure uniqueness after parsing
      attachedEmails = dedupeEmailsById(attachedEmails);
    } finally {
      isLoadingAttached = false;
    }
  }

  /**
   * Attach email to transaction
   */
  async function attachEmail(emailId) {
    try {
      isAttaching = true;
      selectedEmailId = emailId;

      // Find the email to attach
      const emailToAttach = emails.find((email) => email.id === emailId);
      if (!emailToAttach) {
        error = "Email not found";
        return;
      }

      // Prepare email data for attachment
      const emailData = {
        email_id: emailToAttach.id,
        email_subject: emailToAttach.subject || "No Subject",
        email_sender: emailToAttach.sender || "Unknown Sender",
        email_date: emailToAttach.date || new Date().toISOString(),
        email_snippet: emailToAttach.snippet || "",
        email_body_text: emailToAttach.body_text || "",
        email_body_html: emailToAttach.body_html || "",
        email_url:
          emailToAttach.url ||
          `https://mail.google.com/mail/u/0/#inbox/${emailToAttach.id}`,
        email_properties: emailToAttach.properties || {},
      };
      
      

      const result = await emailApi.attachToTransaction(
        transaction.id,
        emailData,
      );

      if (result.success) {
        // Add to attached emails uniquely
        if (!attachedEmails.some((email) => email.id === emailId)) {
          attachedEmails = dedupeEmailsById([...attachedEmails, emailToAttach]);
        }


        // Notify parent about successful attachment (but don't close dialog)
        if (onattach) {
          onattach(emailId);
        }
      } else {
        error = result.error || "Failed to attach email";
        console.error("Attach failed:", result);
      }
    } catch (err) {
      error = "Failed to attach email";
      console.error("Email attachment error:", err);
    } finally {
      isAttaching = false;
      selectedEmailId = null;
    }
  }

  /**
   * Detach email from transaction
   */
  async function detachEmail(emailId) {
    try {
      isDetaching = true;
      selectedEmailId = emailId;

      // Refresh the transaction first to ensure metadata is up-to-date (handles recent auto-attaches)
      try {
        await transactionStore.refreshTransaction(transaction.id);
        // Rebuild attachedEmails from refreshed transaction
        loadAttachedEmails();
      } catch (e) {
        console.warn("Failed to refresh transaction before detach:", e);
      }

      // If after refresh the email isn't actually attached, reconcile UI and exit gracefully
      const isActuallyAttached = attachedEmails.some((e) => e.id === emailId);
      if (!isActuallyAttached) {
        // Remove ghost card if present and add back to candidates
        const ghost = attachedEmails.find((e) => e.id === emailId);
        if (ghost) {
          if (!emails.some((e) => e.id === ghost.id)) {
            emails = [...emails, ghost];
          }
          attachedEmails = attachedEmails.filter((e) => e.id !== ghost.id);
        }
        if (ondetach) ondetach(emailId);
        return; // Nothing to detach server-side
      }

      const result = await emailApi.detachFromTransaction(transaction.id, emailId);

      if (result.success) {
        // Find the detached email and move it back to search results
        const detachedEmail = attachedEmails.find(
          (email) => email.id === emailId,
        );
        if (detachedEmail) {
          // Add back to search results if not already there
          if (!emails.some((email) => email.id === emailId)) {
            emails = [...emails, detachedEmail];
          }
        }

        // Remove from attached emails
        attachedEmails = attachedEmails.filter((email) => email.id !== emailId);

        // Notify parent about successful detachment (but don't close dialog)
        if (ondetach) {
          ondetach(emailId);
        }
      } else {
        // If backend reports 404/not found, treat it as already-detached: sync UI
        if (
          result?.status === 404 ||
          /not found/i.test(result?.error || "") ||
          /No email attachments found/i.test(result?.error || "")
        ) {
          const detachedEmail = attachedEmails.find((email) => email.id === emailId);
          if (detachedEmail) {
            if (!emails.some((email) => email.id === emailId)) {
              emails = [...emails, detachedEmail];
            }
          }
          attachedEmails = attachedEmails.filter((email) => email.id !== emailId);
          if (ondetach) ondetach(emailId);
        } else {
          error = result.error || "Failed to detach email";
          console.error("Detach failed:", result);
        }
      }
    } catch (err) {
      error = "Failed to detach email";
      console.error("Email detachment error:", err);
    } finally {
      isDetaching = false;
      selectedEmailId = null;
    }
  }

  /**
   * Handle search form submission
   */
  function handleSearchSubmit(event) {
    event.preventDefault();
    searchEmails();
  }

  /**
   * Truncate text for display
   */
  function truncateText(text, maxLength = 100) {
    if (!text) return "";
    return text.length > maxLength
      ? text.substring(0, maxLength) + "..."
      : text;
  }

  /**
   * Check if email is already attached
   */
  function isEmailAttached(emailId) {
    // Check in the attachedEmails array first (most reliable)
    if (attachedEmails.some((email) => email.id === emailId)) {
      return true;
    }

    // Fallback to checking transaction metadata
    if (!transaction.metadata || !Array.isArray(transaction.metadata)) {
      return false;
    }

    return transaction.metadata.some((meta) => {
      if (meta.type === "Email" || meta.type === 1) {
        try {
          // Handle both simple dict and complex object structures
          const emailData =
            typeof meta.value === "string"
              ? JSON.parse(meta.value)
              : meta.value;
          return emailData.id === emailId;
        } catch (e) {
          return false;
        }
      }
      return false;
    });
  }

  /**
   * Deduplicate a list of emails by id, preserving first occurrence
   */
  function dedupeEmailsById(list) {
    const seen = new Set();
    const result = [];
    for (const e of list || []) {
      if (!e || !e.id) continue; // skip invalid entries
      if (!seen.has(e.id)) {
        seen.add(e.id);
        result.push(e);
      }
    }
    return result;
  }

  /**
   * Sanitize HTML content to prevent parsing errors and base URL changes
   * Using a simple regex-based approach to avoid DOM manipulation issues
   */
  function sanitizeHtml(html) {
    if (!html) return "";

    // Simple regex-based sanitization to avoid DOM manipulation
    let cleanedHtml = html
      // Remove the most dangerous elements that can change context
      .replace(/<base[^>]*>/gi, "") // Remove base tags (main culprit)
      .replace(/<meta[^>]*>/gi, "") // Remove meta tags
      .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "") // Remove script tags and content
      .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "") // Remove style tags and content
      .replace(/<link[^>]*>/gi, "") // Remove link tags
      .replace(/<iframe[^>]*>[\s\S]*?<\/iframe>/gi, "") // Remove iframes
      .replace(/<object[^>]*>[\s\S]*?<\/object>/gi, "") // Remove objects
      .replace(/<embed[^>]*>/gi, "") // Remove embeds
      .replace(/<form[^>]*>[\s\S]*?<\/form>/gi, "") // Remove forms
      // Remove problematic attributes
      .replace(/\s+on\w+\s*=\s*["'][^"']*["']/gi, "") // Remove event handlers
      .replace(/\s+href\s*=\s*["'][^"']*["']/gi, "") // Remove href attributes
      .replace(/\s+src\s*=\s*["'][^"']*["']/gi, "") // Remove src attributes
      .replace(/\s+style\s*=\s*["'][^"']*["']/gi, "") // Remove inline styles
      .replace(/javascript:[^"'\s]*/gi, ""); // Remove javascript: URLs

    return cleanedHtml;
  }

  /**
   * Get filtered search results (excluding attached emails)
   */
  $: filteredSearchResults = (() => {
    // Exclude only by exact email id; allow other messages in the same thread to remain
    const source = uniqueAttachedEmails && uniqueAttachedEmails.length > 0
      ? uniqueAttachedEmails
      : attachedEmails || [];

    const attachedEmailIds = new Set(source.map((e) => e?.id).filter(Boolean));

    const filtered = emails.filter(
      (email) => email && !attachedEmailIds.has(email.id),
    );

    return filtered;
  })();

  // Always expose a deduped view of attached emails for rendering
  $: uniqueAttachedEmails = dedupeEmailsById(attachedEmails);

  // On mount, refresh transaction first to ensure attachedEmails reflects server state,
  // then load attached list and perform initial search.

  let lastRefreshedTransactionId = null;
  onMount(() => {
    if (transaction?.id && transaction.id !== lastRefreshedTransactionId) {
      lastRefreshedTransactionId = transaction.id;
      (async () => {
        try {
          await transactionStore.refreshTransaction(transaction.id);
        } catch (e) {
          console.warn("Failed to refresh transaction on open:", e);
        } finally {
          loadAttachedEmails();
          searchEmails();
        }
      })();
    } else {
      // If transaction was already refreshed, just load attached emails and search
      loadAttachedEmails();
      searchEmails();
    }
  });

  // Only refresh on bulkResults change if transaction hasn't just been refreshed
  // Remove bulkResults reactive logic to prevent infinite loop
  // The parent MetadataColumn already handles bulk auto-attach refreshes

</script>

<Dialog.Header>
  <Dialog.Title class="flex items-center gap-2">
    <Mail class="h-5 w-5" />
    Email Search for Transaction
  </Dialog.Title>
  <Dialog.Description>
    Find and attach email receipts to this transaction
  </Dialog.Description>
</Dialog.Header>

<div class="flex-1 overflow-y-auto space-y-4 pr-4">
  <!-- Top Section: Settings-style layout with side-by-side panels -->
  <div class="columns-1 lg:columns-3 gap-4 space-y-4">
    <!-- Transaction Context -->
    <Card.Root class="break-inside-avoid">
      <Card.Header class="pb-3">
        <Card.Title class="text-sm font-medium">Transaction Details</Card.Title>
      </Card.Header>
      <Card.Content class="space-y-3">
        <div class="space-y-2 text-sm">
          <div class="flex items-center gap-2">
            <Calendar class="h-4 w-4 text-muted-foreground flex-shrink-0" />
            <span>{formatDate(transaction.date)}</span>
          </div>
          <div class="flex items-center gap-2">
            <DollarSign class="h-4 w-4 text-muted-foreground flex-shrink-0" />
            <span
              class={transaction.amount < 0
                ? "text-red-600 dark:text-red-400"
                : "text-green-600 dark:text-green-400"}
            >
              {formatAmount(transaction.amount)}
            </span>
          </div>
          <div class="flex items-center gap-2">
            <User class="h-4 w-4 text-muted-foreground flex-shrink-0" />
            <span>{getPayeeName(transaction)}</span>
          </div>
          {#if transaction.memo}
            <div class="flex items-center gap-2">
              <FileText class="h-4 w-4 text-muted-foreground flex-shrink-0" />
              <span class="break-words">{transaction.memo}</span>
            </div>
          {/if}
        </div>
      </Card.Content>
    </Card.Root>

    <!-- Search Controls -->
    <Card.Root class="break-inside-avoid">
      <Card.Header class="pb-3">
        <Card.Title class="text-sm font-medium">Search Parameters</Card.Title>
      </Card.Header>
      <Card.Content>
        <form onsubmit={handleSearchSubmit} class="space-y-3">
          <div class="space-y-3">
            <!-- Days and Max Results on one line -->
            <div class="grid grid-cols-2 gap-3">
              <div class="space-y-1">
                <Label for="days-range" class="text-xs">+/- Days</Label>
                <Input
                  id="days-range"
                  type="number"
                  min="0"
                  max="30"
                  bind:value={daysBefore}
                  oninput={() => {
                    daysAfter = daysBefore;
                  }}
                  class="h-8 text-sm w-20"
                  placeholder="± days"
                />
              </div>
              <div class="space-y-1">
                <Label for="max-results" class="text-xs">Max Results</Label>
                <Input
                  id="max-results"
                  type="number"
                  min="1"
                  max="50"
                  bind:value={maxResults}
                  class="h-8 text-sm w-20"
                />
              </div>
            </div>
            <div class="space-y-1">
              <Label for="search-query" class="text-xs"
                >Custom Search Query (optional)</Label
              >
              <div class="flex gap-2">
                <Input
                  id="search-query"
                  type="text"
                  placeholder="Enter additional search terms..."
                  bind:value={searchQuery}
                  class="flex-1 h-8 text-sm"
                />
                <Button type="submit" disabled={isLoading} class="px-3 h-8">
                  {#if isLoading}
                    <div
                      class="animate-spin h-3 w-3 border-2 border-current border-t-transparent rounded-full"
                    ></div>
                  {:else}
                    <Search class="h-3 w-3" />
                  {/if}
                </Button>
              </div>
            </div>
            <div class="space-y-1">
              <Label class="text-xs flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  bind:checked={showPlainTextOnly}
                  class="h-3 w-3"
                />
                <Type class="h-3 w-3" />
                Plain text rendering only
              </Label>
            </div>
          </div>
        </form>
      </Card.Content>
    </Card.Root>

    <!-- Search History -->
    <Card.Root class="break-inside-avoid">
      <EmailSearchHistory
        bind:this={historyComponent}
      />
    </Card.Root>
  </div>

  <!-- Error Display -->
  {#if error}
    <div class="bg-destructive/10 border border-destructive/20 rounded-md p-4">
      <div class="text-destructive text-sm">{error}</div>
    </div>
  {/if}

  <!-- Attached Emails -->
  <div class="space-y-4">
    <div class="flex items-center justify-between">
      <h3 class="text-sm font-medium">
        {#if isLoadingAttached}
          Loading attached emails...
        {:else}
          Attached Emails ({uniqueAttachedEmails.length})
        {/if}
      </h3>
    </div>

    {#if isLoadingAttached}
      <div class="flex items-center justify-center py-4">
        <div
          class="animate-spin h-6 w-6 border-2 border-current border-t-transparent rounded-full"
        ></div>
      </div>
    {:else if attachedEmails.length === 0}
      <div class="text-center py-4 text-muted-foreground text-sm">
        <Mail class="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p>No emails attached to this transaction.</p>
      </div>
    {:else}
      <div
        class="space-y-3 overflow-y-auto"
        style="max-height: {filteredSearchResults.length > 0 ? '40vh' : '60vh'}"
      >
  {#each uniqueAttachedEmails as email (`attached-${email.id}`)}
          <Card.Root class="bg-accent/20 border-accent">
            <Card.Content class="p-4">
              <div class="space-y-3">
                <!-- Header with subject and action buttons -->
                <div class="flex items-start justify-between gap-4">
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-3">
                      <h4 class="font-medium text-sm truncate">
                        {email.subject || "No Subject"}
                      </h4>
                      {#if email?.properties?.relevance?.doubleValue !== undefined}
                        <span class="text-xs text-muted-foreground bg-yellow-100 text-yellow-800 px-2 py-0.5 rounded">Relevance: {email.properties.relevance.doubleValue.toFixed(2)}</span>
                      {/if}
                    </div>
                  </div>

                  <div class="flex items-center gap-2 flex-shrink-0">
                    <Button
                      variant="destructive"
                      size="sm"
                      onclick={() => detachEmail(email.id)}
                      disabled={isDetaching && selectedEmailId === email.id}
                      class="h-7 px-2 text-xs"
                    >
                      {#if isDetaching && selectedEmailId === email.id}
                        <div
                          class="animate-spin h-3 w-3 border-2 border-current border-t-transparent rounded-full mr-1"
                        ></div>
                        Detaching...
                      {:else}
                        <X class="h-3 w-3 mr-1" />
                        Detach
                      {/if}
                    </Button>

                    <Button
                      variant="outline"
                      size="sm"
                      class="h-7 px-2"
                      onclick={() => {
                        if (email.url) {
                          window.open(email.url, "_blank");
                        } else {
                          // Fallback: construct Gmail URL from email ID
                          const gmailUrl = `https://mail.google.com/mail/u/0/#inbox/${email.id}`;
                          window.open(gmailUrl, "_blank");
                        }
                      }}
                      title="Open email in Gmail"
                    >
                      <ExternalLink class="h-3 w-3" />
                    </Button>
                  </div>
                </div>

                <!-- Email metadata -->
                <div class="text-xs text-muted-foreground">
                  <div class="flex items-center gap-4">
                    <span>From: {email.sender || email.from || "Unknown"}</span>
                    <span>Date: {formatDate(email.date)}</span>
                  </div>
                  {#if email.matched_terms}
                    <div class="mt-1">
                      <span class="text-blue-600 dark:text-blue-400">Matched: {email.matched_terms}</span>
                    </div>
                  {/if}
                </div>

                <!-- Email content - full width with more details -->
                <div class="w-full">
                  {#if showPlainTextOnly}
                    <!-- Plain text only mode -->
                    {#if email.body_text}
                      <pre
                        class="text-xs leading-relaxed overflow-y-auto border rounded p-3 bg-background/50 whitespace-pre-wrap font-sans w-full">
                        {email.body_text}
                      </pre>
                    {:else if email.snippet}
                      <p
                        class="text-xs leading-relaxed overflow-y-auto border rounded p-3 bg-background/50 w-full"
                      >
                        {email.snippet}
                      </p>
                    {:else}
                      <p class="text-xs text-muted-foreground italic">
                        No plain text content available
                      </p>
                    {/if}
                  {:else}
                    <!-- Sanitized HTML mode -->
                    {#if email.body_html}
                      <div
                        class="text-xs leading-relaxed max-h-[20.8rem] overflow-y-auto border rounded p-3 bg-background/50 w-full"
                      >
                        <div class="email-content">
                          {@html sanitizeHtml(email.body_html)}
                        </div>
                      </div>
                    {:else if email.body_text}
                      <pre
                        class="text-xs leading-relaxed max-h-[20.8rem] overflow-y-auto border rounded p-3 bg-background/50 whitespace-pre-wrap font-sans w-full">
                        {email.body_text}
                      </pre>
                    {:else if email.snippet}
                      <p
                        class="text-xs leading-relaxed max-h-[20.8rem] overflow-y-auto border rounded p-3 bg-background/50 w-full"
                      >
                        {email.snippet}
                      </p>
                    {/if}
                  {/if}
                </div>
              </div>
            </Card.Content>
          </Card.Root>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Email Search Results -->
  <div class="space-y-4">
    {#if filteredSearchResults.length > 0 || isLoading}
      <div class="flex items-center justify-between">
        <h3 class="text-sm font-medium">
          {#if isLoading}
            Searching emails...
          {:else}
            Search Results ({filteredSearchResults.length} found)
          {/if}
        </h3>
      </div>

      {#if isLoading}
        <div class="flex items-center justify-center py-8">
          <div
            class="animate-spin h-8 w-8 border-2 border-current border-t-transparent rounded-full"
          ></div>
        </div>
      {:else if filteredSearchResults.length === 0}
        <div class="text-center py-8 text-muted-foreground">
          <Mail class="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No emails found for this transaction.</p>
          <p class="text-sm">
            Try adjusting the search parameters or adding a custom query.
          </p>
        </div>
      {:else}
        <div
          class="space-y-3 overflow-y-auto"
          style="max-height: {attachedEmails.length > 0 ? '40vh' : '60vh'}"
        >
          {#each filteredSearchResults as email (`search-${email.id}`)}
            <Card.Root class="hover:bg-accent/50 transition-colors">
              <Card.Content class="p-4">
                <div class="space-y-3">
                  <!-- Header with subject and action buttons -->
                  <div class="flex items-start justify-between gap-4">
                    <div class="flex-1 min-w-0">
                      <div class="flex items-center gap-3">
                        <h4 class="font-medium text-sm truncate">
                          {email.subject || "No Subject"}
                        </h4>
                        {#if email?.properties?.relevance?.doubleValue !== undefined}
                          <span class="text-xs text-muted-foreground bg-yellow-100 text-yellow-800 px-2 py-0.5 rounded">Relevance: {email.properties.relevance.doubleValue.toFixed(2)}</span>
                        {/if}
                      </div>
                    </div>

                    <div class="flex items-center gap-2 flex-shrink-0">
                      <Button
                        size="sm"
                        onclick={() => attachEmail(email.id)}
                        disabled={isAttaching && selectedEmailId === email.id}
                        class="h-7 px-2 text-xs"
                      >
                        {#if isAttaching && selectedEmailId === email.id}
                          <div
                            class="animate-spin h-3 w-3 border-2 border-current border-t-transparent rounded-full mr-1"
                          ></div>
                          Attaching...
                        {:else}
                          <Check class="h-3 w-3 mr-1" />
                          Attach
                        {/if}
                      </Button>

                      <Button
                        variant="outline"
                        size="sm"
                        class="h-7 px-2"
                        onclick={() => {
                          if (email.url) {
                            window.open(email.url, "_blank");
                          } else {
                            // Fallback: construct Gmail URL from email ID
                            const gmailUrl = `https://mail.google.com/mail/u/0/#inbox/${email.id}`;
                            window.open(gmailUrl, "_blank");
                          }
                        }}
                        title="Open email in Gmail"
                      >
                        <ExternalLink class="h-3 w-3" />
                      </Button>
                    </div>
                  </div>

                  <!-- Email metadata -->
                  <div class="text-xs text-muted-foreground">
                    <div class="flex items-center gap-4">
                      <span
                        >From: {email.sender || email.from || "Unknown"}</span
                      >
                      <span>Date: {formatDate(email.date)}</span>
                    </div>
                    {#if email.matched_terms}
                      <div class="mt-1">
                        <span class="text-blue-600 dark:text-blue-400">Matched: {email.matched_terms}</span>
                      </div>
                    {/if}
                  </div>

                  <!-- Email content - full width with more details -->
                  <div class="w-full">
                    {#if showPlainTextOnly}
                      <!-- Plain text only mode -->
                      {#if email.body_text}
                        <pre
                          class="text-xs leading-relaxed max-h-32 overflow-y-auto border rounded p-3 bg-background/50 whitespace-pre-wrap font-sans w-full">
                          {email.body_text}
                        </pre>
                      {:else if email.snippet}
                        <p
                          class="text-xs leading-relaxed max-h-32 overflow-y-auto border rounded p-3 bg-background/50 w-full"
                        >
                          {email.snippet}
                        </p>
                      {:else}
                        <p class="text-xs text-muted-foreground italic">
                          No plain text content available
                        </p>
                      {/if}
                    {:else}
                      <!-- Sanitized HTML mode -->
                      {#if email.body_html}
                        <div
                          class="text-xs leading-relaxed max-h-[20.8rem] overflow-y-auto border rounded p-3 bg-background/50 w-full"
                        >
                          <div class="email-content">
                            {@html sanitizeHtml(email.body_html)}
                          </div>
                        </div>
                      {:else if email.body_text}
                        <pre
                          class="text-xs leading-relaxed max-h-[20.8rem] overflow-y-auto border rounded p-3 bg-background/50 whitespace-pre-wrap font-sans w-full">
                          {email.body_text}
                        </pre>
                      {:else if email.snippet}
                        <p
                          class="text-xs leading-relaxed max-h-[20.8rem] overflow-y-auto border rounded p-3 bg-background/50 w-full"
                        >
                          {email.snippet}
                        </p>
                      {/if}
                    {/if}
                  </div>
                </div>
              </Card.Content>
            </Card.Root>
          {/each}
        </div>
      {/if}
    {/if}
  </div>
</div>

<style>
  /* Email content styling - minimal and targeted */
  :global(.email-content) {
    font-family: inherit;
    font-size: inherit;
    color: inherit;
    /* Isolate email content to prevent CSS bleeding */
    isolation: isolate;
  }

  :global(.email-content p) {
    margin: 0.25rem 0;
    color: inherit;
  }

  :global(.email-content div) {
    margin: 0.125rem 0;
    color: inherit;
  }

  :global(.email-content table) {
    font-size: inherit;
    border-collapse: collapse;
    color: inherit;
  }

  :global(.email-content td, .email-content th) {
    padding: 0.125rem 0.25rem;
    font-size: inherit;
    color: inherit;
  }

  :global(.email-content img) {
    max-width: 100%;
    height: auto;
    /* Prevent images from loading external resources */
    display: none;
  }

  :global(.email-content a) {
    color: hsl(var(--primary));
    text-decoration: underline;
    /* Disable all links to prevent navigation */
    pointer-events: none;
  }

  /* Ensure no external resources can be loaded */
  :global(.email-content *[src]) {
    display: none !important;
  }

  :global(.email-content *[href]) {
    pointer-events: none !important;
  }

  /* Force dialog to be large - override shadcn defaults */
  :global([data-slot="dialog-content"]) {
    width: 95vw !important;
    max-width: 1400px !important;
    height: 90vh !important;
    max-height: 90vh !important;
  }

  @media (min-width: 640px) {
    :global([data-slot="dialog-content"]) {
      max-width: 1400px !important;
    }
  }
</style>
