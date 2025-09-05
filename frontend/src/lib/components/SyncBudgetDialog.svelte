<script>
  import { createEventDispatcher, onMount } from 'svelte';
  import { Button, buttonVariants } from '$lib/components/ui/button/index.js';
  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import { ChevronLeft, ChevronRight, X as XIcon } from '@lucide/svelte';
  import { budgetStore } from "../stores/budget.js";

  // Props
  export let open = false;
  export let title = 'Sync Budget';
  export let loadPreview; // async () => { success, data: { summary, items: [] } }
  export let applyPlan;   // async (plan) => { success }

  const dispatch = createEventDispatcher();

  let loading = false;
  let applying = false;
  let error = null;
  let summary = { add: 0, update: 0, delete: 0, total: 0 };
  // items: [{ id, left, right, status: 'add'|'delete'|'update'|'same', action: 'left'|'right'|'skip' }]
  let items = [];

  function formatAmount(amount) {
    if (amount === null || amount === undefined) return '';
    let n = Number(amount);
    if (Number.isNaN(n)) return String(amount);
    // YNAB uses milliunits
    const dollars = n / 1000;
    try {
      return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(dollars);
    } catch {
      // Fallback formatting
      const sign = dollars < 0 ? '-' : '';
      const val = Math.abs(dollars).toFixed(2);
      return `${sign}$${val}`;
    }
  }

  let fromDate = '';
  // Default the picker to 30 days ago if not set
  function dateToInput(d) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
  }
  onMount(() => {
    if (!fromDate) {
      const d = new Date();
      d.setDate(d.getDate() - 30);
      fromDate = dateToInput(d);
    }
  });

  function rowDate(row) {
    // Prefer right (remote) date, else left (local)
    const r = row?.rightDetails?.date;
    const l = row?.leftDetails?.date;
    return r || l || '';
  }
  function parseRowDate(row) {
    const d = rowDate(row);
    const t = Date.parse(d);
    return Number.isNaN(t) ? -Infinity : t;
  }
  async function refresh() {
    if (!loadPreview) return;
    loading = true; error = null;
    try {
      await budgetStore.refreshFromYNAB();
      const res = await loadPreview({ fromDate: fromDate || undefined });
      if (res?.success && res?.data) {
        summary = res.data.summary || summary;
        items = (res.data.items || [])
          .map(row => ({ ...row, action: defaultAction(row) }))
          // Sort by effective date (newest first)
          .sort((a, b) => parseRowDate(b) - parseRowDate(a));
      } else {
        error = res?.error || 'Failed to load preview';
      }
    } catch (e) {
      error = e?.message || 'Failed to load preview';
    } finally { loading = false; }
  }

  // no reset tracking UI

  function defaultAction(row) {
  // Defaults:
  // - If row.edited (locally edited after original), prefer push local -> right
  // - Otherwise:
  //   add (remote-only): pull remote -> left
  //   delete (local-only): push local -> right
  //   update: prefer remote -> left
  if (row.edited) return 'right';
  if (row.status === 'add') return 'left';
  if (row.status === 'delete') return 'right';
  if (row.status === 'update') return 'left';
    return 'skip';
  }

  function colorFor(status) {
    if (status === 'add') return 'bg-emerald-50 dark:bg-emerald-950/30';
    if (status === 'delete') return 'bg-red-50 dark:bg-red-950/30';
    if (status === 'update') return 'bg-amber-50 dark:bg-amber-950/30';
    return 'bg-transparent';
  }

  async function onConfirm() {
    if (!applyPlan) return;
    applying = true; error = null;
    try {
      const plan = items.filter(r => r.action !== 'skip').map(r => ({ id: r.id, action: r.action }));
      const res = await applyPlan({ plan });
      if (res?.success !== false) {
        dispatch('applied', { plan });
        open = false;
      } else {
        error = res?.error || res?.message || 'Failed to apply sync plan';
      }
    } catch (e) {
      error = e?.message || 'Failed to apply sync plan';
    } finally { applying = false; }
  }

  $: if (open) refresh();

  // Global helpers for select-all controls
  function setAll(action) {
    items = items.map(r => ({ ...r, action }));
  }
  $: allAction = (() => {
    if (!items || items.length === 0) return null;
    const a = items[0].action;
    return items.every(i => i.action === a) ? a : null;
  })();
</script>

<Dialog.Root bind:open>
  <Dialog.Portal>
    <Dialog.Overlay class="fixed inset-0 bg-black/40" />
    <Dialog.Content class="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-background border border-border rounded-lg shadow-xl w-[96vw] max-w-[1400px] max-h-[85vh] overflow-hidden flex flex-col">
      <Dialog.Header class="px-5 py-3 border-b">
        <Dialog.Title class="text-lg font-semibold">{title}</Dialog.Title>
        <Dialog.Description class="text-sm text-muted-foreground">
          <div class="mt-2 grid grid-cols-2 gap-6 items-start">
            <!-- Left column: help text -->
            <div class="flex flex-col gap-2 text-sm text-muted-foreground">
              <div>Review and stage changes. Nothing is saved until you confirm. Choose per-row actions.</div>

              <div class="flex items-center gap-2">
                <ChevronLeft size={16} />
                <div>Imports that Transaction from YNAB</div>
              </div>

              <div class="flex items-center gap-2">
                <ChevronRight size={16} />
                <div>Exports that Transaction to YNAB</div>
              </div>

              <div class="flex items-center gap-2">
                <XIcon size={16} />
                <div>Skips syncing that Transaction</div>
              </div>
            </div>

            <!-- Right column: controls -->
            <div class="flex flex-col items-start gap-3">
              <div class="flex items-center gap-2">
                <label class="text-sm text-muted-foreground" for="from-date">Sync from date:</label>
                <input id="from-date" type="date" bind:value={fromDate} class="h-8 px-2 border border-input rounded-md text-xs" />
                <button class={buttonVariants({ variant: 'secondary', size: 'sm' })} on:click={refresh} disabled={loading}>Refresh Data</button>
              </div>

              <div class="flex items-center gap-2">
                <button type="button" class={buttonVariants({ variant: 'ghost', size: 'default' })} on:click={() => open=false}>Cancel</button>
                <button type="button" class={buttonVariants({ variant: 'default', size: 'default' })} on:click={onConfirm} disabled={applying || loading}>{applying ? 'Applying…' : 'Confirm'}</button>
              </div>
            </div>
          </div>
        </Dialog.Description>
      </Dialog.Header>

  <div class="p-4 flex-1 overflow-y-auto overflow-x-hidden">
        {#if loading}
          <div class="text-sm text-muted-foreground">Loading preview…</div>
        {:else if error}
          <div class="text-sm text-destructive">{error}</div>
        {:else}
          <div class="mb-3 flex items-center justify-between text-sm">
            <div>
              <span class="font-medium">Summary:</span>
              <span class="ml-2">Add: {summary.add}</span>
              <span class="ml-3">Update: {summary.update}</span>
              <span class="ml-3">Delete: {summary.delete}</span>
              <span class="ml-3 text-muted-foreground">Total: {summary.total}</span>
            </div>
            <!-- controls removed: refresh happens on open, no reset tracking -->
          </div>
          <div class="grid grid-cols-[1fr_64px_1fr] gap-3 text-sm w-full">
            <div class="px-2 py-1 text-muted-foreground">Local Data (in app)</div>
            <!-- Global select-all action controls -->
            <div class="px-2 py-1 flex items-center justify-center gap-2">
              <button class={`h-8 w-8 inline-flex items-center justify-center rounded border ${allAction==='left' ? 'bg-foreground text-background' : 'bg-background text-foreground'}`} aria-label="Select all: pull remote (right → left)" title="Select all: pull remote (right → left)" on:click={() => setAll('left')}>
                <ChevronLeft size={16} />
              </button>
              <button class={`h-8 w-8 inline-flex items-center justify-center rounded border ${allAction==='skip' ? 'bg-foreground text-background' : 'bg-background text-foreground'}`} aria-label="Select all: skip" title="Select all: skip" on:click={() => setAll('skip')}>
                <XIcon size={16} />
              </button>
              <button class={`h-8 w-8 inline-flex items-center justify-center rounded border ${allAction==='right' ? 'bg-foreground text-background' : 'bg-background text-foreground'}`} aria-label="Select all: push local (left → right)" title="Select all: push local (left → right)" on:click={() => setAll('right')}>
                <ChevronRight size={16} />
              </button>
            </div>
            <div class="px-2 py-1 text-muted-foreground">Remote Data (in YNAB)</div>
            <div class="col-span-3 border-t"></div>

            {#each items as row}
              <!-- Left card -->
              <div class={`p-2 border rounded ${colorFor(row.status)} flex flex-col gap-1 w-full overflow-hidden`}>
                <div class="font-mono text-xs break-words whitespace-pre-wrap w-full">
                  {(row.leftDetails?.date || '')}
                  {#if row.leftDetails?.amount !== undefined && row.leftDetails?.amount !== null}
                    {' | '}{formatAmount(row.leftDetails?.amount)}
                  {/if}
                </div>
                {#if row.leftDetails?.payee}
                  <div class="font-mono text-xs break-words whitespace-pre-wrap w-full">Payee: {row.leftDetails.payee}</div>
                {/if}
                {#if row.leftDetails?.category}
                  <div class="font-mono text-xs break-words whitespace-pre-wrap w-full">Category: {row.leftDetails.category}</div>
                {/if}
                {#if row.leftDetails?.memo}
                  <div class="font-mono text-xs break-words whitespace-pre-wrap w-full">Memo: {row.leftDetails.memo}</div>
                {/if}
              </div>

              <!-- Action buttons column, prevent overlap -->
              <div class="flex items-center justify-center gap-2">
                <button class={`h-9 w-9 inline-flex items-center justify-center rounded border ${row.action==='left' ? 'bg-foreground text-background' : 'bg-background text-foreground'}`} title="Pull remote (right → left)" on:click={() => row.action='left'}>
                  <ChevronLeft size={18} />
                </button>
                <button class={`h-9 w-9 inline-flex items-center justify-center rounded border ${row.action==='skip' ? 'bg-foreground text-background' : 'bg-background text-foreground'}`} title="Skip" on:click={() => row.action='skip'}>
                  <XIcon size={18} />
                </button>
                <button class={`h-9 w-9 inline-flex items-center justify-center rounded border ${row.action==='right' ? 'bg-foreground text-background' : 'bg-background text-foreground'}`} title="Push local (left → right)" on:click={() => row.action='right'}>
                  <ChevronRight size={18} />
                </button>
              </div>

              <!-- Right card -->
              <div class={`p-2 border rounded ${colorFor(row.status)} flex flex-col gap-1 w-full overflow-hidden`}>
                <div class="font-mono text-xs break-words whitespace-pre-wrap w-full">
                  {(row.rightDetails?.date || '')}
                  {#if row.rightDetails?.amount !== undefined && row.rightDetails?.amount !== null}
                    {' | '}{formatAmount(row.rightDetails?.amount)}
                  {/if}
                </div>
                {#if row.rightDetails?.payee}
                  <div class="font-mono text-xs break-words whitespace-pre-wrap w-full">Payee: {row.rightDetails.payee}</div>
                {/if}
                {#if row.rightDetails?.category}
                  <div class="font-mono text-xs break-words whitespace-pre-wrap w-full">Category: {row.rightDetails.category}</div>
                {/if}
                {#if row.rightDetails?.memo}
                  <div class="font-mono text-xs break-words whitespace-pre-wrap w-full">Memo: {row.rightDetails.memo}</div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>

      
    </Dialog.Content>
  </Dialog.Portal>
  </Dialog.Root>

<style>
  :global(.DialogOverlay) { backdrop-filter: blur(2px); }
</style>
