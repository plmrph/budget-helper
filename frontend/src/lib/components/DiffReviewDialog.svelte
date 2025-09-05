<script>
  import { createEventDispatcher, onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import { Checkbox } from '$lib/components/ui/checkbox/index.js';

  export let open = false;
  export let title = 'Review Changes';
  export let mode = 'import'; // 'import' | 'export'
  export let loadPreview; // async function to load preview data
  export let applyChanges; // async function(selection)

  const dispatch = createEventDispatcher();

  let loading = false;
  let applying = false;
  let error = null;
  let summary = { add: 0, update: 0, delete: 0, total: 0 };
  let diffs = [];
  let selected = new Set();
  let selectAllToggle = false;

  async function refreshPreview() {
    if (!loadPreview) return;
    loading = true;
    error = null;
    try {
      const res = await loadPreview();
      if (res?.success && res?.data) {
        summary = res.data.summary || summary;
        diffs = (res.data.diffs || []).map(d => ({ ...d, checked: true }));
        // default select all
        selected = new Set(diffs.map(d => d.transaction_id));
      } else {
        error = res?.error || 'Failed to load preview';
      }
    } catch (e) {
      error = e?.message || 'Failed to load preview';
    } finally {
      loading = false;
    }
  }

  // Keep selected Set in sync with individual item checked flags
  $: selected = new Set((diffs || []).filter(d => d.checked).map(d => d.transaction_id));

  function toggleAll(checked) {
    if (checked) {
      diffs = (diffs || []).map(d => ({ ...d, checked: true }));
    } else {
      diffs = (diffs || []).map(d => ({ ...d, checked: false }));
    }
  }

  function groupedByAction() {
    const groups = { add: [], update: [], delete: [] };
    for (const d of diffs) {
      if (groups[d.action]) groups[d.action].push(d);
    }
    return groups;
  }

  function formatAmount(mu) {
    if (mu == null) return '';
    const n = Number(mu) / 1000;
    return n.toLocaleString(undefined, { style: 'currency', currency: 'USD' });
  }

  async function onApply(all = false) {
    if (!applyChanges) return;
    applying = true;
    error = null;
    try {
      const groups = groupedByAction();
      const pick = (arr) => all ? arr.map(d => d.transaction_id) : arr.filter(d => selected.has(d.transaction_id)).map(d => d.transaction_id);
      const payload = {
        add: pick(groups.add),
        update: pick(groups.update),
        delete: pick(groups.delete)
      };
      const res = await applyChanges(payload);
      if (res?.success !== false) {
        dispatch('applied', { summary: payload });
        open = false;
      } else {
        error = res?.error || res?.message || 'Failed to apply changes';
      }
    } catch (e) {
      error = e?.message || 'Failed to apply changes';
    } finally {
      applying = false;
    }
  }

  $: if (open) {
    // load preview when opened
    refreshPreview();
  }
</script>

<Dialog.Root bind:open>
  <Dialog.Portal>
    <Dialog.Overlay class="fixed inset-0 bg-black/40" />
    <Dialog.Content class="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-background border border-border rounded-lg shadow-xl w-[95vw] max-w-[1200px] max-h-[85vh] overflow-hidden flex flex-col">
      <Dialog.Header class="px-5 py-3 border-b">
        <Dialog.Title class="text-lg font-semibold">{title}</Dialog.Title>
        <Dialog.Description class="text-sm text-muted-foreground">
          {mode === 'import' ? 'Review changes from YNAB before applying to your local data.' : 'Review local changes before syncing to YNAB.'}
        </Dialog.Description>
      </Dialog.Header>

      <div class="p-4 flex-1 overflow-auto">
        {#if loading}
          <div class="text-sm text-muted-foreground">Loading preview…</div>
        {:else if error}
          <div class="text-sm text-destructive">{error}</div>
        {:else}
          <div class="mb-4 flex items-center justify-between">
            <div class="text-sm">
              <span class="font-medium">Summary:</span>
              <span class="ml-2">Add: {summary.add}</span>
              <span class="ml-3">Update: {summary.update}</span>
              <span class="ml-3">Delete: {summary.delete}</span>
              <span class="ml-3 text-muted-foreground">Total: {summary.total}</span>
            </div>
              <div class="flex items-center gap-3">
              <div class="flex items-center gap-2 text-sm">
                <input type="checkbox" class="size-4" checked={selected.size === diffs.length} indeterminate={selected.size>0 && selected.size<diffs.length} on:change={(e)=>toggleAll(e.currentTarget.checked)} />
                <span>Select all</span>
              </div>
              <Button variant="outline" size="sm" on:click={refreshPreview}>Refresh Data</Button>
            </div>
          </div>

          <div class="space-y-6">
            {#each Object.entries(groupedByAction()) as [action, items]}
              {#if items.length}
                <div class="border rounded-md">
                  <div class="px-3 py-2 border-b bg-muted/40 text-sm font-medium capitalize">{action} ({items.length})</div>
                  <div class="divide-y">
                    {#each items as d}
                      <div class="p-3 grid grid-cols-12 gap-3 items-start">
                        <div class="col-span-1 pt-1">
                          <Checkbox bind:checked={d.checked} aria-label="Select change" />
                        </div>
                        <div class="col-span-11">
                          <div class="flex items-center justify-between text-sm">
                            <div class="font-medium">{d.transaction_id}</div>
                            <div class="text-xs text-muted-foreground">{d.local?.date || d.remote?.date}</div>
                          </div>
                          {#if d.action === 'update'}
                            <div class="mt-2 grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <div class="text-xs font-semibold mb-1">Local</div>
                                <table class="w-full text-xs">
                                  <tbody>
                                    {#each d.changes as c}
                                      <tr>
                                        <td class="pr-2 text-muted-foreground align-top">{c.field}</td>
                                        <td class="font-mono break-all">
                                          {c.field === 'amount' ? formatAmount(c.from_value) : (c.from_value ?? '')}
                                        </td>
                                      </tr>
                                    {/each}
                                  </tbody>
                                </table>
                              </div>
                              <div>
                                <div class="text-xs font-semibold mb-1">Remote</div>
                                <table class="w-full text-xs">
                                  <tbody>
                                    {#each d.changes as c}
                                      <tr>
                                        <td class="pr-2 text-muted-foreground align-top">{c.field}</td>
                                        <td class="font-mono break-all">
                                          {c.field === 'amount' ? formatAmount(c.to_value) : (c.to_value ?? '')}
                                        </td>
                                      </tr>
                                    {/each}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          {:else}
                            <div class="mt-2 text-sm">
                              <table class="w-full text-xs">
                                <tbody>
                                  {#each d.changes as c}
                                    <tr>
                                      <td class="pr-2 text-muted-foreground align-top">{c.field}</td>
                                      <td class="font-mono break-all">
                                        {#if d.action === 'add'}
                                          {c.field === 'amount' ? formatAmount(c.to_value) : (c.to_value ?? '')}
                                        {:else if d.action === 'delete'}
                                          {c.field === 'amount' ? formatAmount(c.from_value) : (c.from_value ?? '')}
                                        {/if}
                                      </td>
                                    </tr>
                                  {/each}
                                </tbody>
                              </table>
                            </div>
                          {/if}
                        </div>
                      </div>
                    {/each}
                  </div>
                </div>
              {/if}
            {/each}
          </div>
        {/if}
      </div>

      <Dialog.Footer class="px-5 py-3 border-t flex items-center justify-between">
        <div class="text-xs text-muted-foreground">Pick changes and apply. This won't proceed until you confirm.</div>
        <div class="flex items-center gap-2">
          <Button variant="ghost" on:click={() => open = false}>Cancel</Button>
          <Button disabled={applying || loading || selected.size === 0} on:click={() => onApply(false)}>
            {applying ? 'Applying…' : 'Apply Selected'}
          </Button>
          <Button disabled={applying || loading || diffs.length === 0} on:click={() => onApply(true)} variant="default">
            {applying ? 'Applying…' : 'Apply All'}
          </Button>
        </div>
      </Dialog.Footer>
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>

<style>
  :global(.DialogOverlay) { backdrop-filter: blur(2px); }
</style>
