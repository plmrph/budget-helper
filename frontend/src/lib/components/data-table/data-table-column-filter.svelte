<script lang="ts" generics="TData, TValue">
  import { CirclePlus, X } from "@lucide/svelte";
  import type { Column } from "@tanstack/table-core";
  import * as Popover from "$lib/components/ui/popover/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";

  let {
    column,
    title,
  }: {
    column: Column<TData, TValue>;
    title: string;
  } = $props();

  const filterValue = $derived((column?.getFilterValue() as string) ?? "");
  const hasFilter = $derived(!!filterValue);
</script>

<Popover.Root>
  <Popover.Trigger>
    {#snippet child({ props })}
      <Button {...props} variant="outline" size="sm" class="h-8 border-dashed">
        <CirclePlus class="mr-2 h-4 w-4" />
        {title}
        {#if hasFilter}
          <Badge variant="secondary" class="ml-2 rounded-sm px-1 font-normal">
            Filtered
          </Badge>
        {/if}
      </Button>
    {/snippet}
  </Popover.Trigger>
  <Popover.Content class="w-[200px] p-3" align="start">
    <div class="space-y-2">
      <div class="text-sm font-medium">Filter {title}</div>
      <Input
        placeholder={`Filter ${title.toLowerCase()}...`}
        value={filterValue}
        oninput={(e) => column?.setFilterValue(e.currentTarget.value)}
        class="h-8"
      />
      {#if hasFilter}
        <Button
          variant="ghost"
          size="sm"
          onclick={() => column?.setFilterValue("")}
          class="h-6 w-full justify-center text-xs"
        >
          <X class="mr-1 h-3 w-3" />
          Clear
        </Button>
      {/if}
    </div>
  </Popover.Content>
</Popover.Root>