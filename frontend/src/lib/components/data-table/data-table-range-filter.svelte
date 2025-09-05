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
    data,
    field,
    step = 1,
    formatValue = (value: number) => value.toString(),
  }: {
    column: Column<TData, TValue>;
    title: string;
    data: TData[];
    field: string;
    step?: number;
    formatValue?: (value: number) => string;
  } = $props();

  // Calculate min and max values from data (memoized, handles empty)
  const values = $derived(
    (data ?? [])
      .map((item: any) => item[field])
      .filter((val) => typeof val === "number") as number[]
  );
  const minValue = $derived(values.length ? Math.min(...values) : 0);
  const maxValue = $derived(values.length ? Math.max(...values) : 0);

  const filterValue = $derived((column?.getFilterValue() as [number, number]) ?? [minValue, maxValue]);
  const hasFilter = $derived(filterValue[0] !== minValue || filterValue[1] !== maxValue);

  let minInput: number = $state(0);
  let maxInput: number = $state(0);

  // Update inputs when filter value or computed range changes
  $effect(() => {
    const mv = minValue;
    const xv = maxValue;
    const fv = filterValue;
    minInput = fv[0] ?? mv;
    maxInput = fv[1] ?? xv;
  });

  function applyFilter() {
    if (minInput === minValue && maxInput === maxValue) {
      column?.setFilterValue(undefined);
    } else {
      column?.setFilterValue([minInput, maxInput]);
    }
  }

  function clearFilter() {
    minInput = minValue;
    maxInput = maxValue;
    column?.setFilterValue(undefined);
  }
</script>

<Popover.Root>
  <Popover.Trigger class="">
    <Button variant="outline" size="sm" class="h-8 border-dashed" disabled={false}>
      <CirclePlus class="mr-2 h-4 w-4" />
      {title}
      {#if hasFilter}
        <Badge variant="secondary" class="ml-2 rounded-sm px-1 font-normal" href={undefined}>
          {formatValue(filterValue[0])} - {formatValue(filterValue[1])}
        </Badge>
      {/if}
    </Button>
  </Popover.Trigger>
  <Popover.Content class="w-[280px] p-3" align="start" portalProps={{}}>
    <div class="space-y-3">
      <div class="text-sm font-medium">Filter {title}</div>
      
      <div class="grid grid-cols-2 gap-2">
        <div>
          <label for="min-input" class="text-xs text-muted-foreground">Min</label>
          <Input
            id="min-input"
            type="number"
            placeholder="Min"
            value={minInput}
            min={minValue}
            max={maxValue}
            step={step}
            oninput={(e) => {
              minInput = Number(e.currentTarget.value);
              applyFilter();
            }}
            onblur={() => {}}
            onkeydown={() => {}}
            class="h-8"
          />
        </div>
        <div>
          <label for="max-input" class="text-xs text-muted-foreground">Max</label>
          <Input
            id="max-input"
            type="number"
            placeholder="Max"
            value={maxInput}
            min={minValue}
            max={maxValue}
            step={step}
            oninput={(e) => {
              maxInput = Number(e.currentTarget.value);
              applyFilter();
            }}
            onblur={() => {}}
            onkeydown={() => {}}
            class="h-8"
          />
        </div>
      </div>

      <div class="text-xs text-muted-foreground">
        Range: {formatValue(minValue)} - {formatValue(maxValue)}
      </div>

      {#if hasFilter}
        <Button
          variant="ghost"
          size="sm"
          onclick={clearFilter}
          disabled={false}
          class="h-6 w-full justify-center text-xs"
        >
          <X class="mr-1 h-3 w-3" />
          Clear Filter
        </Button>
      {/if}
    </div>
  </Popover.Content>
</Popover.Root>