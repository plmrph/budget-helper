<script lang="ts" generics="TData">
  import { X } from "@lucide/svelte";
  import type { Table } from "@tanstack/table-core";
  import DataTableFacetedFilter from "./data-table-faceted-filter.svelte";
  import DataTableColumnFilter from "./data-table-column-filter.svelte";
  import DataTableRangeFilter from "./data-table-range-filter.svelte";
  import DataTableViewOptions from "./data-table-view-options.svelte";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Input } from "$lib/components/ui/input/index.js";

  let { 
    table,
    searchColumn = "name",
    searchPlaceholder = "Filter...",
    facetedFilters = [],
    rangeFilters = [],
    textFilters = [],
    useGlobalFilter = false,
    data = [],
    children
  }: { 
    table: Table<TData>;
    searchColumn?: string;
    searchPlaceholder?: string;
    facetedFilters?: Array<{
      column: string;
      title: string;
      options: Array<{
        label: string;
        value: string;
        icon?: any;
      }>;
    }>;
    rangeFilters?: Array<{
      column: string;
      title: string;
      step?: number;
      formatValue?: (value: number) => string;
    }>;
    textFilters?: Array<{
      column: string;
      title: string;
    }>;
    useGlobalFilter?: boolean;
    data?: TData[];
    children?: any;
  } = $props();

  const isFiltered = $derived(table.getState().columnFilters.length > 0);
</script>

<div class="flex flex-wrap items-start justify-between gap-2">
  <div class="flex flex-wrap flex-1 items-center gap-2 min-w-0">
    <Input
      type="text"
      placeholder={searchPlaceholder}
      value={useGlobalFilter 
        ? (table.getState().globalFilter ?? "")
        : (table.getColumn(searchColumn)?.getFilterValue() as string) ?? ""
      }
      oninput={(e) => {
        if (useGlobalFilter) {
          table.setGlobalFilter(e.currentTarget.value);
        } else {
          table.getColumn(searchColumn)?.setFilterValue(e.currentTarget.value);
        }
      }}
      onblur={() => {}}
      onkeydown={() => {}}
      onchange={(e) => {
        if (useGlobalFilter) {
          table.setGlobalFilter(e.currentTarget.value);
        } else {
          table.getColumn(searchColumn)?.setFilterValue(e.currentTarget.value);
        }
      }}
      class="h-8 min-w-[320px] w-full lg:w-[320px]"
    />
  {#each facetedFilters as filter (filter.column)}
      {@const column = table.getColumn(filter.column)}
      {#if column}
        <DataTableFacetedFilter 
          {column} 
          title={filter.title} 
          options={filter.options} 
        />
      {/if}
    {/each}
  {#each rangeFilters as filter (filter.column)}
      {@const column = table.getColumn(filter.column)}
      {#if column}
        <DataTableRangeFilter 
          {column} 
          title={filter.title} 
          {data}
          field={filter.column}
          step={filter.step}
          formatValue={filter.formatValue}
        />
      {/if}
    {/each}
  {#each textFilters as filter (filter.column)}
      {@const column = table.getColumn(filter.column)}
      {#if column}
        <DataTableColumnFilter 
          {column} 
          title={filter.title} 
        />
      {/if}
    {/each}
    {#if isFiltered}
      <Button
        variant="ghost"
        onclick={() => table.resetColumnFilters()}
  disabled={false}
        class="h-8 px-2 lg:px-3"
      >
        Reset
        <X class="ml-2 h-4 w-4" />
      </Button>
    {/if}
  </div>
  <div class="flex flex-wrap items-center gap-2">
    <!-- Custom buttons slot -->
    {#if children}
      {@render children({ table })}
    {/if}
    <DataTableViewOptions {table} />
  </div>
</div>