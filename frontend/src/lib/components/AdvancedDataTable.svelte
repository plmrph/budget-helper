<script lang="ts" generics="TData">
  import {
    type ColumnDef,
    type ColumnFiltersState,
    type PaginationState,
    type RowSelectionState,
    type SortingState,
    type VisibilityState,
    type ColumnSizingState,
    type Table as TableType,
    getCoreRowModel,
    getFacetedRowModel,
    getFacetedUniqueValues,
    getFilteredRowModel,
    getPaginationRowModel,
    getSortedRowModel,
    type Column,
  } from "@tanstack/table-core";
  import DataTableToolbar from "./data-table/data-table-toolbar.svelte";
  import { createSvelteTable } from "$lib/components/ui/data-table/data-table.svelte.js";
  import FlexRender from "$lib/components/ui/data-table/flex-render.svelte";
  import * as Table from "$lib/components/ui/table/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import {
    ChevronRight,
    ChevronLeft,
    ChevronsLeft,
    ChevronsRight,
  } from "@lucide/svelte";

  type AdvancedDataTableProps<TData> = {
    columns: ColumnDef<TData>[];
    data: TData[];
    searchColumn?: string;
    searchPlaceholder?: string;
    pageSize?: number;
    useGlobalFilter?: boolean;
    initialColumnVisibility?: Record<string, boolean>;
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
  onRowSelectionChange?: (selectedRows: any[]) => void; 
  children?: any; 
  header?: (args: { table: TableType<TData> }) => any; 
  };

  let {
    data,
    columns,
    searchColumn = "name",
    searchPlaceholder = "Filter...",
    pageSize = 10,
    useGlobalFilter = true,
    initialColumnVisibility = {},
    facetedFilters = [],
    rangeFilters = [],
    textFilters = [],
    onRowSelectionChange,
    children,
    header,
  }: AdvancedDataTableProps<TData> = $props();

  let rowSelection = $state<RowSelectionState>({});
  let columnVisibility = $state<VisibilityState>(initialColumnVisibility);
  let columnFilters = $state<ColumnFiltersState>([]);
  let sorting = $state<SortingState>([]);
  let pagination = $state<PaginationState>({ pageIndex: 0, pageSize });
  let globalFilter = $state("");
  let columnSizing = $state<ColumnSizingState>({});

  // Track previous filter values to detect actual changes
  let prevColumnFiltersStr = $state("");
  let prevGlobalFilter = $state("");

  // Reset pagination to page 1 when filters actually change
  $effect(() => {
    // Check if filters have actually changed (not just accessed)
    const currentColumnFiltersStr = JSON.stringify(columnFilters);
    const filtersChanged = currentColumnFiltersStr !== prevColumnFiltersStr;
    const globalFilterChanged = globalFilter !== prevGlobalFilter;
    
    if (filtersChanged || globalFilterChanged) {
      // Reset to page 1 when filters change
      pagination.pageIndex = 0;
      
      // Update previous values
      prevColumnFiltersStr = currentColumnFiltersStr;
      prevGlobalFilter = globalFilter;
    }
  });

  const table = createSvelteTable({
    get data() {
      return data;
    },
    columns,
    state: {
      get sorting() {
        return sorting;
      },
      get columnVisibility() {
        return columnVisibility;
      },
      get rowSelection() {
        return rowSelection;
      },
      get columnFilters() {
        return columnFilters;
      },
      get pagination() {
        return pagination;
      },
      get globalFilter() {
        return globalFilter;
      },
      get columnSizing() {
        return columnSizing;
      },
    },
    enableRowSelection: true,
    enableGlobalFilter: useGlobalFilter,
    enableColumnResizing: true,
  columnResizeMode: "onChange",
    autoResetPageIndex: false,
    onRowSelectionChange: (updater) => {
      if (typeof updater === "function") {
        rowSelection = updater(rowSelection);
      } else {
        rowSelection = updater;
      }

      if (onRowSelectionChange) {
        const selectedRows = table
          .getFilteredSelectedRowModel()
          .rows.map((row) => row.original);
        onRowSelectionChange(selectedRows);
      }
    },
    onSortingChange: (updater) => {
      if (typeof updater === "function") {
        sorting = updater(sorting);
      } else {
        sorting = updater;
      }
    },
    onColumnFiltersChange: (updater) => {
      if (typeof updater === "function") {
        columnFilters = updater(columnFilters);
      } else {
        columnFilters = updater;
      }
    },
    onColumnVisibilityChange: (updater) => {
      if (typeof updater === "function") {
        columnVisibility = updater(columnVisibility);
      } else {
        columnVisibility = updater;
      }
    },
    onPaginationChange: (updater) => {
      if (typeof updater === "function") {
        pagination = updater(pagination);
      } else {
        pagination = updater;
      }
    },
    onGlobalFilterChange: (updater) => {
      if (typeof updater === "function") {
        globalFilter = updater(globalFilter);
      } else {
        globalFilter = updater;
      }
    },
    onColumnSizingChange: (updater) => {
      if (typeof updater === "function") {
        columnSizing = updater(columnSizing);
      } else {
        columnSizing = updater;
      }
    },
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFacetedRowModel: getFacetedRowModel(),
    getFacetedUniqueValues: getFacetedUniqueValues(),
  });
</script>

{#snippet Pagination({ table }: { table: TableType<TData> })}
  <div class="flex items-center justify-between px-2">
    <div class="text-muted-foreground flex-1 text-sm">
      {table.getFilteredSelectedRowModel().rows.length} of{" "}
      {table.getFilteredRowModel().rows.length} row(s) selected.
    </div>
    <div class="flex items-center space-x-6 lg:space-x-8">
      <div class="flex items-center space-x-2">
        <p class="text-sm font-medium">Rows per page</p>
        <select
          class="flex h-8 w-[70px] items-center justify-between rounded-md border border-input bg-background px-3 py-1 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          bind:value={pagination.pageSize}
          onchange={(e) => table.setPageSize(Number(e.currentTarget.value))}
        >
          {#each [5, 10, 20, 30, 40, 50] as pageSizeOption (pageSizeOption)}
            <option
              value={pageSizeOption}
              selected={pageSizeOption === pagination.pageSize}
            >
              {pageSizeOption}
            </option>
          {/each}
        </select>
      </div>
      <div
        class="flex w-[100px] items-center justify-center text-sm font-medium"
      >
        Page {table.getState().pagination.pageIndex + 1} of{" "}
        {table.getPageCount()}
      </div>
      <div class="flex items-center space-x-2">
        <Button
          variant="outline"
          class="hidden size-8 p-0 lg:flex"
          onclick={() => table.setPageIndex(0)}
          disabled={!table.getCanPreviousPage()}
        >
          <span class="sr-only">Go to first page</span>
          <ChevronsLeft class="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          class="size-8 p-0"
          onclick={() => table.previousPage()}
          disabled={!table.getCanPreviousPage()}
        >
          <span class="sr-only">Go to previous page</span>
          <ChevronLeft class="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          class="size-8 p-0"
          onclick={() => table.nextPage()}
          disabled={!table.getCanNextPage()}
        >
          <span class="sr-only">Go to next page</span>
          <ChevronRight class="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          class="hidden size-8 p-0 lg:flex"
          onclick={() => table.setPageIndex(table.getPageCount() - 1)}
          disabled={!table.getCanNextPage()}
        >
          <span class="sr-only">Go to last page</span>
          <ChevronsRight class="h-4 w-4" />
        </Button>
      </div>
    </div>
  </div>
{/snippet}

<div class="space-y-4">
  {@render header?.({ table: table as TableType<any> })}
  <DataTableToolbar
    {table}
    {data}
    {searchColumn}
    {searchPlaceholder}
    {useGlobalFilter}
    {facetedFilters}
    {rangeFilters}
    {textFilters}
    {children}
  />
  <div class="rounded-md border overflow-x-auto">
    <Table.Root class="w-full" style="table-layout: fixed; width: 100%;">
      {#if table.getHeaderGroups().length > 0}
        {@const leafHeaders = table.getHeaderGroups()[table.getHeaderGroups().length - 1].headers}
        <colgroup>
          {#each leafHeaders as leaf (leaf.id)}
            <col style={`width: ${leaf.getSize()}px`} />
          {/each}
        </colgroup>
      {/if}
      <Table.Header class="">
        {#each table.getHeaderGroups() as headerGroup (headerGroup.id)}
          <Table.Row class="">
            {#each headerGroup.headers as header (header.id)}
              <Table.Head
                class=""
                colspan={header.colSpan}
                style="width: {header.getSize()}px; position: relative;"
              >
                {#if !header.isPlaceholder}
                  <FlexRender
                    content={header.column.columnDef.header}
                    context={header.getContext()}
                  />
                  {#if header.column.getCanResize()}
                    <button
                      class="resize-handle absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none hover:bg-primary/20 border-r-2 border-transparent hover:border-primary bg-transparent border-0 p-0"
                      style="user-select: none; margin-right: -1px;"
                      aria-label="Resize column"
                      onmousedown={(e) => {
                        header.getResizeHandler()(e);
                      }}
                      ontouchstart={(e) => {
                        header.getResizeHandler()(e);
                      }}
                    ></button>
                  {/if}
                {/if}
              </Table.Head>
            {/each}
          </Table.Row>
        {/each}
      </Table.Header>
      <Table.Body class="">
        {#each table.getPaginationRowModel().rows as row (row.id)}
          <Table.Row class="" data-state={row.getIsSelected() && "selected"}>
            {#each row.getVisibleCells() as cell (cell.id)}
              <Table.Cell class="" style="overflow: hidden;">
                <FlexRender
                  content={cell.column.columnDef.cell}
                  context={cell.getContext()}
                />
              </Table.Cell>
            {/each}
          </Table.Row>
        {:else}
          <Table.Row class="">
            <Table.Cell colspan={columns.length} class="h-24 text-center">
              No results.
            </Table.Cell>
          </Table.Row>
        {/each}
      </Table.Body>
    </Table.Root>
  </div>
  {@render Pagination({ table: table as TableType<any> })}
</div>
