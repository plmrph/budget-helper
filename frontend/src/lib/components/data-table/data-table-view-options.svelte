<script lang="ts" generics="TData">
  import { Settings2 } from "@lucide/svelte";
  import type { Table } from "@tanstack/table-core";
  import { buttonVariants } from "$lib/components/ui/button/index.js";
  import * as DropdownMenu from "$lib/components/ui/dropdown-menu/index.js";

  let { table }: { table: Table<TData> } = $props();
</script>

<DropdownMenu.Root>
  <DropdownMenu.Trigger
    class={buttonVariants({
      variant: "outline",
      size: "sm",
      class: "ml-auto hidden h-8 lg:flex",
    })}
  >
    <Settings2 class="mr-2 h-4 w-4" />
    View
  </DropdownMenu.Trigger>
  <DropdownMenu.Content align="end">
    <DropdownMenu.Group>
      <DropdownMenu.Label>Toggle columns</DropdownMenu.Label>
      <DropdownMenu.Separator />
      {#each table
        .getAllColumns()
        .filter((col) => typeof col.accessorFn !== "undefined" && col.getCanHide()) as column (column.id)}
        <DropdownMenu.CheckboxItem
          bind:checked={
            () => column.getIsVisible(), (v) => column.toggleVisibility(!!v)
          }
          class="capitalize"
        >
          {column.id}
        </DropdownMenu.CheckboxItem>
      {/each}
    </DropdownMenu.Group>
  </DropdownMenu.Content>
</DropdownMenu.Root>
