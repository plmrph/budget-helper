<script lang="ts" generics="TData">
  import type { Column } from "@tanstack/table-core";
  import * as DropdownMenu from "$lib/components/ui/dropdown-menu/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { ArrowUp, ArrowDown, ChevronsUpDown, EyeOff } from "@lucide/svelte";
  import { cn } from "$lib/utils.js";
  import type { HTMLAttributes } from "svelte/elements";

  let {
    column,
    title,
    class: className,
    ...restProps
  }: {
    column: Column<TData>;
    title: string;
    class?: string;
  } & HTMLAttributes<HTMLDivElement> = $props();
</script>

{#if !column?.getCanSort()}
  <div class={className} {...restProps}>{title}</div>
{:else}
  <div class={cn("flex items-center", className)} {...restProps}>
    <DropdownMenu.Root>
      <DropdownMenu.Trigger>
        {#snippet child({ props })}
          <Button
            {...props}
            variant="ghost"
            size="sm"
            class="data-[state=open]:bg-accent -ml-3 h-8"
          >
            <span>{title}</span>
            {#if column.getIsSorted() === "desc"}
              <ArrowDown class="ml-2 h-4 w-4" />
            {:else if column.getIsSorted() === "asc"}
              <ArrowUp class="ml-2 h-4 w-4" />
            {:else}
              <ChevronsUpDown class="ml-2 h-4 w-4" />
            {/if}
          </Button>
        {/snippet}
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="start">
        <DropdownMenu.Item onclick={() => column.toggleSorting(false)}>
          <ArrowUp class="text-muted-foreground/70 mr-2 size-3.5" />
          Asc
        </DropdownMenu.Item>
        <DropdownMenu.Item onclick={() => column.toggleSorting(true)}>
          <ArrowDown class="text-muted-foreground/70 mr-2 size-3.5" />
          Desc
        </DropdownMenu.Item>
        <DropdownMenu.Separator />
        <DropdownMenu.Item onclick={() => column.toggleVisibility(false)}>
          <EyeOff class="text-muted-foreground/70 mr-2 size-3.5" />
          Hide
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  </div>
{/if}
