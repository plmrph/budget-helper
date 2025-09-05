<script lang="ts">
  import type { ComponentProps } from "svelte";
  import { ArrowUpDown, ArrowUp, ArrowDown } from "@lucide/svelte";
  import { Button } from "$lib/components/ui/button/index.js";

  let { 
    variant = "ghost", 
    sortDirection,
    children,
    ...restProps 
  }: ComponentProps<typeof Button> & {
    sortDirection?: "asc" | "desc" | false;
    children: any;
  } = $props();

  function getSortIcon() {
    if (sortDirection === "asc") return ArrowUp;
    if (sortDirection === "desc") return ArrowDown;
    return ArrowUpDown;
  }
</script>

<Button {variant} {...restProps} class="h-8 data-[state=open]:bg-accent">
  {@render children()}
  {#if sortDirection === "asc"}
    <ArrowUp class="ml-2 size-4" />
  {:else if sortDirection === "desc"}
    <ArrowDown class="ml-2 size-4" />
  {:else}
    <ArrowUpDown class="ml-2 size-4" />
  {/if}
</Button>