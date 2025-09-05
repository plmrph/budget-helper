<script lang="ts" generics="TData, TValue">
  import { CirclePlus, Check } from "@lucide/svelte";
  import type { Column } from "@tanstack/table-core";
  import * as Command from "$lib/components/ui/command/index.js";
  import * as Popover from "$lib/components/ui/popover/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { cn } from "$lib/utils.js";
  import { Separator } from "$lib/components/ui/separator/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import type { Component } from "svelte";

  let {
    column,
    title,
    options,
  }: {
    column: Column<TData, TValue>;
    title: string;
    options: {
      label: string;
      value: string;
      icon?: Component;
    }[];
  } = $props();

  let open = $state(false);
  let facets: Map<string, number> | null = $state(null);
  $effect(() => {
    if (open && !facets) {
      facets = column?.getFacetedUniqueValues?.() ?? null;
    }
  });

  const selectedValues = $derived(new Set<string>((column?.getFilterValue() as string[]) ?? []));

</script>

<Popover.Root bind:open={open}>
  <Popover.Trigger class="">
    <Button variant="outline" size="sm" class="h-8 border-dashed" disabled={false}>
      <CirclePlus class="mr-2 h-4 w-4" />
      {title}
      {#if selectedValues.size > 0}
        <Separator orientation="vertical" class="mx-2 h-4" />
        <Badge variant="secondary" class="rounded-sm px-1 font-normal lg:hidden" href={undefined}>
          {selectedValues.size}
        </Badge>
        <div class="hidden space-x-1 lg:flex">
          {#if selectedValues.size > 2}
            <Badge variant="secondary" class="rounded-sm px-1 font-normal" href={undefined}>
              {selectedValues.size} selected
            </Badge>
          {:else}
            {#each options.filter((opt) => selectedValues.has(opt.value)) as option (option.value)}
              <Badge variant="secondary" class="rounded-sm px-1 font-normal" href={undefined}>
                {option.label}
              </Badge>
            {/each}
          {/if}
        </div>
      {/if}
    </Button>
  </Popover.Trigger>
  <Popover.Content class="w-[200px] p-0" align="start" portalProps={{}}>
    <Command.Root class="">
      <Command.Input placeholder={title} class="h-8 px-3" />
      <Command.List class="">
        <Command.Empty class="">No results found.</Command.Empty>
        <Command.Group class="" heading="Options" value="options">
          {#each options as option (option.value)}
            {@const isSelected = selectedValues.has(option.value)}
            <Command.Item
              class=""
              onSelect={() => {
                const next = new Set(selectedValues);
                if (isSelected) {
                  next.delete(option.value);
                } else {
                  next.add(option.value);
                }
                const filterValues = Array.from(next);
                column?.setFilterValue(filterValues.length ? filterValues : undefined);
              }}
            >
              <div
                class={cn(
                  "border-primary mr-2 flex size-4 items-center justify-center rounded-sm border",
                  isSelected
                    ? "bg-primary text-primary-foreground"
                    : "opacity-50 [&_svg]:invisible"
                )}
              >
                <Check class="size-4" />
              </div>
              {#if option.icon}
                {@const Icon = option.icon}
                <Icon class="text-muted-foreground mr-2 h-4 w-4" />
              {/if}
              <span>{option.label}</span>
              {#if facets && facets.get(option.value)}
                <span
                  class="ml-auto flex size-4 items-center justify-center font-mono text-xs"
                >
                  {facets.get(option.value)}
                </span>
              {/if}
            </Command.Item>
          {/each}
        </Command.Group>
        {#if selectedValues.size > 0}
          <Command.Separator class="" />
          <Command.Group class="" heading="Actions" value="actions">
            <Command.Item
              class="justify-center text-center"
              onSelect={() => column?.setFilterValue(undefined)}
            >
              Clear filters
            </Command.Item>
          </Command.Group>
        {/if}
      </Command.List>
    </Command.Root>
  </Popover.Content>
</Popover.Root>