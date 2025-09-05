<script lang="ts">
  import { Ellipsis } from "@lucide/svelte";
  import { Button } from "$lib/components/ui/button/index.js";
  import * as DropdownMenu from "$lib/components/ui/dropdown-menu/index.js";

  let { id, onEdit, onDelete, onView }: { 
    id: string;
    onEdit?: (id: string) => void;
    onDelete?: (id: string) => void;
    onView?: (id: string) => void;
  } = $props();
</script>

<DropdownMenu.Root>
  <DropdownMenu.Trigger>
    {#snippet child({ props })}
      <Button
        {...props}
        variant="ghost"
        size="icon"
        class="relative size-8 p-0"
      >
        <span class="sr-only">Open menu</span>
        <Ellipsis class="size-4" />
      </Button>
    {/snippet}
  </DropdownMenu.Trigger>
  <DropdownMenu.Content>
    <DropdownMenu.Group>
      <DropdownMenu.Label>Actions</DropdownMenu.Label>
      <DropdownMenu.Item onclick={() => navigator.clipboard.writeText(id)}>
        Copy ID
      </DropdownMenu.Item>
    </DropdownMenu.Group>
    <DropdownMenu.Separator />
    {#if onView}
      <DropdownMenu.Item onclick={() => onView?.(id)}>
        View details
      </DropdownMenu.Item>
    {/if}
    {#if onEdit}
      <DropdownMenu.Item onclick={() => onEdit?.(id)}>
        Edit
      </DropdownMenu.Item>
    {/if}
    {#if onDelete}
      <DropdownMenu.Item onclick={() => onDelete?.(id)} class="text-destructive">
        Delete
      </DropdownMenu.Item>
    {/if}
  </DropdownMenu.Content>
</DropdownMenu.Root>