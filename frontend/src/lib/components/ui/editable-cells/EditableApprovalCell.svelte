<script>
  import { tick } from 'svelte';
  import { cn } from "$lib/utils.js";
  import * as Popover from "$lib/components/ui/popover/index.js";
  import { CheckCircle, CircleHelp, Loader2 } from "@lucide/svelte";
  
  let { 
    value = $bindable(false), 
    onSave, 
    saveStatus = null, 
    saveError = null,
    disabled = false,
    class: className = "",
    ...restProps
  } = $props();
  
  let open = $state(false);

  // Approval options - simple dropdown options
  const approvalOptions = [
    { value: true, label: "Approved" },
    { value: false, label: "Unapproved" }
  ];

  function handleSelect(selectedValue) {
    if (selectedValue !== value && onSave) {
      // Non-blocking save - returns immediately
      onSave(selectedValue);
    }
    open = false;
  }

  // Get display styling for approval status
  function getDisplayInfo(val) {
    const isApproved = val === true || val === 'true';
    return {
      title: isApproved ? "Approved" : "Unapproved",
      icon: isApproved ? 'approved' : 'unapproved',
      iconClass: isApproved 
        ? "text-green-600 dark:text-green-400" 
        : "text-yellow-600 dark:text-yellow-400"
    };
  }

  // Visual feedback based on save status - no layout-shifting borders
  const cellClass = $derived(cn(
    "relative px-1 py-0.5 min-h-[28px] flex items-center justify-center",
    saveStatus === 'error' ? "bg-red-50 rounded" : "",
    !disabled ? "cursor-pointer hover:bg-gray-50 rounded" : "",
    disabled ? "cursor-not-allowed opacity-50" : "",
    className
  ));

  // No need for effects since we're using the value directly

  const displayInfo = $derived(getDisplayInfo(value));
</script>

<div class={cellClass} {...restProps}>
  <div class="w-full flex justify-center">
    <Popover.Root bind:open>
      <Popover.Trigger>
        <button
          type="button"
          class="cursor-pointer border-0 bg-transparent focus:outline-none"
          {disabled}
          aria-label={`Edit approval status: ${displayInfo.title}`}
        >
          {#if saveStatus === 'saving'}
            <Loader2 class="h-4 w-4 animate-spin text-muted-foreground" />
          {:else}
            {#if displayInfo.icon === 'approved'}
              <CheckCircle class={cn("h-4 w-4", displayInfo.iconClass)} title={displayInfo.title} />
            {:else}
              <CircleHelp class={cn("h-4 w-4", displayInfo.iconClass)} title={displayInfo.title} />
            {/if}
          {/if}
        </button>
      </Popover.Trigger>
      <Popover.Content class="w-[150px] p-0" align="center">
        <div class="py-2">
          {#each approvalOptions as option}
            {@const optionInfo = getDisplayInfo(option.value)}
            <button
              type="button"
              class={cn(
                "w-full px-3 py-2 text-left text-sm hover:bg-gray-100 focus:bg-gray-100 focus:outline-none flex items-center gap-2",
                value === option.value ? "bg-blue-50" : ""
              )}
              onclick={() => handleSelect(option.value)}
            >
              {#if optionInfo.icon === 'approved'}
                <CheckCircle class={cn("h-4 w-4", optionInfo.iconClass)} />
              {:else}
                <CircleHelp class={cn("h-4 w-4", optionInfo.iconClass)} />
              {/if}
              <span class="text-xs">{optionInfo.title}</span>
            </button>
          {/each}
        </div>
      </Popover.Content>
    </Popover.Root>
  </div>
  
  <!-- Status indicators -->
  {#if saveStatus === 'error'}
    <div 
      class="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full" 
      title={saveError || "Save failed"}
    ></div>
  {/if}
</div>