<script>
  import { tick } from 'svelte';
  import { cn } from "$lib/utils.js";
  import { ComboBox } from "../combobox/index.js";
  
  let { 
    value = $bindable(""), 
    categories = [],
    onSave, 
    saveStatus = null, 
    saveError = null,
    placeholder = "Select category",
    displayValue = null,
    disabled = false,
    class: className = "",
    ...restProps
  } = $props();
  
  let editing = $state(false);
  let comboValue = $state(value);
  let comboRef = $state(null);
  let comboOpen = $state(false);

  function startEdit(event) {
      console.log("EditableCategoryCell startEdit called, disabled:", disabled, "categories:", categories);
    if (event) {
      event.stopPropagation();
      event.preventDefault();
    }
    if (disabled) return;
    editing = true;
    comboValue = value;
    comboOpen = true; // Open the dropdown directly
    console.log("EditableCategoryCell editing set to true, comboOpen set to true, comboValue:", comboValue);
  }

  function handleSelect(selectedValue, selectedOption) {
    if (selectedValue !== value && onSave) {
      // Non-blocking save - returns immediately
      onSave(selectedValue);
    }
    editing = false;
    comboOpen = false;
  }

  // Get display text for selected category
  function getDisplayText(val) {
    // Use provided displayValue if available (prevents flickering during updates)
    if (displayValue !== null && displayValue !== undefined) {
      return displayValue || placeholder;
    }
    
    if (!val) return placeholder;
    
    const category = categories.find(cat => 
      (typeof cat === 'string' ? cat : cat.value) === val
    );
    
    if (category) {
      return typeof category === 'string' ? category : category.label || category.name || String(category.value);
    }
    
    return String(val);
  }

  // Visual feedback based on save status - no layout-shifting borders
  const cellClass = $derived(cn(
    "relative px-2 py-1 min-h-[32px] flex items-center",
    saveStatus === 'error' ? "bg-red-50 rounded" : "",
    !editing && !disabled ? "cursor-pointer hover:bg-gray-50 rounded" : "",
    disabled ? "cursor-not-allowed opacity-50" : "",
    className
  ));

  // Update combo value when external value changes
  $effect(() => {
    comboValue = value;
  });
</script>

<div class={cellClass} {...restProps}>
  <div class="w-full" bind:this={comboRef}>
    <ComboBox
      bind:value={comboValue}
      bind:open={comboOpen}
      options={categories}
      {placeholder}
      displayValue={getDisplayText(value)}
      searchPlaceholder="Search categories..."
      inline={true}
      onSelect={handleSelect}
      {disabled}
      class="w-full"
    />
  </div>
  
  <!-- Status indicators -->
  {#if saveStatus === 'error'}
    <div 
      class="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full" 
      title={saveError || "Save failed"}
    ></div>
  {/if}
</div>