<script>
  import { tick } from "svelte";
  import { cn } from "$lib/utils.js";

  let {
    value = $bindable(""),
    onSave,
    saveStatus = null,
    saveError = null,
    placeholder = "Click to edit",
    disabled = false,
    class: className = "",
    ...restProps
  } = $props();

  let editing = $state(false);
  let inputValue = $state(value);
  let inputRef = $state(null);

  function startEdit(event) {
    if (event) {
      event.stopPropagation();
      event.preventDefault();
    }
    if (disabled) return;
    editing = true;
    inputValue = value;
    tick().then(() => {
      if (inputRef) {
        inputRef.focus();
        inputRef.select();
      }
    });
  }

  function saveEdit() {
    if (inputValue !== value && onSave) {
      // Non-blocking save - returns immediately
      onSave(inputValue);
    }
    editing = false;
  }

  function cancelEdit() {
    inputValue = value;
    editing = false;
  }

  function handleKeydown(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      saveEdit();
    } else if (event.key === "Escape") {
      event.preventDefault();
      cancelEdit();
    }
  }

  // Visual feedback based on save status - no layout-shifting borders
  const cellClass = $derived(
    cn(
      "relative px-2 py-1 min-h-[32px] flex items-center",
      saveStatus === "error" ? "bg-red-50 rounded" : "",
      editing ? "border border-blue-500 bg-blue-50 rounded" : "",
      !editing && !disabled ? "cursor-pointer hover:bg-gray-50 rounded" : "",
      disabled ? "cursor-not-allowed opacity-50" : "",
      className,
    ),
  );

  // Update input value when external value changes
  $effect(() => {
    if (!editing) {
      inputValue = value;
    }
  });
</script>

<div class={cellClass} {...restProps}>
  {#if editing}
    <input
      bind:this={inputRef}
      bind:value={inputValue}
      class="w-full bg-transparent border-none outline-none text-sm placeholder:text-muted-foreground/60"
      {placeholder}
      onblur={saveEdit}
      onkeydown={handleKeydown}
    />
  {:else}
    <div
      class="w-full text-sm truncate"
      onclick={startEdit}
      role="button"
      tabindex={disabled ? -1 : 0}
      onkeydown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          e.stopPropagation();
          startEdit(e);
        }
      }}
      aria-label={`Edit ${placeholder.toLowerCase()}: ${value || "empty"}`}
    >
      {#if value}
        {value}
      {:else}
        <span class="text-muted-foreground/60">{placeholder}</span>
      {/if}
    </div>
  {/if}

  <!-- Status indicators -->
  {#if saveStatus === "error"}
    <div
      class="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full"
      title={saveError || "Save failed"}
    ></div>
  {/if}
</div>
