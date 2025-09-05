<script lang="ts">
  let {
    value = $bindable(),
    onValueChange,
    allowDeselect = true,
    type = "single",
    children
  }: {
    value?: string;
    onValueChange?: (value: string | undefined) => void;
    allowDeselect?: boolean;
    type?: "single" | "multiple";
    children?: any;
  } = $props();

  function handleValueChange(newValue: string | undefined) {
    value = newValue;
    if (onValueChange) {
      onValueChange(newValue);
    }
  }
</script>

<div class="relative">
  {@render children?.({ value, onValueChange: handleValueChange })}
</div>