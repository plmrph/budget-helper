<script lang="ts">
  import type { HTMLAttributes } from "svelte/elements";

  let {
    value,
    children,
    ...restProps
  }: HTMLAttributes<HTMLOptionElement> & {
    value: string;
    children?: any;
  } = $props();
</script>

<option {value} {...restProps}>
  {#if children}
    {@render children()}
  {/if}
</option>