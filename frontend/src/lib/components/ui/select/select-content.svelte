<script lang="ts">
  let { children }: { children?: any } = $props();
</script>

<!-- Content is handled by the native select, this is just for API compatibility -->
{#if children}
  {@render children()}
{/if}