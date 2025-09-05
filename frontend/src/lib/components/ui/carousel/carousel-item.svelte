<script>
  import { cn } from "$lib/utils.js";
  
  let className = "";
  export { className as class };
</script>

<div class={cn("min-w-0 shrink-0 grow-0 basis-full", className)}>
  <slot />
</div>