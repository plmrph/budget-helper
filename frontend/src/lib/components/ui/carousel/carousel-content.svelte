<script>
  import { cn } from "$lib/utils.js";
  
  let className = "";
  export { className as class };
</script>

<div class={cn("flex", className)}>
  <slot />
</div>