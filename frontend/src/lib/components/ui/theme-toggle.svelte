<!--
  @fileoverview Theme toggle component with light/dark/system mode options
-->
<script>
  import SunIcon from "@lucide/svelte/icons/sun";
  import MoonIcon from "@lucide/svelte/icons/moon";
  import MonitorIcon from "@lucide/svelte/icons/monitor";
  
  import { resetMode, setMode } from "mode-watcher";
  import * as DropdownMenu from "$lib/components/ui/dropdown-menu/index.js";
  import { buttonVariants } from "$lib/components/ui/button/index.js";
</script>

<DropdownMenu.Root>
  <DropdownMenu.Trigger
    class={buttonVariants({ variant: "outline", size: "icon" })}
    data-testid="theme-toggle"
  >
    <SunIcon
      class="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 !transition-all dark:-rotate-90 dark:scale-0"
    />
    <MoonIcon
      class="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 !transition-all dark:rotate-0 dark:scale-100"
    />
    <span class="sr-only">Toggle theme</span>
  </DropdownMenu.Trigger>
  <DropdownMenu.Content align="end">
    <DropdownMenu.Item onclick={() => setMode("light")} data-testid="theme-light">
      <SunIcon class="mr-2 h-4 w-4" />
      Light
    </DropdownMenu.Item>
    <DropdownMenu.Item onclick={() => setMode("dark")} data-testid="theme-dark">
      <MoonIcon class="mr-2 h-4 w-4" />
      Dark
    </DropdownMenu.Item>
    <DropdownMenu.Item onclick={() => resetMode()} data-testid="theme-system">
      <MonitorIcon class="mr-2 h-4 w-4" />
      System
    </DropdownMenu.Item>
  </DropdownMenu.Content>
</DropdownMenu.Root>