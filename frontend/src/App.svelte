<!--
  @fileoverview Main application component with routing and navigation
-->
<script>
  import { onMount } from "svelte";
  import Router from "svelte-spa-router";
  import { location } from "svelte-spa-router";
  import { ModeWatcher } from "mode-watcher";
  import Transactions from "./lib/pages/Transactions.svelte";
  import Settings from "./lib/pages/Settings.svelte";
  import MLTraining from "./lib/pages/MLTraining.svelte";
  import ThemeToggle from "./lib/components/ui/theme-toggle.svelte";
  import { authActions } from "./lib/stores/auth.js";

  /**
   * Application routes configuration
   */
  const routes = {
    "/": Transactions, // Redirect home to transactions
    "/transactions": Transactions,
    "/settings": Settings,
    "/ml-training": MLTraining,
  };

  /**
   * Navigation items
   */
  const navItems = [
    {
      href: "#/transactions",
      label: "Transactions",
      testId: "nav-transactions",
    },
    { href: "#/ml-training", label: "ML Training", testId: "nav-ml-training" },
    { href: "#/settings", label: "Settings", testId: "nav-settings" },
  ];

  /**
   * Check if current route is active
   */
  function isActive(href) {
    const path = href.replace("#", "");
    return $location === path || ($location === "" && path === "/");
  }
</script>

<ModeWatcher />

<div class="min-h-screen bg-background" data-testid="app-container">
  <!-- Navigation -->
  <nav
    class="bg-card border-b border-border shadow-sm"
    data-testid="navigation"
  >
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16">
        <div class="flex items-center">
          <img src="/favicon.svg" alt="App Icon" class="h-6 w-6 mr-2" />
          <h1 class="text-xl font-bold text-foreground" data-testid="app-title">
            Budget Helper
          </h1>
        </div>

        <div class="flex items-center space-x-4">
          {#each navItems as item}
            <a
              href={item.href}
              data-testid={item.testId}
              class="px-3 py-2 rounded-md text-sm font-medium transition-colors
                     {isActive(item.href)
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:text-foreground hover:bg-accent'}"
            >
              {item.label}
            </a>
          {/each}

          <!-- Theme Toggle -->
          <ThemeToggle />
        </div>
      </div>
    </div>
  </nav>

  <!-- Main Content -->
  <main
    class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8 text-foreground"
    data-testid="main-content"
  >
    <Router {routes} />
  </main>
</div>
