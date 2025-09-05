<!--
  @fileoverview Navigation component for the application
  @requires shadcn-svelte components
-->
<script>
  import { Button } from './ui/button/index.js';
  import { location } from 'svelte-spa-router';
  import AuthStatus from './AuthStatus.svelte';

  /**
   * Navigation items configuration
   * @type {Array<{href: string, label: string, icon?: string}>}
   */
  const navItems = [
    { href: '#/transactions', label: 'Transactions' },
    { href: '#/settings', label: 'Settings' }
  ];

  /**
   * Check if current route is active
   * @param {string} href - Route href
   * @returns {boolean} True if route is active
   */
  function isActive(href) {
    const path = href.replace('#', '');
    return $location === path || ($location === '' && path === '/');
  }
</script>

<nav class="bg-card border-b border-border">
  <div class="container mx-auto px-6 py-4">
    <div class="flex items-center justify-between">
      <div class="flex items-center space-x-4">
        <h1 class="text-xl font-bold text-foreground">Budget Helper</h1>
      </div>
      
      <div class="flex items-center space-x-4">
        <!-- Compact authentication status -->
        <AuthStatus compact={true} showRefresh={false} />
        
        <!-- Navigation links -->
        <div class="flex items-center space-x-2">
          {#each navItems as item}
            <Button
              variant={isActive(item.href) ? 'default' : 'ghost'}
              href={item.href}
              class="text-sm"
            >
              {item.label}
            </Button>
          {/each}
        </div>
      </div>
    </div>
  </div>
</nav>