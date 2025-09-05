<!--
  @fileoverview Error display component for showing API and application errors
  @requires shadcn-svelte components
-->
<script>
  import { Button } from '$lib/components/ui/button';

  /**
   * @type {string|null} - Error message to display
   */
  export let error = null;

  /**
   * @type {boolean} - Whether to show retry button
   */
  export let showRetry = true;

  /**
   * @type {Function} - Retry function to call
   */
  export let onRetry = null;

  /**
   * @type {'error'|'warning'|'info'} - Error type for styling
   */
  export let type = 'error';

  /**
   * Clear the error
   */
  function clearError() {
    error = null;
  }

  /**
   * Handle retry action
   */
  function handleRetry() {
    if (onRetry && typeof onRetry === 'function') {
      onRetry();
    }
    clearError();
  }

  /**
   * Get CSS classes based on error type
   * @param {string} type - Error type
   * @returns {string} CSS classes
   */
  function getErrorClasses(type) {
    switch (type) {
      case 'warning':
        return 'bg-yellow-100 border-yellow-400 text-yellow-700';
      case 'info':
        return 'bg-blue-100 border-blue-400 text-blue-700';
      default:
        return 'bg-red-100 border-red-400 text-red-700';
    }
  }
</script>

{#if error}
  <div class="mb-4 p-4 border rounded {getErrorClasses(type)}" role="alert">
    <div class="flex justify-between items-start">
      <div class="flex-1">
        <div class="font-medium mb-1">
          {#if type === 'warning'}
            Warning
          {:else if type === 'info'}
            Information
          {:else}
            Error
          {/if}
        </div>
        <div class="text-sm">
          {error}
        </div>
      </div>
      
      <div class="flex gap-2 ml-4">
        {#if showRetry && onRetry}
          <Button 
            variant="outline" 
            size="sm" 
            onclick={handleRetry}
            class="text-xs"
          >
            Retry
          </Button>
        {/if}
        
        <button 
          type="button"
          class="text-lg leading-none hover:opacity-70"
          onclick={clearError}
          aria-label="Close error"
        >
          ×
        </button>
      </div>
    </div>
  </div>
{/if}