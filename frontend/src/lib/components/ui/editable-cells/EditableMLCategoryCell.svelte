<!--
  @fileoverview Editable ML Category Cell component for transaction table
  Displays predicted categories and allows setting them as the actual category
-->
<script>
  import { Button } from "$lib/components/ui/button/index.js";
  import { ArrowLeft, Brain, Loader2 } from "@lucide/svelte";
  import { mlApi, settingsApi } from "$lib/api/client.js";
  import { bulkMLPredictStore } from "$lib/stores/bulkMLPredict.js";
  import { budgetStore } from "$lib/stores/budget.js";
  import { get } from "svelte/store";
  import { onMount } from "svelte";

  /**
   * @typedef {Object} PredictedCategory
   * @property {string} categoryId - Category ID
   * @property {string} categoryName - Category name
   * @property {number} confidence - Confidence score (0-1)
   */

  // Props
  export let transactionId = "";
  export let onCategorySet = null; // Function to call when setting category from prediction
  export let onApprove = null; // Optional: mark transaction approved after applying ML
  export let saveStatus = null;
  export let saveError = null;

  // State
  let predictions = []; // Array of PredictedCategory
  let isLoading = false;
  let error = null;
  let selectedPrediction = null;
  let selectedCategoryId = ""; // For binding to the select element

  // Get categories from budget store for name lookup
  $: categories = $budgetStore.categories || [];

  // Use $bulkMLPredictStore.hasDefaultModel reactively
  $: hasDefaultModel = $bulkMLPredictStore.hasDefaultModel;

  // In loadPredictions and handleComboboxClick, use hasDefaultModel from store
  async function loadPredictions() {
    if (!transactionId || isLoading) return;

    if (!hasDefaultModel) {
      error = "No default ML model set";
      return;
    }

    try {
      isLoading = true;
      error = null;

      // First check if predictions are already available from bulk store
      const cached = bulkMLPredictStore.getPredictionsForTransaction(transactionId);
      if (cached && Array.isArray(cached)) {
        predictions = cached.slice(0, 3).map((pred) => {
          const categoryInfo = getCategoryInfo(pred.categoryId);
          return {
            categoryId: categoryInfo.id,
            categoryName: categoryInfo.name,
            confidence: pred.confidence || 0,
          };
        });
        if (predictions.length > 0) {
          selectedPrediction = predictions[0];
          selectedCategoryId = selectedPrediction.categoryId;
        }
        return; // use cached predictions
      }

      const response = await mlApi.predict({ transaction_ids: [transactionId] });

      if (!response.success || !response.data) {
        error = response.message || "Failed to get predictions";
        return;
      }

      const predictionResults = response.data.predictions || [];
      if (predictionResults.length === 0) return;

      const result = predictionResults[0];
      if (!result.predictions || !Array.isArray(result.predictions)) return;

      // Convert to our format and take top 3
      predictions = result.predictions
        .slice(0, 3)
        .map(pred => {
          const categoryInfo = getCategoryInfo(pred.categoryId);
          return {
            categoryId: categoryInfo.id,
            categoryName: categoryInfo.name,
            confidence: pred.confidence || 0
          };
        });

      // Set the first prediction as selected by default
      if (predictions.length > 0) {
        selectedPrediction = predictions[0];
        selectedCategoryId = selectedPrediction.categoryId;
      }
    } catch (err) {
      console.error("Error loading predictions:", err);
      error = "Failed to load predictions";
    } finally {
      isLoading = false;
    }
  }

  // Handle setting the predicted category as the actual category
  function handleSetCategory() {
    if (!selectedPrediction || !onCategorySet) return;

    // Non-blocking calls for immediate UI response
    onCategorySet(selectedPrediction.categoryId);
    
    // Also mark as approved if callback provided
    if (onApprove) {
      onApprove(true);
    }
  }

  // Handle combobox click when no predictions are loaded
  async function handleComboboxClick() {
    if (predictions.length === 0 && !isLoading) {
      if (hasDefaultModel) {
        loadPredictions();
      }
    }
  }

  // Handle prediction selection
  function handlePredictionSelect(prediction) {
    selectedPrediction = prediction;
    selectedCategoryId = prediction.categoryId;
  }

  // Initialize component
  onMount(() => {
    // If bulk predictions have already run, hydrate immediately
    const cached = bulkMLPredictStore.getPredictionsForTransaction(transactionId);
    if (cached && Array.isArray(cached) && cached.length > 0) {
      predictions = cached.slice(0, 3).map((pred) => {
        const ci = getCategoryInfo(pred.categoryId);
        return { categoryId: ci.id, categoryName: ci.name, confidence: pred.confidence || 0 };
      });
      selectedPrediction = predictions[0];
      selectedCategoryId = selectedPrediction.categoryId;
    }
  });

  // Reactively update when bulk predictions arrive for this transaction
  $: if ($bulkMLPredictStore && transactionId) {
    const map = $bulkMLPredictStore.predictions || {};
    const incoming = map[transactionId];
    if (incoming && Array.isArray(incoming)) {
      const normalized = incoming.slice(0, 3).map((pred) => {
        const ci = getCategoryInfo(pred.categoryId);
        return { categoryId: ci.id, categoryName: ci.name, confidence: pred.confidence || 0 };
      });
      // Only update if changed to avoid flicker
      const changed = JSON.stringify(normalized) !== JSON.stringify(predictions);
      if (changed) {
        predictions = normalized;
        if (predictions.length > 0) {
          selectedPrediction = predictions[0];
          selectedCategoryId = selectedPrediction.categoryId;
        }
      }
    }
  }

  // Helper to get category info from ID or name
  function getCategoryInfo(categoryIdOrName) {
    // First try to find by ID
    let category = categories.find(c => c.id === categoryIdOrName);
    if (category) {
      return { id: category.id, name: category.name };
    }
    // If not found by ID, try to find by name (for ML predictions)
    category = categories.find(c => c.name === categoryIdOrName);
    if (category) {
      return { id: category.id, name: category.name };
    }
    // If still not found, return the original value as both ID and name
    return { id: categoryIdOrName, name: categoryIdOrName };
  }

  // Helper to format confidence as percentage
  function formatConfidence(confidence) {
    return `${Math.round(confidence * 100)}%`;
  }
</script>

<div class="flex items-center gap-1 min-w-0">
  <!-- Left Arrow Button -->
  <Button
    variant="ghost"
    size="sm"
    class="h-6 w-6 p-0 flex-shrink-0"
    disabled={!selectedPrediction || isLoading || saveStatus === 'saving' || hasDefaultModel === false}
    onclick={handleSetCategory}
    title={hasDefaultModel === false ? "No default ML model set" : "Set predicted category as actual category"}
  >
    {#if saveStatus === 'saving'}
      <Loader2 class="h-3 w-3 animate-spin" />
    {:else}
      <ArrowLeft class="h-3 w-3" />
    {/if}
  </Button>

  <!-- Predicted Category Combobox -->
  <div class="flex-1 min-w-0">
    {#if isLoading}
      <div class="flex items-center gap-1 text-xs text-muted-foreground px-2 py-1">
        <Loader2 class="h-3 w-3 animate-spin" />
        <span>Loading...</span>
      </div>
    {:else if error}
      <div class="text-xs text-muted-foreground px-2 py-1 truncate" title={error}>
        {#if error === "No default ML model set"}
          <span class="text-muted-foreground">No default model</span>
        {:else}
          <span class="text-destructive">Error</span>
        {/if}
      </div>
    {:else if predictions.length === 0}
      <Button
        variant="ghost"
        size="sm"
        class="h-6 w-full justify-start text-xs text-muted-foreground px-2"
        disabled={hasDefaultModel === false}
        onclick={handleComboboxClick}
        title={hasDefaultModel === false ? "No default ML model set" : "Click to predict category"}
      >
        <Brain class="h-3 w-3 mr-1" />
        <span class="truncate">
          {#if hasDefaultModel === false}
            No default model
          {:else}
            Predict
          {/if}
        </span>
      </Button>
    {:else}
      <select
        class="h-6 text-xs border-none shadow-none px-2 bg-transparent hover:bg-accent rounded w-full"
        bind:value={selectedCategoryId}
        onchange={(e) => {
          const value = e.target.value;
          if (value) {
            const prediction = predictions.find(p => p.categoryId === value);
            if (prediction) {
              handlePredictionSelect(prediction);
            }
          }
        }}
      >
        <option value="" disabled>Select prediction</option>
        {#each predictions as prediction (prediction.categoryId)}
          <option value={prediction.categoryId}>
            {prediction.categoryName} ({formatConfidence(prediction.confidence)})
          </option>
        {/each}
      </select>
    {/if}
  </div>
</div>

{#if saveError}
  <div class="text-xs text-destructive mt-1" title={saveError}>
    Save failed
  </div>
{/if}