<script lang="ts">
  import { Slider } from "$lib/components/ui/slider/index.js";

  let {
    column,
    data,
    field,
    formatValue = (value: number) => value.toString(),
    step = 1
  }: {
    column: any;
    data: any[];
    field: string;
    formatValue?: (value: number) => string;
    step?: number;
  } = $props();

  // Calculate min and max values from the data
  const values = data.map(row => row[field]).filter(Boolean);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  
  // Round to nearest step for cleaner ranges
  const roundedMin = Math.floor(minValue / step) * step;
  const roundedMax = Math.ceil(maxValue / step) * step;

  let range = $state([roundedMin, roundedMax]);

  // Handle range changes without reactive loops
  function handleRangeChange(newRange: number[]) {
    range = newRange;
    updateFilter();
  }

  function updateFilter() {
    if (range[0] === roundedMin && range[1] === roundedMax) {
      // If range is at full extent, clear the filter
      column.setFilterValue(undefined);
    } else {
      // Set the custom filter
      column.setFilterValue(range);
    }
  }

  function resetRange() {
    range = [roundedMin, roundedMax];
    updateFilter();
  }
</script>

<div class="space-y-3 p-2">
  <Slider
    bind:value={range}
    onValueChange={handleRangeChange}
    min={roundedMin}
    max={roundedMax}
    {step}
    class="w-full"
  />
  
  <!-- Show current slider values -->
  <div class="flex justify-between text-xs text-muted-foreground">
    <span>{formatValue(range[0])}</span>
    <span>{formatValue(range[1])}</span>
  </div>
  
  {#if range[0] !== roundedMin || range[1] !== roundedMax}
    <button
      onclick={resetRange}
      class="text-xs text-blue-600 hover:text-blue-800 underline"
    >
      Reset range
    </button>
  {/if}
</div>