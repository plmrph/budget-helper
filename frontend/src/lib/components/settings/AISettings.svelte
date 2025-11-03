<!--
  @fileoverview AI/ML Settings component for machine learning configuration
-->
<script>
  import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
  } from "../ui/card/index.js";
  import { Label } from "../ui/label/index.js";
  import { Input } from "../ui/input/index.js";
  import DropdownSelect from "../ui/dropdown-select/DropdownSelect.svelte";

  export let settings = {};
  export let onchange;

  // AI setting keys from backend ConfigKeys
  const AI_KEYS = {
    TRAINING_DATA_MONTHS: "ai.training_data_months",
    TRAINING_TIME_MINUTES: "ai.training_time_minutes",
  };

  // Get setting value helper
  function getSettingValue(key, defaultValue = "") {
    const setting = settings[key];
    if (!setting || !setting.value) return defaultValue;

    // Extract value from ConfigValue union
    const value = setting.value;
    return (
      value.stringValue ??
      value.intValue ??
      value.doubleValue ??
      value.boolValue ??
      defaultValue
    );
  }

  // Update setting helper
  function updateSetting(key, value, description = "") {
    const currentValue = getSettingValue(key);

    // Only update if the value has actually changed
    if (currentValue !== value) {
      onchange?.({ detail: { key, value, type: "AI", description } });
    }
  }
</script>

<div
  class="ai-settings grid grid-cols-1 lg:grid-cols-2 gap-6"
  data-testid="ai-settings"
>
  <!-- Model Training Configuration -->
  <Card>
    <CardHeader>
      <CardTitle>Model Training Configuration</CardTitle>
      <CardDescription
        >Configure how machine learning models are trained for transaction
        categorization</CardDescription
      >
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="space-y-2">
          <Label for="training-data-months">Training Data Period (Months)</Label
          >
          <Input
            id="training-data-months"
            type="number"
            min="1"
            max="24"
            value={getSettingValue(AI_KEYS.TRAINING_DATA_MONTHS, 6)}
            onblur={(e) =>
              updateSetting(
                AI_KEYS.TRAINING_DATA_MONTHS,
                parseInt(e.target.value),
                "Number of months of data to use for training",
              )}
            onkeydown={(e) => {
              if (e.key === "Enter") {
                updateSetting(
                  AI_KEYS.TRAINING_DATA_MONTHS,
                  parseInt(e.target.value),
                  "Number of months of data to use for training",
                );
              }
            }}
            placeholder="Months of historical data"
            data-testid="training-data-months-input"
          />
          <p class="text-sm text-muted-foreground">
            Number of months of historical transaction data to use for model
            training
          </p>
        </div>

        <div class="space-y-2">
          <Label for="training-time-minutes">Maximum Training Time (Minutes)</Label>
          <Input
            id="training-time-minutes"
            type="number"
            min="1"
            max="120"
            step="1"
            value={getSettingValue(AI_KEYS.TRAINING_TIME_MINUTES, 15)}
            onblur={(e) => {
              const minutes = parseInt(e.target.value);
              updateSetting(
                AI_KEYS.TRAINING_TIME_MINUTES,
                isNaN(minutes) ? 15 : Math.max(1, Math.min(120, minutes)),
                "Maximum training time in minutes",
              );
            }}
            onkeydown={(e) => {
              if (e.key === "Enter") {
                const minutes = parseInt(e.target.value);
                updateSetting(
                  AI_KEYS.TRAINING_TIME_MINUTES,
                  isNaN(minutes) ? 15 : Math.max(1, Math.min(120, minutes)),
                  "Maximum training time in minutes",
                );
              }
            }}
            placeholder="e.g., 15"
            data-testid="training-time-minutes-input"
          />
          <p class="text-sm text-muted-foreground">
            Typical training completes in 2–15 minutes. Increase only if you have large datasets or want extra accuracy.
          </p>
        </div>
      </div>
    </CardContent>
  </Card>

  <!-- Training Recommendations -->
  <Card>
    <CardHeader>
      <CardTitle>Training Recommendations</CardTitle>
      <CardDescription>Practical guidelines for better models</CardDescription>
    </CardHeader>
    <CardContent>
      <ul class="list-disc pl-5 space-y-2 text-sm">
        <li>Aim for at least <strong>1,000+ training samples</strong> for stable results.</li>
        <li>Have <strong>several samples per category</strong>; 1–2 per category is usually insufficient.</li>
        <li><strong>Training time</strong> typically completes within <strong>2–15 minutes</strong>.</li>
      </ul>
    </CardContent>
  </Card>

  <!-- Model Performance Metrics -->
  <Card>
    <CardHeader>
      <CardTitle>Performance Monitoring</CardTitle>
      <CardDescription
        >Understanding your model's performance metrics</CardDescription
      >
    </CardHeader>
    <CardContent>
      <div class="space-y-3 text-sm">
        <div>
          <strong>Training Metrics:</strong> How well the model learned from your data during training.
          These show if the model is picking up patterns correctly.
        </div>
        <div>
          <strong>Validation Macro F1 Score (Most Important):</strong> This is the key metric to watch.
          It measures how well the model performs on categories with fewer examples.
          A score of 0.70+ is good, 0.80+ is excellent.
        </div>
        <div>
          <strong>Abstain Rate:</strong> Percentage of predictions where the model wasn't confident enough
          to make a guess. Higher abstain rates mean fewer automatic categorizations but higher accuracy.
        </div>
        <div>
          <strong>Accuracy vs F1 Score:</strong> Accuracy can be misleading with uneven categories.
          Focus on the F1 score instead - it better represents performance across all your categories.
        </div>
      </div>
    </CardContent>
  </Card>
</div>
