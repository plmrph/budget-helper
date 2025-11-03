<script>
  import { onMount } from "svelte";
  import { writable } from "svelte/store";
  import * as Card from "$lib/components/ui/card";
  import * as Table from "$lib/components/ui/table";

  import Button from "$lib/components/ui/button/button.svelte";
  import DropdownSelect from "$lib/components/ui/dropdown-select/DropdownSelect.svelte";
  import { mlApi, budgetsApi } from "$lib/api/client.js";
  import { settingsStore } from "$lib/stores/settings.js";
  import { bulkMLPredictStore } from '../stores/bulkMLPredict.js';

  // Carousel state
  let currentStep = 0;
  const totalSteps = 4;

  // Data stores
  const models = writable([]);
  const trainingStats = writable(null);
  const trainingData = writable(null);
  const validationResults = writable(null);
  const datasets = writable([]);
  const isLoading = writable(false);
  const error = writable(null);

  // Separate loading state for data preparation
  let isPreparingData = false;
  let lastPreparationTime = 0;

  // Form data
  let selectedBudgetId = null;
  let monthsBack = 6; // Will be overridden by AI setting if available
  
  // UI state for metrics display
  let activeMetricTab = 'validation'; // Default to validation metrics
  let modelName = "";
  let selectedStrategy = "pxblendsc";
  let maxTrainingTime = 15; // minutes, will be overridden by AI setting if available
  let trainingInProgress = false;

  // Available model strategies
  const modelStrategies = [
    {
      value: "pxblendsc",
      label: "PXBlendSC",
      description:
        "Advanced ML pipeline with LightGBM+SVM blending and smart categorization",
    },
  ];

  // Available budgets
  let budgets = [];

  // Default model tracking
  let defaultModelName = null;

  // Category mapping for ID to name conversion
  const categoryMap = writable(new Map()); // Map<categoryId, categoryName>

  // Reactive statement to reload datasets when budget changes
  $: if (selectedBudgetId) {
    loadDatasets(selectedBudgetId);
  }

  // Reactive statement to ensure category mapping is loaded when we have training data
  $: if (
    selectedBudgetId &&
    ($trainingData || $trainingStats) &&
    $categoryMap.size === 0
  ) {
    loadCategoryMapping(selectedBudgetId);
  }

  onMount(async () => {
    await loadAISettings();
    await loadInitialData();
    await loadDefaultModelName();
  });

  async function loadAISettings() {
  const monthsSetting = await settingsStore.getSettingValue("ai.training_data_months");
  const monthsVal = monthsSetting ?? 6;
  const months = parseInt(monthsVal, 10);
  if (!isNaN(months) && months > 0) monthsBack = months;

  const minsSetting = await settingsStore.getSettingValue("ai.training_time_minutes");
  maxTrainingTime = Math.min(120, minsSetting ?? 15);
  }

  async function loadInitialData() {
    try {
      $isLoading = true;
      $error = null;

      // Load budgets, selected budget, models, and datasets
      const [budgetsResponse, selectedBudgetResponse, modelsResponse] =
        await Promise.all([
          budgetsApi.getAll(),
          budgetsApi.getSelected(),
          mlApi.getModels(),
        ]);

      if (budgetsResponse.success) {
        // The API returns {success: true, message: "...", data: [...]}
        budgets = budgetsResponse.data || [];
        console.debug("Loaded budgets:", budgets);

        if (budgets.length > 0) {
          // Use the selected budget from config, or fall back to first budget
          let targetBudgetId = null;

          if (
            selectedBudgetResponse.success &&
            selectedBudgetResponse.data?.selected_budget_id
          ) {
            targetBudgetId = selectedBudgetResponse.data.selected_budget_id;
            console.debug("Using selected budget from config:", targetBudgetId);
          } else {
            // Fall back to first budget if no selected budget in config
            targetBudgetId = budgets[0].id;
            console.debug(
              "No selected budget in config, using first budget:",
              targetBudgetId,
            );
          }

          // Verify the target budget exists in the available budgets
          const targetBudget = budgets.find((b) => b.id === targetBudgetId);
          if (targetBudget) {
            selectedBudgetId = targetBudgetId;
            console.debug("Selected budget:", targetBudget);
          } else {
            console.warn(
              "Selected budget not found in available budgets, using first budget",
            );
            selectedBudgetId = budgets[0].id;
          }

          // Load category information for the selected budget
          await loadCategoryMapping(selectedBudgetId);

          // Load datasets for the selected budget
          await loadDatasets(selectedBudgetId);
        } else {
          console.warn(
            "No budgets available. Make sure YNAB is connected and budgets are synced.",
          );
          $error =
            "No budgets available. Please connect to YNAB and sync your budgets first.";
        }
      } else {
        console.error("Failed to load budgets:", budgetsResponse.error);
      }

      if (modelsResponse.success) {
        console.debug("Models response:", modelsResponse);
        // Sort models by trainedDate, latest first
        const sortedModels = (modelsResponse.data || []).sort((a, b) => {
          const dateA = a.trainedDate ? new Date(a.trainedDate) : new Date(0);
          const dateB = b.trainedDate ? new Date(b.trainedDate) : new Date(0);
          return dateB - dateA; // Latest first
        });
        models.set(sortedModels);

        // Check for any models that might be training and resume polling
        await checkAndResumeTraining(modelsResponse.data || []);
      } else {
        console.error("Failed to load models:", modelsResponse.error);
      }

      // Training stats will be loaded after budget selection
    } catch (err) {
      $error = `Failed to load initial data: ${err.message}`;
    } finally {
      $isLoading = false;
    }
  }

  async function prepareTrainingData() {
    // Prevent duplicate calls with debouncing
    const now = Date.now();
    if (isPreparingData) {
      console.debug(
        "Data preparation already in progress, ignoring duplicate call",
      );
      return;
    }
    if (now - lastPreparationTime < 2000) {
      console.debug(
        "Data preparation called too recently, ignoring duplicate call",
      );
      return;
    }

    try {
      isPreparingData = true;
      lastPreparationTime = now;
      $error = null;

      // Store the current budget selection to preserve it
      const currentBudgetId = selectedBudgetId;

      console.debug(
        "Starting data preparation for budget:",
        currentBudgetId,
        "months:",
        monthsBack,
      );

      // Load category mapping for the current budget if it's different
      if (currentBudgetId && $categoryMap.size === 0) {
        await loadCategoryMapping(currentBudgetId);
      }

      // First get training data stats for analysis (with timeout)
      try {
        const statsResponse = await Promise.race([
          mlApi.getTrainingDataStats(currentBudgetId, monthsBack),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error("Stats request timeout")), 30000),
          ),
        ]);

        if (statsResponse.success) {
          trainingStats.set(statsResponse.data);
          console.debug("Stats loaded successfully");
        } else {
          console.warn("Stats request failed:", statsResponse.message);
          // Continue anyway, stats are not critical
        }
      } catch (statsError) {
        console.warn("Stats request error:", statsError.message);
        // Continue anyway, stats are not critical
      }

      // Then prepare the actual training data (with timeout)
      console.log("Preparing training data...");
      const response = await Promise.race([
        mlApi.prepareTrainingData({
          budget_id: currentBudgetId,
          months_back: monthsBack,
          test_split_ratio: 0.2,
          min_samples_per_category: 1,
        }),
        new Promise((_, reject) =>
          setTimeout(
            () =>
              reject(new Error("Data preparation timeout after 60 seconds")),
            60000,
          ),
        ),
      ]);

      if (response.success) {
        trainingData.set(response.data);
        // Ensure budget selection is preserved
        selectedBudgetId = currentBudgetId;
        console.log("Training data prepared successfully:", response.data);

        // Ensure category mapping is loaded for displaying category names
        if ($categoryMap.size === 0) {
          await loadCategoryMapping(currentBudgetId);
        }

        // Refresh the datasets list to show the newly created dataset
        // Add a small delay to ensure backend has fully written the files
        setTimeout(async () => {
          await loadDatasets(currentBudgetId);
        }, 500);

      } else {
        $error = response.message || "Failed to prepare training data";
        console.error("Data preparation failed:", response);
      }
    } catch (err) {
      console.error("Data preparation error:", err);
      $error = `Failed to prepare training data: ${err.message}`;
    } finally {
      console.debug("Data preparation completed, clearing loading state");
      isPreparingData = false;
    }
  }

  async function startTraining() {
    if (!modelName.trim() || !$trainingData?.training_file) {
      $error = "Model name and training data are required";
      return;
    }

    try {
      $isLoading = true;
      trainingInProgress = true;
      $error = null;

      // Start async training
      const response = await mlApi.train({
        ml_model_name: modelName.trim(),
        ml_model_type: "CATEGORICAL",
        training_data_location: $trainingData.training_file,
        training_params: {
          test_data_location: $trainingData.test_file,
          strategy: selectedStrategy,
          time_limit: maxTrainingTime * 60, // Convert minutes to seconds
        },
      });

      if (response.success) {
        // Add the model immediately to frontend state for instant feedback
        const newModel = {
          name: modelName.trim(),
          version: "1.0",
          trainedDate: new Date().toISOString(),
          performanceMetrics: null,
          status: 2, // TrainingStatus.Pending
          trainingStatus: "training",
          trainingMessage: "Training starting..."
        };

        models.update((list) => [newModel, ...list]);

        // Navigate back to Manage Models page
        currentStep = 0;

        // Start polling for training status
        await pollTrainingStatus(modelName.trim());
      } else {
        // Check if it's a 409 conflict (training already in progress)
        if (response.status === 409) {
          $error =
            "Another model is currently being trained. Please wait for it to complete before starting a new training.";
        } else {
          $error = response.message || "Training failed";
        }
        trainingInProgress = false;
      }
    } catch (err) {
      // Check if it's a 409 conflict error
      if (err.message && err.message.includes("409")) {
        $error =
          "Another model is currently being trained. Please wait for it to complete before starting a new training.";
      } else {
        $error = `Training failed: ${err.message}`;
      }
      trainingInProgress = false;
    } finally {
      $isLoading = false;
    }
  }

async function pollTrainingStatus(modelNameToPoll) {
  const maxPollTime = 30 * 60 * 1000; // 30 minutes max
  const pollInterval = 5000; // 5 seconds
  const startTime = Date.now();
  const maxConsecutiveErrors = 3; // Allow 3 consecutive errors before giving up
  let consecutiveErrorCount = 0;

  const poll = async () => {
    try {
      const statusResponse = await mlApi.getTrainingStatus(modelNameToPoll);
      
      // Reset error counter on successful response
      consecutiveErrorCount = 0;

      if (statusResponse.success) {
        const status = statusResponse.data;
        console.debug(`Training status for ${modelNameToPoll}:`, status);

        // Update the model in the list with current status
        models.update((list) =>
          list.map((model) =>
            model.name === modelNameToPoll
              ? {
                  ...model,
                  trainingStatus: status.status,
                  trainingMessage: status.message,
                }
              : model,
          ),
        );

        if (status.status === "completed") {
          // Training completed successfully
          trainingInProgress = false;

          // Reload models to get the complete model data
          const modelsResponse = await mlApi.getModels();
          if (modelsResponse.success) {
            // Sort models by trainedDate, latest first
            const sortedModels = (modelsResponse.data || []).sort((a, b) => {
              const dateA = a.trainedDate ? new Date(a.trainedDate) : new Date(0);
              const dateB = b.trainedDate ? new Date(b.trainedDate) : new Date(0);
              return dateB - dateA; // Latest first
            });
            models.set(sortedModels);
          }
          return;
        } else if (status.status === "failed") {
          // Training failed
          trainingInProgress = false;
          $error = status.message || status.error || "Training failed";

          // Update model status to failed
          models.update((list) =>
            list.map((model) =>
              model.name === modelNameToPoll
                ? {
                    ...model,
                    trainingStatus: "failed",
                    trainingMessage: status.message,
                  }
                : model,
            ),
          );
          return;
        } else if (
          status.status === "training" ||
          status.status === "starting"
        ) {
          // Training still in progress, continue polling
          if (Date.now() - startTime < maxPollTime) {
            setTimeout(poll, pollInterval);
          } else {
            // Timeout
            trainingInProgress = false;
            $error = "Training timeout - please check the status manually";
          }
        }
      } else {
        // Check if it's a 404 error (training session not found)
        const is404Error = statusResponse.status === 404 || 
                          (statusResponse.error && statusResponse.error.includes('404')) ||
                          (statusResponse.error && statusResponse.error.includes('Not Found'));
        
        if (is404Error) {
          consecutiveErrorCount++;
          console.warn(`Training status not found for ${modelNameToPoll} (${consecutiveErrorCount}/${maxConsecutiveErrors})`);
          
          // If we get too many consecutive 404s, assume the training session is stuck
          if (consecutiveErrorCount >= maxConsecutiveErrors) {
            console.warn(`Too many 404s for ${modelNameToPoll}, cleaning up stuck model`);
            
            // Stop training progress
            trainingInProgress = false;
            
            // Try to delete the stuck model from backend first
            try {
              const deleteResponse = await mlApi.deleteModel(modelNameToPoll);
              if (deleteResponse.success) {
                console.debug(`Successfully deleted stuck model: ${modelNameToPoll}`);
              } else {
                console.warn(`Could not delete stuck model ${modelNameToPoll}:`, deleteResponse.message);
              }
            } catch (deleteError) {
              console.warn(`Could not delete stuck model ${modelNameToPoll}:`, deleteError);
            }
            
            // Remove from frontend models list
            models.update((list) => list.filter((m) => m.name !== modelNameToPoll));
            
            // Set a user-friendly message
            $error = `Training session for "${modelNameToPoll}" was lost due to container restart. Model has been cleaned up.`;
            
            return; // Stop polling
          }
        } else {
          // Other errors, increment counter but continue
          consecutiveErrorCount++;
        }
        
        // Continue polling if under error limit and time limit
        if (consecutiveErrorCount < maxConsecutiveErrors && Date.now() - startTime < maxPollTime) {
          setTimeout(poll, pollInterval * 2); // Longer delay for error cases
        } else {
          trainingInProgress = false;
          if (consecutiveErrorCount >= maxConsecutiveErrors) {
            $error = `Lost connection to training session for "${modelNameToPoll}". Please check manually.`;
          } else {
            $error = "Training monitoring timeout - please check the status manually";
          }
        }
      }
    } catch (err) {
      console.error("Error polling training status:", err);
      consecutiveErrorCount++;
      
      // Check if it's a 404-like error in the catch block
      const is404Error = (err.status === 404) || 
                        (err.message && err.message.includes('404')) ||
                        (err.message && err.message.includes('Not Found'));
      
      if (is404Error) {
        console.warn(`Training status not found for ${modelNameToPoll} (${consecutiveErrorCount}/${maxConsecutiveErrors})`);
        
        // If we get too many consecutive 404s, cleaning up
        if (consecutiveErrorCount >= maxConsecutiveErrors) {
          console.warn(`Too many 404s for ${modelNameToPoll}, cleaning up stuck model`);
          
          // Stop training progress
          trainingInProgress = false;
          
          // Try to delete the stuck model from backend first
          try {
            const deleteResponse = await mlApi.deleteModel(modelNameToPoll);
            if (deleteResponse.success) {
              console.debug(`Successfully deleted stuck model: ${modelNameToPoll}`);
            } else {
              console.warn(`Could not delete stuck model ${modelNameToPoll}:`, deleteResponse.message);
            }
          } catch (deleteError) {
            console.warn(`Could not delete stuck model ${modelNameToPoll}:`, deleteError);
          }
          
          // Remove from frontend models list
          models.update((list) => list.filter((m) => m.name !== modelNameToPoll));
          
          // Set a user-friendly message
          $error = `Training session for "${modelNameToPoll}" was lost due to container restart. Model has been cleaned up.`;
          
          return; // Stop polling
        }
      }
      
      // Continue polling if under error limit and time limit
      if (consecutiveErrorCount < maxConsecutiveErrors && Date.now() - startTime < maxPollTime) {
        setTimeout(poll, pollInterval * 2); // Longer delay for error cases
      } else {
        trainingInProgress = false;
        if (consecutiveErrorCount >= maxConsecutiveErrors) {
          $error = `Lost connection to training session for "${modelNameToPoll}". Please check manually.`;
        } else {
          $error = "Error monitoring training progress";
        }
      }
    }
  };

  // Start polling
  setTimeout(poll, pollInterval);
}

  async function loadValidationResults() {
    if (!$trainingData?.test_file) return;

    try {
      // Create placeholder validation results for training workflow
      // These will be replaced with actual metrics once a model is trained
      const placeholderResults = {
        test_samples: $trainingData?.test_samples || 0,
        predictions: [
          {
            transaction_id: "1",
            actual: "Groceries",
            predicted: "Groceries",
            confidence: 0.95,
          },
          {
            transaction_id: "2",
            actual: "Gas",
            predicted: "Transportation",
            confidence: 0.88,
          },
          {
            transaction_id: "3",
            actual: "Restaurant",
            predicted: "Dining",
            confidence: 0.92,
          },
        ],
        accuracy: 0.0, // Will be updated after training
        precision: 0.0, // Will be updated after training
        recall: 0.0, // Will be updated after training
        abstain_rate: 0.0,
        training_samples: $trainingData?.training_samples || 0,
        n_classes: $trainingData?.categories || 0,
        has_lgbm: false,
        has_svm: false,
        metrics_source: "placeholder",
      };

      validationResults.set(placeholderResults);
    } catch (err) {
      console.error("Failed to load validation results:", err);
    }
  }

  async function showModelPerformance(model) {
    try {
      // Parse performance metrics from JSON string
      let metrics = {};
      if (model.performanceMetrics) {
        if (typeof model.performanceMetrics === 'string') {
          metrics = JSON.parse(model.performanceMetrics);
        } else {
          metrics = model.performanceMetrics;
        }
      }
      
      // Parse numeric values from string metrics
      const parseMetric = (key, defaultValue = 0) => {
        const value = metrics[key];
        return value ? parseFloat(value) : defaultValue;
      };

      // Check if we have different metric types
      const hasValidationMetrics = parseMetric("cv_macro_f1") > 0;
      const hasTrainingMetrics = parseMetric("train_macro_f1") > 0;
      const hasFinalMetrics = parseMetric("final_macro_f1") > 0;
      const hasTestMetrics = parseMetric("test_macro_f1") > 0 && parseMetric("test_rows_evaluated") > 0;
      
      const results = {
        modelName: model.name,
        
        // Validation metrics (cross-validation)
        validation_metrics: {
          accuracy: parseMetric("cv_accuracy"),
          macro_f1: parseMetric("cv_macro_f1"),
          balanced_accuracy: parseMetric("cv_balanced_accuracy"),
          abstain_rate: parseMetric("cv_abstain_rate"),
          samples: parseInt(metrics.dataset_info?.validation_samples || "0"),
        },
        
        // Training metrics
        training_metrics: {
          accuracy: parseMetric("train_accuracy"),
          macro_f1: parseMetric("train_macro_f1"),
          balanced_accuracy: parseMetric("train_balanced_accuracy"),
          abstain_rate: parseMetric("train_abstain_rate"),
          samples: parseInt(metrics.dataset_info?.training_samples || "0"),
        },
        
        // Final model metrics (on reserved test set after final retraining)
        final_metrics: {
          accuracy: parseMetric("final_accuracy"),
          macro_f1: parseMetric("final_macro_f1"),
          balanced_accuracy: parseMetric("final_balanced_accuracy"),
          abstain_rate: parseMetric("final_abstain_rate"),
          samples: parseInt(metrics.final_metrics?.test_samples || "0"),
        },
        
        // Legacy test metrics (for backward compatibility)
        test_samples: parseInt(metrics.test_rows_evaluated || "0"),
        accuracy: hasTestMetrics 
          ? parseMetric("test_accuracy") 
          : hasValidationMetrics ? parseMetric("cv_accuracy") : 0,
        precision: hasTestMetrics 
          ? parseMetric("test_macro_f1") 
          : hasValidationMetrics ? parseMetric("cv_macro_f1") : 0,
        recall: hasTestMetrics 
          ? parseMetric("test_balanced_accuracy") 
          : hasValidationMetrics ? parseMetric("cv_balanced_accuracy") : 0,
        abstain_rate: hasTestMetrics 
          ? parseMetric("test_abstain_rate") 
          : hasValidationMetrics ? parseMetric("cv_abstain_rate") : 0,
        
        // Model metadata
        training_samples: parseInt(metrics.dataset_info?.training_samples || metrics.training_samples || "0"),
        n_classes: parseInt(metrics.dataset_info?.n_classes || metrics.n_classes || "0"),
        has_lgbm: metrics.has_lgbm === "True",
        has_svm: metrics.has_svm === "True",
        
        // Indicate which metrics are available
        has_validation_metrics: hasValidationMetrics,
        has_training_metrics: hasTrainingMetrics,
        has_final_metrics: hasFinalMetrics,
        metrics_source: hasTestMetrics ? "test" : hasValidationMetrics ? "cross_validation" : "none",
        
        // Final retraining info
        final_retraining_enabled: metrics.final_retraining?.enabled || false,
        
        // Remove fake predictions - we don't have real prediction data
        predictions: [],
      };

      validationResults.set(results);
      
      // Set default tab based on available metrics - prefer validation as most reliable
      if (hasValidationMetrics) {
        activeMetricTab = 'validation';
      } else if (hasFinalMetrics) {
        activeMetricTab = 'final';
      } else if (hasTrainingMetrics) {
        activeMetricTab = 'training';
      }
      
      currentStep = 3; // Navigate to Model Performance step
    } catch (err) {
      console.error("Failed to load model performance:", err);
      $error = `Failed to load model performance: ${err.message}`;
    }
  }

  async function checkAndResumeTraining(modelsList) {
    // Check for models with Pending status (status = 2) which indicates training in progress
    const pendingModels = modelsList.filter((model) => model.status === 2); // TrainingStatus.Pending

    if (pendingModels.length > 0) {
      console.debug(
        "Found models with Pending status:",
        pendingModels.map((m) => m.name),
      );

      // Before resuming, quickly check if these models actually exist in the backend
      const validPendingModels = [];
      
      for (const model of pendingModels) {
        try {
          const statusResponse = await mlApi.getTrainingStatus(model.name);
          
          if (statusResponse.success) {
            // Model exists and has valid status
            validPendingModels.push(model);
          } else {
            // Check if it's a 404 error (model doesn't exist)
            const is404Error = statusResponse.status === 404 || 
                              (statusResponse.error && statusResponse.error.includes('404')) ||
                              (statusResponse.error && statusResponse.error.includes('Not Found'));
            
            if (is404Error) {
              console.warn(`Pending model ${model.name} not found in backend, cleaning up`);
              
              // Try to delete the model from backend (in case it exists in models list but not training)
              try {
                await mlApi.deleteModel(model.name);
              } catch (deleteErr) {
                console.warn(`Could not delete stuck model ${model.name}:`, deleteErr);
              }
              
              // Remove from frontend models list
              models.update((list) => list.filter((m) => m.name !== model.name));
            } else {
              // Other error, keep the model but log the issue
              console.warn(`Could not check status for pending model ${model.name}:`, statusResponse.error);
              validPendingModels.push(model);
            }
          }
        } catch (err) {
          // Check if it's a 404-like error
          const is404Error = (err.status === 404) || 
                            (err.message && err.message.includes('404')) ||
                            (err.message && err.message.includes('Not Found'));
          
          if (is404Error) {
            console.warn(`Pending model ${model.name} not found in backend (404), cleaning up`);
            
            // Try to delete the model from backend
            try {
              await mlApi.deleteModel(model.name);
            } catch (deleteErr) {
              console.warn(`Could not delete stuck model ${model.name}:`, deleteErr);
            }
            
            // Remove from frontend models list
            models.update((list) => list.filter((m) => m.name !== model.name));
          } else {
            // Other error, keep the model but log the issue
            console.warn(`Error checking status for pending model ${model.name}:`, err);
            validPendingModels.push(model);
          }
        }
      }

      // If we have valid pending models, resume training for the first one
      if (validPendingModels.length > 0) {
        // Update models to show training status
        models.update((list) =>
          list.map((model) =>
            validPendingModels.some(vm => vm.name === model.name)
              ? {
                  ...model,
                  trainingStatus: "training",
                  trainingMessage: "Training in progress...",
                }
              : model,
          ),
        );

        // Set global training state for the first valid pending model
        const firstValidModel = validPendingModels[0];
        trainingInProgress = true;

        // Resume polling for this model
        await pollTrainingStatus(firstValidModel.name);
        return;
      }
    }

    // Fallback: Check if any models are currently training by checking the backend status
    try {
      const currentTrainingResponse = await mlApi.getCurrentTrainingStatus();

      if (
        currentTrainingResponse.success &&
        currentTrainingResponse.data?.current_training
      ) {
        const currentTrainingModel =
          currentTrainingResponse.data.current_training;
        console.debug("Found ongoing training for model:", currentTrainingModel);

        // Update the model in the list to show training status
        models.update((list) =>
          list.map((model) =>
            model.name === currentTrainingModel
              ? {
                  ...model,
                  trainingStatus: "training",
                  trainingMessage:
                    currentTrainingResponse.data.status?.message ||
                    "Training in progress...",
                }
              : model,
          ),
        );

        // Set global training state
        trainingInProgress = true;

        // Resume polling for this model
        await pollTrainingStatus(currentTrainingModel);
      }
    } catch (err) {
      // If we can't check current training status, try checking individual models
      console.debug(
        "Could not check current training status, checking individual models:",
        err,
      );

      // Check each model to see if any are training
      for (const model of modelsList) {
        try {
          const statusResponse = await mlApi.getTrainingStatus(model.name);
          if (
            statusResponse.success &&
            (statusResponse.data.status === "training" ||
              statusResponse.data.status === "starting")
          ) {
            console.debug("Found ongoing training for model:", model.name);

            // Update the model in the list to show training status
            models.update((list) =>
              list.map((m) =>
                m.name === model.name
                  ? {
                      ...m,
                      trainingStatus: statusResponse.data.status,
                      trainingMessage:
                        statusResponse.data.message ||
                        "Training in progress...",
                    }
                  : m,
              ),
            );

            // Set global training state
            trainingInProgress = true;

            // Resume polling for this model
            await pollTrainingStatus(model.name);
            break; // Only one model can be training at a time
          }
        } catch (modelErr) {
          // Check if it's a 404 error for this model - could be stuck
          const is404Error = (modelErr.status === 404) || 
                            (modelErr.message && modelErr.message.includes('404')) ||
                            (modelErr.message && modelErr.message.includes('Not Found'));
          
          if (is404Error && model.status === 2) {
            // This is a pending model that's returning 404 - likely stuck
            console.warn(`Model ${model.name} has pending status but returns 404, cleaning up`);
            
            // Try to delete the model from backend
            try {
              await mlApi.deleteModel(model.name);
            } catch (deleteErr) {
              console.warn(`Could not delete stuck model ${model.name}:`, deleteErr);
            }
            
            // Remove from frontend models list
            models.update((list) => list.filter((m) => m.name !== model.name));
          } else {
            // Skip this model if we can't check its status for other reasons
            console.debug(
              `Could not check training status for model ${model.name}:`,
              modelErr,
            );
          }
        }
      }
    }
  }

  async function deleteModel(model) {
    try {
      $isLoading = true;
      const response = await mlApi.deleteModel(model.name);
      if (response.success) {
        models.update((list) => list.filter((m) => m.name !== model.name));
        
        // If the deleted model was the default, refresh the default model name
        if (defaultModelName === model.name) {
          await loadDefaultModelName();
        }
        // Update bulkMLPredictStore's hasDefaultModel
        await bulkMLPredictStore.checkDefaultModel(true);
      } else {
        $error = response.message || "Failed to delete model";
      }
    } catch (err) {
      $error = `Failed to delete model: ${err.message}`;
    } finally {
      $isLoading = false;
    }
  }

  async function cancelTraining(model) {
    try {
      $isLoading = true;
      const response = await mlApi.cancelTraining(model.name);
      if (response.success) {
        // Remove the model from the list immediately
        models.update((list) => list.filter((m) => m.name !== model.name));
        
        // Clear global training state if this was the training model
        trainingInProgress = false;
        
        // Refresh default model name for consistency (though training models can't be default)
        if (defaultModelName === model.name) {
          await loadDefaultModelName();
        }
        
        console.log(`Training canceled for model: ${model.name}`);

        // Update bulkMLPredictStore's hasDefaultModel
        await bulkMLPredictStore.checkDefaultModel(true);
      } else {
        $error = response.message || "Failed to cancel training";
      }
    } catch (err) {
      $error = `Failed to cancel training: ${err.message}`;
    } finally {
      $isLoading = false;
    }
  }

  async function setDefaultModel(model) {
    try {
      $isLoading = true;
      const response = await mlApi.setDefaultModel(model.name);
      if (response.success) {
        // Update the default model name
        defaultModelName = model.name;
        console.log(`Successfully set ${model.name} as default model`);
        $error = null;
        // Update bulkMLPredictStore's hasDefaultModel
        await bulkMLPredictStore.checkDefaultModel(true);
      } else {
        $error = response.message || "Failed to set default model";
      }
    } catch (err) {
      $error = `Failed to set default model: ${err.message}`;
    } finally {
      $isLoading = false;
    }
  }

  async function loadDefaultModelName() {
    try {
      // Get the default model configuration
      const response = await fetch('/api/settings/?key=system.default_model_name');
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.data?.settings?.['system.default_model_name']?.value?.stringValue) {
          defaultModelName = data.data.settings['system.default_model_name'].value.stringValue;
          console.debug(`Current default model: ${defaultModelName}`);
        }
      }
    } catch (err) {
      console.debug('Could not load default model name:', err);
    }
  }

  function nextStep() {
    if (currentStep < totalSteps - 1) {
      currentStep++;
    }
  }

  function prevStep() {
    if (currentStep > 0) {
      currentStep--;
    }
  }

  function goToStep(step) {
    // Only allow navigation to completed steps or the next step
    if (canNavigateToStep(step)) {
      currentStep = step;
    }
  }

  function canNavigateToStep(step) {
    // Always allow going back to previous steps
    if (step <= currentStep) return true;

    // Check if we can proceed to the next step
    switch (step) {
      case 1: // Data Preparation - always accessible from Model Management
        return true;
      case 2: // Model Training - need training data prepared
        return $trainingData;
      case 3: // Validation Results - need training data prepared
        return $trainingData;
      default:
        return false;
    }
  }

  async function loadCategoryMapping(budgetId) {
    try {
      console.debug("Loading category mapping for budget:", budgetId);

      // Get budget info with categories
      const budgetInfoResponse = await budgetsApi.getBudgetInfo(
        [budgetId],
        ["Category"],
        false,
      );

      if (
        budgetInfoResponse.categories &&
        Array.isArray(budgetInfoResponse.categories)
      ) {
        // Create mapping from category ID to category name
        const newCategoryMap = new Map();
        budgetInfoResponse.categories.forEach((category) => {
          newCategoryMap.set(category.id, category.name);
        });

        categoryMap.set(newCategoryMap);
        console.debug(`Loaded ${newCategoryMap.size} categories for mapping`);
      } else {
        console.warn(
          "Failed to load category mapping:",
          budgetInfoResponse.error ||
            budgetInfoResponse.message ||
            "No categories found",
        );
        console.warn("Full response:", budgetInfoResponse);
      }
    } catch (err) {
      console.error("Error loading category mapping:", err);
    }
  }

  function getCategoryName(categoryId) {
    return $categoryMap.get(categoryId) || categoryId || "Unknown Category";
  }

  async function loadDatasets(budgetId) {
    try {
      console.debug("Loading datasets for budget:", budgetId);

      const datasetsResponse = await mlApi.getDatasets(budgetId);

      if (datasetsResponse.success) {
        datasets.set(datasetsResponse.data || []);
        console.debug(`Loaded ${datasetsResponse.data?.length || 0} datasets`);
      } else {
        console.warn("Failed to load datasets:", datasetsResponse.error);
        datasets.set([]);
      }
    } catch (err) {
      console.error("Error loading datasets:", err);
      datasets.set([]);
    }
  }

  async function selectExistingDataset(dataset) {
    try {
      console.debug("Selecting existing dataset:", dataset.id);

      // Set the training data to the selected dataset
      trainingData.set({
        dataset_id: dataset.id,
        dataset_name: dataset.name,
        total_transactions: dataset.total_transactions,
        training_samples: dataset.training_samples,
        test_samples: dataset.test_samples,
        categories: dataset.categories,
        category_breakdown: dataset.category_breakdown,
        training_file: dataset.training_file,
        test_file: dataset.test_file,
        date_from: dataset.date_from,
        date_to: dataset.date_to,
        created_at: dataset.created_at,
        is_existing: true,
      });

      // Ensure category mapping is loaded for displaying category names
      if (selectedBudgetId && $categoryMap.size === 0) {
        await loadCategoryMapping(selectedBudgetId);
      }

      // Move to training step
      nextStep();
    } catch (err) {
      console.error("Error selecting dataset:", err);
      $error = `Failed to select dataset: ${err.message}`;
    }
  }

  async function deleteDataset(dataset) {
    try {
      console.debug("Deleting dataset:", dataset.id);

      const response = await mlApi.deleteDataset(dataset.id);

      if (response.success) {
        // Check if the deleted dataset is the currently active one
        const wasActiveDataset = $trainingData?.dataset_id === dataset.id || 
                                $trainingData?.dataset_name === dataset.dataset_name;
        
        // Reload datasets
        await loadDatasets(selectedBudgetId);
        
        // If we deleted the active dataset, reset the UI state
        if (wasActiveDataset) {
          trainingData.set(null);
          trainingStats.set(null);
          // Reset to step 1 (Prepare Data) since the current dataset is gone
          currentStep = 1;
        }
        
        console.log("Dataset deleted successfully");
      } else {
        $error = response.message || "Failed to delete dataset";
      }
    } catch (err) {
      console.error("Error deleting dataset:", err);
      $error = `Failed to delete dataset: ${err.message}`;
    }
  }

  function formatDatasetDate(dateString) {
    if (!dateString) return "Unknown";
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return dateString;
    }
  }

  function formatDatasetSize(bytes) {
    if (!bytes) return "Unknown";
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    const mb = kb / 1024;
    return `${mb.toFixed(1)} MB`;
  }

  function parseModelMetrics(model) {
    if (!model.performanceMetrics) return null;
    
    try {
      if (typeof model.performanceMetrics === 'string') {
        return JSON.parse(model.performanceMetrics);
      }
      return model.performanceMetrics;
    } catch (err) {
      console.error('Failed to parse model metrics:', err);
      return null;
    }
  }

  function formatMetricValue(value, isPercentage = false) {
    if (!value || value === "0" || value === "0.0") return "N/A";
    const numValue = parseFloat(value);
    if (isNaN(numValue)) return "N/A";
    return isPercentage ? `${(numValue * 100).toFixed(1)}%` : numValue.toFixed(3);
  }
</script>

<div class="container mx-auto py-6">
  <div class="mb-6">
    <h1 class="text-3xl font-bold">ML Model Training</h1>
    <p class="text-muted-foreground mt-2">
      Train machine learning models to automatically categorize your
      transactions
    </p>
  </div>

  <!-- Progress indicator -->
  <div class="mb-6">
    <div class="flex items-center justify-between">
      {#each Array(totalSteps) as _, i}
        <div class="flex items-center">
          <button
            class="w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors
                   {i === currentStep
              ? 'bg-primary text-primary-foreground'
              : i < currentStep
                ? 'bg-primary/20 text-primary'
                : canNavigateToStep(i)
                  ? 'bg-muted text-muted-foreground hover:bg-muted/80'
                  : 'bg-muted/50 text-muted-foreground/50 cursor-not-allowed'}"
            onclick={() => goToStep(i)}
            disabled={!canNavigateToStep(i)}
          >
            {i + 1}
          </button>
          {#if i < totalSteps - 1}
            <div
              class="w-12 h-0.5 mx-2 {i < currentStep
                ? 'bg-primary'
                : 'bg-muted'}"
            ></div>
          {/if}
        </div>
      {/each}
    </div>
    <div class="flex justify-between mt-2 text-sm text-muted-foreground">
      <span>Manage Models</span>
      <span>Prepare Data</span>
      <span>Train Model</span>
      <span>Model Performance</span>
    </div>
  </div>

  <!-- Error display -->
  {#if $error}
    <div
      class="mb-6 p-4 bg-destructive/10 border border-destructive/20 rounded-lg"
    >
      <p class="text-destructive">{$error}</p>
      <Button
        variant="ghost"
        size="sm"
        class="mt-2"
        onclick={() => error.set(null)}
      >
        Dismiss
      </Button>
    </div>
  {/if}

  <!-- Carousel -->
  <div class="w-full">
    <div class="overflow-hidden">
      <div
        class="flex transition-transform duration-300 ease-in-out"
        style="transform: translateX(-{currentStep * 100}%)"
      >
        <!-- Step 1: Model Management -->
        <div class="min-w-full shrink-0">
          <Card.Root>
            <Card.Header>
              <Card.Title>Model Management</Card.Title>
              <Card.Description>
                Manage your trained models, view performance metrics, and set
                defaults
              </Card.Description>
            </Card.Header>
            <Card.Content class="space-y-4">
              {#if $isLoading}
                <div class="flex items-center justify-center py-8">
                  <div
                    class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"
                  ></div>
                </div>
              {:else if $models.length === 0}
                <div class="text-center py-8">
                  <p class="text-muted-foreground">No trained models found</p>
                  <p class="text-sm text-muted-foreground mt-2">
                    Create your first model by proceeding to the next step
                  </p>
                </div>
              {:else}
                <div class="space-y-3">
                  {#each $models as model}
                    <div
                      class="flex items-center justify-between p-4 border rounded-lg"
                    >
                      <div class="flex-1">
                        <div class="flex items-center gap-2">
                          <h4 class="font-medium">{model.name}</h4>
                          {#if model.trainingStatus === "training" || model.trainingStatus === "starting" || model.status === 2}
                            <div class="flex items-center gap-2">
                              <div
                                class="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"
                              ></div>
                              <span class="text-xs text-muted-foreground"
                                >Training...</span
                              >
                            </div>
                          {:else if model.trainingStatus === "failed" || model.status === 4}
                            <span
                              class="text-xs bg-red-100 text-red-800 px-2 py-1 rounded"
                              >Failed</span
                            >
                          {/if}
                        </div>
                        <p class="text-sm text-muted-foreground">
                          Version {model.version} • Trained {model.trainedDate
                            ? (() => {
                                // Treat the timestamp as UTC if it doesn't have timezone info
                                let date;
                                if (model.trainedDate.includes('Z') || model.trainedDate.includes('+') || model.trainedDate.includes('-', 10)) {
                                  // Already has timezone info
                                  date = new Date(model.trainedDate);
                                } else {
                                  // No timezone info, treat as UTC
                                  date = new Date(model.trainedDate + 'Z');
                                }
                                
                                return date.toLocaleString('en-US', {
                                  year: 'numeric',
                                  month: 'short',
                                  day: 'numeric',
                                  hour: 'numeric',
                                  minute: '2-digit',
                                  second: '2-digit',
                                  timeZoneName: 'short'
                                });
                              })()
                            : "Unknown"}
                        </p>
                        {#if model.trainingStatus === "training" || model.trainingStatus === "starting" || model.status === 2}
                          <div class="mt-2">
                            <p class="text-xs text-muted-foreground">
                              {model.trainingMessage ||
                                "Training in progress..."}
                            </p>
                          </div>
                        {:else if parseModelMetrics(model)}
                          {@const metrics = parseModelMetrics(model)}
                          {@const hasTrainingMetrics = parseFloat(metrics.train_macro_f1 || "0") > 0}
                          {@const hasValidationMetrics = parseFloat(metrics.cv_macro_f1 || "0") > 0}
                          {@const hasFinalMetrics = parseFloat(metrics.final_macro_f1 || "0") > 0}
                          {@const hasTestMetrics = parseFloat(metrics.test_macro_f1 || "0") > 0 && parseInt(metrics.test_rows_evaluated || "0") > 0}
                          
                          <!-- Display all available metric types -->
                          <div class="mt-2 space-y-2">
                            {#if hasTrainingMetrics}
                              <div class="text-xs text-black">
                                <div class="font-medium mb-1">Training: {formatMetricValue(metrics.train_accuracy, true)} Acc • {formatMetricValue(metrics.train_macro_f1, true)} F1</div>
                              </div>
                            {/if}
                            
                            {#if hasValidationMetrics}
                              <div class="text-xs text-green-600">
                                <div class="font-medium mb-1">Validation: {formatMetricValue(metrics.cv_accuracy, true)} Acc • {formatMetricValue(metrics.cv_macro_f1, true)} F1</div>
                              </div>
                            {/if}
                            
                            {#if hasFinalMetrics}
                              <div class="text-xs text-blue-600">
                                <div class="font-medium mb-1">Final: {formatMetricValue(metrics.final_accuracy, true)} Acc • {formatMetricValue(metrics.final_macro_f1, true)} F1</div>
                              </div>
                            {:else if hasTestMetrics}
                              <div class="text-xs text-blue-600">
                                <div class="font-medium mb-1">Test: {formatMetricValue(metrics.test_accuracy, true)} Acc • {formatMetricValue(metrics.test_macro_f1, true)} F1</div>
                              </div>
                            {/if}
                            
                            <!-- Additional details -->
                            <div class="grid grid-cols-2 gap-2 text-xs text-muted-foreground pt-1 border-t">
                              <span>Samples: {metrics.training_samples || "N/A"}</span>
                              <span>Classes: {metrics.n_classes || "N/A"}</span>
                            </div>
                          </div>
                        {/if}
                      </div>
                      <div class="flex gap-2">
                        <Button 
                          variant={defaultModelName === model.name ? "default" : "outline"}
                          size="sm"
                          disabled={model.trainingStatus === "training" ||
                            model.trainingStatus === "starting" ||
                            model.trainingStatus === "failed" ||
                            model.status === 2 ||
                            model.status === 4 ||
                            $isLoading ||
                            defaultModelName === model.name}
                          onclick={() => setDefaultModel(model)}
                        >
                          {defaultModelName === model.name ? "✓ Default" : "Set Default"}
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={model.trainingStatus === "training" ||
                            model.trainingStatus === "starting" ||
                            model.trainingStatus === "failed" ||
                            model.status === 2 ||
                            model.status === 4 ||
                            !model.performanceMetrics}
                          onclick={() => showModelPerformance(model)}
                        >
                          Model Performance
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          onclick={() => {
                            // Check if model is currently training
                            const isTraining = model.trainingStatus === "training" || 
                                             model.trainingStatus === "starting" || 
                                             model.status === 2; // TrainingStatus.Pending
                            
                            if (isTraining) {
                              cancelTraining(model);
                            } else {
                              deleteModel(model);
                            }
                          }}
                        >
                          {#if model.trainingStatus === "training" || model.trainingStatus === "starting" || model.status === 2}
                            Cancel
                          {:else}
                            Delete
                          {/if}
                        </Button>
                      </div>
                    </div>
                  {/each}
                </div>
              {/if}
            </Card.Content>
            <Card.Footer>
              <Button
                onclick={nextStep}
                class="ml-auto"
                disabled={!canNavigateToStep(1)}
              >
                Prepare Training Data
              </Button>
            </Card.Footer>
          </Card.Root>
        </div>

        <!-- Step 2: Data Preparation -->
        <div class="min-w-full shrink-0">
          <Card.Root>
            <Card.Header>
              <Card.Title>Prepare Training Data</Card.Title>
              <Card.Description>
                Pull transaction data from your database and split into
                training/test sets
              </Card.Description>
            </Card.Header>
            <Card.Content class="space-y-4">
              <!-- Existing Datasets Section -->
              {#if $datasets.length > 0}
                <div class="space-y-3">
                  <h4 class="font-medium">Existing Datasets</h4>
                  <div class="space-y-2 max-h-64 overflow-y-auto">
                    {#each $datasets as dataset}
                      <div
                        class="flex items-center justify-between p-3 border rounded-lg bg-muted/30"
                      >
                        <div class="flex-1">
                          <div class="flex items-center gap-2">
                            <h5 class="font-medium text-sm">{dataset.name}</h5>
                            {#if !dataset.files_exist}
                              <span
                                class="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded"
                                >Files Missing</span
                              >
                            {/if}
                          </div>
                          <div class="text-xs text-muted-foreground mt-1">
                            <div class="flex gap-4">
                              <span
                                >{dataset.training_samples} training + {dataset.test_samples}
                                test samples</span
                              >
                              <span>{dataset.categories} categories</span>
                              <span
                                >Created {formatDatasetDate(
                                  dataset.created_at,
                                )}</span
                              >
                            </div>
                            {#if dataset.date_from && dataset.date_to}
                              <div class="mt-1">
                                Data range: {formatDatasetDate(
                                  dataset.date_from,
                                )} - {formatDatasetDate(dataset.date_to)}
                              </div>
                            {/if}
                          </div>
                        </div>
                        <div class="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onclick={() => selectExistingDataset(dataset)}
                            disabled={!dataset.files_exist}
                          >
                            Use Dataset
                          </Button>
                          <Button
                            variant="destructive"
                            size="sm"
                            onclick={() => deleteDataset(dataset)}
                          >
                            Delete
                          </Button>
                        </div>
                      </div>
                    {/each}
                  </div>
                  <div class="border-t pt-3">
                    <h4 class="font-medium mb-2">Create New Dataset</h4>
                  </div>
                </div>
              {/if}

              <!-- Budget Selection and Months Back - Side by Side -->
              <div class="grid grid-cols-2 gap-4">
                <div class="space-y-2">
                  <label for="budget-select" class="text-sm font-medium"
                    >Budget</label
                  >
                  <DropdownSelect
                    id="budget-select"
                    bind:value={selectedBudgetId}
                    options={budgets.map((budget) => ({
                      value: budget.id,
                      label: budget.name,
                    }))}
                    placeholder={$isLoading
                      ? "Loading budgets..."
                      : budgets.length === 0
                        ? "No budgets available - Connect YNAB first"
                        : "Select a budget"}
                    disabled={$isLoading || budgets.length === 0}
                    searchable={true}
                    clearable={false}
                  />
                </div>

                <div class="space-y-2">
                  <label for="months-input" class="text-sm font-medium"
                    >Months of Data</label
                  >
                  <input
                    id="months-input"
                    type="number"
                    bind:value={monthsBack}
                    min="1"
                    max="60"
                    class="w-full p-2 border rounded-md"
                  />
                </div>
              </div>

              <!-- Loading indicator for data preparation -->
              {#if isPreparingData}
                <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div class="flex items-center gap-3">
                    <div
                      class="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"
                    ></div>
                    <div>
                      <h4 class="font-medium text-blue-800">
                        Preparing Training Data
                      </h4>
                      <p class="text-sm text-blue-600">
                        Analyzing transactions and creating training datasets...
                      </p>
                    </div>
                  </div>
                </div>
              {/if}

              <!-- Success indicator for dataset creation -->
              {#if $trainingData && !isPreparingData}
                <div class="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div class="flex items-center gap-3">
                    <div
                      class="w-5 h-5 bg-green-600 rounded-full flex items-center justify-center"
                    >
                      <svg
                        class="w-3 h-3 text-white"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fill-rule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clip-rule="evenodd"
                        ></path>
                      </svg>
                    </div>
                    <div>
                      <h4 class="font-medium text-green-800">
                        Dataset Created Successfully
                      </h4>
                      <p class="text-sm text-green-600">
                        {$trainingData.dataset_name} • {$trainingData.training_samples}
                        training + {$trainingData.test_samples} test samples
                      </p>
                    </div>
                  </div>
                </div>
              {/if}

              <!-- Data Analysis Results (shown after preparing data) -->
              {#if $trainingStats && !isPreparingData}
                <div class="p-4 bg-muted rounded-lg">
                  <h4 class="font-medium mb-2">Data Analysis</h4>
                  <div class="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span class="text-muted-foreground"
                        >Total Transactions:</span
                      >
                      <span class="font-medium ml-2"
                        >{$trainingStats.categorized_transactions}</span
                      >
                    </div>
                    <div>
                      <span class="text-muted-foreground">Categories:</span>
                      <span class="font-medium ml-2"
                        >{$trainingStats.unique_categories}</span
                      >
                    </div>
                    <div class="col-span-2">
                      <span class="text-muted-foreground">Recommendation:</span>
                      <span
                        class="font-medium ml-2 {$trainingStats.suitability
                          ?.sufficient_data
                          ? 'text-green-600'
                          : 'text-yellow-600'}"
                      >
                        {$trainingStats.suitability?.recommendation ||
                          "Analyzing..."}
                      </span>
                      
                      <!-- Display warnings as bullet list if they exist -->
                      {#if $trainingStats.suitability?.warnings && $trainingStats.suitability.warnings.length > 0}
                        <div class="mt-2">
                          <ul class="text-sm text-yellow-600 list-disc list-inside space-y-1">
                            {#each $trainingStats.suitability.warnings as warning}
                              <li>{warning}</li>
                            {/each}
                          </ul>
                        </div>
                      {/if}
                    </div>
                  </div>

                  <!-- Category Breakdown -->
                  {#if $trainingStats.category_breakdown}
                    <div class="mt-4">
                      <h5 class="font-medium mb-2">Category Distribution</h5>
                      <div class="space-y-1 max-h-32 overflow-y-auto">
                        {#each Object.entries($trainingStats.category_breakdown) as [category, count]}
                          <div class="flex justify-between text-xs">
                            <span class="text-muted-foreground truncate"
                              >{getCategoryName(category)}</span
                            >
                            <span class="font-medium">{count}</span>
                          </div>
                        {/each}
                      </div>
                    </div>
                  {/if}
                </div>
              {/if}
            </Card.Content>
            <Card.Footer class="flex justify-between">
              <Button variant="outline" onclick={prevStep}>Back</Button>
              <div class="flex gap-2">
                {#if $trainingData}
                  <!-- Show different buttons based on data quality -->
                  {#if $trainingData.training_samples > 0 && $trainingData.categories > 0}
                    <Button
                      variant="outline"
                      onclick={prepareTrainingData}
                      disabled={isPreparingData || !selectedBudgetId}
                    >
                      {isPreparingData ? "Refreshing..." : "Refresh Data"}
                    </Button>
                    <Button onclick={nextStep} disabled={isPreparingData}
                      >Continue to Training</Button
                    >
                  {:else}
                    <Button
                      variant="outline"
                      onclick={prepareTrainingData}
                      disabled={isPreparingData || !selectedBudgetId}
                    >
                      {isPreparingData
                        ? "Re-analyzing..."
                        : "Try Different Parameters"}
                    </Button>
                    <Button
                      variant="secondary"
                      onclick={nextStep}
                      disabled={isPreparingData}
                    >
                      Continue Anyway
                    </Button>
                  {/if}
                {:else}
                  <Button
                    onclick={prepareTrainingData}
                    disabled={isPreparingData || !selectedBudgetId}
                  >
                    {isPreparingData
                      ? "Analyzing & Preparing..."
                      : "Prepare Data"}
                  </Button>
                {/if}
              </div>
            </Card.Footer>
          </Card.Root>
        </div>

        <!-- Step 3: Model Training -->
        <div class="min-w-full shrink-0">
          <Card.Root>
            <Card.Header>
              <Card.Title>Train Model</Card.Title>
              <Card.Description>
                Configure and start training your machine learning model
              </Card.Description>
            </Card.Header>
            <Card.Content class="space-y-4">
              <!-- Training Data Summary -->
              {#if $trainingData}
                <div
                  class="p-4 {$trainingData.training_samples > 0 &&
                  $trainingData.categories > 0
                    ? 'bg-muted'
                    : 'bg-yellow-50 border border-yellow-200'} rounded-lg"
                >
                  <div class="flex items-center justify-between mb-2">
                    <h4 class="font-medium">
                      {$trainingData.training_samples > 0 &&
                      $trainingData.categories > 0
                        ? "Training Data Ready"
                        : "Training Data Prepared (Limited)"}
                    </h4>
                    {#if $trainingData.is_existing}
                      <span
                        class="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded"
                      >
                        Using Existing Dataset
                      </span>
                    {/if}
                  </div>

                  {#if $trainingData.is_existing}
                    <div
                      class="mb-3 p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-800"
                    >
                      📁 Using dataset: <strong
                        >{$trainingData.dataset_name}</strong
                      >
                      <br />
                      Created: {formatDatasetDate($trainingData.created_at)}
                      {#if $trainingData.date_from && $trainingData.date_to}
                        <br />
                        Data range: {formatDatasetDate($trainingData.date_from)}
                        - {formatDatasetDate($trainingData.date_to)}
                      {/if}
                    </div>
                  {/if}

                  {#if $trainingData.training_samples === 0 || $trainingData.categories === 0}
                    <div
                      class="mb-3 p-2 bg-yellow-100 border border-yellow-300 rounded text-sm text-yellow-800"
                    >
                      ⚠️ Insufficient data for optimal training. Consider:
                      <ul class="mt-1 ml-4 list-disc">
                        <li>Increasing the months of data</li>
                        <li>Ensuring transactions are categorized in YNAB</li>
                        <li>Syncing more recent transaction data</li>
                      </ul>
                    </div>
                  {/if}

                  <div class="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span class="text-muted-foreground"
                        >Training Samples:</span
                      >
                      <span
                        class="font-medium ml-2 {$trainingData.training_samples ===
                        0
                          ? 'text-yellow-600'
                          : ''}">{$trainingData.training_samples}</span
                      >
                    </div>
                    <div>
                      <span class="text-muted-foreground">Test Samples:</span>
                      <span
                        class="font-medium ml-2 {$trainingData.test_samples ===
                        0
                          ? 'text-yellow-600'
                          : ''}">{$trainingData.test_samples}</span
                      >
                    </div>
                    <div>
                      <span class="text-muted-foreground">Categories:</span>
                      <span
                        class="font-medium ml-2 {$trainingData.categories === 0
                          ? 'text-yellow-600'
                          : ''}">{$trainingData.categories}</span
                      >
                    </div>
                  </div>

                  <!-- Category Breakdown from Training Data -->
                  {#if $trainingData.category_breakdown}
                    <div class="mt-4">
                      <h5 class="font-medium mb-2">
                        Training/Test Split by Category
                      </h5>
                      <div class="border rounded-lg max-h-64 overflow-y-auto">
                        <Table.Root>
                          <Table.Header>
                            <Table.Row>
                              <Table.Head class="w-[50%]">Category</Table.Head>
                              <Table.Head class="text-center w-[16%]"
                                >Training</Table.Head
                              >
                              <Table.Head class="text-center w-[16%]"
                                >Test</Table.Head
                              >
                              <Table.Head class="text-center w-[16%]"
                                >Total</Table.Head
                              >
                            </Table.Row>
                          </Table.Header>
                          <Table.Body>
                            {#each Object.entries($trainingData.category_breakdown).sort(([categoryA], [categoryB]) => getCategoryName(categoryA).localeCompare(getCategoryName(categoryB))) as [category, totalCount]}
                              {@const estimatedTraining = Math.round(totalCount * 0.8)}
                              {@const estimatedTest = totalCount - estimatedTraining}
                              <Table.Row>
                                <Table.Cell
                                  class="font-medium"
                                  title={getCategoryName(category)}
                                  >{getCategoryName(category)}</Table.Cell
                                >
                                <Table.Cell class="text-center"
                                  >{estimatedTraining}</Table.Cell>
                                <Table.Cell class="text-center"
                                  >{estimatedTest}</Table.Cell>
                                <Table.Cell class="text-center"
                                  >{totalCount}</Table.Cell>
                              </Table.Row>
                            {/each}
                          </Table.Body>
                        </Table.Root>
                      </div>
                      <div class="text-xs text-muted-foreground mt-2">
                        Training and test samples per category (estimated split
                        based on 80/20 ratio)
                      </div>
                    </div>
                  {/if}
                </div>
              {/if}

              <!-- Model Configuration -->
              <div class="space-y-4">
                <div class="space-y-2">
                  <label for="model-name-input" class="text-sm font-medium"
                    >Model Name</label
                  >
                  <input
                    id="model-name-input"
                    type="text"
                    bind:value={modelName}
                    placeholder="Enter model name (e.g., 'Transaction Classifier v1')"
                    class="w-full p-2 border rounded-md"
                    disabled={trainingInProgress}
                  />
                </div>

                <div class="space-y-2">
                  <label for="strategy-select" class="text-sm font-medium"
                    >Training Strategy</label
                  >
                  <select
                    id="strategy-select"
                    bind:value={selectedStrategy}
                    class="w-full p-2 border rounded-md"
                    disabled={trainingInProgress}
                  >
                    {#each modelStrategies as strategy}
                      <option value={strategy.value}>{strategy.label}</option>
                    {/each}
                  </select>
                  {#if selectedStrategy}
                    {@const strategy = modelStrategies.find(
                      (s) => s.value === selectedStrategy,
                    )}
                    {#if strategy}
                      <p class="text-sm text-muted-foreground">
                        {strategy.description}
                      </p>
                    {/if}
                  {/if}
                </div>

                <div class="space-y-2">
                  <label for="training-time-input" class="text-sm font-medium"
                    >Maximum Training Time (minutes)</label
                  >
                  <input
                    id="training-time-input"
                    type="number"
                    bind:value={maxTrainingTime}
                    min="1"
                    max="120"
                    placeholder="10"
                    class="w-full p-2 border rounded-md"
                    disabled={trainingInProgress}
                  />
                  <p class="text-sm text-muted-foreground">
                    PXBlendSC will train LightGBM and SVM models with
                    feature engineering within this time limit. Longer training
                    typically produces better models.
                  </p>
                </div>
              </div>

              {#if trainingInProgress}
                <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div class="flex items-center gap-3">
                    <div
                      class="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"
                    ></div>
                    <div class="flex-1">
                      <h4 class="font-medium text-blue-800">
                        Training Model: {modelName}
                      </h4>
                      <p class="text-sm text-blue-600">
                        Training is in progress. This may take several minutes
                        depending on your data size and training parameters.
                      </p>
                      <p class="text-xs text-blue-500 mt-1">
                        You can safely navigate away - training will continue in
                        the background.
                      </p>
                    </div>
                  </div>
                </div>
              {/if}
            </Card.Content>
            <Card.Footer class="flex justify-between">
              <Button
                variant="outline"
                onclick={prevStep}
                disabled={trainingInProgress}
              >
                Back
              </Button>
              <Button
                onclick={startTraining}
                disabled={$isLoading ||
                  trainingInProgress ||
                  !modelName.trim() ||
                  !$trainingData ||
                  !selectedStrategy}
              >
                {trainingInProgress ? "Training..." : "Start Training"}
              </Button>
            </Card.Footer>
          </Card.Root>
        </div>

        <!-- Step 4: Model Performance -->
        <div class="min-w-full shrink-0">
          <Card.Root>
            <Card.Header>
              <Card.Title>Model Performance</Card.Title>
              <Card.Description>
                {#if $validationResults?.modelName}
                  Review performance metrics for model: {$validationResults.modelName}
                {:else}
                  Review model performance on test data and validate predictions
                {/if}
              </Card.Description>
            </Card.Header>
            <Card.Content class="space-y-4">
              {#if $validationResults}
                <!-- Performance Metrics with Tabs -->
                <div class="p-4 bg-muted rounded-lg">
                  <div class="flex items-center justify-between mb-3">
                    <h4 class="font-medium">Model Performance</h4>
                    {#if $validationResults.final_retraining_enabled}
                      <div class="text-xs px-2 py-1 bg-green-100 text-green-700 rounded">
                        Final Retraining Enabled
                      </div>
                    {/if}
                  </div>
                  
                  <!-- Metric Type Tabs -->
                  <div class="flex space-x-1 mb-4 border-b">
                    {#if $validationResults.has_training_metrics}
                      <button 
                        class="px-3 py-2 text-sm font-medium border-b-2 transition-colors"
                        class:border-primary={activeMetricTab === 'training'}
                        class:text-primary={activeMetricTab === 'training'}
                        class:border-transparent={activeMetricTab !== 'training'}
                        class:text-muted-foreground={activeMetricTab !== 'training'}
                        onclick={() => activeMetricTab = 'training'}
                      >
                        Training
                      </button>
                    {/if}
                    {#if $validationResults.has_validation_metrics}
                      <button 
                        class="px-3 py-2 text-sm font-medium border-b-2 transition-colors"
                        class:border-primary={activeMetricTab === 'validation'}
                        class:text-primary={activeMetricTab === 'validation'}
                        class:border-transparent={activeMetricTab !== 'validation'}
                        class:text-muted-foreground={activeMetricTab !== 'validation'}
                        onclick={() => activeMetricTab = 'validation'}
                      >
                        Validation
                      </button>
                    {/if}
                    {#if $validationResults.has_final_metrics}
                      <button 
                        class="px-3 py-2 text-sm font-medium border-b-2 transition-colors"
                        class:border-primary={activeMetricTab === 'final'}
                        class:text-primary={activeMetricTab === 'final'}
                        class:border-transparent={activeMetricTab !== 'final'}
                        class:text-muted-foreground={activeMetricTab !== 'final'}
                        onclick={() => activeMetricTab = 'final'}
                      >
                        Final Model
                      </button>
                    {/if}
                  </div>

                  <!-- Training Metrics -->
                  {#if activeMetricTab === 'training' && $validationResults.has_training_metrics}
                    <div class="grid grid-cols-3 gap-4 text-sm mb-4">
                      <div class="text-center">
                        <div class="text-2xl font-bold">
                          {$validationResults.training_metrics.accuracy > 0 
                            ? Math.round($validationResults.training_metrics.accuracy * 100) + "%" 
                            : "N/A"}
                        </div>
                        <div class="text-muted-foreground">Accuracy</div>
                      </div>
                      <div class="text-center">
                        <div class="text-2xl font-bold">
                          {$validationResults.training_metrics.macro_f1 > 0 
                            ? Math.round($validationResults.training_metrics.macro_f1 * 100) + "%" 
                            : "N/A"}
                        </div>
                        <div class="text-muted-foreground">Macro F1</div>
                      </div>
                      <div class="text-center">
                        <div class="text-2xl font-bold">
                          {$validationResults.training_metrics.balanced_accuracy > 0 
                            ? Math.round($validationResults.training_metrics.balanced_accuracy * 100) + "%" 
                            : "N/A"}
                        </div>
                        <div class="text-muted-foreground">Balanced Accuracy</div>
                      </div>
                    </div>

                    <div class="text-xs text-muted-foreground border-t pt-3">
                      <div class="grid grid-cols-2 gap-4">
                        <div>
                          <span class="font-medium">Abstain Rate:</span> 
                          {$validationResults.training_metrics.abstain_rate > 0 
                            ? Math.round($validationResults.training_metrics.abstain_rate * 100) + "%" 
                            : "N/A"}
                        </div>
                        <div>
                          <span class="font-medium">Training Samples:</span> 
                          {$validationResults.training_metrics.samples > 0 
                            ? $validationResults.training_metrics.samples.toLocaleString() 
                            : "N/A"}
                        </div>
                      </div>
                      <div class="mt-2 text-center">
                        <span>Performance on training data (may show overfitting)</span>
                      </div>
                    </div>
                  {/if}

                  <!-- Validation Metrics -->
                  {#if activeMetricTab === 'validation' && $validationResults.has_validation_metrics}
                    <div class="grid grid-cols-3 gap-4 text-sm mb-4">
                      <div class="text-center">
                        <div class="text-2xl font-bold text-green-600">
                          {$validationResults.validation_metrics.accuracy > 0 
                            ? Math.round($validationResults.validation_metrics.accuracy * 100) + "%" 
                            : "N/A"}
                        </div>
                        <div class="text-muted-foreground">Accuracy</div>
                      </div>
                      <div class="text-center">
                        <div class="text-2xl font-bold text-green-600">
                          {$validationResults.validation_metrics.macro_f1 > 0 
                            ? Math.round($validationResults.validation_metrics.macro_f1 * 100) + "%" 
                            : "N/A"}
                        </div>
                        <div class="text-muted-foreground">Macro F1</div>
                      </div>
                      <div class="text-center">
                        <div class="text-2xl font-bold text-green-600">
                          {$validationResults.validation_metrics.balanced_accuracy > 0 
                            ? Math.round($validationResults.validation_metrics.balanced_accuracy * 100) + "%" 
                            : "N/A"}
                        </div>
                        <div class="text-muted-foreground">Balanced Accuracy</div>
                      </div>
                    </div>

                    <div class="text-xs text-muted-foreground border-t pt-3">
                      <div class="grid grid-cols-2 gap-4">
                        <div>
                          <span class="font-medium">Abstain Rate:</span> 
                          {$validationResults.validation_metrics.abstain_rate > 0 
                            ? Math.round($validationResults.validation_metrics.abstain_rate * 100) + "%" 
                            : "N/A"}
                        </div>
                        <div>
                          <span class="font-medium">Validation Samples:</span> 
                          {$validationResults.validation_metrics.samples > 0 
                            ? $validationResults.validation_metrics.samples.toLocaleString() 
                            : "N/A"}
                        </div>
                      </div>
                      <div class="mt-2 text-center">
                        <span class="text-green-600">Cross-validation performance estimate (most reliable)</span>
                      </div>
                    </div>
                  {/if}

                  <!-- Final Model Metrics -->
                  {#if activeMetricTab === 'final' && $validationResults.has_final_metrics}
                    <div class="grid grid-cols-3 gap-4 text-sm mb-4">
                      <div class="text-center">
                        <div class="text-2xl font-bold text-blue-600">
                          {$validationResults.final_metrics.accuracy > 0 
                            ? Math.round($validationResults.final_metrics.accuracy * 100) + "%" 
                            : "N/A"}
                        </div>
                        <div class="text-muted-foreground">Accuracy</div>
                      </div>
                      <div class="text-center">
                        <div class="text-2xl font-bold text-blue-600">
                          {$validationResults.final_metrics.macro_f1 > 0 
                            ? Math.round($validationResults.final_metrics.macro_f1 * 100) + "%" 
                            : "N/A"}
                        </div>
                        <div class="text-muted-foreground">Macro F1</div>
                      </div>
                      <div class="text-center">
                        <div class="text-2xl font-bold text-blue-600">
                          {$validationResults.final_metrics.balanced_accuracy > 0 
                            ? Math.round($validationResults.final_metrics.balanced_accuracy * 100) + "%" 
                            : "N/A"}
                        </div>
                        <div class="text-muted-foreground">Balanced Accuracy</div>
                      </div>
                    </div>

                    <div class="text-xs text-muted-foreground border-t pt-3">
                      <div class="grid grid-cols-2 gap-4">
                        <div>
                          <span class="font-medium">Abstain Rate:</span> 
                          {$validationResults.final_metrics.abstain_rate > 0 
                            ? Math.round($validationResults.final_metrics.abstain_rate * 100) + "%" 
                            : "N/A"}
                        </div>
                        <div>
                          <span class="font-medium">Test Samples:</span> 
                          {$validationResults.final_metrics.samples > 0 
                            ? $validationResults.final_metrics.samples.toLocaleString() 
                            : "N/A"}
                        </div>
                      </div>
                      <div class="mt-2 text-center">
                        <span class="text-blue-600">Final production model retrained on all available data</span>
                      </div>
                    </div>
                  {/if}

                  <!-- Model Summary -->
                  <div class="grid grid-cols-2 gap-4 text-xs text-muted-foreground border-t pt-3 mt-4">
                    <div>
                      <span class="font-medium">Total Samples:</span> 
                      {$validationResults.training_samples > 0 
                        ? $validationResults.training_samples.toLocaleString() 
                        : "N/A"}
                    </div>
                    <div>
                      <span class="font-medium">Categories:</span> 
                      {$validationResults.n_classes > 0 ? $validationResults.n_classes : "N/A"}
                    </div>
                    <div>
                      <span class="font-medium">Models:</span> 
                      {#if $validationResults.has_lgbm && $validationResults.has_svm}
                        LightGBM + SVM
                      {:else if $validationResults.has_lgbm}
                        LightGBM
                      {:else if $validationResults.has_svm}
                        SVM
                      {:else}
                        N/A
                      {/if}
                    </div>
                  </div>
                </div>

                <!-- Model Information -->
                {#if $validationResults.has_validation_metrics}
                  <div class="p-3 bg-green-50 border border-green-200 rounded-lg">
                    <p class="text-sm text-green-800">
          <strong>Validation Evaluation Completed:</strong> This model has been evaluated on held-out validation data, 
          providing the most reliable performance estimates.
                    </p>
                  </div>
                {:else}
                  <div class="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <p class="text-sm text-yellow-800">
          <strong>Validation Results:</strong> These metrics are from validation (cross-validation) during training. 
          Validation evaluation may not have completed successfully.
                    </p>
                  </div>
                {/if}
              {:else}
                <div class="text-center py-8">
                  <p class="text-muted-foreground">
                    No validation results available
                  </p>
                  <p class="text-sm text-muted-foreground mt-2">
                    Complete model training to see validation results
                  </p>
                </div>
              {/if}
            </Card.Content>
            <Card.Footer class="flex justify-between">
              <Button variant="outline" onclick={prevStep}>Back</Button>
              <Button onclick={() => goToStep(0)}>Return to Models</Button>
            </Card.Footer>
          </Card.Root>
        </div>
      </div>
    </div>
  </div>
</div>
