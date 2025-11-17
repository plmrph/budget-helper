# Machine Learning Strategy: PXBlendSC-RF

The Budget Helper uses the **PXBlendSC-RF** (Prefix-eXamination Blend with Semantic Classification + Recency-Frequency) ML strategy for intelligent transaction categorization. This ensemble approach combines multiple machine learning techniques with domain-specific financial transaction understanding and historical pattern recognition.

## Overview

PXBlendSC-RF is a ML pipeline that learns patterns from your historical transaction data to predict categories for new transactions. It combines:

- **Three-Model Ensemble**: LightGBM + SVM + Recency-Frequency models with optimized blending
- **Advanced Feature Engineering**: Text, numerical, temporal, and cross-feature extraction  
- **Prefix Classification**: Pattern matching for vendor/payee recognition
- **Memory Priors**: Learning from recent transaction patterns
- **Recency-Frequency Model**: Historical payee categorization patterns
- **Adaptive Thresholds**: Smart abstention when confidence is low

## Feature Engineering

### Text Features
The system extracts multiple types of text features from transaction descriptions:

#### TF-IDF Features
- **Word-level TF-IDF**: Captures important words and phrases in payee names and memos
- **Character-level TF-IDF**: Learns character patterns for vendor name variations
- **Max Features**: 5,000 word features + 2,500 character features optimized to reduce overfitting

#### Cross-Features (Hashed)
Advanced feature combinations that capture relationships between transaction elements:
- **Unary Features**: Individual field values (payee, account, amount bins)
- **Pairwise Features**: Combinations like payee+account, payee+amount_bin, date+payee
- **Triple Features**: Three-way combinations for complex pattern recognition

#### Pattern Examples the Model Can Learn:
```
Payee Patterns:
- "AMAZON.COM" → Shopping
- "STARBUCKS #1234" → Dining Out
- "SHELL 12345" → Transportation

Amount + Payee Patterns:  
- "NETFLIX" + $15.99 → Entertainment
- "GROCERY STORE" + $127.50 → Groceries
- "GAS STATION" + $45.00 → Transportation

Date + Payee Patterns:
- Weekend + "RESTAURANT" → Dining Out
- Weekday + "COFFEE SHOP" → Dining Out
- Month-end + "RENT" → Housing

Payee + Date + Amount Patterns:
- "GAS STATION" + any day + >$50.00 → Gas
- "GAS STATION" + any day + <$15.00 → Snacks
- "GAS STATION" + Sat + $3.00 → Newspaper

Account + Category Patterns:
- Checking Account + "ATM WITHDRAWAL" → Cash
- Credit Card + "ONLINE PURCHASE" → varies by payee
```

### Numerical Features
- **Amount Binning**: Intelligent amount ranges using per-payee quantiles (15%, 30%, 50%, 70%, 85%)
- **Amount Sign**: Positive/negative transaction handling
- **Amount Scaling**: Normalized amounts for consistent model input

### Temporal Features
- **Date Extraction**: Day of week, month, day of month patterns
- **Recency Weighting**: More recent transactions influence predictions more strongly
- **Seasonal Patterns**: Monthly and seasonal spending pattern recognition

### Recency-Frequency Model
The recency-frequency model learns from your historical payee categorization patterns:

#### How It Works:
- **Frequency Score**: How often you've categorized a payee to each category
- **Recency Score**: How recently you've categorized a payee to each category  
- **Combined Prediction**: Weighted combination favoring recent patterns over historical frequency
- **Smart Filtering**: Only considers patterns used multiple times to avoid one-off mistakes

#### Example Behavior:
```
VENMO (John Smith) - 10 historical transactions:
- 8x "Personal/Friends", 2x "Food/Dining"
- Last 5 transactions: 4x "Personal", 1x "Food"

Frequency Score: Personal=80%, Food=20%
Recency Score: Personal=80%, Food=20%
Final Prediction: Personal=80%, Food=20%

But if recent pattern changes:
- Last 5 transactions: 2x "Personal", 3x "Food"
Recency Score: Personal=40%, Food=60%
Final Prediction: Personal=56%, Food=44% (weighted toward recent change)
```

#### Benefits:
- **Intuitive Predictions**: Matches how users naturally think about categorization
- **Handles Pattern Changes**: Adapts when spending patterns evolve
- **Reduces Edge Case Errors**: Frequency requirement prevents one-off mistakes from dominating
- **Complements ML Models**: Provides different signal than feature-based learning

## PXBLENDSC_CONFIG Parameters

The configuration in `configs.py` controls all aspects of the ML strategy:

### Cross-Validation (cv)
```python
"cv": {
    "n_folds": 3,           # Number of cross-validation folds
    "force_folds": True     # Force exactly 3 folds (slower but better performance)
}
```
- **n_folds**: More folds = better evaluation but slower training. Higher folds also require more minimum samples for categories (each category needs at least n_folds samples for stratified cross-validation)
- **force_folds**: Ensures consistent validation even with small datasets by filtering out categories that don't have enough samples

### Model Configuration (models)
```python
"models": {
    "use_lgbm": True,                    # Enable LightGBM (gradient boosting)
    "use_svm_blend": True,               # Enable SVM component
    "use_recency_frequency": True,       # Enable recency-frequency model
    "lgbm_weight": 0.4,                  # LightGBM weight in ensemble (40%)
    "svm_weight": 0.2,                   # SVM weight in ensemble (20%)
    "recency_freq_weight": 0.4,          # Recency-frequency weight in ensemble (40%)
    "adaptive_splitting": True,          # Adaptive train/test split for small classes
    "final_retraining": True,            # Retrain on all data after evaluation
    "recency_freq_params": {
        "recency_weight": 0.6,           # Weight for recent vs frequent patterns
        "frequency_weight": 0.4,         # Weight for frequency patterns
        "min_frequency": 3,              # Minimum times payee used for pattern
        "lookback_window": 50,           # Consider last 50 transactions per payee
        "recency_window": 5              # Weight last 5 transactions heavily
    }
}
```

#### Key Parameters:
- **n_folds**: Higher folds will typically improve accuracy overall, but will end up dropping categories that don't have enough samples
- **adaptive_splitting**: Improves accuracy on smaller categories, at the risk of having fewer test samples
- **final_retraining**: Increases the overall training time, but helps with accuracy especially on smaller datasets or categories with fewer samples
- **Ensemble weights**: Controls three-model balance (40% LightGBM, 20% SVM, 40% Recency-Frequency)
- **Recency-frequency params**: Tunes how the model weighs recent vs frequent categorizations

### Feature Engineering (features)
```python
"features": {
    "tfidf_word_max_features": 5000,              # Maximum word-level features
    "tfidf_char_max_features": 2500,              # Maximum character-level features
    "hashed_cross": {
        "n_features": 4096,                       # Hash space for cross-features
        "payee_min_count": 3,                     # Minimum payee frequency for features
        "quantiles": [0.15, 0.3, 0.5, 0.7, 0.85], # Amount binning
        "emit_unaries": True,                     # Single field features
        "emit_pairs": True,                       # Two-field combinations
        "emit_triples": True                      # Three-field combinations
    }
}
```

### Prior Knowledge (priors)
```python
"priors": {
    "vendor": {
        "min_count": 3,              # Minimum transactions for vendor memory
        "hard_override_share": 0.85, # Confidence for exact vendor matches
        "beta": 2.0                  # Smoothing parameter
    },
    "recency": {
        "k_last": 8,                 # Consider last 8 transactions from vendor
        "beta": 1.5                  # Recency weighting strength
    },
    "backoff": {
        "global": 0.40               # Global category frequency fallback
    }
}
```

#### How Priors Work:
- **Vendor Memory**: If you've categorized "STARBUCKS" as "Dining Out" 5+ times, new Starbucks transactions get high confidence for that category
- **Recency**: Recent categorizations matter more than old ones
- **Backoff**: When no specific patterns match, fall back to overall category frequencies

### Threshold Learning (thresholds)
```python
"thresholds": {
    "mode": "learn",                              # Learn optimal thresholds
    "global_grid": [0.15, 0.2, 0.25, 0.3, 0.35],  # Confidence thresholds to test
    "tail_support_max": 8,                        # Max examples for "tail" categories
    "tail_f1_max": 0.25,                          # F1 threshold for tail categories
}
```

#### Threshold Strategy:
- **Learned Thresholds**: Model learns when to abstain vs. predict for each category
- **Tail Handling**: Special handling for rare categories with few examples
- **Conservative Defaults**: Lower thresholds prevent over-confident incorrect predictions

## Training Workflow

### 1. Dataset Preparation
Remember, garbage in garbage out, so the better your data the better the model will be.
- **Minimum Recommended**: 500-1000 categorized transactions
- **Optimal Size**: 2000+ transactions with good category distribution
- **Quality Matters**: Clean, consistently categorized data beats quantity

### 2. Feature Engineering Pipeline
1. **Text Preprocessing**: Clean payee names and memos, remove noise patterns
2. **Feature Extraction**: TF-IDF, cross-features, temporal features
3. **Amount Processing**: Binning, normalization, sign extraction
4. **Cross-Feature Generation**: All pairwise and triple combinations

### 3. Model Training
1. **Cross-Validation**: 3-fold validation for robust performance estimation with proper data isolation
2. **Ensemble Training**: Train LightGBM, SVM, and Recency-Frequency components
3. **Threshold Learning**: Learn optimal confidence thresholds per category using corrected F1 optimization
4. **Final Retraining**: Train final model on all available data with early stopping

### 4. Model Evaluation
- **Accuracy Metrics**: Overall accuracy, per-category F1 scores
- **Abstention Analysis**: Performance when model chooses not to predict
- **Confusion Matrix**: Understanding common misclassifications

## Prediction Process

### 1. Feature Extraction
New transactions go through the same feature engineering pipeline as training data.

### 2. Ensemble Prediction
- **LightGBM Prediction**: Gradient boosting model prediction (40% weight)
- **SVM Prediction**: Support vector machine prediction (20% weight)
- **Recency-Frequency Prediction**: Historical pattern prediction (40% weight)
- **Weighted Combination**: Blend predictions using optimized ensemble weights

### 3. Prior Integration
- **Vendor Lookup**: Check if payee has strong historical patterns
- **Recency Weighting**: Apply recent transaction influence
- **Pattern Matching**: Look for exact or fuzzy payee matches

### 4. Threshold Application
- **Confidence Evaluation**: Compare prediction confidence to learned thresholds
- **Abstention Decision**: Decline to predict if confidence is too low
- **Final Prediction**: Return category with confidence score

## Performance Optimization

### For Small Datasets (< 1000 transactions)
- Enable `adaptive_splitting` for better rare category handling
- Use `final_retraining` to maximize use of available data
- Consider lowering `min_count` thresholds for priors

### For Large Datasets (> 5000 transactions)
- Can increase `n_folds` for more robust validation
- Increase `tfidf_max_features` for richer text representation
- Fine-tune `lgbm_weight` based on relative model performance

### For Many Categories (> 50 categories)
- Increase cross-feature hash space (`n_features`)
- Adjust threshold grids for better abstention behavior
- Consider category grouping strategies for rare categories

## Advanced Configuration

### Custom Payee Aliasing
```python
"alias": {
    "generic_noise": True,        # Remove common payment processing noise
    "custom_map_path": None       # Path to custom payee mapping file
}
```

The noise strings that get removed are defined in the `GENERIC_NOISE_PATTERNS` constant at the top of the `pxblendsc_strategy.py` file, where users can update the noise terms to customize payee normalization.

### Sampling Strategy
```python
"sampler": {
    "ros_cap_percentile": 60      # Random oversampling cap at 60th percentile for improving performance on categories with fewer samples
}
```

### Parallel Processing
```python
"parallel": {
    "n_jobs_cv": None,            # Use all cores for cross-validation
    "threads_per_fold": None      # Auto-detect threads per fold
}
```

## Monitoring and Maintenance

### Model Performance Indicators
- **Overall Accuracy**: Should be 75%+ on validation data. Accuracy means the percentage of transactions where the model's predicted category matches the actual category that was manually assigned.
- **Training vs Validation Gap**: Healthy models have <15% gap between training and validation accuracy. Large gaps indicate overfitting.
- **Abstention Rate**: Abstaining means the model chooses not to make a prediction when its confidence is too low, which may happen for categories with few samples or inconsistent categorization. This prevents incorrect categorizations but means some transactions won't get automatic predictions. During training, abstention helps improve the model's precision by avoiding low-confidence guesses. During prediction, the model will always give a result with a minimum confidence level of 10%.
- **Category Coverage**: All major categories should have reasonable F1 scores

### When to Retrain
- **Monthly**: For active users with frequent new transactions
- **Quarterly**: For moderate users with steady transaction patterns  
- **New Categories**: When you start using new budget categories
- **Performance Decay**: When prediction accuracy noticeably drops

## Troubleshooting Common Model Issues (what to try first)

### How to update the Model Configuration
> If you're not getting great results out of the box with the ML Model, you may need to tune some configurations to make it work better for your data. In general try to limit it to 1-2 changes at a time and retrain the model to see if it's improving. If it's not improving, undo the changes and try changing a different configuration. If it's improving, continue to tweak the configuration.

1. Go to the Settings page -> click on "Export Settings" button in the middle of the screen
    * **This contains ALL the app's configurations. Make a backup of this file just in case something gets corrupted so that you can restore this backup.** 
    * **DO NOT SHARE THIS FILE IN ITS ENTIRETY PUBLICALLY, IT MAY CONTAIN YOUR YNAB API AND GMAIL OAUTH CREDENTIALS**
2. Find the "ai.pxblendsc_config" item in the configuration file, this is what you will be changing (see Example Model Configuration below)
3. Make 1-2 changes based on the instructions below for the problem you're seeing with your model. 
    * **Make sure to keep all the escaped slashes like \" otherwise the app can't read the configuration correctly**
4. Go to the Settings page -> click on "Import Settings" button in the middle of the screen
5. Select your updated configuration for upload
6. Go to the ML Training page and train a new model to see how it performs

#### Example Model Configuration
```
    "ai.pxblendsc_config": {
      "key": "ai.pxblendsc_config",
      "type": "AI",
      "value": {
        "stringValue": "{\"columns\": {\"LABEL_COL\": \"category_name\", \"TEXT_COLS\": [\"payee_name\", \"memo\", \"account_name\"], \"NUM_COLS\": [\"amount\"], \"DATE_COL\": \"date\"}, \"random_state\": 42, \"cv\": {\"n_folds\": 3, \"force_folds\": true}, \"parallel\": {\"n_jobs_cv\": null, \"threads_per_fold\": null}, \"models\": {\"use_lgbm\": true, \"use_svm_blend\": true, \"svm_calibration\": \"sigmoid\", \"lgbm_weight\": 0.7, \"adaptive_splitting\": true, \"final_retraining\": true, \"lgbm_params\": {\"learning_rate\": 0.08, \"n_estimators\": 2000, \"subsample\": 0.85, \"colsample_bytree\": 0.85, \"reg_lambda\": 2.0, \"reg_alpha\": 0.5, \"min_child_samples\": 10, \"force_row_wise\": true, \"verbosity\": -1}}, \"features\": {\"tfidf_word_max_features\": 150000, \"tfidf_char_max_features\": 75000, \"hashed_cross\": {\"n_features\": 131072, \"payee_min_count\": 3, \"quantiles\": [0.15, 0.3, 0.5, 0.7, 0.85], \"sign_bins\": \"two_sided\", \"emit_unaries\": true, \"emit_pairs\": true, \"emit_triples\": true}}, \"sampler\": {\"ros_cap_percentile\": 60}, \"alias\": {\"generic_noise\": true, \"custom_map_path\": null}, \"priors\": {\"vendor\": {\"min_count\": 3, \"hard_override_share\": 0.85, \"beta\": 2.0}, \"recency\": {\"k_last\": 8, \"beta\": 1.5}, \"backoff\": {\"global\": 0.4}}, \"thresholds\": {\"mode\": \"learn\", \"global_grid\": [0.15, 0.2, 0.25, 0.3, 0.35], \"tail_support_max\": 8, \"tail_f1_max\": 0.25, \"tail_grid\": [0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]}}"
      },
      "description": "PXBlendSC-RF ML strategy configuration parameters"
    },
```


> **Note on trade-offs:** Improving accuracy/F1 often increases training time. The tips below aim for small, safe steps; monitor both quality *and* runtime after each change.

---

### Too many abstains?

**Lower global confidence threshold**

* Add smaller values to `thresholds.global_grid`.
* **Try:** add `0.10` and `0.12` (so it can choose lower than `0.15`).
* If still high, add `0.08`.

> ⚠️ Minimal runtime impact.

**Relax tail (rare-class) thresholds**

* Add lower options to `thresholds.tail_grid`.
* **Try:** prepend `0.30` and `0.325`.
* **Increase** `thresholds.tail_f1_max` from `0.25 → 0.30–0.35` (lets more tail predictions through).
* **Lower** `thresholds.tail_support_max` from `8 → 6` (fewer classes treated as ultra-rare/strict).

> ⚠️ Minimal runtime impact.

**Oversample minorities a bit more**

* `sampler.ros_cap_percentile: 60 → 70–80` (start with `75`).

> ⚠️ **May increase training time** (more effective sample mass per epoch).

**Example snippet**

```json
"thresholds": {
  "global_grid": [0.10, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35],
  "tail_grid": [0.30, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.55, 0.6],
  "tail_f1_max": 0.3,
  "tail_support_max": 6
},
"sampler": {"ros_cap_percentile": 75}
```

---

### High accuracy but **low macro-F1** (weak on rare classes)?

**Boost minority presence**

* `sampler.ros_cap_percentile: 60 → 80–85`.

> ⚠️ **Likely to increase training time**.

**Make text features catch more rare patterns**

* Bump **char TF-IDF** first (cheaper generalization):
  `tfidf_char_max_features: 75k → 100k–120k`.
* Optionally bump words if you have RAM/time:
  `tfidf_word_max_features: 150k → 180k–200k`.

> ⚠️ **Increases training time and memory** (feature space grows).

**Relax tail thresholds a notch**

* Add `0.30–0.325` to `thresholds.tail_grid`.

> ⚠️ Minimal runtime impact.

**Blend: give SVM a bit more voice (only if enabled)**

* `models.lgbm_weight: 0.7 → 0.6–0.65`.

> ⚠️ Little/no runtime impact.

**Example snippet**

```json
"sampler": {"ros_cap_percentile": 80},
"features": {
  "tfidf_char_max_features": 110000
},
"thresholds": {
  "tail_grid": [0.30, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5]
},
"models": {"lgbm_weight": 0.65}
```

---

### Accuracy just meh (on confident predictions)?

**Increase model capacity conservatively (LightGBM)**

* **Option A (more trees, slower):**
  `n_estimators: 2000 → 2500–3000` **and lower** `learning_rate: 0.08 → 0.06–0.05`.

  > ⚠️ **Increases training time** (more boosting steps).
* **Option B (similar time, a bit sharper):**
  Keep `n_estimators: 2000` and **lower** `learning_rate: 0.08 → 0.06`;
  if too slow, try `n_estimators: 2300` with `learning_rate: 0.07`.

  > ⚠️ Mild time increase.

**Regularization nudges**

* If **underfitting**: slightly **decrease** `reg_lambda: 2.0 → 1.0–1.5`.
* If **overfitting**: **increase** `reg_lambda: 2.0 → 3–5` and/or `reg_alpha: 0.5 → 0.8–1.2`.

> ⚠️ Minimal runtime impact.

**Reduce noisy crosses**

* `hashed_cross.emit_triples: true → false`.
* If vendor coverage is noisy: `payee_min_count: 3 → 5–8`.

> ⚠️ **Often speeds training** (fewer sparse features).

**Use priors for precision wins**

* `priors.vendor.min_count: 3 → 5–8`.
* Keep/raise `hard_override_share: 0.85 → 0.9` when vendor is reliable.

> ⚠️ Minimal runtime impact.

**Example snippet**

```json
"models": {
  "lgbm_params": {"n_estimators": 2800, "learning_rate": 0.06, "reg_lambda": 3.0, "reg_alpha": 0.8}
},
"features": {
  "hashed_cross": {"emit_triples": false, "payee_min_count": 6}
},
"priors": {"vendor": {"min_count": 6, "hard_override_share": 0.9}}
```

---

### Training slow?

**Cut expensive features first**

* `hashed_cross.emit_triples: true → false` *(biggest quick win)*.
* `tfidf_word_max_features: 150k → 80k–120k`.
* `tfidf_char_max_features: 75k → 40k–60k`.
* If memory tight: `hashed_cross.n_features: 131072 → 65536`.

> ⚡ **Speeds training** and lowers RAM.

**Shorten boosting**

* `n_estimators: 2000 → 1200–1600` and **increase** `learning_rate: 0.08 → 0.10–0.12`.

> ⚡ **Speeds training** (fewer trees).

**Skip the blend during iteration**

* `use_svm_blend: true → false` while tuning; re-enable later.

> ⚡ **Speeds training** (fewer models/calibration).

**Cross-validation: fewer folds**

* `cv.n_folds: 3 → 2` (faster).
* **About `force_folds`:**

  * `true` = *exactly* this many folds, which **drops categories** that don’t have enough samples per fold.
  * `false` = the system **auto-reduces fold count** to keep **all categories**.
  * For speed **and** to avoid dropping labels, set `force_folds: false`.

> ⚡ **Speeds training**; `force_folds: false` also preserves labels.

**Skip final retrain during prototyping**

* `final_retraining: true → false`.

> ⚡ **Speeds training**; only recommended to disable this when trying new configurations. Re-enable for your final model to improve performance.

**Example snippet**

```json
"features": {
  "tfidf_word_max_features": 100000,
  "tfidf_char_max_features": 50000,
  "hashed_cross": {"emit_triples": false, "n_features": 65536}
},
"models": {
  "use_svm_blend": false,
  "final_retraining": false,
  "lgbm_params": {"n_estimators": 1400, "learning_rate": 0.1}
},
"cv": {"n_folds": 2, "force_folds": false}
```

---

### Tuning Configurations Using an LLM
If you don't want to do the manual steps above, and are comfortable using LLMs (e.g. chatGPT), you can fill in the prompt below with the following information, and paste it into a chat to have the AI do the adjustments for you:
1. Model Configuration ("ai.pxblendsc_config" shown above)
2. Model Performance Metrics - go to the Model Performance page for a Model and copy paste everything on the Validation tab (accuray, macro F1, balanced accuracy, abstain rate, samples, etc)
3. Copy paste or upload the whole [pxblendsc_strategy.py](pxblendsc_strategy.py) file

#### Prompt:
># Prompt: PXBlendSC Auto-Tuning Assistant
>
>You are helping me tune a transaction-categorization ML pipeline called **PXBlendSC-RF**. I >will provide:
>
>* The current **Model Configuration** JSON (`ai.pxblendsc_config`),
>* **Validation Performance Metrics** (accuracy, macro-F1, balanced accuracy, abstain >rate, sample counts, etc.),
>* The **pxblendsc\_strategy.py** code.
>
>**Your job:** Recommend **small, concrete configuration edits** to improve results, >using safe first-step values. Return changes as a **single JSON object** in the exact >escaped-string format shown below so I can paste it back into my app. Keep explanations >concise (no chain-of-thought).
>
>---
>
>## Inputs
>
>**Model Configuration JSON:**
>
>```json
>{PASTE_CURRENT_ai.pxblendsc_config_HERE}
>```
>
>**Validation Performance Metrics (paste all from the UI):**
>
>```
>ACCURACY: ...
>MACRO_F1: ...
>BALANCED_ACCURACY: ...
>ABSTAIN_RATE: ...
>TOTAL_SAMPLES: ...
>CATEGORIES: ...
>NOTES: ...
>```
>
>**pxblendsc\_strategy.py (paste or attach):**
>
>```
>PASTE_FILE_CONTENT_OR_ATTACH_HERE
>```
>
>**Context Instructions:**
>Copy-paste everything from the README section starting at **“### Too many abstains?”** >through and including **“### Tuning Configurations Using an LLM”** so you can use those >guidelines to decide what to change.
>
>---
>
>## Rules & Constraints
>
>* Use only fields that already exist in the provided config/code.
>* Prefer **small, safe increments** per the pasted guidelines.
>* Respect cross-validation behavior:
>
>  * `force_folds: true` = use exactly `n_folds`, which **drops categories** that lack enough samples per fold.
>  * `force_folds: false` = **auto-reduce** the fold count to **keep all categories**.
>* Avoid verbose reasoning; be implementation-ready.
>
>---
>
>## Output Format (STRICT)
>
>Return **exactly one JSON object** in this structure, where `value.stringValue` is a >**fully-updated, compact, JSON-escaped string** of the *entire* configuration after your >edits. Keep all unrelated fields unchanged. Use double quotes and escape characters >exactly as valid JSON.
>
>```json
>{
>  "ai.pxblendsc_config": {
>    "key": "ai.pxblendsc_config",
>    "type": "AI",
>    "value": {
>      "stringValue": "{\"columns\": {\"LABEL_COL\": \"category_name\", \"TEXT_COLS\": [\"payee_name\", \"memo\", \"account_name\"], \"NUM_COLS\": [\"amount\"], \"DATE_COL\": \"date\"}, \"random_state\": 42, \"cv\": {\"n_folds\": 3, \"force_folds\": true}, \"parallel\": {\"n_jobs_cv\": null, \"threads_per_fold\": null}, \"models\": {\"use_lgbm\": true, \"use_svm_blend\": true, \"svm_calibration\": \"sigmoid\", \"lgbm_weight\": 0.7, \"adaptive_splitting\": true, \"final_retraining\": true, \"lgbm_params\": {\"learning_rate\": 0.08, \"n_estimators\": 2000, \"subsample\": 0.85, \"colsample_bytree\": 0.85, \"reg_lambda\": 2.0, \"reg_alpha\": 0.5, \"min_child_samples\": 10, \"force_row_wise\": true, \"verbosity\": -1}}, \"features\": {\"tfidf_word_max_features\": 150000, \"tfidf_char_max_features\": 75000, \"hashed_cross\": {\"n_features\": 131072, \"payee_min_count\": 3, \"quantiles\": [0.15, 0.3, 0.5, 0.7, 0.85], \"sign_bins\": \"two_sided\", \"emit_unaries\": true, \"emit_pairs\": true, \"emit_triples\": true}}, \"sampler\": {\"ros_cap_percentile\": 60}, \"alias\": {\"generic_noise\": true, \"custom_map_path\": null}, \"priors\": {\"vendor\": {\"min_count\": 3, \"hard_override_share\": 0.85, \"beta\": 2.0}, \"recency\": {\"k_last\": 8, \"beta\": 1.5}, \"backoff\": {\"global\": 0.4}}, \"thresholds\": {\"mode\": \"learn\", \"global_grid\": [0.15, 0.2, 0.25, 0.3, 0.35], \"tail_support_max\": 8, \"tail_f1_max\": 0.25, \"tail_grid\": [0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]}}"
>    },
>    "description": "PXBlendSC-RF ML strategy configuration parameters"
>  }
>}
>```
>
>**What you must do to produce the output:**
>
>1. Parse the provided `value.stringValue` JSON string into an internal JSON object.
>2. Apply minimal edits based on the metrics and the pasted guideline sections.
>3. Serialize the **entire updated config** back to a **compact JSON string** and escape >it as a JSON value under `value.stringValue`.
>4. Return only the one JSON object above with the updated `stringValue`. No extra keys, >text, or commentary.

Once you get a configuration back, follow the steps in "How to update the Model Configuration" to test it.