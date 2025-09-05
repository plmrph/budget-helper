# Machine Learning Strategy: PXBlendSC

The Budget Helper uses the **PXBlendSC** (Prefix-eXamination Blend with Semantic Classification) ML strategy for intelligent transaction categorization. This advanced ensemble approach combines multiple machine learning techniques with domain-specific financial transaction understanding.

## Overview

PXBlendSC is a sophisticated ML pipeline that learns patterns from your historical transaction data to predict categories for new transactions. It combines:

- **Ensemble Learning**: LightGBM + SVM models with optimized blending
- **Advanced Feature Engineering**: Text, numerical, temporal, and cross-feature extraction  
- **Prefix Classification**: Pattern matching for vendor/payee recognition
- **Memory Priors**: Learning from recent transaction patterns
- **Adaptive Thresholds**: Smart abstention when confidence is low

## Feature Engineering

### Text Features
The system extracts multiple types of text features from transaction descriptions:

#### TF-IDF Features
- **Word-level TF-IDF**: Captures important words and phrases in payee names and memos
- **Character-level TF-IDF**: Learns character patterns for vendor name variations
- **Max Features**: 150,000 word features + 75,000 character features for comprehensive coverage

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
    "use_lgbm": True,           # Enable LightGBM (gradient boosting)
    "use_svm_blend": True,      # Enable SVM component
    "lgbm_weight": 0.7,         # LightGBM weight in ensemble (70%)
    "adaptive_splitting": True, # Adaptive train/test split for small classes
    "final_retraining": True    # Retrain on all data after evaluation
}
```

#### Key Parameters:
- **n_folds**: Higher folds will typically improve accuracy overall, but will end up dropping categories that don't have enough samples
- **adaptive_splitting**: Improves accuracy on smaller categories, at the risk of having fewer test samples
- **final_retraining**: Increases the overall training time, but helps with accuracy especially on smaller datasets or categories with fewer samples
- **lgbm_weight**: Controls ensemble balance (0.7 = 70% LightGBM, 30% SVM)

### Feature Engineering (features)
```python
"features": {
    "tfidf_word_max_features": 150000,             # Maximum word-level features
    "tfidf_char_max_features": 75000,              # Maximum character-level features
    "hashed_cross": {
        "n_features": 131072,                      # Hash space for cross-features
        "payee_min_count": 3,                      # Minimum payee frequency for features
        "quantiles": [0.15, 0.3, 0.5, 0.7, 0.85],  # Amount binning
        "emit_unaries": True,                      # Single field features
        "emit_pairs": True,                        # Two-field combinations
        "emit_triples": True                       # Three-field combinations
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
1. **Cross-Validation**: 3-fold validation for robust performance estimation
2. **Ensemble Training**: Train both LightGBM and SVM components
3. **Threshold Learning**: Learn optimal confidence thresholds per category
4. **Final Retraining**: Train final model on all available data

### 4. Model Evaluation
- **Accuracy Metrics**: Overall accuracy, per-category F1 scores
- **Abstention Analysis**: Performance when model chooses not to predict
- **Confusion Matrix**: Understanding common misclassifications

## Prediction Process

### 1. Feature Extraction
New transactions go through the same feature engineering pipeline as training data.

### 2. Ensemble Prediction
- **LightGBM Prediction**: Gradient boosting model prediction
- **SVM Prediction**: Support vector machine prediction  
- **Weighted Combination**: Blend predictions using learned weights

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
- **Overall Accuracy**: Should be 85%+ on validation data. Accuracy means the percentage of transactions where the model's predicted category matches the actual category that was manually assigned.
- **Abstention Rate**: Healthy models abstain on 15-25% of predictions. Abstaining means the model chooses not to make a prediction when its confidence is too low. This prevents incorrect categorizations but means some transactions won't get automatic predictions. During training, abstention helps improve the model's precision by avoiding low-confidence guesses. During prediction, the model will always give a result with a minimum confidence level of 10%.
- **Category Coverage**: All major categories should have reasonable F1 scores

### When to Retrain
- **Monthly**: For active users with frequent new transactions
- **Quarterly**: For moderate users with steady transaction patterns  
- **New Categories**: When you start using new budget categories
- **Performance Decay**: When prediction accuracy noticeably drops

### Troubleshooting Common Issues
- **Low Accuracy**: Increase dataset size, check data quality, verify category consistency
- **High Abstention**: Lower threshold grids, increase training data
- **Memory Issues**: Reduce max_features, decrease n_features for cross-features
- **Slow Training**: Reduce n_estimators, disable final_retraining temporarily