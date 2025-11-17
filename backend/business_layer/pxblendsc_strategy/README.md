# PXBlendSC-RF Strategy Module

This module implements the PXBlendSC-RF (Prefix-Blended Stacked Classifier with Recency-Frequency) machine learning strategy for transaction categorization.

## Module Structure

The implementation is organized into logical components:

### Core Components

- **`strategy.py`** - Main strategy class (`PXBlendSCStrategy`) that implements the ML strategy interface
- **`bundle.py`** - Model bundle class for saving/loading trained models
- **`training.py`** - Main training orchestration and cross-validation logic
- **`training_utils.py`** - Training utility functions (CV fold determination, filtering, metrics)

### Feature Engineering

- **`features.py`** - Feature engineering classes:
  - `AddCombinedText` - Text combination and normalization
  - `AmountFeaturizer` - Amount-based features with per-payee percentiles
  - `DateFeaturizer` - Date-based temporal features
  - `CrossFeaturizer` - Cross-feature generation (payee × amount × date)
- **`pipeline.py`** - Feature pipeline construction
- **`text_processing.py`** - Text normalization and cleaning

### Models and Memory

- **`models.py`** - Model classes:
  - `RecencyFrequencyModel` - Payee history-based predictions
  - `CancellationCallback` - Async cancellation support
  - `CancellableEstimator` - Wrapper for sklearn estimators
- **`memory.py`** - Memory building functions for priors:
  - Payee memory
  - Account memory
  - Payee+amount pattern memory

### Thresholds and Predictions

- **`thresholds.py`** - Threshold calculation and application:
  - Global thresholds
  - Per-prefix thresholds
  - Per-class (tail) thresholds
  - Prefix and prior application

### Utilities

- **`utils.py`** - Common utility functions:
  - Cancellation checking
  - Label prefix extraction
  - Probability padding
  - Type conversion
  - DataFrame preparation
  - ROS strategy
- **`constants.py`** - Pattern constants for text normalization

## Architecture

The strategy combines multiple techniques:

1. **Model Blending**: LightGBM + SVM + Recency-Frequency models
2. **Prefix Classification**: Hierarchical category prediction
3. **Memory Priors**: Payee and account history patterns
4. **Adaptive Thresholds**: Global, per-prefix, and per-class confidence thresholds
5. **Feature Engineering**: Text (TF-IDF), amount, date, and cross-features

## Usage

```python
from backend.business_layer.pxblendsc_strategy import PXBlendSCStrategy

# The strategy is automatically registered and used by the ML engine
strategy = PXBlendSCStrategy()
```

## Backward Compatibility

The original `pxblendsc_strategy.py` file now serves as a compatibility shim that imports from this module, ensuring existing code continues to work without changes.
