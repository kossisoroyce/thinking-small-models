# Thinking Small Models

**Applying LLM reasoning patterns to XGBoost for interpretable fraud detection.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project demonstrates how to make traditional ML models "think" by applying concepts from large language model (LLM) reasoning to XGBoost:

- **Multi-step logic**: Decompose predictions into specialized reasoning heads
- **Self-correction**: A critic model that detects uncertain predictions and triggers refinement
- **Explainability**: Full reasoning trace for every prediction

We also introduce the **Reasoning Quality Score (RQS)** framework—a novel set of metrics for evaluating "thinking" capability in non-LLM models.

## Architecture

```
Input Features
     │
     ▼
┌─────────────────┐
│  Stage 1: XGB   │ → Per-dimension risk scores
│  Reasoning Heads│    (amount, velocity, location, merchant, device, time)
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Stage 2: Hybrid│ → Blended prediction
│  Aggregator     │    (60% weighted avg + 40% XGBoost interactions)
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Stage 3: XGB   │ → "Should I reconsider?"
│  Critic         │    (error detection model)
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Stage 4: XGB   │ → Refined prediction
│  Refiner        │    (only if critic flags uncertainty)
└─────────────────┘
     │
     ▼
Final Decision + Reasoning Trace
```

## Results

| Metric | Baseline XGBoost | Thinking Pipeline |
|--------|-----------------|-------------------|
| ROC-AUC | 0.994 | 0.976 |
| Interpretable | ❌ | ✓ |
| Self-correction | ❌ | ✓ |
| Reasoning trace | ❌ | ✓ |

### Reasoning Quality Score (RQS)

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Overall RQS** | 0.50 | >0.60 | |
| Decomposability | 0.77 | >0.70 | ✓ |
| Self-Correction | 0.33 | >0.30 | ✓ |
| Coherence | 0.57 | >0.50 | ✓ |
| Faithfulness | 0.57 | >0.60 | ✗ |
| Graceful Degradation | 0.91 | <0.50 | ✗ |

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/thinking-small-models.git
cd thinking-small-models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from thinking_pipeline_best import train_pipeline, ThinkingXGBoostPipeline
from data_generator import get_feature_groups
import pandas as pd

# Load data
df = pd.read_csv('fraud_dataset.csv')
X = df[get_all_features()]
y = df['is_fraud']

# Train pipeline
pipeline = train_pipeline(X_train, y_train, get_feature_groups())

# Predict with reasoning
predictions, reasoning_trace = pipeline.predict_with_reasoning(X_test)

# Explain a single prediction
print(pipeline.explain(X_test.iloc[[0]]))
```

**Output:**
```
<REASONING>
  amount       risk: 0.374
  velocity     risk: 0.760
  merchant     risk: 0.715
  location     risk: 0.596
  device       risk: 0.429
  time         risk: 0.484
  --------------------
  Weighted avg:     0.568
  XGBoost pred:     0.815
  Blended:          0.666
  Critic score:     0.669
  [!] REFINEMENT TRIGGERED
</REASONING>

<SOLUTION>
  Probability: 0.054
  Decision:    LEGITIMATE
</SOLUTION>
```

## Project Structure

```
thinking-small-models/
├── README.md                    # This file
├── FRAMEWORK.md                 # RQS framework specification
├── requirements.txt             # Dependencies (pinned)
├── data_generator.py            # Synthetic dataset generator
├── thinking_pipeline_best.py    # Best pipeline implementation
├── rqs_evaluator.py             # RQS evaluation framework
├── thinking_xgboost_fraud.ipynb # Demo notebook
└── fraud_dataset.csv            # Generated dataset (30K samples)
```

## Key Files

| File | Description |
|------|-------------|
| `thinking_pipeline_best.py` | Complete 4-stage thinking pipeline |
| `rqs_evaluator.py` | Reasoning Quality Score evaluator |
| `data_generator.py` | Synthetic fraud dataset generator |
| `FRAMEWORK.md` | Formal RQS metric definitions |

## Reasoning Quality Score (RQS) Framework

We introduce 5 metrics for evaluating "reasoning" in small models:

1. **Decomposability (D)**: Can the prediction be attributed to interpretable sub-components?
2. **Self-Correction (SC)**: Can the model identify its own errors?
3. **Reasoning Coherence (RC)**: Do intermediate signals align with the final decision?
4. **Explanation Faithfulness (EF)**: Do stated reasons actually influence the prediction?
5. **Graceful Degradation (GD)**: Are errors concentrated in identifiable dimensions?

See `FRAMEWORK.md` for full mathematical definitions.

## Usage

### Generate Dataset

```bash
python data_generator.py
```

### Run Notebook

```bash
jupyter notebook thinking_xgboost_fraud.ipynb
```

### Evaluate RQS

```python
from rqs_evaluator import RQSEvaluator

evaluator = RQSEvaluator()
result = evaluator.evaluate(pipeline, X_test, y_test)
print(result)
```

## Contributing

Contributions welcome! Areas of interest:

- Improving Faithfulness and Graceful Degradation metrics
- Applying to other domains (healthcare, credit risk)
- Extending RQS framework to other model types

## Citation

```bibtex
@article{thinking_small_models_2024,
  title={Thinking Small Models: Multi-Stage Reasoning for Interpretable Machine Learning},
  author={Your Name},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.
