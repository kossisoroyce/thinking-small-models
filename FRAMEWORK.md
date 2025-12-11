# Thinking Small Models: A Reasoning Quality Framework

## Abstract

We propose a formal evaluation framework for **reasoning transparency** in small models (XGBoost, Random Forests, MLPs). While large language models are increasingly evaluated on chain-of-thought quality, traditional ML models lack equivalent metrics. This framework introduces five quantifiable dimensions for measuring "thinking" capability in non-LLM models.

---

## 1. Problem Statement

Current small model evaluation:
```
Score = f(accuracy, latency, model_size, calibration)
```

Missing dimension: **How well can we understand WHY the model made a decision?**

LLMs have:
- Chain-of-thought evaluation
- Reasoning trace analysis
- Self-consistency metrics

Small models have: **nothing equivalent**.

---

## 2. Proposed Framework: Reasoning Quality Score (RQS)

### 2.1 Decomposability (D)

**Definition**: The degree to which a model's prediction can be attributed to interpretable sub-components.

**Metric**:
```
D = 1 - (Var(ε) / Var(ŷ))

where:
  ŷ = final prediction
  ŷ_heads = Σ(w_i * head_i predictions)
  ε = ŷ - ŷ_heads (unexplained variance)
```

**Interpretation**:
- D = 1.0: Perfect decomposition (final prediction fully explained by heads)
- D = 0.0: Heads explain nothing (black box)

**Benchmark target**: D ≥ 0.7

---

### 2.2 Self-Correction Capability (SC)

**Definition**: The model's ability to identify and correct its own errors.

**Metrics**:
```
SC_precision = TP / (TP + FP)  # Of flagged samples, how many were actual errors?
SC_recall = TP / (TP + FN)     # Of actual errors, how many were flagged?
SC_F1 = 2 * (precision * recall) / (precision + recall)

where:
  TP = correctly flagged errors
  FP = incorrectly flagged (was actually correct)
  FN = missed errors (not flagged but wrong)
```

**Interpretation**:
- SC_F1 = 1.0: Perfect error detection
- SC_F1 = 0.0: Critic is useless

**Benchmark target**: SC_F1 ≥ 0.3

---

### 2.3 Reasoning Coherence (RC)

**Definition**: Consistency between intermediate reasoning signals and final decision.

**Metric**:
```
RC = mean(|corr(head_i_score, final_pred)|) for all heads

Alternative (disagreement-based):
RC_disagree = 1 - mean(|head_signal - final_decision|)
```

**Interpretation**:
- RC = 1.0: Heads perfectly predict final decision
- RC = 0.0: Heads are disconnected from final decision

**Benchmark target**: RC ≥ 0.5

---

### 2.4 Explanation Faithfulness (EF)

**Definition**: Do the stated reasons actually influence the prediction?

**Metric** (perturbation-based):
```
For each head h:
  1. Perturb input features for head h
  2. Measure Δ_h = change in head_h score
  3. Measure Δ_final = change in final prediction
  
EF_h = corr(Δ_h, Δ_final) across perturbations
EF = mean(EF_h) for all heads
```

**Interpretation**:
- EF = 1.0: Changing head inputs changes final output proportionally
- EF = 0.0: Head explanations are decorative (not causal)

**Benchmark target**: EF ≥ 0.6

---

### 2.5 Graceful Degradation (GD)

**Definition**: When the model is wrong, are errors interpretable?

**Metric**:
```
For each error e:
  1. Identify which heads had highest confidence
  2. Check if error is "concentrated" in specific dimensions

GD = entropy(error_distribution_across_heads) / max_entropy

Alternative:
GD = fraction of errors where exactly 1-2 heads were "wrong"
```

**Interpretation**:
- GD = 1.0: Errors spread across all heads (hard to diagnose)
- GD = 0.0: Errors concentrated (easy to identify failure mode)

**Note**: Lower GD is actually BETTER for debugging.

**Benchmark target**: GD ≤ 0.5

---

## 3. Composite Reasoning Quality Score

```
RQS = α₁D + α₂SC + α₃RC + α₄EF + α₅(1-GD)

Default weights: α = [0.2, 0.25, 0.2, 0.25, 0.1]
```

**Score interpretation**:
| RQS Range | Interpretation |
|-----------|----------------|
| 0.8 - 1.0 | Excellent reasoning transparency |
| 0.6 - 0.8 | Good reasoning transparency |
| 0.4 - 0.6 | Moderate reasoning transparency |
| 0.0 - 0.4 | Poor reasoning transparency (black box) |

---

## 4. Evaluation Protocol

### 4.1 Required Components

A model claiming "thinking" capability must provide:

1. **Reasoning heads**: K interpretable sub-models
2. **Aggregator**: Combines head outputs
3. **Critic**: Estimates prediction uncertainty
4. **Reasoning trace**: Human-readable output format

### 4.2 Test Procedure

```python
def evaluate_reasoning_quality(pipeline, X_test, y_test):
    """
    Returns RQS and component scores.
    """
    # Get predictions and traces
    preds, traces = pipeline.predict_with_reasoning(X_test)
    
    # Compute each metric
    D = compute_decomposability(traces, preds)
    SC = compute_self_correction(pipeline, X_test, y_test)
    RC = compute_reasoning_coherence(traces, preds)
    EF = compute_explanation_faithfulness(pipeline, X_test)
    GD = compute_graceful_degradation(traces, preds, y_test)
    
    # Composite score
    RQS = 0.2*D + 0.25*SC + 0.2*RC + 0.25*EF + 0.1*(1-GD)
    
    return {
        'RQS': RQS,
        'Decomposability': D,
        'Self_Correction': SC,
        'Reasoning_Coherence': RC,
        'Explanation_Faithfulness': EF,
        'Graceful_Degradation': GD
    }
```

---

## 5. Comparison to LLM Evaluation

| Aspect | LLM Evaluation | RQS (Small Models) |
|--------|---------------|-------------------|
| Chain of thought | Text coherence | Head signal coherence |
| Self-consistency | Multiple samples agree | Critic accuracy |
| Faithfulness | Explanations match behavior | Perturbation tests |
| Decomposition | Step-by-step breakdown | Head contributions |

---

## 6. Limitations & Future Work

1. **Feature grouping dependency**: RQS assumes features can be meaningfully grouped
2. **Computational overhead**: Faithfulness requires perturbation tests
3. **Domain specificity**: Weights may need tuning per domain
4. **Causal vs correlational**: Current metrics are correlational

Future extensions:
- Causal faithfulness metrics
- Temporal reasoning evaluation
- Multi-task reasoning transfer

---

## Citation

```bibtex
@article{thinking_small_models_2024,
  title={Thinking Small Models: A Reasoning Quality Framework for Non-LLM Models},
  author={...},
  year={2024}
}
```
