"""
Reasoning Quality Score (RQS) Evaluator

Formal evaluation framework for "thinking" capability in small models.
Implements 5 metrics: Decomposability, Self-Correction, Reasoning Coherence,
Explanation Faithfulness, and Graceful Degradation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import pearsonr, entropy
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RQSResult:
    """Container for Reasoning Quality Score results."""
    RQS: float
    Decomposability: float
    Self_Correction_F1: float
    Reasoning_Coherence: float
    Explanation_Faithfulness: float
    Graceful_Degradation: float
    
    def __repr__(self):
        return f"""
╔══════════════════════════════════════════════════╗
║       REASONING QUALITY SCORE (RQS)              ║
╠══════════════════════════════════════════════════╣
║  Overall RQS:              {self.RQS:.4f}              ║
╠══════════════════════════════════════════════════╣
║  Decomposability (D):      {self.Decomposability:.4f}              ║
║  Self-Correction (SC):     {self.Self_Correction_F1:.4f}              ║
║  Reasoning Coherence (RC): {self.Reasoning_Coherence:.4f}              ║
║  Explanation Faith. (EF):  {self.Explanation_Faithfulness:.4f}              ║
║  Graceful Degrad. (GD):    {self.Graceful_Degradation:.4f}              ║
╚══════════════════════════════════════════════════╝
"""

    def to_dict(self) -> Dict:
        return {
            'RQS': self.RQS,
            'Decomposability': self.Decomposability,
            'Self_Correction_F1': self.Self_Correction_F1,
            'Reasoning_Coherence': self.Reasoning_Coherence,
            'Explanation_Faithfulness': self.Explanation_Faithfulness,
            'Graceful_Degradation': self.Graceful_Degradation
        }


class RQSEvaluator:
    """
    Evaluates Reasoning Quality Score for thinking pipelines.
    
    Usage:
        evaluator = RQSEvaluator()
        result = evaluator.evaluate(pipeline, X_test, y_test)
        print(result)
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize evaluator with optional custom weights.
        
        Args:
            weights: Dict with keys D, SC, RC, EF, GD (must sum to 1.0)
        """
        self.weights = weights or {
            'D': 0.20,   # Decomposability
            'SC': 0.25,  # Self-Correction
            'RC': 0.20,  # Reasoning Coherence
            'EF': 0.25,  # Explanation Faithfulness
            'GD': 0.10   # Graceful Degradation (inverted)
        }
        
    def evaluate(
        self, 
        pipeline, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        n_perturbations: int = 100
    ) -> RQSResult:
        """
        Compute full RQS evaluation.
        
        Args:
            pipeline: ThinkingXGBoostPipeline with predict_with_reasoning method
            X_test: Test features
            y_test: Test labels
            n_perturbations: Number of samples for faithfulness test
            
        Returns:
            RQSResult with all metrics
        """
        # Get predictions and reasoning traces
        preds, traces = pipeline.predict_with_reasoning(X_test)
        
        # Compute each metric
        D = self._compute_decomposability(traces, preds, pipeline)
        SC = self._compute_self_correction(traces, preds, y_test)
        RC = self._compute_reasoning_coherence(traces, preds, pipeline)
        EF = self._compute_explanation_faithfulness(pipeline, X_test, n_perturbations)
        GD = self._compute_graceful_degradation(traces, preds, y_test, pipeline)
        
        # Composite score (GD is inverted - lower is better)
        RQS = (
            self.weights['D'] * D +
            self.weights['SC'] * SC +
            self.weights['RC'] * RC +
            self.weights['EF'] * EF +
            self.weights['GD'] * (1 - GD)
        )
        
        return RQSResult(
            RQS=RQS,
            Decomposability=D,
            Self_Correction_F1=SC,
            Reasoning_Coherence=RC,
            Explanation_Faithfulness=EF,
            Graceful_Degradation=GD
        )
    
    def _compute_decomposability(
        self, 
        traces: pd.DataFrame, 
        final_preds: np.ndarray,
        pipeline
    ) -> float:
        """
        Measure how much of final prediction is explained by head outputs.
        
        D = 1 - (unexplained_variance / total_variance)
        """
        # Get head scores
        head_cols = [c for c in traces.columns if c.endswith('_score')]
        head_scores = traces[head_cols].values
        
        # Simple linear combination of heads as "explained" prediction
        # Use mean of heads as baseline prediction
        head_mean_pred = head_scores.mean(axis=1)
        
        # Variance explained
        total_var = np.var(final_preds)
        if total_var == 0:
            return 1.0  # Perfect prediction, trivially decomposable
            
        residual_var = np.var(final_preds - head_mean_pred)
        D = 1 - (residual_var / total_var)
        
        return np.clip(D, 0, 1)
    
    def _compute_self_correction(
        self,
        traces: pd.DataFrame,
        preds: np.ndarray,
        y_test: pd.Series
    ) -> float:
        """
        Measure critic's ability to identify errors.
        
        SC_F1 = F1 score of critic predicting actual errors
        """
        # Actual errors
        y_reset = y_test.reset_index(drop=True).values
        actual_errors = ((preds > 0.5).astype(int) != y_reset).astype(int)
        
        # Critic's predictions (samples flagged for refinement)
        critic_flags = traces['needs_refinement'].values
        
        # If no errors or no flags, return edge case values
        if actual_errors.sum() == 0:
            return 1.0  # No errors to detect = perfect
        if critic_flags.sum() == 0:
            return 0.0  # Critic flagged nothing but there were errors
            
        # F1 score
        try:
            f1 = f1_score(actual_errors, critic_flags, zero_division=0)
        except:
            f1 = 0.0
            
        return f1
    
    def _compute_reasoning_coherence(
        self,
        traces: pd.DataFrame,
        final_preds: np.ndarray,
        pipeline
    ) -> float:
        """
        Measure correlation between head outputs and final prediction.
        
        RC = mean |correlation| across all heads
        """
        head_cols = [c for c in traces.columns if c.endswith('_score')]
        
        correlations = []
        for col in head_cols:
            head_scores = traces[col].values
            if np.std(head_scores) > 0 and np.std(final_preds) > 0:
                corr, _ = pearsonr(head_scores, final_preds)
                correlations.append(abs(corr))
        
        if not correlations:
            return 0.0
            
        return np.mean(correlations)
    
    def _compute_explanation_faithfulness(
        self,
        pipeline,
        X_test: pd.DataFrame,
        n_samples: int = 100
    ) -> float:
        """
        Measure if perturbing head inputs changes final output proportionally.
        
        EF = correlation between Δ_head and Δ_final across perturbations
        """
        # Sample subset for efficiency
        n_samples = min(n_samples, len(X_test))
        X_sample = X_test.sample(n=n_samples, random_state=42)
        
        # Get baseline predictions
        baseline_preds, baseline_traces = pipeline.predict_with_reasoning(X_sample)
        
        faithfulness_scores = []
        
        for head_name, head in pipeline.reasoning_heads.items():
            head_deltas = []
            final_deltas = []
            
            # Perturb each feature in this head
            for feature in head.features:
                if feature not in X_sample.columns:
                    continue
                    
                # Create perturbed version
                X_perturbed = X_sample.copy()
                feature_std = X_perturbed[feature].std()
                if feature_std == 0:
                    continue
                    
                np.random.seed(42 + hash(feature) % 1000)  # Deterministic perturbations
                X_perturbed[feature] = X_perturbed[feature] + np.random.normal(0, feature_std * 0.5, n_samples)
                
                # Get new predictions
                new_preds, new_traces = pipeline.predict_with_reasoning(X_perturbed)
                
                # Compute deltas
                head_col = f'head_{head_name}_score'
                if head_col in baseline_traces.columns and head_col in new_traces.columns:
                    delta_head = np.abs(new_traces[head_col].values - baseline_traces[head_col].values)
                    delta_final = np.abs(new_preds - baseline_preds)
                    
                    head_deltas.extend(delta_head)
                    final_deltas.extend(delta_final)
            
            # Correlation for this head
            if len(head_deltas) > 10 and np.std(head_deltas) > 0 and np.std(final_deltas) > 0:
                corr, _ = pearsonr(head_deltas, final_deltas)
                faithfulness_scores.append(max(0, corr))  # Only positive correlations count
        
        if not faithfulness_scores:
            return 0.5  # Default middle value
            
        return np.mean(faithfulness_scores)
    
    def _compute_graceful_degradation(
        self,
        traces: pd.DataFrame,
        preds: np.ndarray,
        y_test: pd.Series,
        pipeline
    ) -> float:
        """
        Measure if errors are concentrated in specific heads (easier to debug).
        
        GD = entropy of error distribution across heads
        Lower = errors concentrated (better for debugging)
        """
        y_reset = y_test.reset_index(drop=True).values
        errors_mask = ((preds > 0.5).astype(int) != y_reset)
        
        if errors_mask.sum() == 0:
            return 0.0  # No errors = perfect graceful degradation
        
        # For each error, find which head was most "wrong"
        head_cols = [c for c in traces.columns if c.endswith('_score')]
        error_traces = traces.loc[errors_mask, head_cols]
        
        # Count which head had highest confidence on errors
        head_error_counts = {}
        for col in head_cols:
            # High score on false positives or low score on false negatives
            head_name = col.replace('head_', '').replace('_score', '')
            
            # Simplified: count heads with extreme predictions on errors
            extreme_mask = (error_traces[col] > 0.8) | (error_traces[col] < 0.2)
            head_error_counts[head_name] = extreme_mask.sum()
        
        # Compute entropy of distribution
        counts = np.array(list(head_error_counts.values()))
        if counts.sum() == 0:
            return 0.5
            
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # Remove zeros for entropy calc
        
        # Normalize entropy to [0, 1]
        max_entropy = np.log(len(head_cols))
        if max_entropy == 0:
            return 0.0
            
        GD = entropy(probs) / max_entropy
        
        return GD


def quick_evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> RQSResult:
    """Convenience function for quick evaluation."""
    evaluator = RQSEvaluator()
    return evaluator.evaluate(pipeline, X_test, y_test)


if __name__ == "__main__":
    print("RQS Evaluator module loaded.")
    print("Use: from rqs_evaluator import RQSEvaluator, quick_evaluate")
