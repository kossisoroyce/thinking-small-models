"""
Thinking XGBoost Pipeline - BEST VERSION (V8)

RQS Score: 0.478 (improved from 0.368 baseline, +30%)

Targets Met:
  ✓ Decomposability: 0.77 (target: >0.70)
  ✓ Self-Correction: 0.33 (target: >0.30)  
  ✓ Coherence: 0.57 (target: >0.50)
  ✗ Faithfulness: 0.48 (target: >0.60)
  ✗ Graceful Degradation: 0.91 (target: <0.50)

Key Innovation: Hybrid aggregation blending weighted average (faithful) 
with XGBoost interactions (powerful).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import xgboost as xgb
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ReasoningHead:
    """Reasoning head for one feature dimension."""
    name: str
    features: List[str]
    model: Optional[xgb.XGBClassifier] = None
    weight: float = 1.0
    
    def train(self, X: pd.DataFrame, y: pd.Series, scale_weight: float):
        self.model = xgb.XGBClassifier(
            n_estimators=60, max_depth=5, learning_rate=0.1,
            scale_pos_weight=scale_weight, random_state=42, eval_metric='auc'
        )
        self.model.fit(X[self.features], y)
        self.weight = roc_auc_score(y, self.predict_proba(X))
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]
    
    def get_reasoning(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_proba(X)
        return pd.DataFrame({
            f'head_{self.name}_score': p,
            f'head_{self.name}_signal': (p > 0.5).astype(int)
        })


def get_stage1_outputs(X: pd.DataFrame, heads: Dict[str, ReasoningHead]) -> pd.DataFrame:
    return pd.concat([h.get_reasoning(X) for h in heads.values()], axis=1)


def weighted_average(s1: pd.DataFrame, heads: Dict[str, ReasoningHead]) -> np.ndarray:
    """Direct weighted average of head scores."""
    result = np.zeros(len(s1))
    for name, head in heads.items():
        result += head.weight * s1[f'head_{name}_score'].values
    return result


def build_aggregator_features(s1: pd.DataFrame) -> pd.DataFrame:
    """Build XGBoost aggregator input with interactions."""
    X_agg = s1.copy()
    score_cols = [c for c in s1.columns if '_score' in c]
    for i, h1 in enumerate(score_cols):
        for h2 in score_cols[i+1:]:
            X_agg[f'{h1}_{h2}_int'] = s1[h1] * s1[h2]
    return X_agg


def build_critic_features(s1: pd.DataFrame, wav: np.ndarray, 
                          xgb_pred: np.ndarray, blend: np.ndarray,
                          heads: Dict[str, ReasoningHead]) -> pd.DataFrame:
    """Build critic input features."""
    d = pd.DataFrame()
    d['blend_pred'] = blend
    d['wav_pred'] = wav
    d['xgb_pred'] = xgb_pred
    d['wav_xgb_diff'] = np.abs(wav - xgb_pred)
    d['blend_conf'] = np.abs(blend - 0.5) * 2
    
    for name in heads.keys():
        score = s1[f'head_{name}_score'].values
        d[f'{name}_score'] = score
        d[f'{name}_vs_blend'] = np.abs(score - blend)
    
    score_cols = [c for c in s1.columns if '_score' in c]
    d['head_std'] = s1[score_cols].std(axis=1).values
    d['head_range'] = s1[score_cols].max(axis=1).values - s1[score_cols].min(axis=1).values
    
    return d


@dataclass
class ThinkingXGBoostPipeline:
    """
    4-stage thinking pipeline with hybrid aggregation.
    
    Architecture:
        Stage 1: Reasoning heads (specialized per feature group)
        Stage 2: Hybrid aggregator (60% weighted avg + 40% XGBoost)
        Stage 3: Critic (identifies uncertain predictions)
        Stage 4: Refiner (re-evaluates flagged cases)
    """
    reasoning_heads: Dict[str, ReasoningHead]
    xgb_aggregator: xgb.XGBClassifier
    critic: xgb.XGBClassifier
    refiner: xgb.XGBClassifier
    critic_threshold: float = 0.43
    blend_ratio: float = 0.6
    
    def predict_with_reasoning(self, X: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Run full pipeline with reasoning trace."""
        # Stage 1: Reasoning heads
        s1 = get_stage1_outputs(X, self.reasoning_heads)
        
        # Stage 2: Hybrid aggregation
        wav = weighted_average(s1, self.reasoning_heads)
        X_agg = build_aggregator_features(s1)
        xgb_pred = self.xgb_aggregator.predict_proba(X_agg)[:, 1]
        agg_preds = self.blend_ratio * wav + (1 - self.blend_ratio) * xgb_pred
        
        # Stage 3: Critic
        critic_feats = build_critic_features(s1, wav, xgb_pred, agg_preds, self.reasoning_heads)
        critic_scores = self.critic.predict_proba(critic_feats)[:, 1]
        needs_ref = critic_scores > self.critic_threshold
        
        # Stage 4: Selective refinement
        final = agg_preds.copy()
        if needs_ref.any():
            X_ref = pd.concat([X.reset_index(drop=True), s1], axis=1)
            ref_preds = self.refiner.predict_proba(X_ref)[:, 1]
            final[needs_ref] = ref_preds[needs_ref]
        
        # Build reasoning trace
        trace = s1.copy()
        trace['weighted_avg'] = wav
        trace['xgb_pred'] = xgb_pred
        trace['aggregator_pred'] = agg_preds
        trace['critic_score'] = critic_scores
        trace['needs_refinement'] = needs_ref.astype(int)
        trace['final_pred'] = final
        trace['decision'] = (final > 0.5).astype(int)
        
        return final, trace
    
    def explain(self, X_single: pd.DataFrame, idx: int = 0) -> str:
        """Generate human-readable explanation."""
        _, trace = self.predict_with_reasoning(X_single)
        row = trace.iloc[idx]
        
        lines = ["<REASONING>"]
        for name in self.reasoning_heads.keys():
            score = row[f'head_{name}_score']
            lines.append(f"  {name:12s} risk: {score:.3f}")
        lines.append("  ----")
        lines.append(f"  Weighted avg: {row['weighted_avg']:.3f}")
        lines.append(f"  XGBoost pred: {row['xgb_pred']:.3f}")
        lines.append(f"  Blended:      {row['aggregator_pred']:.3f}")
        lines.append(f"  Critic score: {row['critic_score']:.3f}")
        if row['needs_refinement']:
            lines.append("  [!] REFINEMENT TRIGGERED")
        lines.append("</REASONING>")
        lines.append("")
        lines.append("<SOLUTION>")
        decision = "FRAUD" if row['decision'] == 1 else "LEGITIMATE"
        lines.append(f"  Probability: {row['final_pred']:.3f}")
        lines.append(f"  Decision: {decision}")
        lines.append("</SOLUTION>")
        
        return "\n".join(lines)


def train_pipeline(X_train: pd.DataFrame, y_train: pd.Series, 
                   feature_groups: Dict[str, List[str]]) -> ThinkingXGBoostPipeline:
    """Train complete thinking pipeline."""
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Stage 1: Train reasoning heads
    heads = {}
    for name, features in feature_groups.items():
        heads[name] = ReasoningHead(name, features).train(X_train, y_train, scale_weight)
    
    # Normalize weights
    total_w = sum(h.weight for h in heads.values())
    for h in heads.values():
        h.weight /= total_w
    
    s1_train = get_stage1_outputs(X_train, heads)
    
    # Stage 2: Train XGBoost aggregator
    X_agg_train = build_aggregator_features(s1_train)
    xgb_agg = xgb.XGBClassifier(
        n_estimators=30, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_weight, random_state=42, eval_metric='auc'
    )
    xgb_agg.fit(X_agg_train, y_train.reset_index(drop=True))
    
    # Compute blended predictions for critic training
    BLEND = 0.6
    wav_train = weighted_average(s1_train, heads)
    xgb_train = xgb_agg.predict_proba(X_agg_train)[:, 1]
    
    # Get CV errors
    cv_xgb = cross_val_predict(
        xgb.XGBClassifier(n_estimators=30, max_depth=4, scale_pos_weight=scale_weight, 
                          random_state=42, eval_metric='auc'),
        X_agg_train, y_train.reset_index(drop=True), cv=5, method='predict_proba'
    )[:, 1]
    cv_blend = BLEND * wav_train + (1 - BLEND) * cv_xgb
    agg_wrong = ((cv_blend > 0.5).astype(int) != y_train.reset_index(drop=True).values).astype(int)
    
    # Stage 3: Train critic
    agg_train = BLEND * wav_train + (1 - BLEND) * xgb_train
    critic_train = build_critic_features(s1_train, wav_train, xgb_train, agg_train, heads)
    
    critic = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        random_state=42, eval_metric='auc'
    )
    critic.fit(critic_train, agg_wrong)
    
    # Find optimal threshold
    critic_scores = critic.predict_proba(critic_train)[:, 1]
    best_thresh, best_f1 = 0.3, 0
    for t in np.arange(0.1, 0.7, 0.02):
        flags = (critic_scores > t).astype(int)
        if flags.sum() > 0:
            f1 = f1_score(agg_wrong, flags, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
    
    # Stage 4: Train refiner
    X_ref_train = pd.concat([X_train.reset_index(drop=True), s1_train], axis=1)
    refiner = xgb.XGBClassifier(
        n_estimators=180, max_depth=8, learning_rate=0.03,
        scale_pos_weight=scale_weight, random_state=42, eval_metric='auc'
    )
    weights = np.where(agg_wrong == 1, 8.0, 1.0)
    refiner.fit(X_ref_train, y_train.reset_index(drop=True), sample_weight=weights)
    
    return ThinkingXGBoostPipeline(heads, xgb_agg, critic, refiner, best_thresh, BLEND)


if __name__ == "__main__":
    from data_generator import get_feature_groups, get_all_features
    from rqs_evaluator import RQSEvaluator
    
    # Load data
    df = pd.read_csv('fraud_dataset.csv')
    X = df[get_all_features()]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Thinking XGBoost Pipeline (Best Version)...")
    pipeline = train_pipeline(X_train, y_train, get_feature_groups())
    
    # Traditional metrics
    preds, traces = pipeline.predict_with_reasoning(X_test)
    print(f"\nTraditional Metrics:")
    print(f"  ROC-AUC: {roc_auc_score(y_test, preds):.4f}")
    print(f"  Samples refined: {traces['needs_refinement'].sum()} / {len(traces)}")
    
    # RQS evaluation
    print("\nReasoning Quality Score:")
    evaluator = RQSEvaluator()
    result = evaluator.evaluate(pipeline, X_test, y_test)
    print(result)
    
    # Example
    print("=" * 50)
    print("Example Reasoning Trace:")
    print("=" * 50)
    fraud_idx = y_test[y_test == 1].index[0]
    print(pipeline.explain(X_test.loc[[fraud_idx]]))
