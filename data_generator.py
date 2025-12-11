"""
Synthetic Fraud Detection Dataset Generator v2 - HARDER VERSION

Creates challenging cases that require multi-step reasoning and trigger self-correction:
1. Overlapping distributions (fraud/legit not perfectly separable)
2. Noise in features
3. "Gray zone" ambiguous cases  
4. Different fraud patterns (some obvious, some subtle)
5. Legitimate transactions that LOOK suspicious
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_fraud_dataset_hard(
    n_samples: int = 30000, 
    fraud_rate: float = 0.08,
    noise_level: float = 0.3,
    gray_zone_ratio: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a HARDER synthetic fraud detection dataset.
    
    Args:
        n_samples: Total number of samples
        fraud_rate: Fraction of fraud (higher = more samples to learn from)
        noise_level: How much noise to add (0-1)
        gray_zone_ratio: Fraction of ambiguous cases
        seed: Random seed
    """
    np.random.seed(seed)
    
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud
    
    # Split fraud into different patterns (some easy, some hard)
    n_fraud_obvious = int(n_fraud * 0.4)      # Classic fraud pattern
    n_fraud_subtle = int(n_fraud * 0.35)      # Subtle fraud (looks normal)
    n_fraud_mixed = n_fraud - n_fraud_obvious - n_fraud_subtle  # Mixed signals
    
    # Split legit into normal and suspicious-looking
    n_legit_normal = int(n_legit * (1 - gray_zone_ratio))
    n_legit_suspicious = n_legit - n_legit_normal  # Legit but looks like fraud
    
    def add_noise(arr, level):
        """Add Gaussian noise to array."""
        noise = np.random.normal(0, level * np.std(arr), size=len(arr))
        return arr + noise
    
    # === NORMAL LEGITIMATE TRANSACTIONS ===
    legit_normal = {
        'amount': np.random.lognormal(mean=4, sigma=1, size=n_legit_normal),
        'hour_of_day': np.random.choice(range(7, 22), size=n_legit_normal),
        'day_of_week': np.random.randint(0, 7, size=n_legit_normal),
        'txn_count_1h': np.random.poisson(lam=1.2, size=n_legit_normal),
        'txn_count_24h': np.random.poisson(lam=4, size=n_legit_normal),
        'avg_amount_30d': np.random.lognormal(mean=4, sigma=0.5, size=n_legit_normal),
        'merchant_category_risk': np.random.choice([0.1, 0.2, 0.3], size=n_legit_normal, p=[0.7, 0.2, 0.1]),
        'merchant_age_days': np.random.randint(200, 3000, size=n_legit_normal),
        'merchant_txn_volume': np.random.lognormal(mean=6, sigma=1, size=n_legit_normal),
        'distance_from_home': np.random.exponential(scale=15, size=n_legit_normal),
        'is_foreign_country': np.random.choice([0, 1], size=n_legit_normal, p=[0.97, 0.03]),
        'country_risk_score': np.random.choice([0.1, 0.2, 0.3], size=n_legit_normal, p=[0.85, 0.12, 0.03]),
        'is_new_device': np.random.choice([0, 1], size=n_legit_normal, p=[0.92, 0.08]),
        'failed_attempts_24h': np.random.poisson(lam=0.1, size=n_legit_normal),
        'is_fraud': np.zeros(n_legit_normal)
    }
    
    # === SUSPICIOUS-LOOKING LEGITIMATE (Gray Zone) ===
    # These are LEGIT but have some fraud-like signals
    legit_suspicious = {
        'amount': np.random.lognormal(mean=5, sigma=1.2, size=n_legit_suspicious),  # Higher amounts
        'hour_of_day': np.random.choice(range(0, 24), size=n_legit_suspicious),  # Unusual hours
        'day_of_week': np.random.randint(0, 7, size=n_legit_suspicious),
        'txn_count_1h': np.random.poisson(lam=2.5, size=n_legit_suspicious),  # More velocity
        'txn_count_24h': np.random.poisson(lam=8, size=n_legit_suspicious),
        'avg_amount_30d': np.random.lognormal(mean=4.5, sigma=0.8, size=n_legit_suspicious),
        'merchant_category_risk': np.random.choice([0.2, 0.3, 0.4], size=n_legit_suspicious, p=[0.4, 0.4, 0.2]),
        'merchant_age_days': np.random.randint(50, 1000, size=n_legit_suspicious),
        'merchant_txn_volume': np.random.lognormal(mean=5, sigma=1.2, size=n_legit_suspicious),
        'distance_from_home': np.random.exponential(scale=80, size=n_legit_suspicious),  # Farther
        'is_foreign_country': np.random.choice([0, 1], size=n_legit_suspicious, p=[0.7, 0.3]),
        'country_risk_score': np.random.choice([0.2, 0.3, 0.4], size=n_legit_suspicious, p=[0.5, 0.3, 0.2]),
        'is_new_device': np.random.choice([0, 1], size=n_legit_suspicious, p=[0.6, 0.4]),
        'failed_attempts_24h': np.random.poisson(lam=1, size=n_legit_suspicious),
        'is_fraud': np.zeros(n_legit_suspicious)  # Still legit!
    }
    
    # === OBVIOUS FRAUD ===
    # Clear fraud signals across multiple dimensions
    fraud_obvious = {
        'amount': np.random.lognormal(mean=6, sigma=1.5, size=n_fraud_obvious),
        'hour_of_day': np.random.choice([0, 1, 2, 3, 4, 5, 23], size=n_fraud_obvious),
        'day_of_week': np.random.randint(0, 7, size=n_fraud_obvious),
        'txn_count_1h': np.random.poisson(lam=5, size=n_fraud_obvious),
        'txn_count_24h': np.random.poisson(lam=15, size=n_fraud_obvious),
        'avg_amount_30d': np.random.lognormal(mean=4, sigma=0.5, size=n_fraud_obvious),
        'merchant_category_risk': np.random.choice([0.4, 0.5, 0.6], size=n_fraud_obvious, p=[0.3, 0.4, 0.3]),
        'merchant_age_days': np.random.randint(5, 100, size=n_fraud_obvious),
        'merchant_txn_volume': np.random.lognormal(mean=3, sigma=1.5, size=n_fraud_obvious),
        'distance_from_home': np.random.exponential(scale=300, size=n_fraud_obvious),
        'is_foreign_country': np.random.choice([0, 1], size=n_fraud_obvious, p=[0.2, 0.8]),
        'country_risk_score': np.random.choice([0.5, 0.6, 0.7], size=n_fraud_obvious, p=[0.3, 0.4, 0.3]),
        'is_new_device': np.random.choice([0, 1], size=n_fraud_obvious, p=[0.15, 0.85]),
        'failed_attempts_24h': np.random.poisson(lam=3, size=n_fraud_obvious),
        'is_fraud': np.ones(n_fraud_obvious)
    }
    
    # === SUBTLE FRAUD ===
    # Fraud that looks mostly normal (hard to detect)
    fraud_subtle = {
        'amount': np.random.lognormal(mean=4.5, sigma=1, size=n_fraud_subtle),  # Normal-ish amount
        'hour_of_day': np.random.choice(range(8, 20), size=n_fraud_subtle),  # Normal hours
        'day_of_week': np.random.randint(0, 5, size=n_fraud_subtle),  # Weekdays
        'txn_count_1h': np.random.poisson(lam=2, size=n_fraud_subtle),  # Slightly elevated
        'txn_count_24h': np.random.poisson(lam=6, size=n_fraud_subtle),
        'avg_amount_30d': np.random.lognormal(mean=4, sigma=0.5, size=n_fraud_subtle),
        'merchant_category_risk': np.random.choice([0.2, 0.3, 0.4], size=n_fraud_subtle, p=[0.3, 0.5, 0.2]),
        'merchant_age_days': np.random.randint(100, 800, size=n_fraud_subtle),  # Not too new
        'merchant_txn_volume': np.random.lognormal(mean=5, sigma=1, size=n_fraud_subtle),
        'distance_from_home': np.random.exponential(scale=50, size=n_fraud_subtle),  # Moderate
        'is_foreign_country': np.random.choice([0, 1], size=n_fraud_subtle, p=[0.6, 0.4]),
        'country_risk_score': np.random.choice([0.2, 0.3, 0.4], size=n_fraud_subtle, p=[0.4, 0.4, 0.2]),
        'is_new_device': np.random.choice([0, 1], size=n_fraud_subtle, p=[0.5, 0.5]),
        'failed_attempts_24h': np.random.poisson(lam=1, size=n_fraud_subtle),
        'is_fraud': np.ones(n_fraud_subtle)
    }
    
    # === MIXED SIGNAL FRAUD ===
    # Some signals say fraud, others say legit
    fraud_mixed = {
        'amount': np.random.lognormal(mean=4, sigma=0.8, size=n_fraud_mixed),  # Normal amount
        'hour_of_day': np.random.choice(range(0, 24), size=n_fraud_mixed),
        'day_of_week': np.random.randint(0, 7, size=n_fraud_mixed),
        'txn_count_1h': np.random.poisson(lam=4, size=n_fraud_mixed),  # High velocity (fraud signal)
        'txn_count_24h': np.random.poisson(lam=12, size=n_fraud_mixed),
        'avg_amount_30d': np.random.lognormal(mean=4, sigma=0.5, size=n_fraud_mixed),
        'merchant_category_risk': np.random.choice([0.1, 0.2, 0.3], size=n_fraud_mixed, p=[0.5, 0.3, 0.2]),  # Low risk (legit signal)
        'merchant_age_days': np.random.randint(500, 2000, size=n_fraud_mixed),  # Established (legit signal)
        'merchant_txn_volume': np.random.lognormal(mean=6, sigma=1, size=n_fraud_mixed),
        'distance_from_home': np.random.exponential(scale=150, size=n_fraud_mixed),  # Far (fraud signal)
        'is_foreign_country': np.random.choice([0, 1], size=n_fraud_mixed, p=[0.5, 0.5]),
        'country_risk_score': np.random.choice([0.1, 0.3, 0.5], size=n_fraud_mixed, p=[0.3, 0.4, 0.3]),
        'is_new_device': np.random.choice([0, 1], size=n_fraud_mixed, p=[0.7, 0.3]),  # Often same device
        'failed_attempts_24h': np.random.poisson(lam=2, size=n_fraud_mixed),
        'is_fraud': np.ones(n_fraud_mixed)
    }
    
    # Combine all segments
    dfs = [
        pd.DataFrame(legit_normal),
        pd.DataFrame(legit_suspicious),
        pd.DataFrame(fraud_obvious),
        pd.DataFrame(fraud_subtle),
        pd.DataFrame(fraud_mixed)
    ]
    df = pd.concat(dfs, ignore_index=True)
    
    # Add noise to numeric features
    numeric_cols = ['amount', 'txn_count_1h', 'txn_count_24h', 'avg_amount_30d', 
                    'merchant_age_days', 'merchant_txn_volume', 'distance_from_home',
                    'failed_attempts_24h']
    for col in numeric_cols:
        df[col] = add_noise(df[col].values, noise_level)
        df[col] = np.maximum(df[col], 0)  # No negatives
    
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Add derived features
    df['amount_vs_avg_ratio'] = df['amount'] / (df['avg_amount_30d'] + 1)
    df['velocity_score'] = df['txn_count_1h'] * 2 + df['txn_count_24h'] * 0.5
    df['location_risk'] = df['distance_from_home'] / 100 + df['country_risk_score'] + df['is_foreign_country'] * 0.3
    df['merchant_risk_score'] = df['merchant_category_risk'] + (1000 / (df['merchant_age_days'] + 1))
    
    return df


def get_feature_groups() -> dict:
    """Return feature groupings for multi-head reasoning."""
    return {
        'amount': ['amount', 'avg_amount_30d', 'amount_vs_avg_ratio'],
        'velocity': ['txn_count_1h', 'txn_count_24h', 'velocity_score'],
        'merchant': ['merchant_category_risk', 'merchant_age_days', 'merchant_txn_volume', 'merchant_risk_score'],
        'location': ['distance_from_home', 'is_foreign_country', 'country_risk_score', 'location_risk'],
        'device': ['is_new_device', 'failed_attempts_24h'],
        'time': ['hour_of_day', 'day_of_week']
    }


def get_all_features() -> list:
    """Return all feature names (excluding target)."""
    groups = get_feature_groups()
    all_feats = []
    for feats in groups.values():
        all_feats.extend(feats)
    return list(set(all_feats))


if __name__ == "__main__":
    print("Generating HARDER fraud detection dataset...")
    df = generate_fraud_dataset_hard(
        n_samples=30000, 
        fraud_rate=0.08,
        noise_level=0.3,
        gray_zone_ratio=0.15,
        seed=42
    )
    
    output_path = Path(__file__).parent / "data" / "fraud_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"\nFeature groups: {list(get_feature_groups().keys())}")
    print(f"\nSample:\n{df.head()}")
