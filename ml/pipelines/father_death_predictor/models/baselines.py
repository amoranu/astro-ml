"""Baseline models that the real model must beat.

1. Random (stratified) — predict class proportions from training set
2. Demographic-only — birth_year, birth_lat, gender (3 features)
3. Sun-sign only — one-hot Sun sign (12 features)
4. Season/month — birth_month one-hot (12 features)

GPU-accelerated LightGBM where applicable.
Baselines 1/3/4 are cheap; only demographic uses LightGBM.
"""

import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from ..astro_engine.houses import get_sign
from .classifier import _gpu_params


def train_random_baseline(y_train: np.ndarray, y_val: np.ndarray) -> dict:
    """Baseline 1: Stratified random prediction."""
    class_probs = np.bincount(y_train, minlength=3) / len(y_train)
    rng = np.random.RandomState(42)
    y_pred = rng.choice(3, size=len(y_val), p=class_probs)
    f1 = f1_score(y_val, y_pred, average='macro')
    return {
        'name': 'random_stratified',
        'f1_macro': f1,
        'class_probs': class_probs.tolist(),
    }


def train_demographic_baseline(
    train_records: list, y_train: np.ndarray,
    val_records: list, y_val: np.ndarray
) -> dict:
    """Baseline 2: Demographic-only (birth_year, lat, gender). GPU-accelerated."""
    def _extract(records):
        rows = []
        for r in records:
            year = int(r['birth_date'].split('-')[0])
            lat = r['lat']
            gender = 1.0 if r.get('gender', 'M') == 'M' else 0.0
            rows.append([year, lat, gender])
        return np.array(rows, dtype=np.float64)

    X_tr = _extract(train_records)
    X_va = _extract(val_records)

    model = lgb.LGBMClassifier(
        objective='multiclass', num_class=3,
        n_estimators=500, learning_rate=0.05,
        max_depth=5, num_leaves=31,
        min_child_samples=20,
        class_weight='balanced',
        random_state=42, verbose=-1,
        **_gpu_params(),
    )
    model.fit(
        X_tr, y_train,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    y_pred = model.predict(X_va)
    f1 = f1_score(y_val, y_pred, average='macro')
    return {'name': 'demographic', 'f1_macro': f1, 'model': model}


def train_sun_sign_baseline(
    train_sun_longs: np.ndarray, y_train: np.ndarray,
    val_sun_longs: np.ndarray, y_val: np.ndarray
) -> dict:
    """Baseline 3: Sun sign one-hot (12 features)."""
    def _one_hot(longs):
        signs = np.array([get_sign(l) for l in longs])
        oh = np.zeros((len(signs), 12), dtype=np.float64)
        oh[np.arange(len(signs)), signs] = 1.0
        return oh

    X_tr = _one_hot(train_sun_longs)
    X_va = _one_hot(val_sun_longs)

    model = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    )
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_va)
    f1 = f1_score(y_val, y_pred, average='macro')
    return {'name': 'sun_sign', 'f1_macro': f1, 'model': model}


def train_season_baseline(
    train_records: list, y_train: np.ndarray,
    val_records: list, y_val: np.ndarray
) -> dict:
    """Baseline 4: Birth month one-hot (12 features)."""
    def _one_hot(records):
        months = np.array([int(r['birth_date'].split('-')[1]) - 1 for r in records])
        oh = np.zeros((len(months), 12), dtype=np.float64)
        oh[np.arange(len(months)), months] = 1.0
        return oh

    X_tr = _one_hot(train_records)
    X_va = _one_hot(val_records)

    model = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    )
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_va)
    f1 = f1_score(y_val, y_pred, average='macro')
    return {'name': 'season_month', 'f1_macro': f1, 'model': model}


def run_all_baselines(train_records, y_train, val_records, y_val,
                      train_sun_longs, val_sun_longs):
    """Run all 4 baselines sequentially.

    GPU LightGBM (demographic) must not run concurrently with other GPU
    sessions. The non-LightGBM baselines are cheap (<1s) so threading
    overhead isn't worth it.
    """
    results = []
    results.append(train_random_baseline(y_train, y_val))
    results.append(train_demographic_baseline(
        train_records, y_train, val_records, y_val
    ))
    results.append(train_sun_sign_baseline(
        train_sun_longs, y_train, val_sun_longs, y_val
    ))
    results.append(train_season_baseline(
        train_records, y_train, val_records, y_val
    ))
    return results
