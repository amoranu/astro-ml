"""Full LightGBM classifier with Optuna hyperparameter tuning.

- GPU-accelerated LightGBM (falls back to CPU if unavailable)
- Optuna study persisted to SQLite for resume/caching
- Default max 5 trials
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Detect GPU availability once at import time
_USE_GPU = False
try:
    _m = lgb.LGBMClassifier(device='gpu', n_estimators=2, verbose=-1)
    _m.fit(np.random.rand(20, 3), np.array([0]*10 + [1]*10))
    _USE_GPU = True
    del _m
except Exception:
    pass

DEVICE = 'gpu' if _USE_GPU else 'cpu'

# Persistent study storage
_STUDY_DB = Path(__file__).resolve().parent.parent / 'output' / 'optuna_study.db'


def _gpu_params() -> dict:
    """Return device-related params."""
    if DEVICE == 'gpu':
        return {'device': 'gpu', 'gpu_use_dp': False}
    return {}


def _objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'class_weight': 'balanced',
        'random_state': 42,
        'verbose': -1,
        **_gpu_params(),
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average='macro')


def train_tuned_classifier(X_train: pd.DataFrame, y_train: np.ndarray,
                           X_val: pd.DataFrame, y_val: np.ndarray,
                           n_trials: int = 5) -> dict:
    """Run Optuna hyperparameter search and return best model.

    Study is persisted to SQLite so re-runs resume from previous trials.

    Returns dict with: model, best_params, f1_macro, y_pred, y_prob
    """
    os.makedirs(_STUDY_DB.parent, exist_ok=True)
    storage = f'sqlite:///{_STUDY_DB}'

    study = optuna.create_study(
        study_name='father_death_clf',
        storage=storage,
        direction='maximize',
        load_if_exists=True,
    )

    # Only run new trials if needed (cached trials count)
    existing = len(study.trials)
    remaining = max(0, n_trials - existing)
    if remaining > 0:
        print(f"  Optuna: {existing} cached trials, running {remaining} new "
              f"(device={DEVICE})")
        study.optimize(
            lambda trial: _objective(trial, X_train, y_train, X_val, y_val),
            n_trials=remaining,
        )
    else:
        print(f"  Optuna: {existing} cached trials >= {n_trials} requested, "
              f"skipping search")

    # Retrain with best params
    best = study.best_params
    final_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'n_estimators': 1000,
        'learning_rate': best['lr'],
        'max_depth': best['max_depth'],
        'num_leaves': best['num_leaves'],
        'min_child_samples': best['min_child_samples'],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample'],
        'reg_alpha': best['reg_alpha'],
        'reg_lambda': best['reg_lambda'],
        'class_weight': 'balanced',
        'random_state': 42,
        'verbose': -1,
        **_gpu_params(),
    }
    model = lgb.LGBMClassifier(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')

    return {
        'model': model,
        'best_params': best,
        'f1_macro': f1,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'best_value': study.best_value,
    }
