"""Ranking models: LightGBM LambdaRank + pointwise binary alternative.

GPU-accelerated where available.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from .classifier import DEVICE, _gpu_params


def _rank_params(override: dict = None) -> dict:
    """Default LambdaRank parameters."""
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42,
        'verbose': -1,
        **_gpu_params(),
    }
    if override:
        params.update(override)
    return params


def _binary_params(override: dict = None) -> dict:
    """Default pointwise binary parameters."""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'scale_pos_weight': 23,  # 1 positive vs 23 negatives per group
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'verbose': -1,
        **_gpu_params(),
    }
    if override:
        params.update(override)
    return params


def _make_groups(df: pd.DataFrame) -> np.ndarray:
    """Compute group sizes from group_id column (must be sorted)."""
    return df.groupby('group_id', sort=False).size().values


def train_ranker(X_train: np.ndarray, y_train: np.ndarray,
                 groups_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 groups_val: np.ndarray,
                 params: dict = None,
                 num_boost_round: int = 1000) -> lgb.Booster:
    """Train LambdaRank model.

    Args:
        X_train/X_val: feature arrays
        y_train/y_val: relevance labels (1=correct, 0=wrong)
        groups_train/val: array of group sizes (each = 24)
        params: override default params
        num_boost_round: max boosting rounds

    Returns:
        Trained lgb.Booster
    """
    train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=groups_val,
                           reference=train_data)

    model = lgb.train(
        _rank_params(params), train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(0)],
    )
    return model


def train_binary(X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 params: dict = None,
                 num_boost_round: int = 1000) -> lgb.Booster:
    """Train pointwise binary classifier for ranking.

    Score all 24 candidates by P(correct), rank by probability.
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        _binary_params(params), train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(0)],
    )
    return model


def predict_scores(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """Predict ranking scores for all rows."""
    return model.predict(X)
