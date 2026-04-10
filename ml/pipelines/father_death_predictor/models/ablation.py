"""Ablation study: train separate models per feature group to measure signal.

GPU-accelerated LightGBM. Models trained in parallel via threads.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score
from .classifier import DEVICE, _gpu_params

# Feature group column prefixes
FEATURE_GROUPS = {
    'A_solar': [
        'sun_cos', 'sun_sin', 'sun_house_pos', 'sun_uchcha_bala',
        'sun_sign_dignity', 'sun_speed', 'sun_saturn_angle', 'sun_mars_angle',
        'sun_jupiter_angle', 'sun_rahu_angle', 'sun_malefic_aspect_sum',
        'sun_benefic_aspect_sum', 'sun_net_aspect', 'sun_combust_count',
        'sun_in_kendra', 'sun_in_trikona', 'sun_in_dusthana',
        'sun_d12_cos', 'sun_d12_sin',
    ],
    'B_ninth': [
        'h9_lord_cos', 'h9_lord_sin', 'h9_lord_dignity',
        'h9_lord_sign_dignity', 'h9_lord_house', 'h9_lord_in_kendra',
        'h9_lord_in_trikona', 'h9_lord_in_dusthana', 'h9_lord_retrograde',
        'h9_occupant_count', 'h9_malefic_count', 'h9_benefic_count',
        'h9_malefic_aspect', 'h9_benefic_aspect', 'h9_net_aspect',
        'h9_lord_sat_aspect', 'h9_lord_jup_aspect',
        'h9_sun_bav', 'h9_sat_bav', 'h9_sav', 'h9_lord_vargottama',
    ],
    'C_derived': [
        'h4_malefic_count', 'h4_benefic_count', 'h4_lord_uchcha',
        'h4_lord_in_dusthana', 'h10_malefic_count', 'h10_lord_uchcha',
        'h3_malefic_count', 'h3_lord_uchcha', 'h5_benefic_count',
        'h8_lord_in_9th', 'father_maraka_max', 'malefics_in_h4_and_h9',
        'sun_sat_mars_in_2_12', 'sun_sat_mars_in_5_9', 'lagna_lord_in_9th',
    ],
    'D_temporal': [
        f'age{a}_{s}'
        for a in [10, 15, 20, 25, 30, 35, 40, 50]
        for s in ['maha_is_maraka', 'antar_is_maraka', 'maha_lord_strength']
    ],
}


def _default_params():
    return dict(
        objective='multiclass', num_class=3,
        n_estimators=500, learning_rate=0.05,
        max_depth=5, num_leaves=31,
        min_child_samples=20,
        class_weight='balanced',
        random_state=42, verbose=-1,
        **_gpu_params(),
    )


def _train_one_group(name: str, cols: list,
                     X_train: pd.DataFrame, y_train: np.ndarray,
                     X_val: pd.DataFrame, y_val: np.ndarray) -> tuple:
    """Train a single ablation model. Returns (name, result_dict)."""
    avail = [c for c in cols if c in X_train.columns]
    if not avail:
        return name, None

    model = lgb.LGBMClassifier(**_default_params())
    model.fit(
        X_train[avail], y_train,
        eval_set=[(X_val[avail], y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    y_pred = model.predict(X_val[avail])
    return name, {
        'f1_macro': f1_score(y_val, y_pred, average='macro'),
        'kappa': cohen_kappa_score(y_val, y_pred),
        'model': model,
        'n_features': len(avail),
    }


def run_ablation(X_train: pd.DataFrame, y_train: np.ndarray,
                 X_val: pd.DataFrame, y_val: np.ndarray) -> dict:
    """Run ablation study in parallel: one model per feature group + full.

    Returns dict of group_name -> {f1_macro, kappa, model, n_features}.
    """
    all_cols = list(X_train.columns)

    # Build task list: A-D groups + E_full + F_full_geo
    tasks = list(FEATURE_GROUPS.items())
    tasks.append(('E_full', all_cols))
    if 'birth_year' in X_train.columns:
        tasks.append(('F_full_geo', all_cols))

    # GPU LightGBM must run sequentially (concurrent GPU sessions deadlock).
    # GPU already parallelises internally so this is still fast.
    results = {}
    for name, cols in tasks:
        name, result = _train_one_group(
            name, cols, X_train, y_train, X_val, y_val
        )
        if result is not None:
            results[name] = result

    return results
