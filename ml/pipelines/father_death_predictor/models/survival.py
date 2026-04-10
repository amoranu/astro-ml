"""Cox Proportional Hazards survival model for Stage 2 timing prediction."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


def train_survival_model(X_train: pd.DataFrame, durations_train: np.ndarray,
                         events_train: np.ndarray,
                         X_val: pd.DataFrame, durations_val: np.ndarray,
                         events_val: np.ndarray,
                         penalizer: float = 0.1) -> dict:
    """Train Cox PH model and evaluate on validation set.

    Args:
        X_train/X_val: feature DataFrames
        durations_train/val: native's age at father's death (years)
        events_train/val: 1 = father died, 0 = censored
        penalizer: L2 penalty strength

    Returns:
        Dict with: model, c_index_train, c_index_val, summary
    """
    # Drop near-zero-variance columns to avoid convergence failures
    variance = X_train.var()
    low_var_cols = variance[variance < 1e-6].index.tolist()
    X_train_clean = X_train.drop(columns=low_var_cols)
    X_val_clean = X_val.drop(columns=low_var_cols)
    if low_var_cols:
        print(f"    Survival: dropped {len(low_var_cols)} low-variance cols: "
              f"{low_var_cols[:5]}{'...' if len(low_var_cols) > 5 else ''}")

    # Prepare survival DataFrame
    surv_train = X_train_clean.copy()
    surv_train['duration'] = durations_train
    surv_train['event'] = events_train.astype(int)

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(surv_train, duration_col='duration', event_col='event')

    # Evaluate
    c_train = concordance_index(
        durations_train,
        -cph.predict_partial_hazard(X_train_clean).values.ravel(),
        events_train,
    )
    c_val = concordance_index(
        durations_val,
        -cph.predict_partial_hazard(X_val_clean).values.ravel(),
        events_val,
    )

    return {
        'model': cph,
        'c_index_train': c_train,
        'c_index_val': c_val,
        'summary': cph.summary,
    }
