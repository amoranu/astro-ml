#!/usr/bin/env python3
"""
Father Death Prediction — V4 Graph/Temporal Approach
=====================================================
Fundamentally new architecture:
  1. Exact-aspect proximity features (peaked, continuous)
  2. Per-chart rank/percentile features (within-chart relative standing)
  3. Coincidence density features (multi-system convergence)
  4. Temporal context model (Transformer sees all 25 months at once)
  5. Ensemble of temporal + flat models

Train: v2 (1000 subjects), Test: v1 (500 subjects)
Target: 90% top-1 accuracy at ±1 month tolerance
"""
import os, sys, pickle, json, time, math
import numpy as np
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from investigation.father_death_ml import FEATURE_NAMES, N_FEATURES
from investigation.father_death_factors import (
    EXTRA_FEATURE_NAMES, N_EXTRA, EXTRA2_FEATURE_NAMES, N_EXTRA2,
)

N_TOTAL = N_FEATURES + N_EXTRA + N_EXTRA2  # 210
AGE_NORM_IDX = N_FEATURES + N_EXTRA + 14


# ============================================================
#  Data loading
# ============================================================

def load_monthly_data(cache_path):
    """Load monthly feature caches and split into train/test by v1/v2."""
    with open(cache_path, "rb") as f:
        mc = pickle.load(f)
    with open(cache_path.replace(".pkl", "_test.pkl"), "rb") as f:
        tc = pickle.load(f)
    all_v11 = mc + tc

    ep = cache_path.replace(".pkl", "_extras.pkl")
    e2p = cache_path.replace(".pkl", "_extras2.pkl")
    el = pickle.load(open(ep, "rb")) if os.path.exists(ep) else None
    e2l = pickle.load(open(e2p, "rb")) if os.path.exists(e2p) else None

    merged = []
    for si, d in enumerate(all_v11):
        f = d["features"]
        nm = f.shape[0]
        ex = (el[si] if el and si < len(el) and el[si] is not None
              and el[si].shape[0] == nm else np.zeros((nm, N_EXTRA), dtype=np.float32))
        ex2 = (e2l[si] if e2l and si < len(e2l) and e2l[si] is not None
               and e2l[si].shape[0] == nm else np.zeros((nm, N_EXTRA2), dtype=np.float32))
        merged.append({
            "name": d["name"],
            "death_month_idx": d["death_month_idx"],
            "features": np.concatenate([f, ex, ex2], axis=1),
            "dates": d.get("dates", []),
        })

    v1_path = os.path.join(PROJECT_ROOT, "ml", "father_passing_date.json")
    v2_path = os.path.join(PROJECT_ROOT, "ml", "father_passing_date_v2.json")
    with open(v1_path, encoding="utf-8") as f:
        v1_names = {s["name"] for s in json.load(f)}
    with open(v2_path, encoding="utf-8") as f:
        v2_names = {s["name"] for s in json.load(f)}

    train = [d for d in merged if d["name"] in v2_names]
    test = [d for d in merged if d["name"] in v1_names]
    return train, test


def load_daily_data():
    """Load pre-computed daily feature caches."""
    daily_dir = os.path.join(PROJECT_ROOT, "investigation", ".daily_cache")
    train_path = os.path.join(daily_dir, "daily_train_v2.pkl")
    test_path = os.path.join(daily_dir, "daily_test_v1.pkl")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return None, None
    with open(train_path, "rb") as f:
        train = pickle.load(f)
    with open(test_path, "rb") as f:
        test = pickle.load(f)
    return train, test


# ============================================================
#  NEW FEATURES: Exact aspect proximity from daily data
# ============================================================

def compute_exact_aspect_features(daily_data, monthly_data, tight_orb=0.8):
    """For each month, compute proximity to nearest exact degree hit from daily data.

    Instead of binary "hit within 3 deg", compute:
    - days_to_nearest_exact_hit: continuous, peaked signal
    - inverse_proximity: 1/(1+days) -> strongest at exact aspect day
    - peak_density: how many exact hits cluster in a +/-7 day window

    Returns list of (n_months, N_EXACT) arrays, one per subject.
    """
    N_EXACT = 30  # exact aspect features per month

    daily_lookup = {d["name"]: d for d in daily_data}
    results = []

    for md in monthly_data:
        name = md["name"]
        nm = md["features"].shape[0]
        dd = daily_lookup.get(name)

        exact = np.zeros((nm, N_EXACT), dtype=np.float32)
        if dd is None:
            results.append(exact)
            continue

        daily_feats = dd["features"]  # (n_days, 48)
        daily_dates = dd["dates"]
        n_days = daily_feats.shape[0]

        # Group days into months
        month_to_days = {}
        for di, date_str in enumerate(daily_dates):
            ym = date_str[:7]
            if ym not in month_to_days:
                month_to_days[ym] = []
            month_to_days[ym].append(di)

        if not daily_dates:
            results.append(exact)
            continue

        first_ym = daily_dates[0][:7]
        parts = first_ym.split("-")
        start_y, start_m = int(parts[0]), int(parts[1])

        # Pre-compute: for each degree-hit feature (0-15), find all activation days
        hit_days = {}
        for fi in range(16):
            hit_days[fi] = np.where(daily_feats[:, fi] > 0)[0]

        # Pre-compute: transition days (feature 24-27 > 0)
        dasha_trans_days = np.where(daily_feats[:, 27] > 0)[0]

        # Pre-compute: mrityubhaga activation days
        mrityu_days = np.where(daily_feats[:, 39] > 0)[0]

        # Pre-compute: station days
        station_days = np.where(
            (daily_feats[:, 30] > 0.5) | (daily_feats[:, 31] > 0.5) |
            (daily_feats[:, 32] > 0.5) | (daily_feats[:, 33] > 0.5)
        )[0]

        # Pre-compute: eclipse proximity days
        eclipse_days = np.where(daily_feats[:, 28] > 0.5)[0]

        for mi in range(nm):
            y = start_y + (start_m + mi - 1) // 12
            m = ((start_m + mi - 1) % 12) + 1
            ym = f"{y}-{m:02d}"

            day_indices = month_to_days.get(ym, [])
            if not day_indices:
                continue

            mid_day = day_indices[len(day_indices) // 2]  # month midpoint

            # Feature group 1: Proximity to nearest degree hit for each planet (0-7)
            planet_groups = [(0, 4), (4, 8), (8, 12), (12, 16)]
            for pi, (fi_start, fi_end) in enumerate(planet_groups):
                all_planet_hits = np.concatenate(
                    [hit_days[fi] for fi in range(fi_start, fi_end)]
                ) if any(len(hit_days[fi]) > 0 for fi in range(fi_start, fi_end)) else np.array([])

                if len(all_planet_hits) > 0:
                    dists = np.abs(all_planet_hits - mid_day)
                    min_dist = dists.min()
                    exact[mi, pi] = 1.0 / (1.0 + min_dist)  # peaked proximity
                    exact[mi, 4 + pi] = min_dist  # raw distance in days

            # Feature group 2: Per-target proximity (Sun, 9L, ASC, 8L) (8-15)
            for ti in range(4):
                target_hits = np.concatenate(
                    [hit_days[pi * 4 + ti] for pi in range(4)]
                ) if any(len(hit_days[pi * 4 + ti]) > 0 for pi in range(4)) else np.array([])

                if len(target_hits) > 0:
                    dists = np.abs(target_hits - mid_day)
                    min_dist = dists.min()
                    exact[mi, 8 + ti] = 1.0 / (1.0 + min_dist)
                    exact[mi, 12 + ti] = min_dist

            # Feature group 3: Dasha transition proximity (16-17)
            if len(dasha_trans_days) > 0:
                dists = np.abs(dasha_trans_days - mid_day)
                min_dist = dists.min()
                exact[mi, 16] = 1.0 / (1.0 + min_dist)
                exact[mi, 17] = min_dist

            # Feature group 4: Convergence density (18-23)
            window = 7
            lo = max(0, mid_day - window)
            hi = min(n_days, mid_day + window + 1)
            window_feats = daily_feats[lo:hi]

            n_planet_hits = sum(1 for fi in range(16) if window_feats[:, fi].max() > 0)
            exact[mi, 18] = n_planet_hits / 16.0

            n_triggers = 0
            if window_feats[:, 27].max() > 0: n_triggers += 1  # dasha
            if window_feats[:, 39].max() > 0: n_triggers += 1  # mrityu
            if window_feats[:, 28].max() > 0.5: n_triggers += 1  # eclipse
            if (window_feats[:, 30:34] > 0.5).any(): n_triggers += 1  # station
            if window_feats[:, 44].max() > 0: n_triggers += 1  # double transit
            exact[mi, 19] = n_triggers / 5.0

            daily_trigger_counts = window_feats[:, 45]
            exact[mi, 20] = daily_trigger_counts.max() if len(daily_trigger_counts) > 0 else 0

            if len(mrityu_days) > 0:
                dists = np.abs(mrityu_days - mid_day)
                exact[mi, 21] = 1.0 / (1.0 + dists.min())

            if len(station_days) > 0:
                dists = np.abs(station_days - mid_day)
                exact[mi, 22] = 1.0 / (1.0 + dists.min())

            if len(eclipse_days) > 0:
                dists = np.abs(eclipse_days - mid_day)
                exact[mi, 23] = 1.0 / (1.0 + dists.min())

            # Feature group 5: Monthly aggregates (24-29)
            month_total_hits = sum(
                1 for fi in range(16)
                if any(daily_feats[d, fi] > 0 for d in day_indices)
            )
            exact[mi, 24] = month_total_hits / 16.0

            month_double = max(
                (daily_feats[d, 44] for d in day_indices), default=0
            )
            exact[mi, 25] = month_double

            month_concurrent = max(
                (daily_feats[d, 47] for d in day_indices), default=0
            )
            exact[mi, 26] = month_concurrent

            month_malefic = max(
                (daily_feats[d, 42] for d in day_indices), default=0
            )
            exact[mi, 27] = month_malefic

            month_new = sum(
                daily_feats[d, 46] for d in day_indices
            )
            exact[mi, 28] = month_new

            month_gandanta = sum(
                daily_feats[d, 23] for d in day_indices
            )
            exact[mi, 29] = month_gandanta

        results.append(exact)

    return results, N_EXACT


# ============================================================
#  NEW FEATURES: Per-chart rank features
# ============================================================

def compute_rank_features(monthly_data, n_top_features=30):
    """For each month, compute its percentile rank within the chart."""
    all_stds = np.zeros(N_TOTAL)
    for d in monthly_data:
        all_stds += d["features"].std(axis=0)
    all_stds /= len(monthly_data)
    all_stds[AGE_NORM_IDX] = 0
    top_feat_idx = np.argsort(-all_stds)[:n_top_features]

    results = []
    for d in monthly_data:
        f = d["features"]
        nm = f.shape[0]
        ranks = np.zeros((nm, n_top_features), dtype=np.float32)
        for fi_out, fi_in in enumerate(top_feat_idx):
            col = f[:, fi_in]
            order = np.argsort(np.argsort(col))
            ranks[:, fi_out] = order / max(nm - 1, 1)
        results.append(ranks)
    return results, n_top_features, top_feat_idx


# ============================================================
#  NEW FEATURES: Anomaly/uniqueness score per month
# ============================================================

def compute_anomaly_features(monthly_data):
    """For each month within a chart, compute how unusual it is."""
    mask = np.ones(N_TOTAL, dtype=bool)
    mask[AGE_NORM_IDX] = False

    results = []
    for d in monthly_data:
        f = d["features"][:, mask]
        nm = f.shape[0]
        anom = np.zeros((nm, 4), dtype=np.float32)

        mu = f.mean(0); sig = f.std(0); sig[sig < 1e-8] = 1.0
        z = (f - mu) / sig

        anom[:, 0] = (z ** 2).sum(axis=1)        # Mahalanobis-like
        anom[:, 1] = np.abs(z).max(axis=1)        # Max |z|
        anom[:, 2] = (np.abs(z) > 2.0).sum(axis=1)  # Count |z|>2
        anom[:, 3] = (np.abs(z) > 1.5).sum(axis=1)  # Count |z|>1.5

        results.append(anom)
    return results, 4


# ============================================================
#  Build combined feature matrix
# ============================================================

def build_enhanced_flat(monthly_data, exact_features, rank_features,
                        anomaly_features, wpm=1):
    """Build flat X, y with ALL feature types combined and z-normalized."""
    mask = np.ones(N_TOTAL, dtype=bool)
    mask[AGE_NORM_IDX] = False

    X_list, y_list, gs = [], [], []

    for si, d in enumerate(monthly_data):
        f_raw = d["features"][:, mask]
        nm = f_raw.shape[0]
        di = d["death_month_idx"]

        mu = f_raw.mean(0); sig = f_raw.std(0); sig[sig < 1e-8] = 1.0
        f_z = (f_raw - mu) / sig

        ex = exact_features[si]
        ex_mu = ex.mean(0); ex_sig = ex.std(0); ex_sig[ex_sig < 1e-8] = 1.0
        ex_z = (ex - ex_mu) / ex_sig

        rk = rank_features[si] - 0.5

        an = anomaly_features[si]
        an_mu = an.mean(0); an_sig = an.std(0); an_sig[an_sig < 1e-8] = 1.0
        an_z = (an - an_mu) / an_sig

        combined = np.concatenate([f_z, ex_z, rk, an_z], axis=1)

        dm = set()
        for o in range(-wpm, wpm + 1):
            m = di + o
            if 0 <= m < nm:
                dm.add(m)

        X_list.append(combined.astype(np.float32))
        y_list.append(np.array([1 if m in dm else 0 for m in range(nm)], np.int32))
        gs.append(nm)

    return np.vstack(X_list), np.concatenate(y_list), gs


def _split(flat, gs):
    s, o = [], 0
    for n in gs:
        s.append(flat[o:o + n])
        o += n
    return s


def evaluate(data, scores, wpm=1, label=""):
    n = len(data)
    t1 = t3 = t5 = 0
    ranks = []
    for si, d in enumerate(data):
        s = scores[si]
        di = d["death_month_idx"]
        nm = len(s)
        dm = set()
        for o in range(-wpm, wpm + 1):
            m = di + o
            if 0 <= m < nm:
                dm.add(m)
        rk = np.argsort(-s)
        br = nm
        for r, mi in enumerate(rk):
            if mi in dm:
                br = r + 1
                break
        ranks.append(br)
        if br <= 1: t1 += 1
        if br <= 3: t3 += 1
        if br <= 5: t5 += 1
    ranks = np.array(ranks)
    bl = (2 * wpm + 1) / np.mean([d["features"].shape[0] for d in data])
    print(f"\n  {label}")
    print(f"  Top-1: {100*t1/n:.1f}% ({t1}/{n}) [base {100*bl:.1f}%]")
    print(f"  Top-3: {100*t3/n:.1f}% | Top-5: {100*t5/n:.1f}%")
    print(f"  Mean rank: {ranks.mean():.1f} | Median: {np.median(ranks):.0f}")
    return {"top1": t1 / n, "top3": t3 / n, "top5": t5 / n,
            "mean_rank": ranks.mean(), "median_rank": np.median(ranks)}


# ============================================================
#  Model 1: Enhanced LogReg
# ============================================================

def run_logreg(X_tr, y_tr, gs_tr, X_te, gs_te, C=0.01, label="LogReg"):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        penalty="l1", C=C, solver="saga", max_iter=5000,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    model.fit(X_tr, y_tr)
    nz = (np.abs(model.coef_[0]) > 1e-6).sum()
    print(f"  {label}: {nz}/{X_tr.shape[1]} non-zero features, C={C}")
    ts = _split(model.predict_proba(X_te)[:, 1], gs_te)
    trs = _split(model.predict_proba(X_tr)[:, 1], gs_tr)
    return ts, trs, model


# ============================================================
#  Model 2: LightGBM
# ============================================================

def run_lgbm(X_tr, y_tr, gs_tr, X_te, gs_te, seed=42, label="LGBM"):
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    scale = float((y_tr == 0).sum()) / max(y_tr.sum(), 1)
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        num_leaves=8, min_child_samples=50, subsample=0.7,
        colsample_bytree=0.3, reg_alpha=1.0, reg_lambda=5.0,
        scale_pos_weight=scale,
        random_state=seed, n_jobs=-1, verbose=-1
    )
    X_t, X_v, y_t, y_v = train_test_split(
        X_tr, y_tr, test_size=0.12, random_state=seed, stratify=y_tr)
    model.fit(X_t, y_t, eval_set=[(X_v, y_v)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"  {label}: best_iter={model.best_iteration_}")
    ts = _split(model.predict_proba(X_te)[:, 1], gs_te)
    trs = _split(model.predict_proba(X_tr)[:, 1], gs_tr)
    return ts, trs, model


# ============================================================
#  Model 3: Per-chart Temporal Transformer
# ============================================================

def run_transformer(train_data, test_data, train_feats, test_feats,
                    n_feat, epochs=300, lr=3e-4, label="Transformer"):
    """Small Transformer that sees all months of a chart at once."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  PyTorch not available -- skipping Transformer")
        return None, None, None

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  {label}: device={device}, features={n_feat}")

    max_len = 30

    def make_tensors(data_list, feat_list):
        n = len(data_list)
        X = np.zeros((n, max_len, n_feat), dtype=np.float32)
        y = np.zeros((n, max_len), dtype=np.float32)
        mask = np.zeros((n, max_len), dtype=np.float32)

        for i, (d, feats) in enumerate(zip(data_list, feat_list)):
            nm = min(feats.shape[0], max_len)
            di = d["death_month_idx"]

            mu = feats[:nm].mean(0); sig = feats[:nm].std(0); sig[sig < 1e-8] = 1.0
            X[i, :nm] = (feats[:nm] - mu) / sig
            mask[i, :nm] = 1.0

            # Soft label: Gaussian centered on death month
            for m in range(nm):
                dist = abs(m - di)
                if dist <= 1: y[i, m] = 1.0
                elif dist <= 3: y[i, m] = 0.3
                elif dist <= 5: y[i, m] = 0.1
            s = y[i, :nm].sum()
            if s > 0: y[i, :nm] /= s

        return (torch.tensor(X).to(device),
                torch.tensor(y).to(device),
                torch.tensor(mask).to(device))

    X_tr, y_tr, mask_tr = make_tensors(train_data, train_feats)
    X_te, y_te, mask_te = make_tensors(test_data, test_feats)

    class ChartTransformer(nn.Module):
        def __init__(self, d_in, d_model=64, nhead=4, nlayers=3, dropout=0.3):
            super().__init__()
            self.input_proj = nn.Linear(d_in, d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=128,
                dropout=dropout, batch_first=True, activation='gelu'
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
            self.output = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1)
            )

        def forward(self, x, mask=None):
            h = self.input_proj(x) + self.pos_emb[:, :x.size(1)]
            if mask is not None:
                src_key_padding_mask = (mask == 0)
            else:
                src_key_padding_mask = None
            h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
            logits = self.output(h).squeeze(-1)
            if mask is not None:
                logits = logits * mask + (-1e9) * (1 - mask)
            return logits

    model = ChartTransformer(n_feat, d_model=64, nhead=4, nlayers=3, dropout=0.35).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    kl = nn.KLDivLoss(reduction='batchmean')

    loader = DataLoader(TensorDataset(X_tr, y_tr, mask_tr),
                        batch_size=64, shuffle=True)

    best_loss, best_state, no_improve = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        total = 0
        for xb, yb, mb in loader:
            logits = model(xb, mb)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = kl(log_probs, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * xb.size(0)
        scheduler.step()
        avg = total / len(X_tr)
        if avg < best_loss - 1e-5:
            best_loss = avg; no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= 40: break
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: loss={avg:.5f} best={best_loss:.5f}")

    if best_state: model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        ts_raw = model(X_te, mask_te).cpu().numpy()
        trs_raw = model(X_tr, mask_tr).cpu().numpy()

    test_scores = [ts_raw[i, :min(d["features"].shape[0], max_len)]
                   for i, d in enumerate(test_data)]
    train_scores = [trs_raw[i, :min(d["features"].shape[0], max_len)]
                    for i, d in enumerate(train_data)]
    return test_scores, train_scores, model


# ============================================================
#  Model 4: 1D-CNN on enhanced features
# ============================================================

def run_cnn(train_data, test_data, train_feats, test_feats,
            n_feat, epochs=200, lr=5e-4, label="CNN"):
    """1D-CNN that processes the monthly sequence."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  PyTorch not available -- skipping CNN")
        return None, None, None

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  {label}: device={device}")

    max_len = 30

    def make_tensors(data_list, feat_list):
        n = len(data_list)
        X = np.zeros((n, max_len, n_feat), dtype=np.float32)
        y = np.zeros((n, max_len), dtype=np.float32)
        mask = np.zeros((n, max_len), dtype=np.float32)
        for i, (d, feats) in enumerate(zip(data_list, feat_list)):
            nm = min(feats.shape[0], max_len)
            di = d["death_month_idx"]
            mu = feats[:nm].mean(0); sig = feats[:nm].std(0); sig[sig < 1e-8] = 1.0
            X[i, :nm] = (feats[:nm] - mu) / sig
            mask[i, :nm] = 1.0
            for m in range(nm):
                dist = abs(m - di)
                if dist <= 1: y[i, m] = 1.0
                elif dist <= 3: y[i, m] = 0.3
            s = y[i, :nm].sum()
            if s > 0: y[i, :nm] /= s
        return (torch.tensor(X).to(device),
                torch.tensor(y).to(device),
                torch.tensor(mask).to(device))

    X_tr, y_tr, mask_tr = make_tensors(train_data, train_feats)
    X_te, y_te, mask_te = make_tensors(test_data, test_feats)

    class MonthlyCNN(nn.Module):
        def __init__(self, nf):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(nf, 128, 3, padding=1),
                nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.4),
                nn.Conv1d(128, 64, 3, padding=1),
                nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.3),
                nn.Conv1d(64, 32, 3, padding=1),
                nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.3),
                nn.Conv1d(32, 1, 1),
            )
        def forward(self, x, mask=None):
            logits = self.net(x.permute(0, 2, 1)).squeeze(1)
            if mask is not None:
                logits = logits * mask + (-1e9) * (1 - mask)
            return logits

    model = MonthlyCNN(n_feat).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    kl = nn.KLDivLoss(reduction='batchmean')

    loader = DataLoader(TensorDataset(X_tr, y_tr, mask_tr), batch_size=64, shuffle=True)
    best_loss, best_state, no_improve = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        total = 0
        for xb, yb, mb in loader:
            logits = model(xb, mb)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = kl(log_probs, yb)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * xb.size(0)
        scheduler.step()
        avg = total / len(X_tr)
        if avg < best_loss - 1e-5:
            best_loss = avg; no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= 30: break
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: loss={avg:.5f} best={best_loss:.5f}")

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ts_raw = model(X_te, mask_te).cpu().numpy()
        trs_raw = model(X_tr, mask_tr).cpu().numpy()

    test_scores = [ts_raw[i, :min(d["features"].shape[0], max_len)]
                   for i, d in enumerate(test_data)]
    train_scores = [trs_raw[i, :min(d["features"].shape[0], max_len)]
                    for i, d in enumerate(train_data)]
    return test_scores, train_scores, model


# ============================================================
#  Model 5: XGBoost with GPU
# ============================================================

def run_xgb(X_tr, y_tr, gs_tr, X_te, gs_te, seed=42, label="XGB"):
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    scale = float((y_tr == 0).sum()) / max(y_tr.sum(), 1)
    try:
        model = XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.3,
            reg_alpha=1.0, reg_lambda=5.0, min_child_weight=50,
            scale_pos_weight=scale, random_state=seed,
            device="cuda", tree_method="hist",
            eval_metric="logloss", early_stopping_rounds=50, verbosity=0
        )
    except Exception:
        model = XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.3,
            reg_alpha=1.0, reg_lambda=5.0, min_child_weight=50,
            scale_pos_weight=scale, random_state=seed,
            eval_metric="logloss", early_stopping_rounds=50, verbosity=0
        )

    X_t, X_v, y_t, y_v = train_test_split(
        X_tr, y_tr, test_size=0.12, random_state=seed, stratify=y_tr)
    model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
    print(f"  {label}: best_iter={model.best_iteration}")
    ts = _split(model.predict_proba(X_te)[:, 1], gs_te)
    trs = _split(model.predict_proba(X_tr)[:, 1], gs_tr)
    return ts, trs, model


# ============================================================
#  Model 6: LambdaRank
# ============================================================

def run_rank(X_tr, y_tr, gs_tr, X_te, gs_te, seed=42, label="Rank"):
    import lightgbm as lgb
    ds = lgb.Dataset(X_tr, label=y_tr, group=gs_tr)
    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3],
        "learning_rate": 0.05, "num_leaves": 8, "max_depth": 3,
        "min_child_samples": 50, "subsample": 0.7, "colsample_bytree": 0.3,
        "reg_alpha": 1.0, "reg_lambda": 5.0,
        "verbose": -1, "n_jobs": -1, "seed": seed,
    }
    model = lgb.train(params, ds, num_boost_round=500, valid_sets=[ds],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(0)])
    print(f"  {label}: best_iter={model.best_iteration}")
    ts = _split(model.predict(X_te), gs_te)
    trs = _split(model.predict(X_tr), gs_tr)
    return ts, trs, model


# ============================================================
#  Grand Ensemble
# ============================================================

def ensemble_scores(score_lists, weights=None):
    """Combine multiple score lists via weighted z-norm averaging."""
    n_models = len(score_lists)
    if weights is None:
        weights = [1.0] * n_models
    n_charts = len(score_lists[0])
    combined = []
    for ci in range(n_charts):
        avg = None
        total_w = 0
        for mi in range(n_models):
            s = np.array(score_lists[mi][ci], dtype=np.float64)
            mu, std = s.mean(), s.std()
            if std > 1e-8: s = (s - mu) / std
            else: s = s - mu
            if avg is None:
                avg = np.zeros_like(s)
            min_len = min(len(avg), len(s))
            avg[:min_len] += weights[mi] * s[:min_len]
            total_w += weights[mi]
        combined.append((avg / total_w).astype(np.float32))
    return combined


# ============================================================
#  Feature importance analysis
# ============================================================

def analyze_feature_signal(train_data, enhanced_features, n_feat_total, feature_names=None):
    """Which features separate death month from non-death months?"""
    print(f"\n  Feature Signal Analysis (death month vs non-death)")
    print(f"  {'='*60}")

    death_feats, non_death_feats = [], []
    for si, d in enumerate(train_data):
        f = enhanced_features[si]
        nm = f.shape[0]; di = d["death_month_idx"]
        mu = f.mean(0); sig = f.std(0); sig[sig < 1e-8] = 1.0
        z = (f - mu) / sig
        for m in range(nm):
            if abs(m - di) <= 1: death_feats.append(z[m])
            else: non_death_feats.append(z[m])

    death_feats = np.array(death_feats)
    non_death_feats = np.array(non_death_feats)

    d_mean = death_feats.mean(0)
    nd_mean = non_death_feats.mean(0)
    pooled_std = np.sqrt((death_feats.std(0)**2 + non_death_feats.std(0)**2) / 2)
    pooled_std[pooled_std < 1e-8] = 1.0
    cohens_d = (d_mean - nd_mean) / pooled_std

    order = np.argsort(-np.abs(cohens_d))
    print(f"  Top 20 features by effect size (Cohen's d):")
    for r, fi in enumerate(order[:20]):
        name = feature_names[fi] if feature_names and fi < len(feature_names) else f"feat_{fi}"
        print(f"    {r+1:>2}. {name:<40s}  d={cohens_d[fi]:+.4f}  "
              f"death={d_mean[fi]:+.3f}  non_death={nd_mean[fi]:+.3f}")

    return cohens_d


# ============================================================
#  Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["logreg", "lgbm", "xgb", "rank", "transformer", "cnn"],
                        help="Models to run")
    parser.add_argument("--quick", action="store_true",
                        help="Only run logreg + lgbm")
    parser.add_argument("--analyze", action="store_true",
                        help="Run feature signal analysis")
    args = parser.parse_args()

    if args.quick:
        args.models = ["logreg", "lgbm"]

    cache_path = os.path.join(PROJECT_ROOT, "investigation",
                              "features_cache_v11_father_passing_date_plus500_w1.pkl")

    print(f"{'#'*90}")
    print(f"  FATHER DEATH v4 — TEMPORAL + EXACT ASPECT + CONVERGENCE")
    print(f"{'#'*90}")

    # --- Load data ---
    print(f"\n  Loading monthly data...")
    t0 = time.time()
    train, test = load_monthly_data(cache_path)
    print(f"  Train: {len(train)}, Test: {len(test)} ({time.time()-t0:.1f}s)")

    print(f"\n  Loading daily data...")
    daily_train, daily_test = load_daily_data()
    has_daily = daily_train is not None
    if has_daily:
        print(f"  Daily: {len(daily_train)} train, {len(daily_test)} test")
    else:
        print(f"  No daily cache -- skipping exact aspect features")

    # --- Compute new features ---
    if has_daily:
        print(f"\n  Computing exact aspect proximity features...")
        t0 = time.time()
        train_exact, N_EXACT = compute_exact_aspect_features(daily_train, train)
        test_exact, _ = compute_exact_aspect_features(daily_test, test)
        print(f"  Exact features: {N_EXACT} per month ({time.time()-t0:.1f}s)")
    else:
        N_EXACT = 0
        train_exact = [np.zeros((d["features"].shape[0], 0), dtype=np.float32) for d in train]
        test_exact = [np.zeros((d["features"].shape[0], 0), dtype=np.float32) for d in test]

    print(f"\n  Computing rank features...")
    N_RANK = 30
    train_rank, _, top_rank_idx = compute_rank_features(train, n_top_features=N_RANK)
    test_rank, _, _ = compute_rank_features(test, n_top_features=N_RANK)

    print(f"\n  Computing anomaly features...")
    train_anom, N_ANOM = compute_anomaly_features(train)
    test_anom, _ = compute_anomaly_features(test)

    # --- Build combined per-chart feature arrays ---
    print(f"\n  Assembling enhanced features per chart...")
    mask = np.ones(N_TOTAL, dtype=bool)
    mask[AGE_NORM_IDX] = False
    n_base = mask.sum()
    n_total_feat = n_base + N_EXACT + N_RANK + N_ANOM
    print(f"  Total features per month: {n_total_feat} "
          f"(base={n_base} + exact={N_EXACT} + rank={N_RANK} + anom={N_ANOM})")

    train_enhanced = []
    for si, d in enumerate(train):
        f = d["features"][:, mask]
        combined = np.concatenate([f, train_exact[si], train_rank[si], train_anom[si]], axis=1)
        train_enhanced.append(combined.astype(np.float32))

    test_enhanced = []
    for si, d in enumerate(test):
        f = d["features"][:, mask]
        combined = np.concatenate([f, test_exact[si], test_rank[si], test_anom[si]], axis=1)
        test_enhanced.append(combined.astype(np.float32))

    # --- Feature signal analysis ---
    if args.analyze:
        all_names = [n for i, n in enumerate(
            list(FEATURE_NAMES) + list(EXTRA_FEATURE_NAMES) + list(EXTRA2_FEATURE_NAMES)
        ) if mask[i]] + [f"exact_{i}" for i in range(N_EXACT)] + \
            [f"rank_{i}" for i in range(N_RANK)] + [f"anom_{i}" for i in range(N_ANOM)]
        analyze_feature_signal(train, train_enhanced, n_total_feat, all_names)

    # --- Build flat dataset ---
    print(f"\n  Building flat datasets...")
    t0 = time.time()
    X_tr, y_tr, gs_tr = build_enhanced_flat(train, train_exact, train_rank, train_anom)
    X_te, y_te, gs_te = build_enhanced_flat(test, test_exact, test_rank, test_anom)
    print(f"  Train: {X_tr.shape} (pos={y_tr.sum()}) | Test: {X_te.shape}")
    print(f"  Build time: {time.time()-t0:.1f}s")

    results = {}
    all_test_scores = {}

    # ---- Flat Models ----
    if "logreg" in args.models:
        print(f"\n{'='*80}\n  LogReg (L1) — Enhanced Features\n{'='*80}")
        for C in [0.001, 0.005, 0.01, 0.05]:
            ts, trs, m = run_logreg(X_tr, y_tr, gs_tr, X_te, gs_te, C=C,
                                    label=f"LogReg C={C}")
            key = f"logreg_C{C}"
            results[f"{key}_test"] = evaluate(test, ts, 1, f"LogReg C={C} — TEST")
            results[f"{key}_train"] = evaluate(train, trs, 1, f"LogReg C={C} — TRAIN")
            all_test_scores[key] = ts

    if "lgbm" in args.models:
        print(f"\n{'='*80}\n  LightGBM — Enhanced Features\n{'='*80}")
        ts, trs, _ = run_lgbm(X_tr, y_tr, gs_tr, X_te, gs_te, label="LGBM")
        results["lgbm_test"] = evaluate(test, ts, 1, "LGBM — TEST")
        results["lgbm_train"] = evaluate(train, trs, 1, "LGBM — TRAIN")
        all_test_scores["lgbm"] = ts

    if "xgb" in args.models:
        print(f"\n{'='*80}\n  XGBoost — Enhanced Features\n{'='*80}")
        ts, trs, _ = run_xgb(X_tr, y_tr, gs_tr, X_te, gs_te, label="XGB")
        results["xgb_test"] = evaluate(test, ts, 1, "XGB — TEST")
        results["xgb_train"] = evaluate(train, trs, 1, "XGB — TRAIN")
        all_test_scores["xgb"] = ts

    if "rank" in args.models:
        print(f"\n{'='*80}\n  LambdaRank — Enhanced Features\n{'='*80}")
        ts, trs, _ = run_rank(X_tr, y_tr, gs_tr, X_te, gs_te, label="Rank")
        results["rank_test"] = evaluate(test, ts, 1, "Rank — TEST")
        results["rank_train"] = evaluate(train, trs, 1, "Rank — TRAIN")
        all_test_scores["rank"] = ts

    # ---- Temporal Models ----
    if "transformer" in args.models:
        print(f"\n{'='*80}\n  Transformer — Enhanced Features\n{'='*80}")
        ts, trs, _ = run_transformer(
            train, test, train_enhanced, test_enhanced, n_total_feat,
            epochs=300, lr=3e-4, label="Transformer"
        )
        if ts is not None:
            results["transformer_test"] = evaluate(test, ts, 1, "Transformer — TEST")
            results["transformer_train"] = evaluate(train, trs, 1, "Transformer — TRAIN")
            all_test_scores["transformer"] = ts

    if "cnn" in args.models:
        print(f"\n{'='*80}\n  CNN — Enhanced Features\n{'='*80}")
        ts, trs, _ = run_cnn(
            train, test, train_enhanced, test_enhanced, n_total_feat,
            epochs=200, lr=5e-4, label="CNN"
        )
        if ts is not None:
            results["cnn_test"] = evaluate(test, ts, 1, "CNN — TEST")
            results["cnn_train"] = evaluate(train, trs, 1, "CNN — TRAIN")
            all_test_scores["cnn"] = ts

    # ---- Grand Ensemble ----
    if len(all_test_scores) >= 2:
        print(f"\n{'='*80}\n  ENSEMBLES\n{'='*80}")
        # All models
        ts = ensemble_scores(list(all_test_scores.values()))
        results["ensemble_all_test"] = evaluate(test, ts, 1, "Ensemble (all) — TEST")

        # Flat models only
        flat_keys = [k for k in all_test_scores if k not in ("transformer", "cnn")]
        if len(flat_keys) >= 2:
            ts = ensemble_scores([all_test_scores[k] for k in flat_keys])
            results["ensemble_flat_test"] = evaluate(test, ts, 1, "Ensemble (flat) — TEST")

        # Temporal only
        temp_keys = [k for k in all_test_scores if k in ("transformer", "cnn")]
        if len(temp_keys) >= 2:
            ts = ensemble_scores([all_test_scores[k] for k in temp_keys])
            results["ensemble_temporal_test"] = evaluate(test, ts, 1, "Ensemble (temporal) — TEST")

    # ---- SUMMARY ----
    print(f"\n\n{'#'*90}")
    print(f"  FINAL SUMMARY — TEST ({len(test)} subjects, +/-1 month)")
    print(f"{'#'*90}")
    avg_months = np.mean([d["features"].shape[0] for d in test])
    bl = 3.0 / avg_months
    print(f"\n  Random baseline: {100*bl:.1f}% | Avg months: {avg_months:.1f}")
    print(f"\n  {'Model':<35s}  {'Top-1':>7}  {'Top-3':>7}  {'Top-5':>7}  {'MnRk':>6}  {'Overfit':>8}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*8}")
    for key in sorted(results.keys()):
        if "_test" in key:
            name = key.replace("_test", "")
            r = results[key]
            tr_key = name + "_train"
            of = ""
            if tr_key in results:
                of = f"{100*(results[tr_key]['top1'] - r['top1']):+.1f}pp"
            print(f"  {name:<35s}  {100*r['top1']:>6.1f}%  {100*r['top3']:>6.1f}%  "
                  f"{100*r['top5']:>6.1f}%  {r['mean_rank']:>5.1f}  {of:>8}")


if __name__ == "__main__":
    main()
