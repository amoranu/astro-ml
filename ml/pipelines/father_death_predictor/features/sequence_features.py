"""Sequence context features — neighbor-aware within the candidate window.

The current pipeline treats each candidate independently. But a maraka period
surrounded by other maraka periods is qualitatively different from an isolated
maraka spike. These features capture local context:

- Is this candidate a local danger maximum?
- How many consecutive maraka-tier neighbors surround it?
- Danger score relative to immediate neighbors
- Position within the window (early/middle/late)
- Duration-based: short periods = more precise timing signals
- Group-level: total danger density, candidate count, global rank

~25 features per candidate.
"""


def extract_sequence_features(candidates, ci):
    """Sequence context features for candidate at index ci.

    Args:
        candidates: full list of candidates in the window (each must have
                    'tier', 'danger_score', 'start_jd', 'end_jd' keys)
        ci: index of current candidate in the list

    Returns:
        dict of ~25 features.
    """
    n = len(candidates)
    cand = candidates[ci]
    tier = cand['tier']
    dscore = cand['danger_score']
    duration = cand['end_jd'] - cand['start_jd']

    f = {}

    # ── Immediate neighbors (window of 1 on each side) ──────────────
    prev_tier = candidates[ci - 1]['tier'] if ci > 0 else 0
    next_tier = candidates[ci + 1]['tier'] if ci < n - 1 else 0
    prev_ds = candidates[ci - 1]['danger_score'] if ci > 0 else 0.0
    next_ds = candidates[ci + 1]['danger_score'] if ci < n - 1 else 0.0

    f['seq_ds_vs_prev'] = dscore - prev_ds
    f['seq_ds_vs_next'] = dscore - next_ds

    f['seq_is_local_peak'] = 1.0 if (dscore > prev_ds and dscore > next_ds
                                      and dscore > 0) else 0.0

    is_danger = tier > 0 and tier <= 4
    prev_danger = prev_tier > 0 and prev_tier <= 4
    next_danger = next_tier > 0 and next_tier <= 4
    f['seq_isolated_maraka'] = 1.0 if (is_danger and not prev_danger
                                        and not next_danger) else 0.0

    # ── Consecutive maraka run ──────────────────────────────────────
    run_len = 0
    if is_danger:
        run_len = 1
        j = ci - 1
        while j >= 0 and 0 < candidates[j]['tier'] <= 4:
            run_len += 1
            j -= 1
        j = ci + 1
        while j < n and 0 < candidates[j]['tier'] <= 4:
            run_len += 1
            j += 1
    f['seq_maraka_run'] = run_len

    # ── Local window stats (5 candidates centered) ──────────────────
    win_start = max(0, ci - 2)
    win_end = min(n, ci + 3)
    local_scores = [candidates[j]['danger_score']
                    for j in range(win_start, win_end)]

    local_max = max(local_scores) if local_scores else 0
    local_mean = sum(local_scores) / len(local_scores) if local_scores else 0

    f['seq_local_max'] = local_max
    f['seq_is_local_max'] = 1.0 if (dscore == local_max
                                     and dscore > 0) else 0.0
    f['seq_ds_vs_local_mean'] = dscore - local_mean

    local_danger_count = sum(
        1 for j in range(win_start, win_end)
        if 0 < candidates[j]['tier'] <= 4
    )
    f['seq_local_danger_density'] = local_danger_count / max(
        win_end - win_start, 1)

    # ── Duration features ───────────────────────────────────────────
    # Shorter PDs = more precise timing. Astrologically, a short PD that
    # is a maraka is more "pointed" than a long one.
    f['seq_duration_days'] = duration

    # Duration relative to group mean
    durations = [c['end_jd'] - c['start_jd'] for c in candidates]
    mean_dur = sum(durations) / n if n > 0 else 1.0
    f['seq_dur_vs_mean'] = duration / max(mean_dur, 0.1)

    # Short period = more specific (log scale to dampen outliers)
    import math
    f['seq_dur_log'] = math.log1p(duration)

    # Duration-weighted danger: danger_score / duration (intensity)
    f['seq_danger_intensity'] = dscore / max(duration, 0.1)

    # ── Global (group-level) features ───────────────────────────────
    # These are the same for every candidate in a group, but the model
    # can learn to weight candidates differently based on group context.

    all_scores = [c['danger_score'] for c in candidates]
    all_scores_sorted = sorted(all_scores, reverse=True)

    # Global rank of this candidate's danger score
    # (1 = highest in group)
    f['seq_global_rank'] = all_scores_sorted.index(dscore) + 1

    # How many tier 1-4 candidates in the entire window
    n_danger_total = sum(1 for c in candidates if 0 < c['tier'] <= 4)
    f['seq_n_danger_total'] = n_danger_total

    # Danger density across entire window
    f['seq_danger_frac'] = n_danger_total / n

    # Total candidates in window (group size)
    f['seq_n_candidates'] = n

    # Is this the global max danger score in the window?
    f['seq_is_global_max'] = 1.0 if (dscore == all_scores_sorted[0]
                                      and dscore > 0) else 0.0

    # How many candidates have the SAME danger score (tie count)?
    # More ties = harder to distinguish = lower confidence
    f['seq_tie_count'] = sum(1 for s in all_scores if s == dscore)

    # Distance from global max (0 if this IS the max)
    f['seq_ds_gap_from_max'] = all_scores_sorted[0] - dscore

    return f
