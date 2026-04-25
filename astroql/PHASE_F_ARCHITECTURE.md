# Phase F Architecture: Per-Tradition Query Planner

This doc describes the implemented Phase F architecture (sub-phases F.1
through F.15). For the original design plan see `PHASE_F_PLAN.md`.

## 1. Goals (recap)

- **Independent per-tradition execution** — Parashari, Jaimini, KP each
  produce their own ranked windows. No cross-tradition fusion.
- **Reasoning-based scoring** — within each tradition, structured
  argument per window: support evidence + benefic/malefic attacks +
  classical exceptions.
- **RRF fusion** across reasoning lines within a tradition.
- **Structured explanations** for every ranked window.

## 2. Module map

```
astroql/planner/
  plan.py        QueryPlan, ReasoningLine, ExceptionRule, ExceptionLibrary,
                 load_plan(school, plan_name), load_exception_library(school)
  lines.py       evaluate_line(line, candidate, window, fired_index, bundle)
                 apply_exceptions(line, line_ev, fired_exceptions)
                 (within-line scorer: support / attack / fact-graded / noisy-OR)
  exceptions.py  evaluate_chart_static(library, bundle, ids)
                 evaluate_window_dynamic(library, bundle, candidate, ids)
  fusion.py      fuse_rrf(per_line_scores, weights, k=60)
                 fuse_weighted_sum(per_line_scores, weights)
                 annotate_line_ranks(per_line_scores)
  runner.py      run_tradition(query, school, plan_name, max_windows)
                 run_multi_tradition(query, plan_name, max_windows)

astroql/schemas/results.py
  YogaFiring, ExceptionFiring, LineEvidence,
  RankedWindow, TraditionResult, MultiTraditionResult

astroql/query_plans/
  parashari/longevity.yaml + exceptions.yaml
  jaimini/longevity.yaml   + exceptions.yaml
  kp/longevity.yaml        + exceptions.yaml

astroql/benchmark/
  retrodiction_per_tradition.py   per-school Hit@K + ±6mo metrics
  retrodiction.py                 (legacy combined aggregator — preserved)
```

## 3. Data flow per tradition (single chart)

```
FocusQuery
   │
   │ FocusResolver.resolve(query)
   ▼
ResolvedFocus
   │
   │ ChartComputer.compute(birth, config, vargas, dashas, need_kp/need_jaimini)
   ▼
Chart
   │
   │ <School>FeatureExtractor().extract(chart, resolved)
   ▼
FeatureBundle
   │           ─────────────────────────┐
   │ RuleEngine.apply(bundle, rules)    │ load_plan(school)
   ▼                                    ▼
List[FiredRule]                     QueryPlan + ExceptionLibrary
   │                                    │
   │ index by rule_id                   │
   ▼                                    ▼
fired_by_rule_id  ───────►  [ for each candidate window: ]
                              for line in plan.reasoning_lines:
                                  line_ev = evaluate_line(line, cand, w,
                                                          fired_by_rule_id,
                                                          bundle)
                                  apply_exceptions(line, line_ev,
                                                   chart_static_excs +
                                                   window_dynamic_excs)

per_line_scores: {line_id → {window → LineEvidence}}
   │
   │ fuse_rrf(per_line_scores, weights, k=60)
   ▼
{window → rrf_score}
   │
   │ rank, build RankedWindow w/ structured_argument
   ▼
TraditionResult
```

`run_multi_tradition()` calls `run_tradition()` once per requested
school, each with its own resolver + chart + extractor + rule library
+ plan + exception library — fully isolated.

## 4. Within-line scorer

For one reasoning line on one candidate window:

```
support_yogas = [tier_weighted(fr.strength) for fr in fired
                 where fr.rule_id ∈ line.supporting_yogas
                 and fr.window matches candidate (or static)]
attack_yogas  = [...]   # same, line.attacking_yogas

if line.supporting_facts:
    add a virtual graded yoga from chart_facts to support_yogas
if line.attacking_facts:
    add a virtual graded yoga from chart_facts to attack_yogas

raw_support = noisy_or(s.tier_weighted_strength for s in support_yogas)
raw_attack  = noisy_or(a.tier_weighted_strength for a in attack_yogas)
net = raw_support * (1 - alpha * raw_attack)
```

The "virtual graded yoga from chart_facts" is computed by treating the
list of clauses as a graded rule:
`strength = base_strength * (sum_matched_weights / sum_total_weights)`.

`alpha` (default 0.5) controls how aggressively benefic attacks reduce
the net. `alpha=0` = ignore attacks; `alpha=1` = a saturated attack
(noisy-OR=1) zeros support entirely.

## 5. Exception system

```yaml
exceptions:
  - id: vipreet_raja_yoga
    scope: line          # or 'yoga'
    applies_to_lines: [static_promise]
    applies_to_yogas: []
    condition_type: chart_static     # or 'window_dynamic'
    condition:
      - feature: primary_house_data.vipreet_raja_yoga_present
        op: eq
        value: true
    attenuation: 0.4
    source: BPHS Ch.36
```

Two scopes:
- **scope=line** — multiplies the line's final `net_strength` by
  `attenuation` (after support/attack aggregation).
- **scope=yoga** — recomputes raw_support after multiplying matching
  yogas' tier-weighted strengths by `attenuation`. Used when the
  classical override targets specific yogas (e.g. Neecha Bhanga
  cancels classical_maraka_yoga's debilitation but not the entire
  static_promise line).

Two condition types:
- **chart_static** — evaluated once per chart (e.g. natal Vipreet
  Raja Yoga, Sade Sati Saturn natal status).
- **window_dynamic** — evaluated per candidate window (e.g. transit
  configuration at the window midpoint — references `candidate.*`).

Conditions are lists of antecedent clauses (AND-combined), evaluated
by the same `_evaluate_antecedent` machinery as graded rule factors.

## 6. RRF fusion (default)

```
rrf_score(window) = Σ over lines L:  weight[L] / (k + rank[L][window])
```

with `k=60` per Cormack-Clarke-Buettcher 2009. Windows that did not
fire on a line are absent from that line's ranking (no contribution).

Per-line ranks are based on `LineEvidence.net_strength` after exception
attenuation. Windows with `net_strength <= 0` don't get ranked in their
line.

`weighted_sum` is also available as `aggregation.method: weighted_sum`
in the plan YAML — uses raw net_strengths weighted by line weight, no
ranking. RRF is the default because it's robust to score-scale
differences between lines.

## 7. Per-tradition independence

The `MultiTraditionResult` is purely a container — `parashari`,
`jaimini`, `kp` are each `Optional[TraditionResult]`. No combined
metric. The benchmark reports each school's Hit@K independently:

```
PARASHARI:  hit@10 = 0.367, hit@10_pm6mo = 0.467, median_months_off = 28.4
KP:         hit@10 = 0.300, hit@10_pm6mo = 0.367, median_months_off = 41.0
JAIMINI:    inconclusive (deferred — needs Chara Dasha implementation)
```

## 8. Phase F.9 engine extensions (Parashari)

- **Mrityu Bhaga** — `MRITYU_BHAGA_DEG` table in `features/classical.py`,
  `in_mrityu_bhaga(planet, longitude, ±1°)` helper. Per-candidate
  fields `chain_lord_in_mrityu_bhaga` and `n_chain_in_mrityu_bhaga`
  exposed for the new yoga `mrityu_bhaga_chain_lord_active`.

- **Vipreet Raja Yoga** — `detect_vipreet_raja_yoga(positions, signs)`
  returns one entry per dusthana whose lord sits in another dusthana
  (BPHS Ch.36). Exposed as static facts
  `primary_house_data.vipreet_raja_yoga_present` (bool) and
  `_count` (int). Used by `vipreet_raja_yoga` exception (line-scope,
  attenuation 0.4 on `static_promise`).

- **Neecha Bhanga** — `detect_neecha_bhanga(positions, signs)` returns
  cancelled debilities: a debilitated planet whose dispositor or
  exalt-lord sits in kendra/trikona (Phaladeepika Ch.6). Exposed as
  `primary_house_data.neecha_bhanga_present` and
  `_planets`. Used by `neecha_bhanga_chain_lord` exception
  (yoga-scope, attenuation 0.5 on classical_maraka / role_overlay /
  chain_dignity).

## 9. Deferred (Phase F.10 / F.11)

These were scoped in `PHASE_F_PLAN.md` but require substantial
ephemeris / significator engineering and were marked as follow-up work
to keep F deliverable in scope:

- **F.10 (Jaimini)** — Chara Dasha computation, PiK extraction
  (8-karaka scheme), Rashi Drishti graph, Argala-on-PiK with
  Virodhargala. Without these, Jaimini reports 0 candidate windows
  and `inconclusive`. The plan structure (`jaimini/longevity.yaml`)
  is in place, ready for new yogas once features land.

- **F.11 (KP)** — Full significator hierarchy (occupants > occupants
  in lord's star > house lord > lord's nakshatra dispositors), per-
  window CSL evaluation, Ruling Planets at window midpoint, Star Lord
  vs Sub Lord agreement matrix. Without these, KP windows tie on
  static evidence and only `dbas_fully_signifies` differentiates.

## 10. Adding a new tradition (e.g. Tajik / Annual chart)

1. Add `School.TAJIK` to `astroql/schemas/enums.py`.
2. Create `astroql/features/tajik.py` (extractor) producing a
   `FeatureBundle`.
3. Wire into `_EXTRACTORS` in `astroql/planner/runner.py`.
4. Author yogas in `astroql/rules/tajik/longevity.yaml`.
5. Author plan + exceptions in
   `astroql/query_plans/tajik/longevity.yaml` +
   `exceptions.yaml`.
6. Add Tajik branch to `MultiTraditionResult` and
   `run_multi_tradition`.

No changes to `lines.py`, `exceptions.py`, `fusion.py`, or
`runner.run_tradition()` itself.

## 11. Adding a new reasoning line to an existing plan

1. Identify (or author) the yogas in the school's
   `rules/<school>/<life_area>.yaml`.
2. Add a `reasoning_lines:` entry in the plan YAML referencing those
   rule_ids in `yogas.supporting` / `yogas.attacking`.
3. Optionally add `chart_facts.supporting` clauses for facts not yet
   wrapped as full graded yogas.
4. Add the line's weight to `aggregation.weights`.

The planner picks it up next run — no code changes.

## 12. Adding a new exception

1. Add the override to `query_plans/<school>/exceptions.yaml`.
2. Add its id to the plan's `exceptions:` list.
3. If the condition references a new chart fact, add the fact to the
   appropriate feature extractor and declare the path in
   `rules/features_schema.yaml`.

## 13. Backward compatibility

The legacy combined-aggregator path (`Aggregator.aggregate()` and
`benchmark/retrodiction.py`) is unchanged. Both code paths can run
side-by-side; the new planner is opt-in via `run_tradition()` /
`run_multi_tradition()`.

## 14. Baseline numbers (Phase F.14, charts 200-230, n=30)

```
PARASHARI:
  hit@10        = 0.000
  hit@20        = 0.033
  hit@10_pm6mo  = 0.033
  hit@20_pm6mo  = 0.067
  median months off @rank1 = 400

KP:
  hit@10        = 0.000
  hit@20        = 0.033 (within 6mo)
  median months off @rank1 = 470
```

These are **below** the legacy combined aggregator's ~17.2% hit@10_pm6mo
on the same charts. The architecture is functional; the **v1 plan
tuning is the gap**. Diagnosis:

1. **Static lines saturate uniformly** — every window in a chart sees
   the same `static_promise` net_strength because the line's yogas
   have `window=None`. Within RRF this becomes near-random ordering
   (Python sort is stable but sequential-ties carry no signal),
   diluting the per-window discriminator lines.

2. **`dasha_chain_strength` saturates at noisy-OR=1.0** — once 5+
   role-bearing yogas fire on a window, support → 1.0; many candidate
   windows tie, RRF rank-1 is essentially arbitrary among them.

3. **No cross-school confluence** — the legacy aggregator's per-school
   noisy-OR + cross-school weighted-combination + confluence_bonus
   captured signal that per-tradition isolation discards by design.

4. **Per-line weights need calibration** — current weights came from
   doctrinal intuition, not data.

Tuning roadmap (post-F):

- Strip static yogas from RRF lines that include them, or compute a
  per-window scaling against the static base. (Static promise is "is
  the chart afflicted at all" — should be a global multiplier, not a
  per-window line.)
- Reduce `dasha_chain_strength`'s 17 yogas to a ranked top-3 to
  preserve discrimination at saturation.
- Add per-window CSL evaluation for KP (F.11) so the KP plan has more
  than `dbas_fully_signifies` to differentiate windows.
- Calibrate `aggregation.weights` on a held-out training slice.

The Phase F architecture supports these tuning iterations without
restructuring — just edit the YAML plans + rerun
`retrodiction_per_tradition`.

## 15. Status of sub-phases

| Sub-phase | Status |
|---|---|
| F.1  Schemas | ✓ |
| F.2  Within-line scorer | ✓ |
| F.3  Exception evaluator | ✓ |
| F.4  RRF fusion | ✓ |
| F.5  Per-tradition runner | ✓ |
| F.6  Parashari plan | ✓ |
| F.7  Jaimini plan | ✓ |
| F.8  KP plan | ✓ |
| F.9  Parashari engine ext (Mrityu Bhaga + Vipreet/Neecha) | ✓ |
| F.10 Jaimini engine ext (Chara Dasha, PiK, Rashi Drishti) | DEFERRED |
| F.11 KP engine ext (full significator hierarchy + RPs) | DEFERRED |
| F.12 Per-tradition benchmark | ✓ |
| F.13 Legacy aggregator backward compat smoke test | ✓ |
| F.14 Baseline benchmark run | ✓ (numbers in §14 above) |
| F.15 Architecture doc | ✓ (this doc) |
