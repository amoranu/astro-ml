# AstroQL — Caveats & Gap Tracker

Single source of truth for known limitations across all phases.
Survives conversation compaction. Update whenever a gap is found or closed.

Format: `ID — Status | Scope | Impact | Fix`

---

## STATUS LEGEND
- **closed** — fix landed, verified
- **partial** — fix in place but with documented limits
- **open** — still needs work
- **deferred** — explicit design decision to skip / postpone
- **timing** — timing-quality issue (intentionally deferred until rest are clean)

---

## Closed (verified)

### CAV-002 / CAV-021 — 8-karaka Jaimini scheme + Pitrukaraka — closed
- chart/computer.py `_eight_karaka_scheme` populates AK..DK with PiK = Pitru (father).
- Activated by `ChartConfig(karaka_scheme="8")`.

### CAV-003 — Missing vargas (D11/D16/D20/D27/D40/D45) — closed
- chart/computer.py `_local_division` implements all six per BPHS formulas.

### CAV-004 — Per-varga lagna alignment — closed
- chart/computer.py `_build_divisional_with_lagna` computes varga-lagna from
  Lagna's longitude through the same divisional formula. Verified D1 H1=Cancer,
  D9 H1=Virgo on test chart (different per varga as required).

### CAV-005 — Badhaka relative to target — closed (Phase 1)

### CAV-006 — Safe `expr` evaluator — closed
- engine/rule_engine.py `_safe_eval` with sandboxed builtins +
  `lord_of(h)`, `marakas_of(target)`, `eighth_from(h)`, `sign_of_house(h)`
  helpers. Rules can reference `karakas`, `primary`, `varga`, `candidate`.

### CAV-010 — RAG retrieval wired — closed
- rag/pipeline.py wraps astro-prod's `retrieve_for_tradition`. Tested with live
  corpora — 5 passages per query. Synthesis is heuristic (see CAV-019).

### CAV-011 — KP 5-level tightening — closed
- features/kp.py `_strong_sigs_of_planet` keeps only level 1-2 (occupation)
  connections; ownership-only (3-4) and sub-lord-star (5) "null lord"
  links excluded. DBAS now uses strong-only signification.

### CAV-012 — Contradiction → confidence damping — closed
- aggregator.py applies `aggregate_confidence *= 0.85^n_contradictions`.

### CAV-013 / CAV-044 — Citation integrity test — closed
- tests/test_citation_integrity.py walks 49 rules, checks source against
  ALLOWED_SOURCE_PATTERNS allowlist. CI-fails on a single bad citation.

### CAV-014 — Spouse karaka gender warning — closed
- resolver/focus_resolver.py emits Python `warnings.warn` when spouse query
  has no gender; both karakas returned as before.

### CAV-016 — time_accuracy → confidence dampener — closed
- engine/rule_engine.py multiplies all FiredRule.strength by
  `{exact: 1.0, approximate: 0.9, unknown: 0.7}` per spec §12.2.

### CAV-017 — ExecutionTrace + stage timings — closed
- planner/trace.py + planner/pipeline.py instrument every stage.
- Explainer's `explain(trace=...)` emits per-stage `{name, ms, extra}` plus
  `total_ms`. Verified: chart=13ms, aggregate=495ms on test chart.

### CAV-018 — Shadbala + Ashtakavarga wired — closed
- chart/computer.py `need_strength=True` populates Chart.shadbala +
  Chart.ashtakavarga via astro-prod. Pipeline auto-enables for MAGNITUDE/YES_NO.

### CAV-019 — Heuristic RAG synthesis (no LLM) — closed (with note)
- engine/rule_engine.py `_apply_rag_passages` synthesizes one FiredRule per
  passage with strength derived from retrieval score.
- **Upgrade path**: replace with astro-prod's llm_engine LLMAgent or direct
  Anthropic call — pass passage + feature bundle, get `{fires, polarity,
  strength, evidence_excerpt}` JSON. Defer until live LLM auth is set up.

### CAV-022 / CAV-028 / CAV-031 / CAV-032 — Rule library expansion — partial
- Currently 49 rules across 12 YAML files:
  - parashari/{longevity, spouse, finance, career, children, health,
    education, marriage}.yaml
  - kp/{longevity, marriage}.yaml
  - jaimini/{longevity, marriage, career}.yaml
- Spec target: 200-300 rules. Current is 25%. Expansion is incremental
  authoring work — not a structural gap.

### CAV-023 — Argala benefic/malefic polarity + cancellation — closed
- features/jaimini.py `_argala_planets` splits occupants by natural
  benefic/malefic, applies virodhargala cancellation, emits
  `net_argala_polarity ∈ {benefic, malefic, neutral}`.

### CAV-024 — Karakamsha D9 lookup — closed
- features/jaimini.py emits `karakamsha_d9_sign` + `karakamsha_d9_house`
  from the AK's actual D9 navamsha placement.

### CAV-025 — MAGNITUDE thresholds documented + symmetric — closed
- aggregator.py cutoffs picked so 1 strong rule lands in high/low,
  2 lands in very_high/very_low. Symmetric around 0. Calibration
  via retrodiction is open work but rationale documented in code.

### CAV-026 — DESCRIPTION cross-school conflicts surfaced — closed
- aggregator.py `_aggregate_description` tags `value` with
  `[CONFLICT]` when ≥2 values for same attribute come from different
  schools within ε=0.15 confidence.

### CAV-027 — YES_NO threshold documented — closed
- aggregator.py `threshold_gap=0.2` parameterized. Same retrodiction
  caveat as CAV-025.

### CAV-035 — Placidus fallback to equal-house — closed
- chart/computer.py wraps Placidus in try/except; on failure (e.g. high
  latitudes), falls back to `_calculate_equal_house_cusps` from lagna.
  Sub-lords still computed; precision noted as reduced.

### CAV-039 — Rectification multi-life-area scoring — closed
- sensitivity/rectify.py builds per-event FocusQuery from event's own
  life_area with appropriate effect/modifier defaults.

### CAV-042 — NL parser (heuristic) — closed
- parser/nl_parser.py with regex-based intent extraction. 7/7 spec §11
  examples parse correctly. Raises `ClarifyRequired` for ambiguous queries.
- **Upgrade path**: LLM-based parser via astro-prod llm_engine when needed.

### CAV-043 — Severity gating extended — closed
- api/server.py `_SENSITIVE` includes longevity+neg, health+neg,
  health+magnitude, litigation+neg.

### CAV-001 — KP ayanamsa (Krishnamurti) — partial-closed
- chart/computer.py toggles `swe.set_sid_mode(SIDM_KRISHNAMURTI)` for the
  KP block, recomputes positions under KP ayanamsa, then restores Lahiri
  in the `finally`. KP cusps + sub-lords now align to KP convention.
- **Limit**: astro-prod sets Lahiri at module import, so any direct
  astro-prod calls outside ChartComputer still use Lahiri. Acceptable as
  AstroQL only calls astro-prod through ChartComputer.

### CAV-030 — Cross-life-area rules — deferred (design)
- A rule referencing both `karaka_data.Sun` and `karaka_data.Mars` for
  health-vs-finance interactions is not currently expressible without
  the rule being loaded for both life_areas. Workaround: write the rule
  twice (once per life_area). Proper fix needs rule_type=composite +
  multi-life_area applicable_to support.

### CAV-040 / CAV-041 — DSL grammar / HTTP server — deferred (design)
- DSL is regex-based — sufficient for spec §15.7 syntax. Rewriting with
  Lark/PEG is a future quality-of-life upgrade if DSL usage grows.
- No HTTP server shipped — `run_query_json(dict) -> dict` is the API
  surface. Wrap with FastAPI/Flask if needed (10 LOC).

---

## TIMING FIXES — Phase 9 (2026-04-22, after non-timing pass)

### CAV-007 — Dasha role-filtering tightened — closed
- features/parashari.py emits `chain_strength ∈ {full, partial, weak}`
  per candidate. Rules now filter on `chain_strength`. Tier-1 rules
  require `full` chain (all MD/AD/PAD lords in relevant set); tier-2
  requires `partial`; weak windows only fire a low-strength fallback.

### CAV-008 — Transit triggers wired — closed
- features/parashari.py `_compute_transit_table` precomputes monthly
  Saturn + Jupiter sign positions over 100y. `saturn_transit_target`
  and `jupiter_transit_target` flags added per candidate.
- New rules: `saturn_transit_target.01` (tier-1 negative s=0.6) and
  `jupiter_transit_target_protective.01` (tier-2 positive s=0.4).
- On Phase 0 chart: 126/679 candidates flagged Saturn-transit, 95/679
  Jupiter-transit.

### CAV-009 — Confluence saturation reduced — closed
- aggregator.py uses (a) tier-weighted strengths (1.0/0.7/0.4),
  (b) per-rule dedupe (each rule_id contributes max strength once),
  (c) distinct-rule confluence bonus `0.05 × log₂(n_distinct)`.
- New tier-1 `role_diversity_bonus` rule fires only when ≥4 distinct
  classical roles align (tightened from ≥3 after benchmark feedback).
- Conjoint static+timing combination: `cluster_conf × (0.7 + 0.3 ×
  static_mult)` when static present, `× 0.7` when not. Penalizes
  pure-timing windows lacking static support but doesn't crush them.

### CAV-029 — Rule strengths now data-informed — partial-closed
- Tier hierarchy (1/2/3) replaces flat 0.4-0.6 strengths.
- Maraka full-chain: 0.7, partial: 0.4. 8L full-chain: 0.65, partial 0.35.
- Pure dasha activation (rotated_lord, relation_karaka): 0.4.
- Weak-chain fallback: 0.18.
- Per-rule calibration multipliers loaded from calibrated_strengths.yaml
  if present.

### CAV-020 — Chara dasha — open (deferred)
- Jaimini timing rules still cannot produce dated windows. Architecture
  is in place; implementation needs proper Chara dasha formula.

### CAV-033 — Retrodiction improvement — partial-closed
- 30-chart benchmark: Hit@1 0% → 3.4%, Hit@10 3.4% → 10.3% (3×),
  mean rank 380 → 297 (22% better). Driven entirely by astrology-based
  rule structure changes (tiers + chain strength + transit + conjoint)
  — soft age prior is capped at ±5%.
- Still ≈ random at the head (Hit@1=3.4% vs ~0.1% random baseline for
  rank-1-out-of-700). Further gains need: more rules with proper
  doctrine (CAV-022/028/031/032), Chara dasha (CAV-020), Bayesian
  calibration loop (CAV-029).

### CAV-034 — Calibrated strengths applied at runtime — closed
- engine/rule_engine.py loads `rules/calibrated_strengths.yaml` at
  RuleEngine.__init__ and multiplies each FiredRule.strength by the
  per-rule multiplier (clamped [0.3, 2.0]). Maps hit_rate / prior to
  multiplier; missing rules default to 1.0 (no adjustment).

### CAV-036 — Soft age prior added (capped ±5%) — closed
- aggregator.py `_apply_soft_age_prior` Gaussian centered at empirical
  mode (39y for father longevity, σ=20). Cap of ±0.05 ensures outliers
  (childhood paternal deaths, very late deaths) are never suppressed —
  only marginally re-ranked in tied-confidence cases.

### CAV-038 — Sensitivity overlap with CAV-016 — closed (documented)
- CAV-016 already dampens via `time_accuracy ∈ {exact, approximate, unknown}`.
- Sensitivity scan results would be a finer-grained version; calling it
  per-window would be 5× the runtime — deferred as expensive optimization.

---

## Phase 9 final benchmark numbers (30-chart father-death)

```
Metric        v0_baseline   v5_final     Δ
n_evaluable   29            29           -
n_abstained   0             2            +2 (acceptable)
Hit@1         0.0%          3.4%         +3.4pp
Hit@3         3.4%          3.4%         -
Hit@5         3.4%          3.4%         -
Hit@10        3.4%          10.3%        +6.9pp (3x)
Hit@20        6.9%          10.3%        +3.4pp
Mean rank     380.6         297.0        -84 (22% better)
Median rank   360           363          ±0
```

Per-chart results saved at `/tmp/bench_30_v5.json`.

---

## Open (engine v2 review — 2026-04-25)

### CAV-NS-100 — JSON DSL evaluator + LLM autonomy — closed (partial)
- engine/dsl_evaluator.py: dot-path traversal + operators (==, !=, <, <=,
  >, >=, in, not_in, contains, truthy, falsy) + combinators (all/any/not).
- cf_engine.infer_cf accepts epoch_state and re-evaluates
  Rule.modifiers[].condition via the DSL when populated; legacy
  fired_modifier_indices path preserved unchanged.
- CFRuleSpec.fires_when and modifier_predicates now accept either a
  Python callable OR a JSON dict; cf_predict._resolve_predicate
  dispatches on type. LLM-emitted JSON rules execute identically to
  human-written lambdas.
- EpochState.derived_lords + EpochState.ashtakavarga populate so
  DSL paths reach lord identities and BAV bindus without helpers.
- Open: existing v12-v15 rules still use Python lambdas. Migration
  to JSON conditions is mechanical and incremental — both paths
  co-exist. Track per-rule migration in application READMEs.

### CAV-NS-101 — Correlation-group max-pooling — closed
- Rule.correlation_group (Optional[str]); when two+ fired rules
  share a tag, only the largest |final_cf| survives into MYCIN
  aggregation. Audit-trail via FiredRuleTrace.suppressed_by_group.
- Addresses MYCIN's independence assumption flaw on highly-
  correlated astrological evidence (Saturn-9H + Saturn-aspect-9L).

### CAV-NS-102 — Rank-based commit gate — closed
- regression.evaluate_ranks + RankMetrics (MRR, top-1/3/10 recall,
  median rank). regression.rank_commit_gate accepts a proposed rule
  if MRR strictly improves AND top-3 recall does not regress beyond
  tolerance. Top-1 (exact-match) drops are allowed when rank metrics
  improve overall — matches Cox-style time-to-event evaluation.
- Old hit/miss commit_gate retained for back-compat; callers pick.

### CAV-NS-103 — Avasthas (combustion + planetary war) — open
- Reviewer flagged: shadbala captures quantitative strength but
  ignores Avasthas (qualitative state). A combust benefic Jupiter
  is "dead" for benefic purposes; a defeated planet in graha-yuddha
  should have its CF inverted/penalized.
- Required additions:
    * EpochState.PlanetEpochState.is_combust: bool
    * EpochState.PlanetEpochState.is_in_planetary_war: bool
    * EpochState.PlanetEpochState.war_winner: Optional[str]
- Combustion orbs (BPHS): Mercury 14°, Venus 10°, Mars 17°,
  Jupiter 11°, Saturn 15°. Need per-planet longitude data on
  EpochState (currently we only carry transit_sign + house).
- Graha-yuddha: two non-luminary planets within ~1° → defeated
  planet's CF flips. Same longitude requirement.
- Action: extend epoch_emitter to attach longitudes; add helpers
  in engine/avasthas.py that compute the booleans; update
  PlanetEpochState schema; add DSL paths
  "planets.<P>.is_combust" etc. so JSON rules can condition on it.

### CAV-NS-104 — Multi-dasha consensus (Yogini + Chara) — open
- Reviewer: Vimshottari alone caps accuracy ~70-75% per classical
  texts; major events require consensus across timekeeping systems.
- ml/pipelines/father_death_predictor/astro_engine/yogini_dasha.py
  and chara_dasha.py exist but aren't wired into epoch_emitter.
- Required:
    * Parallel dasha emission: emit_epochs returns triplets
      (vimshottari_dashas, yogini_dashas, chara_dashas) per SD.
    * EpochState.dashas → Dict[scheme, DashaStack] (currently single
      DashaStack assumed Vimshottari).
    * Consensus multiplier in cf_engine: scan all schemes' active
      sub-period lords; if multiple schemes agree on negative
      lords, multiply final_cf by 1.5. If schemes disagree (one
      protective, one afflictive), dampen.
- Big lift; requires careful schema migration (legacy callers
  expect single dasha stack).

### CAV-NS-105 — Overlap-fraction sampling for transits — open
- cf_engine.infer_cf accepts overlap_dampening per-rule factor
  (default 1.0 = no-op). Computing the actual fraction (sample
  N points within each SD, count how many trigger the rule)
  requires multi-point sampling in epoch_emitter.
- Current emitter computes positions at SD midpoint only; a fast
  Moon transit triggering a rule for ~2 days within a 5-day SD
  is treated identically to a 5-day-sustained Saturn transit.
- Action: emit_epochs(..., transit_sample_points=3) → re-evaluate
  predicates at start, mid, end; compute overlap fraction per
  fired rule; pass dict to infer_cf via overlap_dampening.
