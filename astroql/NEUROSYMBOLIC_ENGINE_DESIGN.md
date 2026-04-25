# RAG-Augmented Neuro-Symbolic Astrological Engine — Design

**Scope (v1):** Parashari tradition only.
**Status:** Design locked pending implementation mapping against existing `astroql/` and astro-prod `rag_engine`.

## Key scope decisions

- **Tradition:** Parashari only for v1. Other traditions deferred.
- **Regression testing:** Held-out split (not global accuracy on full GT set) — avoids overfitting rules to training set.
- **RAG:** Reuse astro-prod's `rag_engine.retrieve_for_tradition` rather than building a new vector DB. Parashari corpus already indexed there.
- **Ephemeris/state:** Reuse astro-prod `AstroEngine` (per standing memory rule) — do not use the thinner `ml/pipelines/father_death_predictor/astro_engine/`.
- **Engine reuse:** Map onto existing `astroql/` modules where possible (engine, rules, schemas, parser, planner, rag). Audit before writing new code.

## 1. System Architecture Overview

Two primary loops:

- **Inference Loop (Forward Pass):** Deterministic time-slicing, astronomical state generation, probabilistic rule evaluation.
- **Learning Loop (Backward Pass):** RAG-assisted automated theory revision where an LLM acts as critic to patch logic failures based on classical text retrieval.

## 2. Core Data Schemas (JSON)

### 2.1 Epoch State (`epoch_state.json`)

Pre-calculated astronomical snapshot for a discrete time window. **Granularity: sookshma dasha level** (MD → AD → PD → SD), locked 2026-04-24.

```json
{
  "epoch_id": "string",
  "start_time": "ISO8601",
  "end_time": "ISO8601",
  "dashas": {
    "maha": "string",
    "antar": "string",
    "pratyantar": "string",
    "sookshma": "string"
  },
  "planets": {
    "[PlanetName]": {
      "transit_sign": "string",
      "transit_house": "integer",
      "natal_house": "integer",
      "shadbala_coefficient": "float (0.0 to 1.0, normalized)",
      "shadbala_virupas": "float (raw, optional for debugging)",
      "is_retrograde": "boolean",
      "aspects_receiving": ["string"]
    }
  }
}
```

**Epoch boundary events:** sookshma lord change OR major transit event (slow-planet sign change, station, exact aspect formation) within a sookshma window. A new epoch is emitted on any such event.

**Cost budget:** ~100–500 sookshma periods per 24-month query window (see [run_dasha_depth.py](ml/pipelines/father_death_predictor/run_dasha_depth.py)). A full-lifetime query ≈ 4k–20k epochs. Module E must evaluate within target event window ± margin rather than all epochs, and cache epoch states per chart.

### 2.2 Rule Schema (`rule.json`)

Directed acyclic graph (DAG) node representing an astrological principle.

```json
{
  "rule_id": "string",
  "target_aspect": "string",
  "is_veto": "boolean",
  "base_cf": "float (-1.0 to 1.0)",
  "conditions": {
    "logic": "AND | OR",
    "nodes": [
      { "path": "string", "operator": "string", "value": "any" }
    ]
  },
  "modifiers": [
    {
      "condition": { "path": "string", "operator": "string", "value": "any" },
      "effect_cf": "float",
      "explanation": "string"
    }
  ],
  "subsumes_rules": ["string (rule_ids)"],
  "provenance": {
    "author": "human | llm_critic",
    "confidence": "float (0.0 to 1.0)",
    "citations": [
      { "text_chunk": "string", "source_id": "string" }
    ]
  }
}
```

### 2.3 Execution Trace (`trace.json`)

Log required by the LLM Critic to understand why the engine failed.

```json
{
  "query_id": "string",
  "target_aspect": "string",
  "final_score": "float",
  "rules_fired": [
    {
      "rule_id": "string",
      "initial_cf": "float",
      "strength_multiplier": "float",
      "modifiers_applied": ["string"],
      "final_cf": "float"
    }
  ],
  "rules_subsumed": ["string"]
}
```

## 3. Core Modules & Math Specifications

### Module A — Time-Slicer & State Generator

- **Input (all required):** Birth date, time, timezone, query window start, query window end. **No lifetime / open-ended query is ever accepted** — the time range is a mandatory input contract at every layer (CLI, API, engine, regression harness). Locked 2026-04-24.
- **Function:** Uses ephemeris (astro-prod `AstroEngine` / Swiss Ephemeris) to compute exact timestamps for Dasha changes and major planetary sign changes within the query window. Use `AstroEngine.calculate_dasha_sequence(..., include_sookshma=True)` for sookshma-depth output.
- **Output:** Array of Epoch State objects strictly covering `[query_start, query_end]`. Epochs that partially overlap the window are clipped.
- **Validation:** Reject `query_end − query_start > MAX_WINDOW` (default 10 years — tunable per use case; prevents accidental lifetime queries). Reject zero/negative windows.

### Module B — Inference Engine (Forward Pass)

Evaluates rules against an Epoch State using **defeasible reasoning + MYCIN certainty factors**.

1. **Filter:** Retrieve all rules matching `target_aspect`.
2. **Evaluate base logic:** Traverse JSON conditions tree.
3. **Defeasibility pruning (yoga-bhanga aware, locked 2026-04-24):** If Rule A is active and Rule B is active, and Rule A's `rule_id` ∈ Rule B's `subsumes_rules`, drop Rule A — **except** if Rule A has `is_veto: true`, then Rule B must *also* have `is_veto: true` to subsume it. Rationale: Parashari yoga-bhanga doctrine — "it takes a yoga to break a yoga." A veto (absolute denial like durmarana confluence, hard Balarishta) can only be neutralized by another veto (counter-yoga like Mahamrityunjaya, Vipareeta Raja), never by ordinary mitigating factors. Non-veto rules subsume normally.
4. **Veto check:** After pruning, if any surviving active rule has `is_veto: true`, return final score −1.0 immediately. (Subsumed vetoes are already dropped in step 3, so they do not fire.)
5. **CF modulation:** For each active rule, multiply `base_cf` by primary planet's normalized `shadbala_coefficient` (μ ∈ [0, 1]). See shadbala normalization below.

   **Shadbala normalization (locked 2026-04-24):** Classical Parashara per-planet required-strength thresholds in virupas:

   | Planet | Required (virupas) |
   |---|---|
   | Sun | 390 |
   | Moon | 360 |
   | Mars | 300 |
   | Mercury | 420 |
   | Jupiter | 390 |
   | Venus | 330 |
   | Saturn | 300 |
   | Rahu/Ketu | via dispositor (see below) |

   Formula (classical seven): `μ = min(shadbala_virupas / required[planet], 1.0)`

   **Rahu/Ketu (locked 2026-04-24, Parashari dispositor rule):** Nodes inherit their dispositor's normalized μ — "Rahu acts like its dispositor" is standard Parashari heuristic. `μ_Rahu = μ_of(sign_lord(Rahu's sign))`, and likewise for Ketu. If the dispositor is itself Rahu/Ketu (impossible classically since nodes don't own signs, but defensive), fall back to μ = 0.5.

   Rationale: μ = 0 when planet is utterly weak, μ = 1 when it meets/exceeds classical required strength. Guarantees `|μ * base_cf| < 1` since μ ≤ 1 and `|base_cf| < 1` (rule invariant). Replaces the old "0.0 to 2.0" range in the schema — raw virupas can still be stored as `shadbala_virupas` for debugging but aren't used in CF math.
6. **CF aggregation:** Combine CFs iteratively. **Locked 2026-04-24.**

   **Invariant:** Non-veto rules must have `base_cf ∈ (-1, 1)` (strictly open). Vetos are the only mechanism for absolute certainty (±1). Combined with μ ∈ [0, 1] (see §3.B shadbala normalization), this guarantees all pre-aggregation CFs lie in `(-1, 1)`, so the mixed-sign denominator is strictly positive.

   - Both positive: `CF_new = CF1 + CF2 * (1 - CF1)`
   - Both negative: `CF_new = CF1 + CF2 * (1 + CF1)`
   - Mixed signs: `CF_new = (CF1 + CF2) / (1 - min(|CF1|, |CF2|))`

   **Post-aggregation clip:** After each combination, clamp `CF_new` to `[-1 + ε, 1 - ε]` with `ε = 1e-9` to prevent floating-point drift and preserve the invariant for subsequent combinations.

   **Order invariance:** MYCIN combination is associative/commutative under the invariant, so aggregation order must not affect the result. Unit test this explicitly.

### Module C — RAG & Vector Database

- **v1 decision:** Use astro-prod `rag_engine.retrieve_for_tradition(tradition="parashari", ...)`. Wrap via existing `astroql/rag/pipeline.py`.
- **Metadata (in astro-prod corpus):** topic tags (e.g., marriage, debilitation), planet tags for precision filtering.

### Module D — LLM Critic (Backward Pass)

Triggered when a system prediction fails against a ground-truth chart.

1. **Query gen:** LLM produces 3 search queries from the execution trace.
2. **Retrieval:** astro-prod RAG returns top-K chunks (Parashari-only corpus).
3. **Synthesis:** LLM emits a new `rule.json` object. Prompt must enforce schema-valid JSON and populate `provenance.citations`.

### Module E — Regression Tester

- **Function:** Ingest LLM-proposed rule. Run Module B against a **held-out split** of ground-truth charts (not the full GT set).
- **Split (locked 2026-04-24):** 50/50 train/holdout. Train surfaces failures → critic input. Holdout is frozen eval set. Fixed seed per GT file, recorded in split manifest alongside chart IDs.
- **Commit logic:**
  - New rule commits iff `holdout_accuracy_new > holdout_accuracy_old` AND `train_accuracy_new >= train_accuracy_old - ε`.
  - Also require per-aspect accuracy doesn't regress beyond threshold on holdout, to avoid aggregate-masking.
- **Evaluation window:** Per the Module A contract, every GT record must carry a query window (e.g., for father-death, birth → birth+85y or a narrower window around the known event ± margin). No chart is ever evaluated over a lifetime default.

## 4. Open questions

**All resolved 2026-04-24.**

1. ~~CF mixed-sign formula~~ — standard MYCIN with strict `(-1, 1)` invariant on pre-aggregation CFs + ε-clip after each combine. See §3.B step 6.
2. ~~Shadbala coefficient range~~ — classical per-planet required-strength normalization to [0, 1]. See §3.B step 5.
3. ~~Epoch granularity~~ — sookshma dasha level (MD→AD→PD→SD) with intra-sookshma transit sub-events. See §2.1.
4. ~~Veto vs defeasibility~~ — Parashari yoga-bhanga model: only another veto can subsume a veto. Pruning happens before veto check, so subsumed vetoes don't fire. See §3.B step 3.
5. ~~Train/holdout split~~ — 50/50 with fixed per-GT-file seed. See Module E.
6. ~~Rahu/Ketu shadbala~~ — Parashari dispositor rule: nodes inherit their sign-lord's μ. See §3.B step 5.

**Hard input contract (locked 2026-04-24):** All queries — engine, regression harness, critic loop, CLI — MUST supply an explicit `[query_start, query_end]` time window. No lifetime or open-ended queries are accepted anywhere. Enforced at Module A validation. Default max window is 10 years; callers must opt in to longer via explicit parameter.

## 5. Mapping to existing code (audit 2026-04-24)

| Module | Reuse | Gap |
|---|---|---|
| **A** time-slicer/state | `astroql/chart/computer.py` (Swiss Ephemeris, all 16 vargas) + `astroql/features/parashari.py` (dasha candidates already extracted) | No `epoch_state.json` serializer; need thin wrapper emitting epoch_id + start/end_time windows. |
| **B** inference | `astroql/engine/rule_engine.py` has forward-chaining over YAML rules | **Biggest gap.** Uses scalar `strength` [0,1] — no MYCIN CF, no defeasibility pruning, no veto short-circuit, no mixed-sign CF formula. `resolver_engine/aggregator.py` does noisy-OR but not MYCIN. Must build. |
| **C** RAG | `astroql/rag/pipeline.py` wraps `rag_engine.retrieve_for_tradition` **already wired to Parashari**. 8 base passes + 2 longevity-specific + optional dynamic pass. Returns `[{text, source, score}]`. | `source` field maps to `provenance.citations[].source_id`; `text` → `text_chunk`. No explicit topic/planet tags in response (filtering is via input params). |
| **D** LLM critic | `astroql/discovery/` does symbolic pattern-mining + lift ranking, emits YAML rules — precedent for rule synthesis, **not** LLM-driven. | Must build LLM critic loop: trace → 3 queries → astro-prod RAG → schema-valid `rule.json` with `provenance.citations` populated from RAG `source`. |
| **E** regression | `astroql/benchmark/retrodiction_per_tradition.py` — per-tradition Hit@K + ±6mo, slices by `--start`/`--end` | No formal held-out split, no fixed seed, no per-aspect accuracy. Refactor to deterministic train/holdout partition with locked seed. |
| **Schemas** | `astroql/schemas/rules.py:Rule` has `confidence`, `priority_tier`, `antecedent`, `consequent` | Missing `base_cf`, `is_veto`, `subsumes_rules[]`, `modifiers[]` (with `effect_cf`), `provenance.citations[]`. Retrofit existing Parashari rules in `astroql/rules/parashari/longevity.yaml` (~85 KB, curated) to new schema. |

**Parashari-only scoping:** `astroql/planner/runner.py` uses an extractors dict keyed by `School.PARASHARI | JAIMINI | KP`. CLI `--schools` flag already accepts subsets. v1 wiring = skip Jaimini/KP entries in `_EXTRACTORS` and rule loader.

**Concrete RAG call shape for Module D:**
```python
import rag_engine
chunks = rag_engine.retrieve_for_tradition(
    query=critic_query,
    tradition="parashari",
    houses=[7, 8],          # from focus resolver
    planets=["Venus"],      # karakas
    category="marriage",    # or "longevity"
    sub_topics=[...],       # optional, preferred over category
    natal_context={...},    # chart dict
)
# chunks: list of {text, source, score}; top-K is 9 (11 for longevity queries)
```

**Existing assets worth seeding from:**
- `astroql/rules/parashari/longevity.yaml` — curated Parashari rules, retrofit target for new schema.
- `astroql/rules/calibrated_strengths.yaml` — existing per-rule calibration, may inform `base_cf` seed values.
- `astroql/PHASE_F_ARCHITECTURE.md`, `astroql/PHASE_F_PLAN.md`, `astroql/CAVEATS.md` (CAV-019 = LLM rule synthesis, CAV-034 = per-rule calibration, CAV-036 = age priors) — pre-existing design context.

## 6. Suggested v1 build order

1. **Schema retrofit** (Module B scaffolding): extend `astroql/schemas/rules.py` with `base_cf`, `is_veto`, `subsumes_rules`, `modifiers`, `provenance`. Backfill a small Parashari subset from `longevity.yaml` to prove round-trip.
2. **Epoch state emitter** (Module A): wrap `chart/computer.py` + `features/parashari.py` into `epoch_state.json` producer. Decide epoch granularity (open Q3).
3. **CF inference engine** (Module B): new evaluator doing filter → base logic → defeasibility pruning → veto → CF modulation → CF aggregation. Lock mixed-sign formula first (open Q1).
4. **Trace emitter**: log every fired rule + aggregation step to `trace.json`.
5. **Held-out regression harness** (Module E): fork `retrodiction_per_tradition.py` with fixed-seed train/holdout split + per-aspect accuracy gate.
6. **LLM critic loop** (Module D): trace → query gen → astro-prod RAG → rule synthesis with citations → regression harness gate.
