# Phase F: Per-Tradition Query Planner Architecture — End-to-End Plan

## 1. Goals

1. **Independent per-tradition execution** — Parashari, Jaimini, KP each produce their own ranked result set. No cross-tradition fusion.
2. **Reasoning-based scoring** — within each tradition, structured argument per window: support evidence + benefic/malefic attacks + classical exceptions.
3. **RRF fusion** across reasoning lines within a tradition (independent-evidence reward).
4. **Comprehensive astrological coverage** — cover all major classical factors from each tradition.
5. **Structured explanations** — every top window has a paragraph showing which lines confirmed it and which mitigators applied.

## 2. Astrological factors to cover (by tradition)

### 2.1 PARASHARI factors

**Already implemented:**
- Target/direct lord dignity, house, aspects, occupants, combust
- Sun (relation karaka) status
- Saturn (domain karaka, ayushkaraka)
- Maraka houses (2H, 7H from target), badhaka house
- Vimshottari MD/AD/PAD/SD chain analysis
- Chain composition (n_distinct_roles, full/partial/weak)
- Role overlay (n_md_roles, n_pad_roles, n_max_level_roles)
- Nakshatra-lord shadow chain
- Saturn-over-natal-Sun, Saturn over target sign, Jupiter over target
- Sade Sati, Ashtama Shani, Kantaka Shani
- Window duration / md=ad / md=pad

**To add:**
- **Mrityu Bhaga** — each planet has a "death degree" range; chain lord's longitude in mrityu bhaga = activates death significance
- **Vipreet Raja Yoga** — when 6L, 8L, 12L are in mutual exchange or in each other's signs → bad significations invert
- **Neecha Bhanga Raja Yoga** — debility cancelled when {dispositor in kendra/trikona, debilitated planet's lord exchanges, exalted planet aspects}
- **Vargottama** — same sign in D1 and D9 → amplifies that planet's significations
- **Argala** — planets in 2/4/11 from a house influence it; Virodhargala (in 12/10/3) blocks
- **Ashtakavarga** — Sarvashtakavarga points of target houses (low SAV = vulnerable)
- **D8 dignity of chain lords** — Ashtamamsha is the longevity divisional; debility here is critical
- **D60 dignity of chain lords** — Shastiamsa is the deepest karmic divisional
- **Mars-Saturn conjunction within 8 degrees** (Mrityu Yoga)
- **Sun-Saturn-Rahu cluster** (Pitri Dosha)
- **Eclipse axis crossing natal Sun or Moon** (Rahu-Ketu transit hits karaka)
- **Vedha** (transit obstruction) — Saturn transit "obstructed" by another planet 7 houses away
- **Yogini Dasha sub-period of afflictive lord** (cross-dasha confluence)

### 2.2 JAIMINI factors

**Already implemented (limited):**
- Chara karakas (7-karaka or 8-karaka scheme with PiK)
- Arudha lagna (A1) and other arudhas
- Atmakaraka

**To add:**
- **PiK (Pitrukaraka) dignity, house, aspects** — direct father karaka in 8-karaka scheme
- **Karakamsha lagna analysis** — sign of AK in D9 = personality + life events
- **9H from PiK** (father's own father — gives info about paternal lineage strength)
- **Argala on PiK** (planets in 2/4/11 from PiK strengthen; in 12/10/3 weaken)
- **Virodhargala on PiK** (counter-influences blocking father's longevity)
- **Sign-based aspect (rashi drishti)** — movable signs aspect fixed (except adjacent), fixed aspect movable, dual aspect dual. Different from planetary aspects.
- **Sthira karakas** (fixed assignments per house) vs Chara karakas (variable per chart)
- **Karako Bhava Nashaya** — when karaka occupies its own bhava, it spoils that bhava (e.g., PiK in 9H spoils 9H = father)
- **Padmaragas** — specific sign-occupancy patterns
- **Chara Dasha analysis** — sign-based dasha; for father, Chara MD lord vs PiK position
- **Niryana Sool Dasha** — specifically for death prediction (if computable; needs ephemeris of period boundaries)
- **Trikona Dasha** — relevant for major events
- **Drig Dasha** — for spirituality but used in some longevity calcs
- **Yogada / Kevala / Karaka Yoga** patterns

### 2.3 KP (Krishnamurti Paddhati) factors

**Already implemented (limited):**
- KP cuspal sub-lords (CSL) per house
- KP significators (planet → houses signified)
- Placidus cusps + sub-lord assignments

**To add:**
- **9H CSL signification analysis** — does 9H CSL signify 6/8/12 from itself? (= father will face illness/death/loss)
- **Star lord → Sub lord → Sub-sub lord hierarchy** for each chain lord
- **Significator strength matrix** — planets graded by how strongly they signify a house (occupants > occupants in lord's star > house lord > lord's nakshatra dispositors)
- **All-Signify rule** — when MD/AD/PAD all signify the "death cluster" (6+8+12 from focus) → strong death timing
- **Sub Lord agreement with Star Lord** — for each chain lord, do star and sub agree on signification? Disagreement = weakness
- **Sub-Sub Lord narrowing** for sub-period precision (KP claim: SSL gives ±2 days)
- **Ruling Planets at moment of consultation/queried-date** — KP horary method: planets ruling the moment confirm or deny event
- **Cuspal Sub Lord stellar position** — CSL's own star lord matters for effect
- **Significator hierarchy** (4 levels):
  1. Planets in the house (occupants)
  2. Planets in the star of occupants (most powerful)
  3. House lord
  4. Planets in the star of house lord
- **Negative significators** — planets signifying houses that NEGATE the focus
- **Promise vs trigger** — CSL gives promise (yes/no), dasha+transit gives trigger (when)

## 3. Independent-tradition execution

Output type changes from `QueryResult` (single ranked list) to:

```python
@dataclass
class MultiTraditionResult:
    parashari: TraditionResult
    jaimini: TraditionResult
    kp: TraditionResult

@dataclass
class TraditionResult:
    school: School
    plan: QueryPlan
    windows: List[RankedWindow]
    explanation: str            # per-tradition narrative
    n_lines_total: int
    aggregation_method: str     # "rrf" or "weighted_sum" etc.

@dataclass
class RankedWindow:
    start: datetime
    end: datetime
    rrf_score: float
    final_rank: int
    line_evidence: Dict[str, LineEvidence]   # per-reasoning-line breakdown
    structured_argument: str                  # multi-paragraph reasoning

@dataclass
class LineEvidence:
    line_id: str
    net_strength: float       # after support × attack × exception
    rank_in_line: int
    support: List[YogaFiring]   # [{yoga_id, strength, factors_matched}]
    attacks: List[YogaFiring]   # benefic/protective firings
    exceptions_fired: List[ExceptionFiring]   # [{exception_id, attenuation, reason}]
```

Benchmark reports per-tradition Hit@K independently:

```
PARASHARI:  Hit@10 ±6mo = X%
JAIMINI:    Hit@10 ±6mo = Y%
KP:         Hit@10 ±6mo = Z%
```

No combined metric. User compares traditions side-by-side.

## 4. Within-line scorer (graded + override-aware)

```
LineScorer.evaluate(candidate, chart_context):

  STEP 1 — Collect evidence
    supporting = [yoga firings with negative polarity matched against candidate]
    attacks    = [yoga firings with positive polarity matched against candidate]

  STEP 2 — Aggregate strengths
    support_strength = noisy_or(s.strength for s in supporting)
    attack_strength  = noisy_or(a.strength for a in attacks)

  STEP 3 — Apply attack mitigation
    net = support_strength * (1 - alpha * attack_strength)
    # alpha is per-line config (default 0.5);
    # benefic attacks subtract proportionally

  STEP 4 — Apply exceptions (gates)
    fired_exceptions = [e for e in exceptions
                        if e.condition(candidate, chart)]
    for e in fired_exceptions:
        if e.scope == "line":
            net *= e.attenuation
        elif e.scope == "yoga":
            # attenuate just the yogas the exception targets
            for yoga_id in e.applies_to_yogas:
                if yoga_id in supporting_ids:
                    recompute support_strength with that yoga's
                    strength reduced

  return LineEvidence(net, supporting, attacks, fired_exceptions)
```

## 5. Exception system (classical overrides)

```yaml
# astroql/query_plans/parashari/exceptions.yaml
exceptions:
  - id: neecha_bhanga_chain_lord
    scope: yoga
    applies_to_yogas: [classical_maraka_yoga, role_overlay_yoga]
    condition_type: chart_static
    condition: |
      any chain lord debilitated AND
      (its dispositor in kendra/trikona OR
       lord of debilitation sign in kendra/trikona OR
       exalted planet aspects the debilitated lord)
    attenuation: 0.3
    source: BPHS Ch.40, Phaladeepika 6

  - id: vipreet_raja_yoga
    scope: line
    applies_to_lines: [dasha_chain_strength]
    condition_type: chart_static
    condition: |
      6L in 8H/12H AND 8L in 6H/12H AND 12L in 6H/8H (mutual exchange)
    attenuation: 0.0   # hard override — bad becomes good
    source: BPHS Ch.36

  - id: sade_sati_exalted_saturn
    scope: yoga
    applies_to_yogas: [sade_sati, ashtama_shani, kantaka_shani]
    condition_type: chart_static
    condition: |
      Saturn natally in [Libra (exalted), Capricorn, Aquarius (own)]
      AND Saturn aspected by Jupiter
    attenuation: 0.4
    source: Phaladeepika Ch.26 v.18

  - id: jupiter_in_kendra_protection
    scope: line
    applies_to_lines: [dasha_chain_strength, transit_overlay]
    condition_type: chart_static
    condition: |
      Jupiter in 1/4/7/10 from lagna in own/exalted/friendly sign
    attenuation: 0.7   # 30% softening
    source: Saravali 30.50

  - id: pak_in_own_bhava
    # Jaimini-specific
    scope: line
    applies_to_lines: [pik_analysis]
    condition_type: chart_static
    condition: |
      Pitrukaraka (PiK) occupies the 9th house from lagna
    attenuation: 1.5   # AMPLIFIES (Karako Bhava Nashaya — karaka in own house spoils it)
    source: Jaimini Sutras Ch.1
```

Conditions can be expressions over chart facts (planet positions, dignities, house lords) — same evaluator as graded yoga factor conditions.

## 6. Reasoning lines per tradition

### 6.1 Parashari (8 lines)

```yaml
# astroql/query_plans/parashari/longevity.yaml
goal: predict father longevity event_negative timing
school: parashari
relationship: father
life_area: longevity

reasoning_lines:
  - id: static_promise
    description: "Is the natal father-longevity arc fundamentally afflicted?"
    yogas:
      supporting: [afflicted_target_lord_pratyantar]
      attacking:  [jupiter_aspects_direct]
    chart_facts:
      supporting:
        - rotated_lord in dusthana with dignity affliction
        - direct_lord in dusthana
        - Sun afflicted (combust + dusthana)
      attacking:
        - 9H aspected by exalted Jupiter
        - rotated_lord in own/exalted sign
    alpha: 0.5

  - id: dasha_chain_strength
    description: "Does this window's chain activate the affliction?"
    yogas:
      supporting: [classical_maraka_yoga, role_overlay_yoga,
                   eighth_from_target_dasha_full, maraka_dasha_full,
                   md_lord_is_8l, md_lord_is_maraka, md_lord_is_badhaka]
    alpha: 0.5

  - id: dasha_chain_concentration
    description: "Is activation time-compressed (sudden event)?"
    yogas:
      supporting: [time_compressed_yoga, same_lord_chain,
                   md_pad_same_planet_with_role, short_pad_role_confluence]
    alpha: 0.5

  - id: transit_overlay
    description: "Are slow-mover transits hitting sensitive points?"
    yogas:
      supporting: [father_specific_yoga, sade_sati, ashtama_shani,
                   saturn_transit_target, saturn_over_natal_sun_father]
      attacking:  [jupiter_transit_target_protective]
    alpha: 0.5

  - id: divisional_dignity
    description: "Are chain lords debilitated in D8 (longevity)?"
    chart_facts:
      supporting:
        - any chain lord debilitated in D8
        - any chain lord in 6/8/12 in D8
        - any chain lord debilitated in D60
    alpha: 0.5

  - id: yoga_mrityu_bhaga
    description: "Are chain lords in Mrityu Bhaga degrees?"
    chart_facts:
      supporting:
        - any chain lord longitude in mrityu_bhaga[lord]
    alpha: 0.5

  - id: weak_chain_carrier
    description: "Rescue line for charts with weak-chain death windows"
    yogas:
      supporting: [weak_chain_shadow_yoga,
                   weak_chain_with_nak_lord_relevant,
                   nak_lord_role_confluence]
    alpha: 0.5

  - id: ashtakavarga_check
    description: "Are target houses ashtakavarga-weak in this transit?"
    chart_facts:
      supporting:
        - SAV of 9H below 25 (weak house)
        - SAV of 4H below 25
        - chain lord's BAV in target houses < 4

aggregation:
  method: rrf
  k: 60
  weights:
    static_promise: 1.0
    dasha_chain_strength: 1.5
    dasha_chain_concentration: 1.0
    transit_overlay: 1.3
    divisional_dignity: 1.0
    yoga_mrityu_bhaga: 0.8
    weak_chain_carrier: 0.7
    ashtakavarga_check: 0.6

exceptions: [neecha_bhanga_chain_lord, vipreet_raja_yoga,
             sade_sati_exalted_saturn, jupiter_in_kendra_protection]
```

### 6.2 Jaimini (6 lines)

```yaml
# astroql/query_plans/jaimini/longevity.yaml
goal: predict father longevity event_negative timing
school: jaimini
relationship: father

reasoning_lines:
  - id: pik_analysis
    description: "Pitrukaraka (PiK) condition — direct father karaka"
    chart_facts:
      supporting:
        - PiK debilitated
        - PiK in dusthana (6/8/12)
        - PiK conjunct Rahu/Ketu/Saturn
        - 9th from PiK afflicted (paternal lineage weak)
      attacking:
        - PiK in own/exalted sign in kendra
        - benefic argala on PiK

  - id: karakamsha_analysis
    description: "Karakamsha lagna afflictions affecting father"
    chart_facts:
      supporting:
        - 9H from karakamsha contains malefics
        - 9H from karakamsha lord in dusthana
        - PiK in 6/8/12 from karakamsha
      attacking:
        - 9H from karakamsha contains exalted Jupiter

  - id: chara_dasha_father
    description: "Chara Dasha sign activation for father"
    chart_facts:
      supporting:
        - current Chara MD sign is 8th from PiK
        - current Chara MD lord aspects PiK
        - current Chara AD lord = lord of 8th from PiK
    requires_dasha: chara

  - id: rashi_drishti_overlay
    description: "Sign-based aspects on PiK / 9H"
    chart_facts:
      supporting:
        - PiK aspected (rashi drishti) by malefic-occupied sign
        - 9H aspected by 6/8/12 occupants

  - id: argala_on_pik
    description: "Argala / Virodhargala on PiK"
    chart_facts:
      supporting:
        - malefic argala on PiK (planets in 2/4/11 from PiK)
        - virodhargala blocks beneficial argala on PiK

  - id: arudha_pada_affliction
    description: "Arudha lagna and arudha-pada conditions"
    chart_facts:
      supporting:
        - A9 (father's arudha) in dusthana
        - A9 lord debilitated

aggregation:
  method: rrf
  k: 60
  weights:
    pik_analysis: 1.5
    karakamsha_analysis: 1.0
    chara_dasha_father: 1.3
    rashi_drishti_overlay: 0.8
    argala_on_pik: 0.9
    arudha_pada_affliction: 0.7

exceptions: [pak_in_own_bhava, jaimini_neecha_bhanga, ...]
```

### 6.3 KP (5 lines)

```yaml
# astroql/query_plans/kp/longevity.yaml
goal: predict father longevity event_negative timing
school: kp
relationship: father

reasoning_lines:
  - id: ninth_csl_signification
    description: "9H Cuspal Sub Lord signifies death cluster?"
    chart_facts:
      supporting:
        - 9H CSL signifies any of [6, 8, 12] from focus
        - 9H CSL's star lord signifies death cluster
        - 9H CSL is malefic and signifies dusthanas
      attacking:
        - 9H CSL signifies only auspicious houses (1, 5, 9)

  - id: dasha_chain_significators
    description: "Do MD/AD/PAD all signify the death cluster?"
    chart_facts:
      supporting:
        - MD lord signifies 6/8/12 from 9H
        - AD lord signifies 6/8/12 from 9H
        - PAD lord signifies 6/8/12 from 9H
        - all three agree (All-Signify rule)

  - id: significator_hierarchy_strength
    description: "How strongly do chain lords signify dusthanas?"
    chart_facts:
      supporting:
        - chain lord is occupant of 6H or 8H from 9H
        - chain lord is in star of dusthana occupant (most powerful KP signification)
        - chain lord is dusthana lord

  - id: sublord_starlord_agreement
    description: "Sub Lord and Star Lord agreement on signification"
    chart_facts:
      supporting:
        - chain lord's star lord and sub lord both signify dusthana
      attacking:
        - sub lord signifies auspicious house against star lord

  - id: ruling_planets_overlay
    description: "Ruling Planets at window midpoint confirm death cluster?"
    chart_facts:
      supporting:
        - Ruling Planets at midpoint include dusthana significators
        - Ascendant lord at midpoint signifies 6/8/12

aggregation:
  method: rrf
  k: 60
  weights:
    ninth_csl_signification: 2.0     # KP's primary indicator
    dasha_chain_significators: 1.5
    significator_hierarchy_strength: 1.2
    sublord_starlord_agreement: 0.9
    ruling_planets_overlay: 1.0

exceptions: [csl_signifies_only_auspicious, ...]
```

## 7. Implementation file map

```
astroql/
├── planner/
│   ├── __init__.py
│   ├── plan.py              # QueryPlan, ReasoningLine, ExceptionRule schemas
│   ├── lines.py             # ReasoningLine evaluator (within-line scorer)
│   ├── exceptions.py        # Exception condition evaluator
│   ├── fusion.py            # RRF + alternatives
│   └── runner.py            # Per-tradition planner.execute()
├── query_plans/
│   ├── parashari/
│   │   ├── longevity.yaml
│   │   └── exceptions.yaml
│   ├── jaimini/
│   │   ├── longevity.yaml
│   │   └── exceptions.yaml
│   └── kp/
│       ├── longevity.yaml
│       └── exceptions.yaml
├── schemas/
│   └── results.py           # Add MultiTraditionResult, TraditionResult, LineEvidence
├── benchmark/
│   ├── retrodiction.py      # Update to call planner.execute() per tradition,
│   │                        # report Hit@K per tradition independently
│   └── retrodiction_per_tradition.py   # New: focused per-school benchmark
└── rules/
    ├── parashari/longevity.yaml   # Existing yogas — no changes needed
    ├── jaimini/longevity.yaml     # Add new Jaimini-specific yogas
    └── kp/longevity.yaml          # Add new KP-specific yogas
```

## 8. New chart facts to expose (engine extensions)

For the new reasoning lines to evaluate, the chart computation needs to expose:

**Parashari additions:**
- `chart.mrityu_bhaga` — dict `{planet: (start_deg, end_deg)}`
- `chart.vargas['D8'].planet_dignities` — already there, ensure all chain lords have D8 dignity
- `chart.vargas['D60']` — extend if not computed
- `chart.argala` — dict `{house: [planets influencing], counter_argala: [planets blocking]}`
- `chart.ashtakavarga_per_house` — already partially there (SAV)
- `chart.vipreet_raja_yoga_present` — boolean + which lords
- `chart.neecha_bhanga_present` — boolean + which lord cancelled

**Jaimini additions:**
- `chart.pik_planet` — already there if 8-karaka scheme
- `chart.karakamsha_sign` — already partial
- `chart.chara_dasha` — needs computation (astro-prod might have it)
- `chart.rashi_drishti` — sign-based aspect graph
- `chart.argala_on_planet` — generalised argala calculator
- `chart.arudhas` — already there

**KP additions:**
- `chart.kp_significators` — already there as planet→houses
- `chart.kp_house_significators` — inverse: house→[planets, ranked by strength]
- `chart.ruling_planets_at(date)` — function to compute RPs for any moment
- `chart.kp_csl_signifies` — dict `{house: [houses_signified_by_its_csl]}`
- `chart.kp_sublord_starlord_agreement` — per planet, do star and sub agree?

## 9. Per-tradition independence guarantees

- Each tradition's planner runs in isolation; no cross-tradition state
- Each tradition's exceptions only apply to its own lines
- Each tradition reports its own confidence threshold + windows
- The benchmark output is `{parashari: BenchResult, jaimini: BenchResult, kp: BenchResult}` keyed by tradition
- No "school weight" combination logic anywhere in the new planner code path
- Legacy `Aggregator.aggregate()` (combined output) still exists for backward compatibility but isn't called by the new planner

## 10. Implementation phases (within Phase F)

| Sub-phase | Deliverable | Effort | Dependencies |
|---|---|---|---|
| F.1 | Schemas: QueryPlan, ReasoningLine, ExceptionRule, MultiTraditionResult, LineEvidence | 1.5 hr | none |
| F.2 | Within-line scorer (support/attack/exception) | 2 hr | F.1 |
| F.3 | Exception evaluator (chart-static + window-dynamic conditions) | 2 hr | F.1 |
| F.4 | RRF fusion module | 30 min | F.1 |
| F.5 | Per-tradition runner with isolated execution | 2 hr | F.1-F.4 |
| F.6 | Author Parashari plan (longevity.yaml + exceptions.yaml) using existing yogas | 1.5 hr | F.1-F.5 |
| F.7 | Author Jaimini plan + add 4-5 new Jaimini yogas | 4 hr | F.6 + Jaimini engine extensions |
| F.8 | Author KP plan + add 4-5 new KP yogas | 4 hr | F.6 + KP engine extensions |
| F.9 | Engine extensions: Mrityu Bhaga, Argala calculator, Vipreet/Neecha detection (Parashari) | 3 hr | F.6 |
| F.10 | Engine extensions: Chara Dasha, Rashi Drishti, generalized Argala (Jaimini) | 4 hr | F.7 |
| F.11 | Engine extensions: KP house significators, Ruling Planets, CSL signification analysis | 4 hr | F.8 |
| F.12 | Updated retrodiction benchmark — per-tradition Hit@K, ±6mo metric | 1 hr | F.5 |
| F.13 | Smoke-test legacy aggregator path still works (backward compat) | 30 min | F.12 |
| F.14 | Run v12 (Parashari only first), v13 (with Jaimini), v14 (with KP) — comparative benchmark | 1 hr runtime + analysis | F.12 |
| F.15 | Document architecture + per-tradition reasoning structure | 1 hr | F.14 |

**Total effort: ~32 hours = 4 focused days**

## 11. Expected outcomes

**Hit@K per tradition (target after Phase F):**

| Metric | Current (combined v9) | Parashari only (target) | Jaimini only (target) | KP only (target) |
|---|---|---|---|---|
| Hit@1 ±6mo | 13.8% | 25-35% | 15-25% | 25-35% |
| Hit@10 ±6mo | 17.2% | 40-50% | 20-30% | 35-45% |
| Hit@20 ±6mo | 24.1% | 55-65% | 30-40% | 45-55% |
| Median months off | 93 | 30-50 | 40-60 | 30-50 |

KP often outperforms Parashari for precise timing in classical practice; Parashari is best for promise + dasha; Jaimini is mid (good for relationship-specific via PiK).

**Structural wins:**
- Each tradition becomes interpretable on its own — astrologers can audit the reasoning per their school
- Adding new yogas means adding to one line, not redesigning the engine
- Adding new traditions (e.g., Tajik for annual chart) is just authoring a new query plan
- Exceptions are explicit and traceable — every override has a citation

## 12. Out of scope (Phase G+ later)

- Chart rectification (Phase E) — separate effort
- LLM-based explanation generation (per-line summaries currently template-based)
- Yogini Dasha (Phase D) — only if KP/Jaimini results don't already give the desired lift
- Cross-tradition consensus reporting (per user requirement: keep separate)
- Annual chart (Varshaphal/Tajik) tradition — could be added as 4th tradition later

---

**Implementation order: F.1 → F.15.** Checkpoint reports after each major sub-phase.
