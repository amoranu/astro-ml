# Father Longevity — Parashari CF-Rules Application

Predicts the dasha window (sookshma granularity) of a native's
father's death from natal chart data, using classical Parashari
techniques expressed as MYCIN-style certainty-factor (CF) rules.

Built on `astroql/engine` (CF inference, dasha emitter, shadbala,
sign-aspects) and `astroql/schemas` (Rule, EpochState, BirthDetails).

## Layout

```
father_longevity/
├── __init__.py          # exports RULES (= current canonical = v15)
├── eval.py              # evaluation harness (Hit@MD/AD/PD/SD + distance)
├── diagnose.py          # per-chart diagnostic dump
├── analyze.py           # batch failure analysis (rule firings, lord roles)
├── rules/
│   ├── __init__.py      # RULES_V12..V15 + RULES (current)
│   ├── v12.py           # Sun-karaka baseline
│   ├── v13.py           # + 9th/8th lord transit affliction
│   ├── v14.py           # + W-2 derived-lagna helpers + W-4 protective-PAD
│   └── v15.py           # + derived-F-lord AD rule + standalone PD = F-loss-lord
└── data/
    └── verified.json    # 19 web-verified father-death GT records
```

## Running

```bash
# Full evaluation on verified set
python -X utf8 -m astroql.applications.father_longevity.eval
python -X utf8 -m astroql.applications.father_longevity.eval v12  # ablation

# Per-chart diagnostic
python -X utf8 -m astroql.applications.father_longevity.diagnose 2

# Batch failure analysis with rule-firing comparison
python -X utf8 -m astroql.applications.father_longevity.analyze
```

## Rule version evolution

| Version | Rules | Hit@AD | Hit@PD | Hit@SD | ≤180d | Notes |
|---------|-------|--------|--------|--------|-------|-------|
| v12 | 13 | 6/19 | 1/19 | 1/19 | 4/19 | Sun-karaka baseline (BPHS Ch. 41) |
| v13 | 16 | 6/19 | 1/19 | 1/19 | 2/19 | + 9L/8L/lagna-lord transit family |
| v14 | 16 | 7/19 | 1/19 | 1/19 | 3/19 | + W-2 derived-lagna helpers, W-4 protective-PAD on `sun_dusthana` and `saturn_aspects` |
| v15 | 18 | 6/19 | 1/19 | 1/19 | 4/19 | + derived F-lord AD rule + standalone PD = F-loss-lord rule (BPHS Ch. 8 + Sanjay Rath) |

(Hit@MD = 18/19 = 95% across all versions.)

## Classical sources

- **BPHS Ch. 41** — Maraka and ayurdaya analysis
- **BPHS Ch. 8** — Derived-lagna theory for parents/relatives
- **BPHS Ayurdaya Adhyaya** — Longevity computation; 8L precedence
- **Sanjay Rath, *Crux of Vedic Astrology — Timing of Events* (1998)** — chapters on derived chart for father; dasha activation
- **Phaladeepika** — Saturn-Sun gochara; sign-based aspects
- **Sanjay Rath, Sukshma/Prana Dasa chapter** — W-4 PAD theory
- `astro-prod/tools/death_window_selection_methodology.txt` — operational rulebook (W-1 through W-12) used as primary RAG source

## Known limits

1. **Data noise floor.** `data/verified.json` filters the original
   ground truth to records with precise birth times (no hourly
   rounding) and web-search-verified father-death dates. 19 records
   is small; PD/SD numbers are sample-noise-bound.
2. **Pre-natal events excluded.** Vimshottari can't predict events
   prior to birth (no dasha defined). Bill Clinton (father died 3
   months before birth) is dropped.
3. **Vimshottari only.** Jaimini Niryana Shoola dasha + Pitrukaraka
   (W-5 in the methodology) is an orthogonal trigger system that's
   not yet implemented; some failures may be Jaimini-classical.
4. **Single karaka.** Atmakaraka/Pitrukaraka cross-check not
   implemented; rules use Sun-karaka and derived-lagna only.

## How to extend

Add a new ruleset under `rules/` (`v16.py`), then register it in
`rules/__init__.py`. Run `eval.py v16` to compare. Use `analyze.py
v16` to see per-chart firing differences vs the canonical version.

For new ground-truth data: add records to `data/verified.json` with
precise birth times and web-verified `father_death_date`. The eval
harness applies a deterministic seed-based 3-year window per record.
