"""Aggregate window-feature backtraces into rule candidates.

Why per-chart precision matters more than raw frequency:
    A pattern P that matches the death window in only 5 charts out of 200,
    but in each of those charts is the *only* window matching P (1 of ~700
    windows), is an extremely strong specialist rule. Population frequency
    would discard it; per-chart precision recovers it.

For each candidate pattern P (a feature=value constraint, or a set of them)
we compute:
    support           - # charts where P matches the death window
    n_charts_fired    - # charts where P matches *any* window
    cross_precision   - support / n_charts_fired
                        ("when P fires in a chart at all, how often is it
                         selecting the death window?")
    mean_specificity  - over the `support` charts, the average value of
                        (1 / windows_in_chart_matching_P)
                        High specificity = the pattern narrows down to a
                        tight subset of the chart's lifetime.
    avg_chart_windows - sanity check; ~700 typical
    coverage          - support / n_charts_total — how broadly the rule
                        speaks (population recall)

We surface two ranked lists:
    GENERALIST: high support × cross_precision (drives mass accuracy)
    SPECIALIST: high cross_precision × mean_specificity, support >= 3
                (rare patterns that nail their charts; emit as
                 narrow-antecedent rules with high strength)

The YAML emitter assigns:
    strength = clamp(0.3 + 0.6 * cross_precision, 0.3, 0.95)
    priority_tier = 1 if support >= 20 and cross_precision >= 0.6
                    else 2 if support >= 5
                    else 3
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .backtrace import DeathBacktrace, WindowFeatures


# ── Pattern definitions ─────────────────────────────────────────────
# A pattern is a callable WindowFeatures -> bool with a stable string key.

@dataclass
class Pattern:
    key: str                            # stable id for YAML emission
    description: str                    # human-readable
    matches: Callable[[WindowFeatures], bool]
    yaml_antecedent: List[Dict[str, Any]]  # rule.antecedent skeleton
    tags: List[str] = field(default_factory=list)


def _pattern_chain_role(role: str) -> Pattern:
    """Pattern: at least one chain lord plays `role`."""
    return Pattern(
        key=f"role_in_chain.{role}",
        description=f"chain contains a lord playing role={role}",
        matches=lambda w: role in w.roles_present,
        yaml_antecedent=[
            {"feature": "candidate.level", "op": "in",
             "value": ["AD", "PAD"]},
            {"feature": "candidate.lord_roles_present",
             "op": "contains", "value": role},
        ],
        tags=[role, "chain_role"],
    )


def _pattern_chain_role_at_level(role: str, level: str) -> Pattern:
    """Pattern: a SPECIFIC chain level (md/ad/pad) plays `role`."""
    field_map = {"md": "md_roles", "ad": "ad_roles", "pad": "pad_roles"}
    f = field_map[level]
    return Pattern(
        key=f"role_at_level.{level}.{role}",
        description=f"{level.upper()} lord plays role={role}",
        matches=lambda w, _f=f, _r=role: _r in getattr(w, _f),
        yaml_antecedent=[
            # we approximate at-level via the existing `lord_roles_present`
            # plus a chain_strength filter — fine for surfaced lifts but not
            # exact; the engine doesn't yet expose per-level role lists, so
            # this is a documented approximation for the discovery output.
            {"feature": "candidate.level", "op": "in",
             "value": ["AD", "PAD"]},
            {"feature": "candidate.lord_roles_present",
             "op": "contains", "value": role},
        ],
        tags=[role, "chain_role", level],
    )


def _pattern_chain_strength(strength: str) -> Pattern:
    return Pattern(
        key=f"chain_strength.{strength}",
        description=f"chain_strength = {strength}",
        matches=lambda w, _s=strength: w.chain_strength == _s,
        yaml_antecedent=[
            {"feature": "candidate.level", "op": "in",
             "value": ["AD", "PAD"]},
            {"feature": "candidate.chain_strength", "op": "eq",
             "value": strength},
        ],
        tags=["chain_strength", strength],
    )


def _pattern_n_distinct_gte(n: int) -> Pattern:
    return Pattern(
        key=f"n_distinct_roles_gte.{n}",
        description=f"n_distinct_roles >= {n}",
        matches=lambda w, _n=n: w.n_distinct_roles >= _n,
        yaml_antecedent=[
            {"feature": "candidate.level", "op": "in",
             "value": ["AD", "PAD"]},
            {"feature": "candidate.n_distinct_roles", "op": "gte",
             "value": n},
        ],
        tags=["confluence", f"n_distinct>={n}"],
    )


def _pattern_transit(planet: str) -> Pattern:
    flag = f"{planet.lower()}_transit_target"
    return Pattern(
        key=f"transit.{planet.lower()}",
        description=f"{planet} transits natal target sign at window mid",
        matches=lambda w, _f=flag: bool(getattr(w, _f, False)),
        yaml_antecedent=[
            {"feature": "candidate.level", "op": "in",
             "value": ["AD", "PAD"]},
            {"feature": f"candidate.{flag}", "op": "eq", "value": True},
        ],
        tags=["transit", planet.lower()],
    )


def _pattern_karaka_in_chain(planet: str) -> Pattern:
    flag_map = {
        "Saturn": "saturn_in_chain", "Sun": "sun_in_chain",
        "Jupiter": "jupiter_in_chain", "Mars": "mars_in_chain",
    }
    f = flag_map[planet]
    # Best YAML approximation: a chain hit on the planet → the planet must
    # carry one of its expected roles to register; closest current proxy is
    # role-in-chain "domain_karaka" (Saturn for longevity), "relation_karaka"
    # (Sun for father). Emit as a documented filter on roles.
    role_alias = {
        "Saturn": "domain_karaka", "Sun": "relation_karaka",
        "Jupiter": None, "Mars": None,
    }.get(planet)
    antecedent = [
        {"feature": "candidate.level", "op": "in", "value": ["AD", "PAD"]},
    ]
    if role_alias:
        antecedent.append({
            "feature": "candidate.lord_roles_present",
            "op": "contains", "value": role_alias,
        })
    return Pattern(
        key=f"chain_has.{planet.lower()}",
        description=f"chain includes {planet}",
        matches=lambda w, _f=f: bool(getattr(w, _f)),
        yaml_antecedent=antecedent,
        tags=["chain_has", planet.lower()],
    )


def _pattern_pad_in_dusthana_from_md() -> Pattern:
    return Pattern(
        key="pad_in_dusthana_from_md",
        description="PAD lord in 6/8/12 from MD lord (chain-internal "
                    "affliction)",
        matches=lambda w: w.pad_in_dusthana_from_md,
        yaml_antecedent=[
            {"feature": "candidate.level", "op": "eq", "value": "PAD"},
            # NOTE: requires extending features_schema to expose
            # candidate.pad_house_from_md_house. Documented in CAVEATS.
            {"feature": "candidate.pad_house_from_md_house",
             "op": "in", "value": [6, 8, 12]},
        ],
        tags=["chain_geometry", "pad_dusthana_from_md"],
    )


def _pattern_chain_dignity_count(field_name: str, n: int) -> Pattern:
    return Pattern(
        key=f"{field_name}_gte.{n}",
        description=f"{field_name} >= {n} in chain",
        matches=lambda w, _f=field_name, _n=n: getattr(w, _f) >= _n,
        yaml_antecedent=[
            {"feature": "candidate.level", "op": "in",
             "value": ["AD", "PAD"]},
            # Approximated via `candidate.lord_roles_present` for now;
            # we'd need a schema extension for true chain-dignity filters.
        ],
        tags=["chain_dignity", field_name],
    )


def _pattern_combo(p1: Pattern, p2: Pattern) -> Pattern:
    return Pattern(
        key=f"combo({p1.key}+{p2.key})",
        description=f"{p1.description} AND {p2.description}",
        matches=lambda w, a=p1, b=p2: a.matches(w) and b.matches(w),
        yaml_antecedent=p1.yaml_antecedent + [
            c for c in p2.yaml_antecedent
            if c not in p1.yaml_antecedent
        ],
        tags=sorted(set(p1.tags + p2.tags)),
    )


# ── Amplifier patterns (extended factors from backtrace) ──────────
# These are the candidate "narrowing" features. The compound-discovery
# loop crosses each broad pattern with each amplifier.

def _bool_pattern(field: str, label: str = None) -> Pattern:
    lbl = label or field
    return Pattern(
        key=f"amp.{field}",
        description=f"{lbl}",
        matches=lambda w, _f=field: bool(getattr(w, _f, False)),
        yaml_antecedent=[
            {"feature": f"candidate.{field}", "op": "eq", "value": True},
        ],
        tags=["amplifier", field],
    )


def _int_gte_pattern(field: str, n: int) -> Pattern:
    return Pattern(
        key=f"amp.{field}_gte_{n}",
        description=f"{field} >= {n}",
        matches=lambda w, _f=field, _n=n: getattr(w, _f, 0) >= _n,
        yaml_antecedent=[
            {"feature": f"candidate.{field}", "op": "gte", "value": n},
        ],
        tags=["amplifier", field, f">={n}"],
    )


def _str_eq_pattern(field: str, value: str) -> Pattern:
    return Pattern(
        key=f"amp.{field}_eq_{value}",
        description=f"{field} == {value}",
        matches=lambda w, _f=field, _v=value: getattr(w, _f, "") == _v,
        yaml_antecedent=[
            {"feature": f"candidate.{field}", "op": "eq", "value": value},
        ],
        tags=["amplifier", field, value],
    )


def amplifier_set() -> List[Pattern]:
    """Candidate amplifier features. Each is a per-window feature that
    might narrow a broad pattern's death-window selection."""
    out: List[Pattern] = []
    # Boolean amplifiers.
    for f in [
        "chain_lord_in_8h", "chain_lord_in_12h",
        "chain_lord_in_dusthana", "chain_lord_in_8h_from_target",
        "md_pad_same_planet", "md_ad_same_planet",
        "window_short", "window_long",
        "saturn_over_sun_at_mid",
        "saturn_or_jupiter_transit", "saturn_and_jupiter_transit",
        "saturn_in_chain", "sun_in_chain", "jupiter_in_chain",
        "rahu_or_ketu_in_chain", "mars_in_chain",
        "saturn_transit_target", "jupiter_transit_target",
        "pad_in_dusthana_from_md",
    ]:
        out.append(_bool_pattern(f))
    # Integer amplifiers.
    for n in (1, 2, 3):
        out.append(_int_gte_pattern("n_malefics_in_chain", n))
        out.append(_int_gte_pattern("n_benefics_in_chain", n))
    for n in (3, 4, 5):
        out.append(_int_gte_pattern("n_distinct_roles", n))
    for n in (1, 2):
        out.append(_int_gte_pattern("n_debilitated_in_chain", n))
        out.append(_int_gte_pattern("n_combust_in_chain", n))
    # Categorical amplifiers.
    for s in ("full", "partial", "weak"):
        out.append(_str_eq_pattern("chain_strength", s))
    for b in ("child", "young", "mid", "old"):
        out.append(_str_eq_pattern("age_band", b))
    return out


def broad_set_for_compounding() -> List[Pattern]:
    """The broad patterns whose precision we want to amplify.
    Each compound = (broad AND amplifier)."""
    return [
        _pattern_chain_role("domain_karaka"),
        _pattern_chain_role("maraka"),
        _pattern_chain_role("eighth_from_target"),
        _pattern_chain_role("rotated_lord"),
        _pattern_chain_role("badhaka_lord"),
        _pattern_n_distinct_gte(3),
        _pattern_chain_strength("full"),
        _pattern_chain_strength("partial"),
    ]


def default_pattern_set() -> List[Pattern]:
    """Curated set of patterns to evaluate. Easy to extend."""
    role_names = [
        "rotated_lord", "direct_lord", "badhaka_lord",
        "eighth_from_target", "maraka", "relation_karaka",
        "domain_karaka",
    ]
    out: List[Pattern] = []
    for r in role_names:
        out.append(_pattern_chain_role(r))
        for lvl in ("md", "ad", "pad"):
            out.append(_pattern_chain_role_at_level(r, lvl))
    for s in ("full", "partial", "weak"):
        out.append(_pattern_chain_strength(s))
    for n in (3, 4, 5, 6):
        out.append(_pattern_n_distinct_gte(n))
    out.append(_pattern_transit("saturn"))
    out.append(_pattern_transit("jupiter"))
    for k in ("Saturn", "Sun", "Jupiter", "Mars"):
        out.append(_pattern_karaka_in_chain(k))
    out.append(_pattern_pad_in_dusthana_from_md())
    out.append(_pattern_chain_dignity_count("n_debilitated_in_chain", 1))
    out.append(_pattern_chain_dignity_count("n_combust_in_chain", 1))

    # A small set of meaningful combos to surface joint conditions.
    maraka = _pattern_chain_role("maraka")
    eighth = _pattern_chain_role("eighth_from_target")
    sat_tr = _pattern_transit("saturn")
    full = _pattern_chain_strength("full")
    out += [
        _pattern_combo(maraka, sat_tr),
        _pattern_combo(eighth, sat_tr),
        _pattern_combo(maraka, full),
        _pattern_combo(eighth, full),
    ]
    return out


# ── Lift computation ───────────────────────────────────────────────

@dataclass
class FeatureLift:
    pattern_key: str
    description: str
    support: int                  # charts where pattern matches death window
    n_charts_fired: int           # charts where pattern matches >=1 window
    n_charts_total: int
    cross_precision: float        # support / max(1, n_charts_fired)
    mean_specificity: float       # avg 1/coverage_in_chart over death-match charts
    coverage: float               # support / n_charts_total
    avg_windows_per_chart: float
    yaml_antecedent: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


def aggregate_lifts(
    backtraces: List[DeathBacktrace],
    patterns: Optional[List[Pattern]] = None,
) -> List[FeatureLift]:
    """For each pattern, compute support/precision/specificity across train.

    Charts whose death window couldn't be located are excluded from
    `support`/`cross_precision` calculations but still contribute to the
    "any-window" count if the pattern fires in any of their windows.
    """
    if patterns is None:
        patterns = default_pattern_set()

    valid = [b for b in backtraces if b.death_window is not None]
    n_total = len(valid)
    if n_total == 0:
        return []

    out: List[FeatureLift] = []
    avg_windows = (
        sum(len(b.all_windows) for b in valid) / n_total
        if n_total else 0.0
    )

    for p in patterns:
        support = 0
        n_fired = 0
        spec_sum = 0.0
        for b in valid:
            n_match = sum(1 for w in b.all_windows if p.matches(w))
            if n_match > 0:
                n_fired += 1
            assert b.death_window is not None  # narrowed above
            if p.matches(b.death_window):
                support += 1
                spec_sum += 1.0 / max(1, n_match)
        cross_prec = support / n_fired if n_fired > 0 else 0.0
        mean_spec = spec_sum / support if support > 0 else 0.0
        coverage = support / n_total
        out.append(FeatureLift(
            pattern_key=p.key,
            description=p.description,
            support=support,
            n_charts_fired=n_fired,
            n_charts_total=n_total,
            cross_precision=cross_prec,
            mean_specificity=mean_spec,
            coverage=coverage,
            avg_windows_per_chart=avg_windows,
            yaml_antecedent=p.yaml_antecedent,
            tags=p.tags,
        ))
    return out


# ── YAML rule emission ─────────────────────────────────────────────

def _strength_for(lift: FeatureLift) -> float:
    s = 0.3 + 0.6 * lift.cross_precision
    return max(0.3, min(0.95, round(s, 2)))


def _tier_for(lift: FeatureLift) -> int:
    if lift.support >= 20 and lift.cross_precision >= 0.6:
        return 1
    if lift.support >= 5:
        return 2
    return 3


@dataclass
class CompoundLift:
    """A compound (broad ∧ amplifier) pattern's discriminative metrics."""
    broad_key: str
    amplifier_key: str
    broad_support: int            # broad's death-window matches (baseline)
    compound_support: int         # compound's death-window matches
    compound_n_fired: int         # compound matches >=1 window in N charts
    compound_precision: float     # compound_support / compound_n_fired
    compound_specificity: float   # mean 1/(windows matching compound)
    broad_precision: float        # baseline for comparison
    precision_uplift: float       # compound_precision - broad_precision
    yaml_antecedent: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)


def discover_compounds(
    backtraces: List[DeathBacktrace],
    *,
    broads: Optional[List[Pattern]] = None,
    amps: Optional[List[Pattern]] = None,
    min_compound_support: int = 5,
) -> List[CompoundLift]:
    """For each (broad, amp) pair, compute compound vs broad uplift.

    The interesting compounds are ones where adding `amp` to `broad`:
      - keeps support >= min_compound_support (still useful)
      - lifts precision noticeably above broad alone
      - improves specificity (narrows the in-chart match set)
    """
    if broads is None:
        broads = broad_set_for_compounding()
    if amps is None:
        amps = amplifier_set()
    valid = [b for b in backtraces if b.death_window is not None]
    if not valid:
        return []

    # Pre-compute per-broad baseline.
    broad_metrics: Dict[str, Tuple[int, int, float]] = {}
    for B in broads:
        sup = sum(1 for b in valid if B.matches(b.death_window))
        n_fired = sum(
            1 for b in valid
            if any(B.matches(w) for w in b.all_windows)
        )
        prec = sup / n_fired if n_fired else 0.0
        broad_metrics[B.key] = (sup, n_fired, prec)

    out: List[CompoundLift] = []
    for B in broads:
        b_sup, _, b_prec = broad_metrics[B.key]
        if b_sup < min_compound_support:
            continue
        for A in amps:
            comp_match = (
                lambda w, _b=B, _a=A: _b.matches(w) and _a.matches(w)
            )
            sup = sum(1 for b in valid if comp_match(b.death_window))
            if sup < min_compound_support:
                continue
            n_fired = 0
            spec_sum = 0.0
            for b in valid:
                n_match = sum(1 for w in b.all_windows if comp_match(w))
                if n_match > 0:
                    n_fired += 1
                if comp_match(b.death_window):
                    spec_sum += 1.0 / max(1, n_match)
            prec = sup / n_fired if n_fired else 0.0
            spec = spec_sum / sup if sup else 0.0
            antecedent = list(B.yaml_antecedent) + [
                c for c in A.yaml_antecedent
                if c not in B.yaml_antecedent
            ]
            out.append(CompoundLift(
                broad_key=B.key,
                amplifier_key=A.key,
                broad_support=b_sup,
                compound_support=sup,
                compound_n_fired=n_fired,
                compound_precision=prec,
                compound_specificity=spec,
                broad_precision=b_prec,
                precision_uplift=prec - b_prec,
                yaml_antecedent=antecedent,
                description=f"{B.description} AND {A.description}",
                tags=sorted(set(B.tags + A.tags + ["compound"])),
            ))
    return out


def render_compound_rules(
    compounds: List[CompoundLift],
    *,
    min_uplift: float = 0.15,
    min_support: int = 5,
) -> List[Dict[str, Any]]:
    """Convert compound lifts into YAML rule entries.

    Filters to compounds whose precision exceeds the broad baseline by
    at least `min_uplift`. Strength = clamp(0.4 + 0.5 * uplift, 0.4, 0.9).
    Tier = 1 if uplift >= 0.25 and support >= 10, else 2.
    """
    out: List[Dict[str, Any]] = []
    for c in compounds:
        if c.compound_support < min_support:
            continue
        if c.precision_uplift < min_uplift:
            continue
        strength = max(0.4, min(0.9, round(0.4 + 0.5 * c.precision_uplift, 2)))
        tier = (1 if c.precision_uplift >= 0.25 and c.compound_support >= 10
                else 2)
        rid_amp = c.amplifier_key.replace(".", "_")
        rid_broad = c.broad_key.replace(".", "_")
        out.append({
            "rule_id": (
                f"parashari.longevity.compound."
                f"{rid_broad}_x_{rid_amp}.01"
            ),
            "school": "parashari",
            "source": (
                f"ml/discovery (compound) — broad={c.broad_key} "
                f"amp={c.amplifier_key} sup={c.compound_support} "
                f"prec={c.compound_precision:.2f} "
                f"uplift=+{c.precision_uplift:.2f}"
            ),
            "rule_type": "dasha_timing",
            "applicable_to": {
                "relationships": [
                    "self", "father", "mother", "spouse", "children",
                    "elder_sibling", "younger_sibling",
                ],
                "life_areas": ["longevity"],
                "effects": ["event_negative"],
            },
            "antecedent": c.yaml_antecedent,
            "consequent": {
                "polarity": "negative",
                "strength": strength,
                "timing_hint": (
                    f"compound: {c.broad_key} narrowed by "
                    f"{c.amplifier_key}; precision {c.compound_precision:.2f} "
                    f"vs broad {c.broad_precision:.2f}"
                ),
            },
            "confidence": strength,
            "tags": c.tags,
            "priority_tier": tier,
            "notes": (
                f"Discovered compound. Broad support={c.broad_support} "
                f"(prec {c.broad_precision:.2f}); "
                f"compound support={c.compound_support} "
                f"(prec {c.compound_precision:.2f}, "
                f"specificity {c.compound_specificity:.3f})."
            ),
        })
    return out


def render_rule_candidates(
    lifts: List[FeatureLift],
    *,
    min_support_general: int = 20,
    min_support_specialist: int = 3,
    min_precision_specialist: float = 0.5,
    min_specificity_specialist: float = 0.05,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert lifts into two YAML rule lists: generalists + specialists.

    Returns (generalists, specialists). A specialist rule is a low-support
    high-precision pattern that nails its few charts narrowly; ship it
    with a strong strength but a narrow antecedent.
    """
    generals: List[Dict[str, Any]] = []
    specs: List[Dict[str, Any]] = []

    for lift in lifts:
        if lift.support == 0:
            continue
        rule = {
            "rule_id": f"parashari.longevity.discovered.{lift.pattern_key}",
            "school": "parashari",
            "source": "ml/discovery — backtrace from train set",
            "rule_type": "dasha_timing",
            "applicable_to": {
                "relationships": [
                    "self", "father", "mother", "spouse", "children",
                    "elder_sibling", "younger_sibling",
                ],
                "life_areas": ["longevity"],
                "effects": ["event_negative"],
            },
            "antecedent": lift.yaml_antecedent,
            "consequent": {
                "polarity": "negative",
                "strength": _strength_for(lift),
                "timing_hint": (
                    f"discovered: support={lift.support}/"
                    f"{lift.n_charts_total}, "
                    f"precision={lift.cross_precision:.2f}, "
                    f"specificity={lift.mean_specificity:.3f}"
                ),
            },
            "confidence": _strength_for(lift),
            "tags": lift.tags + ["discovered"],
            "priority_tier": _tier_for(lift),
            "notes": lift.description,
        }
        is_generalist = lift.support >= min_support_general
        is_specialist = (
            lift.support >= min_support_specialist
            and lift.cross_precision >= min_precision_specialist
            and lift.mean_specificity >= min_specificity_specialist
            and not is_generalist
        )
        if is_generalist:
            generals.append(rule)
        elif is_specialist:
            specs.append(rule)
    return generals, specs
