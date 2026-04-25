"""Rule + FiredRule (spec §5.5, §5.6).

Neuro-symbolic CF extensions (NEUROSYMBOLIC_ENGINE_DESIGN.md, 2026-04-24):
Rule adds `base_cf`, `is_veto`, `subsumes_rules`, `modifiers`, `provenance`
alongside the legacy `consequent.polarity + consequent.strength` form.
Legacy rules remain valid: `effective_base_cf` derives CF from polarity +
strength when `base_cf` is not explicitly set.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .enums import School


@dataclass
class Citation:
    source_id: str
    text_chunk: str


@dataclass
class Provenance:
    author: str = "human"
    confidence: float = 0.6
    citations: List[Citation] = field(default_factory=list)


@dataclass
class CFModifier:
    condition: Dict[str, Any]
    effect_cf: float
    explanation: str = ""


@dataclass
class Rule:
    rule_id: str
    school: School
    source: str
    source_uri: Optional[str] = None
    rule_type: str = "static"
    applicable_to: Dict[str, Any] = field(default_factory=dict)
    antecedent: List[Dict[str, Any]] = field(default_factory=list)
    consequent: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.6
    # Classical hierarchy:
    #   1 = primary indicator (8L MD as maraka, CSL of 9H signifying 8H)
    #   2 = secondary (rotated_lord, badhaka, karaka activation)
    #   3 = circumstantial (sun-saturn yoga, distant Argala)
    priority_tier: int = 2
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    # ── Graded-rule fields (optional; classical yoga form) ───────────
    # When `factors` is non-empty the engine evaluates this rule using the
    # graded path: `required` is a hard gate (all must match), `factors`
    # is a pool of weighted clauses contributing to a fractional strength
    # = base_strength * (sum_matched_weights / sum_total_weights).
    # `min_factors` is the minimum count of factors that must match for
    # the rule to fire at all (after the required gate passes).
    #
    # Legacy rules continue to work: if `factors` is empty, the engine
    # uses `antecedent` (all clauses required, fixed strength).
    required: List[Dict[str, Any]] = field(default_factory=list)
    factors: List[Dict[str, Any]] = field(default_factory=list)
    min_factors: int = 1
    base_strength: float = 0.0
    scoring: str = "weighted_fraction"   # weighted_fraction | linear_count

    # ── Chart-applicability (v38) ────────────────────────────────────
    # Evaluated ONCE per (rule, chart) before any per-window matching.
    # If any clause fails, the rule is skipped entirely for that chart —
    # it doesn't contribute to any window's score. Each clause has the
    # same shape as an antecedent clause (feature / op / value).
    #
    # Use cases:
    #   - saturn_transit_target: applicable_when Saturn is functional
    #     malefic for this lagna (skip for Taurus/Libra where Saturn=YK)
    #   - mars_in_chain_without_role_attack: applicable_when Mars is NOT
    #     yogakaraka (skip for Cancer/Leo where Mars=YK)
    #
    # Default = universal applicability (empty list).
    applicable_when: List[Dict[str, Any]] = field(default_factory=list)

    # ── CF neuro-symbolic fields (v1, 2026-04-24) ────────────────────
    # Invariant: non-veto rules MUST satisfy -1 < base_cf < 1 strictly
    # (vetos are the only |CF|=1 mechanism, see design §3.B step 6).
    # Vetos must set is_veto=True AND base_cf ∈ {-1.0, +1.0} explicitly.
    # subsumes_rules: rule_ids this rule defeats during pruning.
    #   Yoga-bhanga invariant (§3.B step 3): only a veto can subsume a
    #   veto. Loader enforces this when both rules are known.
    # modifiers: list of conditional CF adjustments; each has a single
    #   antecedent-style condition clause + effect_cf + explanation.
    # primary_planet: which planet's shadbala μ modulates this rule's
    #   CF at fire time. Required for CF-native non-veto rules (vetoes
    #   short-circuit so μ is irrelevant). Must be in the 9-planet
    #   Parashari set.
    base_cf: Optional[float] = None
    is_veto: bool = False
    subsumes_rules: List[str] = field(default_factory=list)
    modifiers: List[CFModifier] = field(default_factory=list)
    provenance: Optional[Provenance] = None
    primary_planet: Optional[str] = None
    # ── Correlation grouping (max-pool, design v2) ──────────────────
    # MYCIN's combine() assumes evidence pieces are statistically
    # independent. In astrology many rules are highly correlated —
    # e.g. "Saturn transits 9H" and "Saturn aspects 9L" both ride
    # on the same underlying planet placement. Combining them via
    # MYCIN compounds toward ±1 (false-positive amplification).
    #
    # When two or more fired rules share the same `correlation_group`
    # tag, the engine MAX-POOLS them: only the rule with the largest
    # |effective_cf| in the group survives into the MYCIN aggregation
    # step. Independent groups still combine via MYCIN.
    #
    # Suggested tag taxonomy (string is opaque to the engine):
    #   "saturn_affliction_to_father_bhava"
    #   "rahu_axis_on_sun"
    #   "jupiter_protection_to_sun"
    # None = rule contributes independently (legacy behavior).
    correlation_group: Optional[str] = None

    @property
    def effective_base_cf(self) -> float:
        """Static CF ceiling for this rule.

        Priority:
          1. Explicit `base_cf` (CF-native rule).
          2. Legacy fixed rule: sign(polarity) * consequent.strength.
          3. Legacy graded rule: sign(polarity) * base_strength.
             This is a ceiling — the CF engine scales by the matched /
             total factor-weight ratio at fire time.
        """
        if self.base_cf is not None:
            return self.base_cf
        consequent = self.consequent or {}
        polarity = consequent.get("polarity", "neutral")
        sign = {"positive": 1.0, "negative": -1.0}.get(polarity, 0.0)
        if sign == 0.0:
            return 0.0
        raw_strength = consequent.get("strength")
        if raw_strength is None and self.factors:
            raw_strength = self.base_strength
        try:
            mag = float(raw_strength) if raw_strength is not None else 0.0
        except (TypeError, ValueError):
            mag = 0.0
        return sign * mag


@dataclass
class FiredRule:
    rule: Rule
    bindings: Dict[str, Any] = field(default_factory=dict)
    polarity: str = "neutral"
    strength: float = 0.0
    window: Optional[Tuple] = None
    evidence_excerpt: str = ""
    # Indices into rule.modifiers for those whose condition matched in
    # the current epoch. The CF engine combines each fired modifier's
    # effect_cf into base_cf via cf_math.combine before μ modulation.
    fired_modifier_indices: List[int] = field(default_factory=list)
