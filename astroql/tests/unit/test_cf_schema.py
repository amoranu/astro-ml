"""CF neuro-symbolic schema tests (NEUROSYMBOLIC_ENGINE_DESIGN.md §2.2).

Covers the v1 schema retrofit:
  - CF-native rules load with explicit base_cf / is_veto / modifiers /
    provenance / subsumes_rules.
  - Legacy rules (polarity + strength) and graded rules (factors +
    base_strength) continue to load and derive effective_base_cf.
  - Loader rejects invariant violations:
      * non-veto with |base_cf| >= 1
      * veto without explicit ±1.0 base_cf
      * base_cf == 0.0 (no-op)
      * modifier effect_cf outside (-1, 1)
      * provenance citation missing source_id or text_chunk
      * yoga-bhanga violation: non-veto subsuming veto
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from astroql.rules.loader import (
    RuleLoadError,
    StructuredRuleLibrary,
    _validate_rule,
    validate_yoga_bhanga,
)
from astroql.schemas.enums import School


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def schema() -> dict:
    path = Path(__file__).resolve().parents[2] / "rules" / (
        "features_schema.yaml"
    )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _base_rule(**overrides) -> dict:
    raw = {
        "rule_id": "test.rule.01",
        "school": "parashari",
        "source": "test",
        "antecedent": [{
            "feature": "primary_house_data.rotated.lord_house",
            "op": "in",
            "value": [6, 8, 12],
        }],
        "applicable_to": {
            "relationships": ["self"],
            "life_areas": ["longevity"],
            "effects": ["event_negative"],
        },
        # Default primary_planet so CF-native tests pass without each
        # test having to re-declare it. Tests that want to exercise
        # the missing-primary-planet path override this to None.
        "primary_planet": "Sun",
    }
    raw.update(overrides)
    return raw


# ── Happy path: production rule library loads cleanly ───────────────

def test_full_parashari_library_loads():
    lib = StructuredRuleLibrary()
    rules = lib.all_rules(School.PARASHARI)
    # 90 legacy + 4 CF-native samples = 94 at time of writing; new
    # curated rules may be added later so assert a floor.
    assert len(rules) >= 90
    cf_native = [r for r in rules if r.base_cf is not None]
    assert len(cf_native) >= 4
    vetoes = [r for r in rules if r.is_veto]
    assert len(vetoes) >= 2  # durmarana + mahamrityunjaya samples


def test_legacy_rule_derives_effective_base_cf():
    lib = StructuredRuleLibrary()
    rules = lib.all_rules(School.PARASHARI)
    legacy = next(
        r for r in rules
        if r.base_cf is None
        and r.consequent.get("polarity") == "negative"
        and r.consequent.get("strength") is not None
    )
    assert legacy.effective_base_cf == -float(legacy.consequent["strength"])


def test_graded_rule_derives_ceiling_from_base_strength():
    lib = StructuredRuleLibrary()
    rules = lib.all_rules(School.PARASHARI)
    graded = next(
        r for r in rules
        if r.factors
        and r.base_cf is None
        and r.consequent.get("polarity") == "negative"
    )
    assert graded.effective_base_cf == -graded.base_strength


def test_cf_native_rule_uses_explicit_base_cf():
    lib = StructuredRuleLibrary()
    rules = lib.all_rules(School.PARASHARI)
    r = next(
        r for r in rules
        if r.rule_id == "parashari.longevity.rotated_lord_afflicted.cf01"
    )
    assert r.base_cf == -0.55
    assert r.effective_base_cf == -0.55
    assert not r.is_veto
    assert len(r.modifiers) == 1
    assert r.modifiers[0].effect_cf == -0.15
    assert r.provenance is not None
    assert r.provenance.author == "human"
    assert len(r.provenance.citations) == 1


def test_yoga_bhanga_veto_subsumes_veto_passes():
    lib = StructuredRuleLibrary()
    rules = lib.all_rules(School.PARASHARI)
    mmy = next(
        r for r in rules
        if r.rule_id == "parashari.longevity.mahamrityunjaya_yoga.veto01"
    )
    assert mmy.is_veto
    assert "parashari.longevity.durmarana_confluence.veto01" in (
        mmy.subsumes_rules
    )


# ── Error paths: invariant violations ───────────────────────────────

def test_non_veto_base_cf_at_one_rejected(schema):
    raw = _base_rule(rule_id="test.bad_cf_pos", base_cf=1.0)
    with pytest.raises(RuleLoadError, match="open interval"):
        _validate_rule(raw, schema)


def test_non_veto_base_cf_at_minus_one_rejected(schema):
    raw = _base_rule(rule_id="test.bad_cf_neg", base_cf=-1.0)
    with pytest.raises(RuleLoadError, match="open interval"):
        _validate_rule(raw, schema)


def test_non_veto_base_cf_zero_rejected(schema):
    raw = _base_rule(rule_id="test.bad_cf_zero", base_cf=0.0)
    with pytest.raises(RuleLoadError, match="no-op"):
        _validate_rule(raw, schema)


def test_veto_without_base_cf_rejected(schema):
    raw = _base_rule(rule_id="test.veto_missing_cf", is_veto=True)
    with pytest.raises(RuleLoadError, match="is_veto=true requires"):
        _validate_rule(raw, schema)


def test_veto_with_mid_base_cf_rejected(schema):
    raw = _base_rule(
        rule_id="test.veto_half",
        is_veto=True,
        base_cf=-0.5,
    )
    with pytest.raises(RuleLoadError, match="is_veto=true requires"):
        _validate_rule(raw, schema)


def test_modifier_effect_cf_out_of_range_rejected(schema):
    raw = _base_rule(
        rule_id="test.bad_modifier",
        base_cf=-0.5,
        modifiers=[{
            "condition": {
                "feature": "primary_house_data.rotated.lord_retrograde",
                "op": "eq",
                "value": True,
            },
            "effect_cf": 1.0,  # invalid: must be in open interval
            "explanation": "",
        }],
    )
    with pytest.raises(RuleLoadError, match="effect_cf"):
        _validate_rule(raw, schema)


def test_modifier_explanation_roundtrips(schema):
    raw = _base_rule(
        rule_id="test.modifier_explanation",
        base_cf=-0.5,
        modifiers=[{
            "condition": {
                "feature": "primary_house_data.rotated.lord_retrograde",
                "op": "eq",
                "value": True,
            },
            "effect_cf": -0.1,
            "explanation": "retrograde deepens affliction",
        }],
    )
    rule = _validate_rule(raw, schema)
    assert rule.modifiers[0].explanation == (
        "retrograde deepens affliction"
    )


def test_cf_native_non_veto_requires_primary_planet(schema):
    raw = _base_rule(
        rule_id="test.missing_primary_planet",
        base_cf=-0.5,
        primary_planet=None,
    )
    with pytest.raises(RuleLoadError, match="primary_planet"):
        _validate_rule(raw, schema)


def test_primary_planet_unknown_rejected(schema):
    raw = _base_rule(
        rule_id="test.bad_primary_planet",
        base_cf=-0.5,
        primary_planet="Chiron",
    )
    with pytest.raises(RuleLoadError, match="primary_planet"):
        _validate_rule(raw, schema)


def test_veto_without_primary_planet_ok(schema):
    # Vetoes short-circuit to ±1.0, so μ is irrelevant and
    # primary_planet is optional for them.
    raw = _base_rule(
        rule_id="test.veto_no_primary_planet",
        is_veto=True,
        base_cf=-1.0,
        primary_planet=None,
    )
    rule = _validate_rule(raw, schema)
    assert rule.primary_planet is None
    assert rule.is_veto


def test_veto_with_modifiers_rejected(schema):
    raw = _base_rule(
        rule_id="test.veto_with_modifiers",
        is_veto=True,
        base_cf=-1.0,
        modifiers=[{
            "condition": {
                "feature": "primary_house_data.rotated.lord_retrograde",
                "op": "eq",
                "value": True,
            },
            "effect_cf": -0.1,
        }],
    )
    with pytest.raises(RuleLoadError, match="veto rules cannot carry"):
        _validate_rule(raw, schema)


def test_modifier_missing_condition_rejected(schema):
    raw = _base_rule(
        rule_id="test.modifier_no_cond",
        base_cf=-0.5,
        modifiers=[{"effect_cf": -0.1}],
    )
    with pytest.raises(RuleLoadError, match="condition"):
        _validate_rule(raw, schema)


def test_provenance_citation_missing_source_id_rejected(schema):
    raw = _base_rule(
        rule_id="test.prov_missing_sid",
        base_cf=-0.5,
        provenance={
            "author": "human",
            "confidence": 0.5,
            "citations": [{"text_chunk": "hello"}],
        },
    )
    with pytest.raises(RuleLoadError, match="source_id"):
        _validate_rule(raw, schema)


def test_provenance_citation_missing_text_chunk_rejected(schema):
    raw = _base_rule(
        rule_id="test.prov_missing_txt",
        base_cf=-0.5,
        provenance={
            "author": "human",
            "confidence": 0.5,
            "citations": [{"source_id": "abc"}],
        },
    )
    with pytest.raises(RuleLoadError, match="text_chunk"):
        _validate_rule(raw, schema)


def test_provenance_bad_author_rejected(schema):
    raw = _base_rule(
        rule_id="test.prov_bad_author",
        base_cf=-0.5,
        provenance={"author": "unknown_source"},
    )
    with pytest.raises(RuleLoadError, match="author"):
        _validate_rule(raw, schema)


# ── Yoga-bhanga cross-rule validation ───────────────────────────────

def _veto(rid: str, subsumes=None) -> object:
    """Build a minimal Rule-like object for yoga-bhanga tests."""
    from astroql.schemas.rules import Rule
    return Rule(
        rule_id=rid,
        school=School.PARASHARI,
        source="test",
        is_veto=True,
        base_cf=-1.0,
        subsumes_rules=subsumes or [],
    )


def _non_veto(rid: str, subsumes=None) -> object:
    from astroql.schemas.rules import Rule
    return Rule(
        rule_id=rid,
        school=School.PARASHARI,
        source="test",
        is_veto=False,
        base_cf=-0.5,
        subsumes_rules=subsumes or [],
    )


def test_yoga_bhanga_veto_subsumes_veto_allowed():
    v1 = _veto("test.v1")
    v2 = _veto("test.v2", subsumes=["test.v1"])
    validate_yoga_bhanga([v1, v2])  # no raise


def test_yoga_bhanga_veto_subsumes_non_veto_allowed():
    # A veto subsuming a non-veto is fine (strong rule defeats weaker).
    n1 = _non_veto("test.n1")
    v1 = _veto("test.v1", subsumes=["test.n1"])
    validate_yoga_bhanga([n1, v1])  # no raise


def test_yoga_bhanga_non_veto_subsumes_non_veto_allowed():
    n1 = _non_veto("test.n1")
    n2 = _non_veto("test.n2", subsumes=["test.n1"])
    validate_yoga_bhanga([n1, n2])  # no raise


def test_yoga_bhanga_non_veto_subsumes_veto_rejected():
    v1 = _veto("test.v1")
    n1 = _non_veto("test.n1", subsumes=["test.v1"])
    with pytest.raises(RuleLoadError, match="yoga-bhanga"):
        validate_yoga_bhanga([v1, n1])


def test_yoga_bhanga_forward_ref_to_unknown_rule_skipped():
    # Cross-tradition or unresolved references do not fail load;
    # engine reports at eval time.
    n1 = _non_veto("test.n1", subsumes=["does.not.exist"])
    validate_yoga_bhanga([n1])  # no raise
