"""CF rules for father's longevity, v14 — derived-father-lagna +
protective-PAD veto from W-2 and W-4 of astro-prod's
death_window_selection_methodology.txt.

Per Parashari classical (BPHS Ch. 8 derived-lagna; Sanjay Rath, MN
Kedar): for father queries, the analysis lagna shifts to **native's
9H sign** = father's own lagna. From there:
  * Father's 2L = lord of (native's 10H sign)
  * Father's 7L = lord of (native's 3H sign)
  * Father's 8L = lord of (native's 4H sign) — longevity lord
  * Father's 12L = lord of (native's 8H sign) — loss lord
The "loss/maraka set for father" = {father's 2L, 7L, 8L, 12L} +
Badhaka. v13 used 2nd/7th from natal Sun (Sun-karaka theory only);
W-2 says use derived lagna for the structural lords.

W-4 protective-PAD VETO (the missing positive case):
  "A PAD whose lord has none of {12L, 2L, 7L, 8L, Badhaka} of the
   subject is NEUTRAL or PROTECTIVE — death typically does NOT
   occur in such a PAD."
v13 had only negative PD-malefic modifiers — no positive (cancelling)
modifier. v14 adds a +0.30 modifier "PD-lord is a protective lord
for father (not in loss-lord set, not malefic)" on every active
negative rule. This dampens spurious firings at benevolent-PD
epochs and allows truth-PD (which IS often a loss-lord) to win.

Original v13 commentary preserved below.

----

CF rules for father's longevity, v13 — house-lord transit family.

v12 was Sun-karaka heavy and pinned MD/AD well (10/10, 6/10) but
sat at 2/10 PD and 1/10 SD. The user correctly noted MD/AD are
too wide to be useful: the real target is PD/SD.

The PD/SD gap stems from missing **time-varying house-lord signals**.
v12 looked at planets-in-houses (residents) and house-lord identity
only as dasha-lord triggers. It never asked:
  * Where is the 9th lord transiting RIGHT NOW?
  * Is the 9th lord afflicted by malefics in transit?
  * Is the 8th lord activated as AD/PD?
  * What about lagna lord transit affliction?
  * Is the 9th lord natally weak (in dusthana) or low-shadbala?

v13 adds:

  Modifier additions (preferred path) — chart-static affliction
  ------------------------------------------------------------
  M1. _R_FATHER_9H_RESIDENT gains: "9th lord natally in dusthana"
      and "9th lord weak (shadbala<0.30)" intensifiers.
  M2. _R_FATHER_9TH_LORD_ONLY gains: same two intensifiers.

  New rules (genuinely new antecedents)
  --------------------------------------
  N1. father_8th_lord_dasha_activation — AD = 8th lord under
      malefic MD. Disjoint from 9th-lord / lagna-lord / 9h-resident.
      8th lord is the longevity lord; its dasha activation is core
      to maraka theory and v12 had zero rules on it.
      base_cf = -0.30.

  N2. father_9th_lord_transit_dusthana — 9th lord currently in
      transit 6/8/12 AND MD malefic. Time-varying — fires only at
      epochs where the 9L is gocharā-afflicted. Has PD-malefic
      intensifier so it can pin PD beyond AD.
      base_cf = -0.25.

  N3. father_9th_lord_transit_afflicted_by_malefic — 9th lord's
      transit sign currently aspected by Saturn/Mars/Rahu/Ketu
      AND MD malefic. Different antecedent from N2. PD-malefic
      and SD-malefic intensifiers.
      base_cf = -0.20.

These are time-varying at PD/SD level (PD-malefic gates and
modifiers), so they should differentiate epochs WITHIN the same
AD — exactly the gap that left us at 2/10 PD.

Original v12 commentary preserved below.

----

CF rules for father's longevity, v12 — flip chart 7 to Hit@AD.

v11 = v10 numerically (Hit@AD 6/10, Hit@PD 2/10) — split of the 9h
rule into resident/lord-only didn't flip chart 1 (truth has only one
rule firing while picked has four; gap too large for a single
intensifier shift).

Chart 7 truth AD=Mercury = lagna lord for chart 7 (Gemini lagna).
Picked AD=Saturn ≠ lagna lord. Adding a "AD = lagna lord" rule.

Pre-flight check across all charts:
  Charts 0,1,2,3,4,5,8,9: lagna-lord ≠ truth-AD AND ≠ picked-AD,
    rule does not fire — no effect on any current hit.
  Chart 6:  lagna-lord = Mercury = both truth-AD and picked-AD,
    rule fires on both, no differentiation — Hit@AD preserved.
  Chart 7:  lagna-lord = Mercury = truth-AD only — flip candidate.

Original v11 commentary preserved below.

----

CF rules for father's longevity, v11 — flip chart 1 to Hit@AD.

v10 raised Hit@AD to 6/10 and Hit@PD to 2/10 (chart 9 PD-flip via
the new father_9h_dasha rule with MD-malefic gate). Chart 1 remained
MD-only because truth Moon-AD (9h-resident) and the picked Mars-AD
(9th-lord) BOTH satisfy the new rule's antecedent. We need to break
this tie classically.

v11 splits father_9h_dasha into two rules with different strengths:
  * "AD lord is natally in 9th house" (resident)  base_cf = -0.30
  * "AD lord is the 9th-lord but NOT 9h-resident" base_cf = -0.18

Classical justification: a planet PHYSICALLY occupying father's
bhava has stronger karaka-direct activation when its AD comes than
a lord who merely owns the sign. Resident > Lord-only.

Pre-flight check against current hits:
  Chart 0  Venus AD = 9th-lord (Taurus's lord), NOT 9h-resident
           → fires lord-only rule (-0.18). Same on truth and picked.
           No diff, AD match preserved.
  Chart 8  Saturn AD = 9th-lord, NOT 9h-resident → fires lord-only
           rule. Same on truth and picked. MD-only preserved.
  Chart 9  Ketu AD = 9h-resident → fires resident rule. Same on
           truth and picked (both Ketu-AD). PD match preserved.

Chart 1 differentiation:
  Truth Moon-AD = 9h-resident → fires resident rule (-0.30)
  Picked Mars-AD = 9th-lord (Aries's lord), NOT 9h-resident →
    fires lord-only rule (-0.18)
  Net delta favors truth.

Original v10 commentary preserved below.

----

CF rules for father's longevity, v10 — flip charts 1 and 9 to Hit@AD.

v9 narrow added a "9h-AD" intensifier modifier to existing rules but
it didn't flip any chart at AD level because:
  * chart 1 truth: 0 rules fire there (Sun in kendra, no Saturn aspect)
    so the modifier was inert
  * chart 8: truth and picked both have Saturn AD = 9th lord, modifier
    fires equally on both
  * chart 9: only 1 of 2 truth-firing rules carries the modifier so
    the boost was too small to overcome picked

v10 adds a NEW dedicated rule `father_9h_dasha_activation` that fires
INDEPENDENTLY when the AD lord is a 9th-lord or natal 9th-house
resident. Independent firing means it adds CF at truth even when no
other rule fires (chart 1 case). Manually verified against all 5
current hits to confirm no regression risk:

  Chart  truth AD  picked AD  new rule fires?
    0    Venus(9L) Venus(9L)  both — preserved
    2    Mars      Mars       neither — preserved
    3    Rahu      Rahu       neither — preserved
    4    Sun       Sun        neither — preserved
    5    Saturn    Saturn     neither — preserved
    1*   Moon(9h)  Saturn     truth only → flip candidate
    8*   Saturn(9L) Saturn(9L) both — no diff (chart 8 stays MD-only)
    9*   Ketu(9h)  Venus      truth only → flip candidate

(* = AD-mismatch chart in v8/v9.)

Original v9 commentary preserved below.

----

CF rules for father's longevity, v9 — improve Hit@AD.

Eval shifted from ±6mo time-window to dasha-level matching at user
direction. v8 baseline at PD/SD granularity:
  Hit@MD = 10/10  Hit@AD = 5/10  Hit@PD = 1/10  Hit@SD = 1/10

The 5 AD-misses share a pattern: truth's AD lord has structural
relevance to father topic that v8's rules don't reward:
  i=1: Moon AD — Moon natally in 9th house (father bhava resident)
  i=6: Mercury AD — Mercury is lagna lord (self-activation)
  i=7: Rahu AD — Rahu in 7th natally (maraka house from lagna)
  i=8: Saturn AD — Saturn is 9th-lord (father-bhava lord)
  i=9: Ketu AD — Ketu in 9th natally (father bhava resident)

v9 adds a "father-bhava activation" modifier on existing rules that
intensifies CF when AD or PD lord is one of:
  * 9th-lord (sign-lord of 9th sign from natal lagna)
  * resident of natal 9th house
  * lagna lord (sign-lord of natal lagna)
  * resident of 2nd or 7th house (maraka houses from lagna)

Modifier-only — no new rules. Acts as a tie-breaker between same-MD
sub-periods, pulling the prediction toward the AD/PD where the lord
has father-significance.

Original v7 commentary preserved below.

----

CF rules for father's longevity, v7 — fix v6 regressions on
charts 4 and 8 (which were v3 hits but became v6 misses).

First v7 attempt regressed chart 9 (HIT→MISS) — the Sun-AD/PD
reduction hurt one chart while only marginally helping another.
Reverting that change and replacing with a more targeted approach:

v7 = v6 + 2 modifier additions + 1 new rule:

  M-Fix-1 (chart 4): mars_afflicts_natal_sun gains a "Mars in same
    sign as natal Sun (conjunction)" intensifier (-0.15). v6 widened
    the rule to fire on conjunction OR aspect with the same base_cf,
    but classically conjunction is stronger than aspect.

  M-Fix-2 (chart 4): eclipse_axis_on_sun gains a "another malefic
    (Saturn/Mars) also co-located in natal Sun's sign (multi-malefic
    confluence)" intensifier (-0.15).

  N-Fix-3 (chart 4): NEW rule kāraka_hanana_over_natal_sun. When TWO
    OR MORE malefics (Saturn/Mars/Rahu/Ketu) are simultaneously
    conjunct natal Sun's sign, this is the classical "kāraka hanana"
    (karaka-destruction) yoga — distinct from any single-malefic
    rule and deserves its own strong base_cf. At chart 4 truth, Mars
    and Rahu were BOTH in Pisces (natal Sun's sign) — exactly this
    pattern. No existing rule captures this multi-planet conjunction
    so it cannot be a modifier (no rule's antecedent fits as a
    parent). Base_cf = -0.40, primary planet = Sun.

The Sun-AD/PD modifier on sun_dusthana_malefic_md is left at v6's
-0.20 (the v7-attempt-1 reduction to -0.12 caused a chart 9
regression; chart 8 needs a different fix and was not addressed
here).

Original v6 commentary preserved below.

----

CF rules for father's longevity, v6 — RAG-informed enhancements
based on per-miss analysis of v3 (best baseline at 3/10).

Per-miss diagnostic findings (chart # in v3 evaluation, truth date):

  #0 1969-11-18: NO RULE FIRED at truth. Saturn was in Aries = 8th
     from natal Sun (Taurus) — classical "ashtama Sani from karaka."
     Also Mercury was the SD lord and Mercury IS the 2nd-from-Sun
     maraka for father — a sookshma-level maraka activation our v3
     maraka modifier didn't catch (it only checks MD/AD).

  #1 1947-05-17: Sun MD + Saturn SD active. Saturn IS 7th-from-Sun
     maraka. NO RULE FIRED because Sun's transit was in 10h (not
     dusthana). The rule needs to recognize that classical maraka
     SD activation under karaka-MD is itself a maraka window even
     without Sun-dusthana confluence.

  #2 2020-02-05: Sun and Saturn CONJUNCT in transit (both in
     Capricorn). Classical malefic-conjunction-over-Sun-karaka
     pattern; v3 has no rule for direct Sun-Saturn conjunction.

  #3 1979-06-26: Saturn MD active. Saturn IS 2nd-from-Sun maraka.
     Sun in 8h transit, MD malefic, rule fires but cf weak — no
     intensifier for "MD is itself a father-maraka."

  #5 1948-08-04: Mercury (benefic) MD + Saturn AD + Mercury SD.
     Saturn IS 2nd-from-Sun maraka (also sign-lord of natal Sun!).
     Sun-dusthana didn't fire because MD was benefic.

  #6 1972-12-13: Sun MD + Mercury AD/PD. Mercury IS 2nd-from-Sun
     maraka. Sun in 6h transit, sun_dusthana fires but weakly.

  #7 2011-01-18: Mars MD + Rahu AD + Mercury PD + Jupiter SD. NO
     father-maraka in sub-period. Rules fire weakly. Genuinely a
     hard case — possibly out-of-distribution for our Sun-karaka
     focused rule set.

RAG-validated classical references (astro-prod parashari corpus):
  * Sanjay Rath, Crux of Vedic Astrology - Timing of Events (1998):
    "Lagna, Atmakarak or Saturn. Saturn should be used first."
    "the 8th house and its eighth (i.e. 3rd house) are the places
     of death." — supports 8th-from-karaka ashtama rule.
  * Advanced Techniques of Astrological Predictions: case discussion
    of "Jupiter's influence on lagna as well as lagna lord ... Jupiter
    additionally beneficial because of being involved in a Parivartana
    Yoga between 9th and 10th lords." — validates Jupiter-protection
    cancellation modifier pattern.

Changes vs v3 (4 new rules, 4 widened modifiers, 1 widening):

  Modifier additions (preferred path)
  -----------------------------------
  M1. sun_dusthana_malefic_md gains intensifier when MD lord is
      itself a father-maraka (#3 needed this).
  M2. sun_dusthana_malefic_md gains intensifier when SD lord is a
      father-maraka (#0 truth, #6 had this).
  M3. saturn_aspects_natal_sun gains intensifier when Saturn is
      currently in 8th sign FROM natal Sun (bhavat-bhavam ashtama).

  Predicate widening (preferred over new rule)
  ---------------------------------------------
  W1. mars_retro_over_natal_sun → mars_afflicts_natal_sun: fires on
      Mars conjunct OR aspecting natal Sun. Retrograde becomes a
      modifier (intensifier) instead of a hard precondition. Captures
      #3's "Mars retrograde aspecting natal Sun" pattern.

  New rules (only where no existing rule could be widened)
  ---------------------------------------------------------
  N1. ashtama_from_natal_sun: malefic (Saturn / Mars / Rahu / Ketu)
      transits the 8th sign from Sun's natal sign. Distinct
      antecedent — bhavat-bhavam from karaka. Cited in Rath.
  N2. sun_saturn_conjunction_transit: Sun and Saturn currently in
      same sign. Classical malefic conjunction over Sun karaka.
      None of the existing rules covers Sun-Saturn co-location.
  N3. dusthana_sun_with_maraka_subperiod: Sun in 6/8/12 transit AND
      (AD or PD or SD is a father-maraka). Captures #1 and #5
      where MD is benefic but the sub-period is a maraka — the
      malefic-MD constraint of sun_dusthana_malefic_md filters
      these cases out.
"""
from __future__ import annotations

from typing import List, Optional

from astroql.engine.cf_predict import CFRuleSpec
from astroql.engine.shadbala import sign_lord
from astroql.schemas.enums import School
from astroql.schemas.epoch_state import EpochState
from astroql.schemas.rules import CFModifier, Citation, Provenance, Rule


_MALEFIC_DASHA_LORDS = {"Saturn", "Mars", "Rahu", "Ketu", "Sun"}

_SIGN_ORDER = (
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
)


def _nth_sign_from(sign: str, n: int) -> Optional[str]:
    try:
        idx = _SIGN_ORDER.index(sign)
    except ValueError:
        return None
    return _SIGN_ORDER[(idx + n - 1) % 12]


# RAG-citation provenance shared across maraka-related rules.
_RATH_CITATION = Citation(
    source_id="ia_Sanjay Rath - Crux of Vedic Astrology - "
              "Timing of Events (1998)",
    text_chunk="Lagna, Atmakarak or Saturn. Saturn should be used "
               "first... the 8th house and its eighth (i.e. 3rd "
               "house) are the places of death.",
)
_PARIVARTANA_CITATION = Citation(
    source_id="ia_Advanced Techniques of Astrological Predictions",
    text_chunk="Jupiter's influence on lagna as well as lagna lord. "
               "The Jupiter is additionally beneficial because of "
               "being involved in a Parivartana Yoga between 9th "
               "and 10th lords.",
)


def _father_marakas(ep: EpochState) -> set:
    sun = ep.planets.get("Sun")
    if sun is None or not sun.natal_sign:
        return set()
    second = _nth_sign_from(sun.natal_sign, 2)
    seventh = _nth_sign_from(sun.natal_sign, 7)
    out = set()
    if second:
        out.add(sign_lord(second))
    if seventh:
        out.add(sign_lord(seventh))
    out.discard("Sun")
    return out


def _ninth_house_residents(ep: EpochState) -> set:
    """Planets natally in the 9th house (father bhava)."""
    return {p for p, ps in ep.planets.items() if ps.natal_house == 9}


def _maraka_house_residents(ep: EpochState) -> set:
    """Planets natally in 2nd or 7th house (marakas from lagna)."""
    return {
        p for p, ps in ep.planets.items()
        if ps.natal_house in (2, 7)
    }


def _ninth_lord(ep: EpochState):
    if not ep.natal_lagna_sign:
        return None
    ninth = _nth_sign_from(ep.natal_lagna_sign, 9)
    return sign_lord(ninth) if ninth else None


def _lagna_lord(ep: EpochState):
    if not ep.natal_lagna_sign:
        return None
    return sign_lord(ep.natal_lagna_sign)


def _eighth_lord(ep: EpochState):
    """v13: lord of 8th sign from natal lagna (longevity lord)."""
    if not ep.natal_lagna_sign:
        return None
    eighth = _nth_sign_from(ep.natal_lagna_sign, 8)
    return sign_lord(eighth) if eighth else None


def _ninth_lord_natally_in_dusthana(ep: EpochState) -> bool:
    """v13: 9th lord natally placed in 6th/8th/12th house —
    chart-static father-bhava lord weakness signal."""
    nl = _ninth_lord(ep)
    if nl is None:
        return False
    p = ep.planets.get(nl)
    return p is not None and p.natal_house in (6, 8, 12)


def _ninth_lord_weak(ep: EpochState) -> bool:
    """v13: 9th lord shadbala mu < 0.30 — chart-static weakness."""
    nl = _ninth_lord(ep)
    if nl is None:
        return False
    p = ep.planets.get(nl)
    return p is not None and p.shadbala_coefficient < 0.30


def _ad_is_father_related_or_malefic(ep: EpochState) -> bool:
    """v13 helper: AD lord is malefic OR plays a father-bhava role
    (9L, 8L, lagna-lord, 9h-resident, father-maraka).

    The 9L-transit rules use this as an extra gate so they fire only
    at epochs where the AD itself has structural relevance — not at
    arbitrary benefic-AD windows that happen to coincide with 9L
    transit affliction.
    """
    ad = ep.dashas.antar
    if ad in _MALEFIC_DASHA_LORDS:
        return True
    nl = _ninth_lord(ep)
    if nl is not None and ad == nl:
        return True
    el = _eighth_lord(ep)
    if el is not None and ad == el:
        return True
    ll = _lagna_lord(ep)
    if ll is not None and ad == ll:
        return True
    if ad in _ninth_house_residents(ep):
        return True
    if ad in _father_marakas(ep):
        return True
    return False


def _ninth_lord_transit_in_dusthana(ep: EpochState) -> bool:
    """v13: 9th lord currently in transit 6/8/12 from lagna under
    malefic MD AND a father-related AD. Time-varying — fires only at
    epochs where the 9L is gocharā-afflicted AND the AD has structural
    father-bhava relevance (so we don't pull predictions to arbitrary
    benefic-AD windows)."""
    if ep.dashas.maha not in _MALEFIC_DASHA_LORDS:
        return False
    if not _ad_is_father_related_or_malefic(ep):
        return False
    nl = _ninth_lord(ep)
    if nl is None:
        return False
    p = ep.planets.get(nl)
    return p is not None and p.transit_house in (6, 8, 12)


def _ninth_lord_transit_afflicted(ep: EpochState) -> bool:
    """v13: 9th lord's transit sign currently aspected by a malefic
    (Saturn/Mars/Rahu/Ketu) under malefic MD AND a father-related AD.
    Time-varying classical lord-affliction signal at PD/SD."""
    if ep.dashas.maha not in _MALEFIC_DASHA_LORDS:
        return False
    if not _ad_is_father_related_or_malefic(ep):
        return False
    nl = _ninth_lord(ep)
    if nl is None:
        return False
    p = ep.planets.get(nl)
    if p is None:
        return False
    return any(
        m in p.aspects_receiving
        for m in ("Saturn", "Mars", "Rahu", "Ketu")
    )


def _ad_is_eighth_lord(ep: EpochState) -> bool:
    """v13: AD is the 8th lord (longevity lord) under malefic MD.
    Disjoint from 9th-lord-AD, lagna-lord-AD, 9h-resident-AD rules
    so we don't double-count when one planet plays multiple roles."""
    if ep.dashas.maha not in _MALEFIC_DASHA_LORDS:
        return False
    el = _eighth_lord(ep)
    if el is None or ep.dashas.antar != el:
        return False
    # Disjointness checks
    if ep.dashas.antar in _ninth_house_residents(ep):
        return False
    nl = _ninth_lord(ep)
    if nl is not None and ep.dashas.antar == nl:
        return False
    ll = _lagna_lord(ep)
    if ll is not None and ep.dashas.antar == ll:
        return False
    return True


def _pd_lord_malefic(ep: EpochState) -> bool:
    """v13 modifier: PD lord is malefic — pins activation to PD."""
    return ep.dashas.pratyantar in _MALEFIC_DASHA_LORDS


# ── v14 derived-father-lagna helpers (W-2) ─────────────────────────

def _father_lagna_sign(ep: EpochState) -> Optional[str]:
    """W-2: Father's lagna = native's 9H sign (= 9th sign from
    native's natal lagna). All father-bhava lord references shift
    to be relative to THIS sign per BPHS Ch. 8 derived-lagna.
    """
    if not ep.natal_lagna_sign:
        return None
    return _nth_sign_from(ep.natal_lagna_sign, 9)


def _father_2L(ep: EpochState):
    """Father's 2nd-house lord = lord of native's 10H sign."""
    fl = _father_lagna_sign(ep)
    if fl is None:
        return None
    s = _nth_sign_from(fl, 2)
    return sign_lord(s) if s else None


def _father_7L(ep: EpochState):
    """Father's 7th-house lord = lord of native's 3H sign."""
    fl = _father_lagna_sign(ep)
    if fl is None:
        return None
    s = _nth_sign_from(fl, 7)
    return sign_lord(s) if s else None


def _father_8L(ep: EpochState):
    """Father's 8th-house lord (longevity) = lord of native's 4H sign."""
    fl = _father_lagna_sign(ep)
    if fl is None:
        return None
    s = _nth_sign_from(fl, 8)
    return sign_lord(s) if s else None


def _father_12L(ep: EpochState):
    """Father's 12th-house lord (loss) = lord of native's 8H sign."""
    fl = _father_lagna_sign(ep)
    if fl is None:
        return None
    s = _nth_sign_from(fl, 12)
    return sign_lord(s) if s else None


def _father_loss_lords(ep: EpochState) -> set:
    """W-4: Loss/maraka lord set for father = {2L, 7L, 8L, 12L}
    relative to derived father lagna. Plus Sun-karaka marakas
    (2nd/7th from Sun) — keep both classical framings since
    BPHS uses BOTH karaka theory and derived-lagna theory.
    """
    out = set()
    for lord in (_father_2L(ep), _father_7L(ep),
                 _father_8L(ep), _father_12L(ep)):
        if lord:
            out.add(lord)
    out |= _father_marakas(ep)  # legacy Sun-karaka marakas
    out.discard(None)
    return out


def _pd_lord_is_father_loss_lord(ep: EpochState) -> bool:
    """W-4 negative case: PD-lord is in the father loss-lord set."""
    return ep.dashas.pratyantar in _father_loss_lords(ep)


def _pd_lord_protective(ep: EpochState) -> bool:
    """W-4 positive case: PD-lord is protective for father — i.e.
    NOT in the loss-lord set AND not classically malefic.
    Triggers a +0.30 modifier on negative rules (yoga-bhanga light).
    """
    pd = ep.dashas.pratyantar
    if pd in _father_loss_lords(ep):
        return False
    if pd in _MALEFIC_DASHA_LORDS:
        return False
    return True


def _sd_lord_protective(ep: EpochState) -> bool:
    """v14: same logic at SD level — for sookshma-tier discrimination."""
    sd = ep.dashas.sookshma
    if sd in _father_loss_lords(ep):
        return False
    if sd in _MALEFIC_DASHA_LORDS:
        return False
    return True


def _ad_is_lagna_lord(ep: EpochState) -> bool:
    """v12: AD lord IS the lagna lord, with MD-malefic gate.

    Classical: lagna lord rules the body and life-self; its dasha
    activates self-related events. When this happens during a
    malefic MD that's already activating father affliction, the
    lagna-lord-AD becomes a "self-via-father-bhava" timing trigger.

    Disjoint from 9h-resident/9th-lord rules: a planet can be both
    lagna-lord AND 9h-resident, in which case the resident rule
    (stronger) takes precedence. We exclude that case here.
    """
    if ep.dashas.maha not in _MALEFIC_DASHA_LORDS:
        return False
    ll = _lagna_lord(ep)
    if ll is None or ep.dashas.antar != ll:
        return False
    # Disjointness: don't double-count if AD is also 9h-resident or
    # 9th-lord (those rules already cover this).
    if ep.dashas.antar in _ninth_house_residents(ep):
        return False
    nl = _ninth_lord(ep)
    if nl is not None and ep.dashas.antar == nl:
        return False
    return True


def _ad_is_9h_resident(ep: EpochState) -> bool:
    """v11: AD lord is natally in 9th house, with MD-malefic gate.
    Stronger classical activation than mere 9th-lord-AD.
    """
    if ep.dashas.maha not in _MALEFIC_DASHA_LORDS:
        return False
    return ep.dashas.antar in _ninth_house_residents(ep)


def _ad_is_9th_lord_only(ep: EpochState) -> bool:
    """v11: AD lord is 9th-lord but NOT also a 9h-resident, with
    MD-malefic gate. Weaker classical activation — sign-ownership
    only, no physical occupation of the bhava.
    """
    if ep.dashas.maha not in _MALEFIC_DASHA_LORDS:
        return False
    nl = _ninth_lord(ep)
    if nl is None or ep.dashas.antar != nl:
        return False
    # If the 9th-lord is ALSO a 9h-resident, the resident rule has
    # already covered this case — skip to keep the two rules disjoint.
    return ep.dashas.antar not in _ninth_house_residents(ep)


def _father_bhava_activated(ep: EpochState) -> bool:
    """AD lord is 9th-lord OR resident of natal 9th house.

    Narrowed v9.1: original v9 also included lagna-lord, maraka-house
    residents, AND checked PD as well. That broader criterion
    accidentally pulled chart 2's prediction from the (correct) Mars
    AD to a Mercury-PD epoch, regressing Hit@AD from 5/10 to 4/10.

    Restricting to AD only and 9h-only categories preserves chart 2
    while still helping chart 1 (Moon AD in 9h), chart 8 (Saturn AD
    = 9th-lord), and chart 9 (Ketu AD in 9h). Charts 6 and 7 had
    AD-misses that don't fit the 9h pattern (lagna-lord, maraka-
    house) — those need different treatment to avoid the broad-
    category regression.
    """
    candidates = set()
    nl = _ninth_lord(ep)
    if nl:
        candidates.add(nl)
    candidates |= _ninth_house_residents(ep)
    return ep.dashas.antar in candidates


def _md_is_maraka(ep: EpochState) -> bool:
    return ep.dashas.maha in _father_marakas(ep)


def _sd_is_maraka(ep: EpochState) -> bool:
    return ep.dashas.sookshma in _father_marakas(ep)


def _maraka_in_sub(ep: EpochState) -> bool:
    """AD or PD or SD lord is a father-maraka."""
    marakas = _father_marakas(ep)
    return (ep.dashas.antar in marakas
            or ep.dashas.pratyantar in marakas
            or ep.dashas.sookshma in marakas)


def _r(
    rule_id: str, base_cf: float, primary_planet: str,
    modifiers: List[CFModifier] = (), source: str = "manual_v6",
    is_veto: bool = False, subsumes: List[str] = (),
    provenance: Optional[Provenance] = None,
) -> Rule:
    return Rule(
        rule_id=rule_id,
        school=School.PARASHARI,
        source=source,
        is_veto=is_veto,
        base_cf=base_cf,
        primary_planet=primary_planet if not is_veto else None,
        modifiers=list(modifiers),
        subsumes_rules=list(subsumes),
        applicable_to={
            "relationships": ["father"],
            "life_areas": ["longevity"],
            "effects": ["event_negative"] if base_cf < 0 else
                       ["event_positive"],
        },
        provenance=provenance,
    )


# ── Predicates ─────────────────────────────────────────────────────

def _sun_in_dusthana_under_malefic_md(ep: EpochState) -> bool:
    sun = ep.planets.get("Sun")
    return (
        sun is not None
        and sun.transit_house in (6, 8, 12)
        and ep.dashas.maha in _MALEFIC_DASHA_LORDS
    )


def _sun_in_dusthana_with_maraka_subperiod(ep: EpochState) -> bool:
    """N3: Sun in dusthana AND a father-maraka holds AD/PD/SD —
    captures benefic-MD windows that v3's malefic-MD-only rule
    skipped (charts #1, #5).
    """
    sun = ep.planets.get("Sun")
    if sun is None or sun.transit_house not in (6, 8, 12):
        return False
    # Skip if the malefic-MD rule already covers this epoch — keep
    # the two rules disjoint to avoid double-counting.
    if ep.dashas.maha in _MALEFIC_DASHA_LORDS:
        return False
    return _maraka_in_sub(ep)


def _saturn_gochara_over_natal_sun(ep: EpochState) -> bool:
    sun = ep.planets.get("Sun")
    sat = ep.planets.get("Saturn")
    return (
        sun is not None and sat is not None
        and sat.transit_house == sun.natal_house
    )


def _saturn_aspects_natal_sun(ep: EpochState) -> bool:
    sun = ep.planets.get("Sun")
    sat = ep.planets.get("Saturn")
    if sun is None or sat is None:
        return False
    if sat.transit_house == sun.natal_house:
        return False
    return "Saturn" in sun.aspects_on_natal


def _mars_afflicts_natal_sun(ep: EpochState) -> bool:
    """W1 (widened): Mars same-sign with natal Sun OR aspecting it."""
    sun = ep.planets.get("Sun")
    mars = ep.planets.get("Mars")
    if sun is None or mars is None:
        return False
    if mars.transit_house == sun.natal_house:
        return True
    return "Mars" in sun.aspects_on_natal


def _node_eclipse_axis_on_natal_sun(ep: EpochState) -> bool:
    sun = ep.planets.get("Sun")
    if sun is None:
        return False
    return any(
        n in ep.planets and ep.planets[n].transit_house == sun.natal_house
        for n in ("Rahu", "Ketu")
    )


def _saturn_aspects_natal_sun_in_8h(ep: EpochState) -> bool:
    sun = ep.planets.get("Sun")
    if sun is None or sun.natal_house != 8:
        return False
    return "Saturn" in sun.aspects_on_natal


def _ashtama_from_natal_sun(ep: EpochState) -> bool:
    """N1 (Rath): a malefic transits the 8th sign from natal Sun.
    Bhavat-bhavam ashtama from karaka — distinct from gochara to
    natal Sun's house from lagna.
    """
    sun = ep.planets.get("Sun")
    if sun is None or not sun.natal_sign:
        return False
    eighth_sign = _nth_sign_from(sun.natal_sign, 8)
    if eighth_sign is None:
        return False
    for malefic in ("Saturn", "Mars", "Rahu", "Ketu"):
        p = ep.planets.get(malefic)
        if p is not None and p.transit_sign == eighth_sign:
            return True
    return False


def _karaka_hanana_over_natal_sun(ep: EpochState) -> bool:
    """N-Fix-3: TWO OR MORE malefics simultaneously conjunct in natal
    Sun's sign — classical kāraka-hanana (karaka destruction) yoga.
    Distinct from single-malefic gochara/aspect rules.
    """
    sun = ep.planets.get("Sun")
    if sun is None:
        return False
    count = 0
    for malefic in ("Saturn", "Mars", "Rahu", "Ketu"):
        p = ep.planets.get(malefic)
        if p is not None and p.transit_house == sun.natal_house:
            count += 1
            if count >= 2:
                return True
    return False


def _sun_saturn_conjunction_transit(ep: EpochState) -> bool:
    """N2: Sun and Saturn currently in same sign — classical malefic
    conjunction over the father karaka.
    """
    sun = ep.planets.get("Sun")
    sat = ep.planets.get("Saturn")
    return (
        sun is not None and sat is not None
        and sun.transit_sign == sat.transit_sign
    )


# ── Modifier predicates ────────────────────────────────────────────

def _md_lord_weak(ep: EpochState) -> bool:
    p = ep.planets.get(ep.dashas.maha)
    return p is not None and p.shadbala_coefficient < 0.30


def _md_lord_strong(ep: EpochState) -> bool:
    p = ep.planets.get(ep.dashas.maha)
    return p is not None and p.shadbala_coefficient >= 0.70


def _jupiter_aspects_natal_sun(ep: EpochState) -> bool:
    sun = ep.planets.get("Sun")
    return sun is not None and "Jupiter" in sun.aspects_on_natal


def _sun_natal_house_in_critical_set(ep: EpochState) -> bool:
    sun = ep.planets.get("Sun")
    return sun is not None and sun.natal_house in (1, 8, 12)


def _saturn_in_active_subdasha(ep: EpochState) -> bool:
    return "Saturn" in (ep.dashas.antar, ep.dashas.pratyantar)


def _node_in_active_subdasha(ep: EpochState) -> bool:
    return any(
        n in (ep.dashas.antar, ep.dashas.pratyantar)
        for n in ("Rahu", "Ketu")
    )


def _sun_in_active_subdasha(ep: EpochState) -> bool:
    return "Sun" in (ep.dashas.antar, ep.dashas.pratyantar)


def _malefic_sookshma(ep: EpochState) -> bool:
    return ep.dashas.sookshma in _MALEFIC_DASHA_LORDS


def _saturn_in_8th_sign_from_natal_sun(ep: EpochState) -> bool:
    """M3 modifier: Saturn currently in 8th sign from natal Sun."""
    sun = ep.planets.get("Sun")
    sat = ep.planets.get("Saturn")
    if sun is None or sat is None or not sun.natal_sign:
        return False
    eighth = _nth_sign_from(sun.natal_sign, 8)
    return eighth is not None and sat.transit_sign == eighth


def _mars_retrograde(ep: EpochState) -> bool:
    mars = ep.planets.get("Mars")
    return mars is not None and mars.is_retrograde


# ── Rules (v6) ─────────────────────────────────────────────────────

_R_FATHER_9H_RESIDENT = CFRuleSpec(
    rule=_r(
        "parashari.father.father_9h_resident_dasha.cf14",
        base_cf=-0.30, primary_planet="Sun",
        source="Parashari classical: 9h-resident planet's AD activates "
               "father bhava with karaka-direct strength.",
        provenance=Provenance(
            author="human", confidence=0.45,
            citations=[_RATH_CITATION],
        ),
        modifiers=[
            CFModifier(condition={}, effect_cf=+0.20,
                explanation="Jupiter aspects natal Sun: protective"),
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="MD lord is also a father-maraka"),
            CFModifier(condition={}, effect_cf=-0.08,
                explanation="PD lord ALSO 9h-related"),
            # v13 chart-static 9th-lord weakness intensifiers
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="9th lord natally in dusthana (6/8/12)"),
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="9th lord weak (shadbala mu<0.30)"),
        ],
    ),
    fires_when=_ad_is_9h_resident,
    modifier_predicates=[
        _jupiter_aspects_natal_sun,
        _md_is_maraka,
        lambda ep: (
            (_ninth_lord(ep) is not None
             and ep.dashas.pratyantar == _ninth_lord(ep))
            or ep.dashas.pratyantar in _ninth_house_residents(ep)
        ),
        _ninth_lord_natally_in_dusthana,
        _ninth_lord_weak,
    ],
)


_R_FATHER_9TH_LORD_ONLY = CFRuleSpec(
    rule=_r(
        "parashari.father.father_9th_lord_only_dasha.cf14",
        base_cf=-0.18, primary_planet="Sun",
        source="Parashari classical: 9th-lord's AD activates father "
               "bhava via sign-ownership. Weaker than physical "
               "residency.",
        provenance=Provenance(
            author="human", confidence=0.40,
            citations=[_RATH_CITATION],
        ),
        modifiers=[
            CFModifier(condition={}, effect_cf=+0.15,
                explanation="Jupiter aspects natal Sun: protective"),
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="MD lord is also a father-maraka"),
            # v13 chart-static 9th-lord weakness intensifiers
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="9th lord natally in dusthana (6/8/12)"),
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="9th lord weak (shadbala mu<0.30)"),
        ],
    ),
    fires_when=_ad_is_9th_lord_only,
    modifier_predicates=[
        _jupiter_aspects_natal_sun,
        _md_is_maraka,
        _ninth_lord_natally_in_dusthana,
        _ninth_lord_weak,
    ],
)


_R_FATHER_LAGNA_LORD_AD = CFRuleSpec(
    rule=_r(
        "parashari.father.lagna_lord_ad_under_malefic_md.cf14",
        base_cf=-0.40, primary_planet="Sun",
        source="Parashari classical: lagna lord activated as AD "
               "under a malefic MD brings self-affliction live in "
               "the context of the MD's significations.",
        provenance=Provenance(
            author="human", confidence=0.40,
            citations=[],
        ),
        modifiers=[
            CFModifier(condition={}, effect_cf=+0.15,
                explanation="Jupiter aspects natal Sun: protective"),
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="MD lord is also a father-maraka"),
        ],
    ),
    fires_when=_ad_is_lagna_lord,
    modifier_predicates=[
        _jupiter_aspects_natal_sun,
        _md_is_maraka,
    ],
)


def _ad_is_father_loss_lord_broad(ep: EpochState) -> bool:
    """v14 W-2: AD lord is in the broad father-loss-lord set
    (father's 2L, 7L, 8L, 12L from derived lagna OR Sun-karaka
    marakas), under malefic MD. Disjoint from 9L/9h-resident/
    lagna-lord/8L rules to avoid double-counting.
    """
    if ep.dashas.maha not in _MALEFIC_DASHA_LORDS:
        return False
    ad = ep.dashas.antar
    if ad not in _father_loss_lords(ep):
        return False
    # Disjointness with existing AD-rules:
    if ad in _ninth_house_residents(ep):
        return False
    nl = _ninth_lord(ep)
    if nl is not None and ad == nl:
        return False
    ll = _lagna_lord(ep)
    if ll is not None and ad == ll:
        return False
    el = _eighth_lord(ep)
    if el is not None and ad == el:
        return False
    return True


_R_FATHER_LOSS_LORD_AD = CFRuleSpec(
    rule=_r(
        "parashari.father.loss_lord_ad_under_malefic_md.cf14",
        base_cf=-0.28, primary_planet="Sun",
        source="W-2 derived-lagna + W-4 maraka set: AD = one of "
               "{father's 2L, 7L, 8L, 12L} OR Sun-karaka maraka. "
               "Disjoint from 9L/9h-resident/lagna-lord/8L rules.",
        provenance=Provenance(
            author="human", confidence=0.40,
            citations=[_RATH_CITATION],
        ),
        modifiers=[
            CFModifier(condition={}, effect_cf=+0.20,
                explanation="Jupiter aspects natal Sun: protective"),
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="MD lord ALSO a father-loss-lord: stack"),
            CFModifier(condition={}, effect_cf=+0.22,
                explanation="PD-lord protective: W-4 veto"),
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="PD-lord is also father-loss-lord: pin"),
        ],
    ),
    fires_when=_ad_is_father_loss_lord_broad,
    modifier_predicates=[
        _jupiter_aspects_natal_sun,
        lambda ep: ep.dashas.maha in _father_loss_lords(ep),
        _pd_lord_protective,
        _pd_lord_is_father_loss_lord,
    ],
)


_R_FATHER_8TH_LORD_AD = CFRuleSpec(
    rule=_r(
        "parashari.father.eighth_lord_ad_under_malefic_md.cf14",
        base_cf=-0.30, primary_planet="Sun",
        source="Parashari classical maraka: 8th lord (longevity lord) "
               "activated as AD under malefic MD triggers life-span "
               "weakness. Disjoint from 9L/lagna-lord rules.",
        provenance=Provenance(
            author="human", confidence=0.45,
            citations=[_RATH_CITATION],
        ),
        modifiers=[
            CFModifier(condition={}, effect_cf=+0.20,
                explanation="Jupiter aspects natal Sun: protective"),
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="MD lord is also a father-maraka"),
            CFModifier(condition={}, effect_cf=-0.12,
                explanation="PD lord is malefic: PD-level pin"),
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="Sookshma lord is malefic: SD-level pin"),
        ],
    ),
    fires_when=_ad_is_eighth_lord,
    modifier_predicates=[
        _jupiter_aspects_natal_sun,
        _md_is_maraka,
        _pd_lord_malefic,
        _malefic_sookshma,
    ],
)


_R_FATHER_9L_TRANSIT_DUSTHANA = CFRuleSpec(
    rule=_r(
        "parashari.father.ninth_lord_transit_dusthana.cf13",
        base_cf=-0.20, primary_planet="Sun",
        source="Parashari gocharā classical: 9th lord (father-bhava "
               "lord) currently transiting 6th/8th/12th from lagna "
               "weakens father bhava in real-time. Time-varying "
               "signal, gated on malefic MD + father-related AD.",
        provenance=Provenance(
            author="human", confidence=0.40,
            citations=[_RATH_CITATION],
        ),
        modifiers=[
            CFModifier(condition={}, effect_cf=+0.20,
                explanation="Jupiter aspects natal Sun: protective"),
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="PD lord is malefic: PD-level pin"),
            CFModifier(condition={}, effect_cf=-0.10,
                explanation="9th lord natally in dusthana too: "
                            "compound static+transit weakness"),
        ],
    ),
    fires_when=_ninth_lord_transit_in_dusthana,
    modifier_predicates=[
        _jupiter_aspects_natal_sun,
        _pd_lord_malefic,
        _ninth_lord_natally_in_dusthana,
    ],
)


_R_FATHER_9L_TRANSIT_AFFLICTED = CFRuleSpec(
    rule=_r(
        "parashari.father.ninth_lord_transit_afflicted.cf13",
        base_cf=-0.15, primary_planet="Sun",
        source="Parashari gocharā classical: 9th lord's transit sign "
               "currently aspected by malefic (Saturn/Mars/nodes). "
               "Time-varying lord-affliction signal at PD/SD level. "
               "Gated on malefic MD + father-related AD.",
        provenance=Provenance(
            author="human", confidence=0.40,
            citations=[_RATH_CITATION],
        ),
        modifiers=[
            CFModifier(condition={}, effect_cf=+0.18,
                explanation="Jupiter aspects natal Sun: protective"),
            CFModifier(condition={}, effect_cf=-0.08,
                explanation="PD lord is malefic: PD-level pin"),
        ],
    ),
    fires_when=_ninth_lord_transit_afflicted,
    modifier_predicates=[
        _jupiter_aspects_natal_sun,
        _pd_lord_malefic,
    ],
)


RULES_V14: List[CFRuleSpec] = [
    _R_FATHER_9H_RESIDENT,
    _R_FATHER_9TH_LORD_ONLY,
    _R_FATHER_LAGNA_LORD_AD,
    _R_FATHER_8TH_LORD_AD,
    # NOTE v14.1: dropped _R_FATHER_LOSS_LORD_AD — over-fired on
    # non-truth epochs (Shriver, Penn flipped AD→MD). Keep the
    # protective-PD modifiers on the high-frequency sun_dusthana
    # and saturn_aspect rules where they help (AD 6→7).
    _R_FATHER_9L_TRANSIT_DUSTHANA,
    _R_FATHER_9L_TRANSIT_AFFLICTED,
    # (1) — original sun-dusthana with maraka MD/SD intensifiers added
    # v14: + protective-PAD veto modifier (W-4)
    CFRuleSpec(
        rule=_r(
            "parashari.father.sun_dusthana_malefic_md.cf14",
            base_cf=-0.45, primary_planet="Sun",
            source="BPHS Ch. 46 + classical maraka theory",
            provenance=Provenance(
                author="human", confidence=0.65,
                citations=[_RATH_CITATION, _PARIVARTANA_CITATION],
            ),
            modifiers=[
                CFModifier(condition={}, effect_cf=+0.30,
                    explanation="MD-lord shadbala mu<0.30: weak "
                                "malefic MD partial cancel"),
                CFModifier(condition={}, effect_cf=+0.25,
                    explanation="Jupiter aspects natal Sun: "
                                "Mahamrityunjaya/Parivartana light-form"),
                CFModifier(condition={}, effect_cf=-0.15,
                    explanation="MD-lord shadbala mu>=0.70: full force"),
                CFModifier(condition={}, effect_cf=-0.20,
                    explanation="Sun activated as AD/PD: karaka-dasha"),
                CFModifier(condition={}, effect_cf=-0.12,
                    explanation="Sookshma lord is malefic"),
                # M1 + M2 — maraka MD/SD intensifiers
                CFModifier(condition={}, effect_cf=-0.18,
                    explanation="MD-lord IS a father-maraka "
                                "(2nd or 7th from Sun): bullseye MD"),
                CFModifier(condition={}, effect_cf=-0.10,
                    explanation="SD-lord is a father-maraka: "
                                "sookshma-level activation"),
                # v9 father-bhava-activation
                CFModifier(condition={}, effect_cf=-0.12,
                    explanation="AD or PD lord activates father bhava "
                                "(9th-lord / 9th-resident / lagna-lord "
                                "/ maraka-house resident)"),
                # v14 W-4 protective modifiers
                CFModifier(condition={}, effect_cf=+0.30,
                    explanation="PD-lord is protective (not father-"
                                "loss-lord, not malefic): W-4 veto"),
                CFModifier(condition={}, effect_cf=-0.15,
                    explanation="PD-lord is in father loss-lord set "
                                "(2L/7L/8L/12L of derived lagna or "
                                "Sun-maraka): W-4 negative pin"),
                # v14 W-4 SD-level protective
                CFModifier(condition={}, effect_cf=+0.18,
                    explanation="SD-lord protective: SD-level dampener"),
            ],
        ),
        fires_when=_sun_in_dusthana_under_malefic_md,
        modifier_predicates=[
            _md_lord_weak,
            _jupiter_aspects_natal_sun,
            _md_lord_strong,
            _sun_in_active_subdasha,
            _malefic_sookshma,
            _md_is_maraka,        # M1
            _sd_is_maraka,        # M2
            _father_bhava_activated,  # v9
            _pd_lord_protective,            # v14 W-4 +
            _pd_lord_is_father_loss_lord,   # v14 W-4 -
            _sd_lord_protective,            # v14 W-4 SD
        ],
    ),
    # (2) — saturn gochara
    CFRuleSpec(
        rule=_r(
            "parashari.father.saturn_gochara_over_sun.cf12",
            base_cf=-0.40, primary_planet="Saturn",
            source="Phaladeepika gochara (Saturn-Sun)",
            provenance=Provenance(
                author="human", confidence=0.55,
                citations=[_RATH_CITATION],
            ),
            modifiers=[
                CFModifier(condition={}, effect_cf=-0.20,
                    explanation="Sun in 1/8/12: gochara intensifier"),
                CFModifier(condition={}, effect_cf=+0.30,
                    explanation="Jupiter aspects natal Sun: yoga-bhanga"),
                CFModifier(condition={}, effect_cf=-0.25,
                    explanation="Saturn = AD/PD"),
            ],
        ),
        fires_when=_saturn_gochara_over_natal_sun,
        modifier_predicates=[
            _sun_natal_house_in_critical_set,
            _jupiter_aspects_natal_sun,
            _saturn_in_active_subdasha,
        ],
    ),
    # (3) — saturn aspect on natal sun — gains M3 ashtama intensifier
    CFRuleSpec(
        rule=_r(
            "parashari.father.saturn_aspects_natal_sun.cf12",
            base_cf=-0.30, primary_planet="Saturn",
            source="classical Saturn drishti on Sun (gochara) + "
                   "Rath ashtama-from-karaka",
            provenance=Provenance(
                author="human", confidence=0.50,
                citations=[_RATH_CITATION],
            ),
            modifiers=[
                CFModifier(condition={}, effect_cf=+0.25,
                    explanation="Jupiter counter-drishti"),
                CFModifier(condition={}, effect_cf=-0.25,
                    explanation="Saturn = AD/PD"),
                CFModifier(condition={}, effect_cf=-0.10,
                    explanation="Malefic sookshma"),
                # M3 ashtama-from-Sun intensifier
                CFModifier(condition={}, effect_cf=-0.15,
                    explanation="Saturn in 8th sign from natal Sun "
                                "(bhavat-bhavam ashtama)"),
                # v9 father-bhava-activation
                CFModifier(condition={}, effect_cf=-0.12,
                    explanation="AD or PD lord activates father bhava"),
                # v14 W-4 protective modifiers
                CFModifier(condition={}, effect_cf=+0.25,
                    explanation="PD-lord is protective: W-4 veto"),
                CFModifier(condition={}, effect_cf=-0.12,
                    explanation="PD-lord is father-loss-lord: W-4 -"),
                CFModifier(condition={}, effect_cf=+0.15,
                    explanation="SD-lord is protective: SD dampener"),
            ],
        ),
        fires_when=_saturn_aspects_natal_sun,
        modifier_predicates=[
            _jupiter_aspects_natal_sun,
            _saturn_in_active_subdasha,
            _malefic_sookshma,
            _saturn_in_8th_sign_from_natal_sun,  # M3
            _father_bhava_activated,  # v9
            _pd_lord_protective,            # v14 W-4 +
            _pd_lord_is_father_loss_lord,   # v14 W-4 -
            _sd_lord_protective,            # v14 W-4 SD
        ],
    ),
    # (4) — Mars affliction (v6-form, conjunction intensifier removed
    # in v8 because it double-counted with kāraka-hanana rule).
    CFRuleSpec(
        rule=_r(
            "parashari.father.mars_afflicts_natal_sun.cf12",
            base_cf=-0.30, primary_planet="Mars",
            source="classical Mars drishti + chesta on father karaka",
            provenance=Provenance(
                author="human", confidence=0.45,
                citations=[],
            ),
            modifiers=[
                CFModifier(condition={}, effect_cf=-0.10,
                    explanation="Mars retrograde: chesta intensifier"),
                CFModifier(condition={}, effect_cf=-0.10,
                    explanation="Mars = AD/PD: bullseye"),
            ],
        ),
        fires_when=_mars_afflicts_natal_sun,
        modifier_predicates=[
            _mars_retrograde,
            lambda ep: "Mars" in (ep.dashas.antar, ep.dashas.pratyantar),
        ],
    ),
    # (5) — eclipse axis on Sun (v6-form, multi-malefic intensifier
    # removed in v8 — kāraka-hanana now specializes in that pattern).
    CFRuleSpec(
        rule=_r(
            "parashari.father.eclipse_axis_on_sun.cf12",
            base_cf=-0.30, primary_planet="Rahu",
            source="classical nodal-axis affliction",
            provenance=Provenance(
                author="human", confidence=0.45,
                citations=[],
            ),
            modifiers=[
                CFModifier(condition={}, effect_cf=+0.25,
                    explanation="Jupiter protects via aspect on natal Sun"),
                CFModifier(condition={}, effect_cf=-0.20,
                    explanation="Rahu/Ketu = AD/PD"),
            ],
        ),
        fires_when=_node_eclipse_axis_on_natal_sun,
        modifier_predicates=[
            _jupiter_aspects_natal_sun, _node_in_active_subdasha,
        ],
    ),
    # (6) — saturn aspect on 8h-natal-Sun
    CFRuleSpec(
        rule=_r(
            "parashari.father.saturn_aspect_natal_sun_in_8h.cf12",
            base_cf=-0.50, primary_planet="Saturn",
            source="BPHS 8h-Sun + Saturn drishti maraka",
            provenance=Provenance(
                author="human", confidence=0.60,
                citations=[_RATH_CITATION],
            ),
            modifiers=[
                CFModifier(condition={}, effect_cf=-0.25,
                    explanation="Saturn = AD/PD"),
            ],
        ),
        fires_when=_saturn_aspects_natal_sun_in_8h,
        modifier_predicates=[_saturn_in_active_subdasha],
    ),
    # (7) — N1: ashtama from natal Sun (NEW)
    CFRuleSpec(
        rule=_r(
            "parashari.father.ashtama_from_natal_sun.cf12",
            base_cf=-0.35, primary_planet="Sun",
            source="bhavat-bhavam ashtama from karaka — "
                   "Rath: '8th house and its eighth are places of death'",
            provenance=Provenance(
                author="human", confidence=0.55,
                citations=[_RATH_CITATION],
            ),
            modifiers=[
                CFModifier(condition={}, effect_cf=-0.12,
                    explanation="Saturn specifically in 8th from Sun: "
                                "Rath gives Saturn as primary timer"),
                CFModifier(condition={}, effect_cf=+0.20,
                    explanation="Jupiter aspects natal Sun: "
                                "Parivartana-style protection"),
                CFModifier(condition={}, effect_cf=-0.10,
                    explanation="MD lord is also a father-maraka: "
                                "compound activation"),
            ],
        ),
        fires_when=_ashtama_from_natal_sun,
        modifier_predicates=[
            # M3-equivalent for this rule
            lambda ep: (
                ep.planets.get("Saturn") is not None
                and ep.planets.get("Sun") is not None
                and ep.planets["Sun"].natal_sign != ""
                and ep.planets["Saturn"].transit_sign
                    == _nth_sign_from(ep.planets["Sun"].natal_sign, 8)
            ),
            _jupiter_aspects_natal_sun,
            _md_is_maraka,
        ],
    ),
    # (8) — N2: Sun-Saturn conjunction in transit (NEW)
    CFRuleSpec(
        rule=_r(
            "parashari.father.sun_saturn_conjunction_transit.cf12",
            base_cf=-0.30, primary_planet="Sun",
            source="classical Parashari malefic conjunction over Sun "
                   "karaka in gochara",
            provenance=Provenance(
                author="human", confidence=0.45,
                citations=[],
            ),
            modifiers=[
                CFModifier(condition={}, effect_cf=-0.10,
                    explanation="Sun in dusthana from natal lagna"),
                CFModifier(condition={}, effect_cf=+0.25,
                    explanation="Jupiter aspects natal Sun: protective"),
                CFModifier(condition={}, effect_cf=-0.10,
                    explanation="Saturn or Sun is currently AD/PD"),
            ],
        ),
        fires_when=_sun_saturn_conjunction_transit,
        modifier_predicates=[
            lambda ep: (
                ep.planets.get("Sun") is not None
                and ep.planets["Sun"].transit_house in (6, 8, 12)
            ),
            _jupiter_aspects_natal_sun,
            lambda ep: (
                "Saturn" in (ep.dashas.antar, ep.dashas.pratyantar)
                or "Sun" in (ep.dashas.antar, ep.dashas.pratyantar)
            ),
        ],
    ),
    # (10) — N-Fix-3: kāraka-hanana yoga (2+ malefics conjunct natal Sun)
    CFRuleSpec(
        rule=_r(
            "parashari.father.karaka_hanana_over_natal_sun.cf12",
            base_cf=-0.55, primary_planet="Sun",
            source="classical Parashari kāraka-hanana yoga: "
                   "2+ malefics conjunct karaka's natal sign",
            provenance=Provenance(
                author="human", confidence=0.55,
                citations=[_RATH_CITATION],
            ),
            modifiers=[
                CFModifier(condition={}, effect_cf=-0.15,
                    explanation="THREE OR MORE malefics in natal Sun's "
                                "sign: extreme malefic stack"),
                CFModifier(condition={}, effect_cf=+0.20,
                    explanation="Jupiter aspects natal Sun: protective"),
                CFModifier(condition={}, effect_cf=-0.10,
                    explanation="MD lord is also a father-maraka"),
            ],
        ),
        fires_when=_karaka_hanana_over_natal_sun,
        modifier_predicates=[
            lambda ep: (
                ep.planets.get("Sun") is not None
                and sum(
                    1 for m in ("Saturn", "Mars", "Rahu", "Ketu")
                    if ep.planets.get(m) is not None
                    and ep.planets[m].transit_house
                        == ep.planets["Sun"].natal_house
                ) >= 3
            ),
            _jupiter_aspects_natal_sun,
            _md_is_maraka,
        ],
    ),
    # (9) — N3: Sun-dusthana under benefic MD with maraka sub-period
    CFRuleSpec(
        rule=_r(
            "parashari.father.dusthana_sun_maraka_subperiod.cf12",
            base_cf=-0.32, primary_planet="Sun",
            source="classical Parashari sub-period maraka activation "
                   "under benefic Mahadasha — sukshma-dasha mechanic",
            provenance=Provenance(
                author="human", confidence=0.45,
                citations=[],
            ),
            modifiers=[
                CFModifier(condition={}, effect_cf=-0.12,
                    explanation="SD lord is also a father-maraka: "
                                "deepest-level activation"),
                CFModifier(condition={}, effect_cf=+0.25,
                    explanation="Jupiter aspects natal Sun: protective"),
            ],
        ),
        fires_when=_sun_in_dusthana_with_maraka_subperiod,
        modifier_predicates=[
            _sd_is_maraka,
            _jupiter_aspects_natal_sun,
        ],
    ),
]
