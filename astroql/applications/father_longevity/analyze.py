"""Per-chart deep astrological analysis at AD-miss AND PD-miss level.

For each chart:
  * AD-miss case: compare truth-AD epoch vs picked-AD epoch — why
    did the picked AD beat the truth AD at AD-level?
  * AD-hit/PD-miss case: compare truth-PD vs picked-PD within the
    same matched AD — why did picked-PD beat truth-PD?

Dumps: dasha lords + roles, planet positions (transit + natal),
house lord positions, afflictions on Sun and on 9H, benefic/malefic
status of dasha lords, derived-father-lagna lords.

Run:
  python -X utf8 -m astroql.applications.father_longevity.analyze [version]
where [version] is v12 | v13 | v14 | v15 (default: current = v15).
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from astroql.engine import epoch_emitter as _ee
from astroql.engine.cf_predict import predict_extreme_epoch
from astroql.engine.shadbala import sign_lord
from astroql.schemas.birth import BirthDetails

from .rules import ALL_VERSIONS, RULES

DATA_PATH = Path(__file__).resolve().parent / "data" / "verified.json"
WINDOW_YEARS = 3.0
SPLIT_SEED = 42

_SIGN_ORDER = (
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
)

_MALEFICS = {"Saturn", "Mars", "Rahu", "Ketu", "Sun"}
_BENEFICS = {"Jupiter", "Venus", "Mercury", "Moon"}


def _nth(sign: str, n: int) -> Optional[str]:
    try: i = _SIGN_ORDER.index(sign)
    except ValueError: return None
    return _SIGN_ORDER[(i + n - 1) % 12]


def _polarity(planet: str) -> str:
    if planet in _MALEFICS: return "MALEFIC"
    if planet in _BENEFICS: return "benefic"
    return "?"


def _to_birth(rec):
    return BirthDetails(
        date=date.fromisoformat(rec["birth_date"][:10]),
        time=rec["birth_time"], tz=rec["tz"],
        lat=float(rec["lat"]), lon=float(rec["lon"]),
    )


def _window_for(truth, birth_tz, rng):
    o = rng.uniform(0.0, WINDOW_YEARS)
    db = int(o * 365.25); da = int((WINDOW_YEARS - o) * 365.25)
    tdt = datetime(truth.year, truth.month, truth.day, tzinfo=ZoneInfo(birth_tz))
    return tdt - timedelta(days=db), tdt + timedelta(days=da), tdt


def _chart_lords(ep) -> Dict[str, Any]:
    """Compute all lord identities for a chart."""
    lagna = ep.natal_lagna_sign
    if not lagna: return {}
    sun = ep.planets.get("Sun")
    sun_sign = sun.natal_sign if sun else "?"
    out = {
        "lagna_sign": lagna,
        "lagna_lord": sign_lord(lagna),
        "9L": sign_lord(_nth(lagna, 9)),
        "8L": sign_lord(_nth(lagna, 8)),
        "12L": sign_lord(_nth(lagna, 12)),
        "10L": sign_lord(_nth(lagna, 10)),
        "father_lagna_sign": _nth(lagna, 9),
        "father_2L": sign_lord(_nth(_nth(lagna, 9), 2)) if _nth(lagna, 9) else None,
        "father_7L": sign_lord(_nth(_nth(lagna, 9), 7)) if _nth(lagna, 9) else None,
        "father_8L": sign_lord(_nth(_nth(lagna, 9), 8)) if _nth(lagna, 9) else None,
        "father_12L": sign_lord(_nth(_nth(lagna, 9), 12)) if _nth(lagna, 9) else None,
        "sun_sign": sun_sign,
        "sun_house": sun.natal_house if sun else "?",
        "sun_2nd_maraka": sign_lord(_nth(sun_sign, 2)) if sun_sign != "?" else None,
        "sun_7th_maraka": sign_lord(_nth(sun_sign, 7)) if sun_sign != "?" else None,
    }
    return out


def _planet_role(p_name: str, lords: Dict) -> str:
    """Get all roles a planet plays in this chart."""
    roles = []
    if p_name == lords.get("lagna_lord"): roles.append("LL")
    if p_name == lords.get("9L"): roles.append("9L")
    if p_name == lords.get("8L"): roles.append("8L")
    if p_name == lords.get("12L"): roles.append("12L")
    if p_name == lords.get("10L"): roles.append("10L")
    if p_name == lords.get("father_2L"): roles.append("F2L")
    if p_name == lords.get("father_7L"): roles.append("F7L")
    if p_name == lords.get("father_8L"): roles.append("F8L")
    if p_name == lords.get("father_12L"): roles.append("F12L")
    if p_name == lords.get("sun_2nd_maraka"): roles.append("Sun2M")
    if p_name == lords.get("sun_7th_maraka"): roles.append("Sun7M")
    return ",".join(roles) if roles else "-"


def _dump_dasha_stack(ep, lords) -> str:
    lines = []
    for level, name in (("MD", ep.dashas.maha), ("AD", ep.dashas.antar),
                        ("PD", ep.dashas.pratyantar), ("SD", ep.dashas.sookshma)):
        p = ep.planets.get(name)
        roles = _planet_role(name, lords)
        pol = _polarity(name)
        if p is None:
            lines.append(f"   {level}={name:<8s} [{pol:8s}] roles={roles}")
            continue
        lines.append(
            f"   {level}={name:<8s} [{pol:8s}] roles={roles:<22} "
            f"natal=H{p.natal_house}/{p.natal_sign[:3]}  "
            f"transit=H{p.transit_house}/{p.transit_sign[:3]}  "
            f"mu={p.shadbala_coefficient:.2f}  "
            f"retro={'Y' if p.is_retrograde else '-'}  "
            f"on_natal_Sun={'Y' if 'Sun' in (ep.planets.get('Sun').aspects_on_natal if ep.planets.get('Sun') else []) and name in ep.planets.get('Sun').aspects_on_natal else '-'}"
        )
    return "\n".join(lines)


def _dump_key_signifiers(ep, lords) -> str:
    """Dump Sun, 9L, 8L, lagna lord transit/affliction state."""
    out = []
    for label, name in (("Sun", "Sun"),
                        ("9L=" + str(lords.get("9L", "?")), lords.get("9L")),
                        ("8L=" + str(lords.get("8L", "?")), lords.get("8L")),
                        ("LL=" + str(lords.get("lagna_lord", "?")), lords.get("lagna_lord")),
                        ("F8L=" + str(lords.get("father_8L", "?")), lords.get("father_8L"))):
        if not name: continue
        p = ep.planets.get(name)
        if p is None:
            out.append(f"   {label:<22}: <missing>")
            continue
        on_natal = ",".join(p.aspects_on_natal) or "-"
        recv = ",".join(p.aspects_receiving) or "-"
        out.append(
            f"   {label:<22}: H{p.transit_house:2d}/{p.transit_sign[:3]}  "
            f"mu={p.shadbala_coefficient:.2f}  "
            f"retro={'Y' if p.is_retrograde else '-'}  "
            f"recv_aspects={recv}"
        )
    return "\n".join(out)


def _find_eps_at_ad(epochs, md, ad):
    return [ep for ep in epochs if ep.dashas.maha == md and ep.dashas.antar == ad]


def main():
    ruleset = (sys.argv[1] if len(sys.argv) > 1 else "current").lower()
    if ruleset == "current":
        rules = RULES
        ruleset = "current (v15)"
    elif ruleset in ALL_VERSIONS:
        rules = ALL_VERSIONS[ruleset]
    else:
        print(f"Unknown ruleset: {ruleset}. Valid: current, "
              f"{list(ALL_VERSIONS.keys())}")
        sys.exit(1)
    print(f"=== Failure analysis under {ruleset} ({len(rules)} rules) ===\n")

    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rng = random.Random(SPLIT_SEED)

    for i, rec in enumerate(records):
        truth = date.fromisoformat(rec["father_death_date"][:10])
        birth = _to_birth(rec)
        start, end, truth_dt = _window_for(truth, birth.tz, rng)
        try:
            pred = predict_extreme_epoch(birth, start, end, rules,
                polarity="negative", max_window_years=WINDOW_YEARS + 0.5)
            all_eps = _ee.emit_epochs(birth, start, end,
                max_window_years=WINDOW_YEARS + 0.5)
        except Exception as e:
            print(f"[{i}] {rec.get('name')}: ERROR {e}\n")
            continue

        truth_ep = next((ep for ep in all_eps
                         if ep.start_time <= truth_dt <= ep.end_time), None)
        pred_ep = (next((ep for ep in all_eps
                         if ep.epoch_id == pred.extreme_epoch.epoch_id), None)
                   if pred else None)

        if not truth_ep or not pred_ep:
            print(f"[{i}] {rec.get('name')}: missing epochs\n")
            continue

        td, pd = truth_ep.dashas, pred_ep.dashas
        ad_match = (td.maha == pd.maha and td.antar == pd.antar)
        pd_match = ad_match and td.pratyantar == pd.pratyantar

        if pd_match:
            continue  # skip fully matched

        kind = "AD-MISS" if not ad_match else "PD-MISS (AD-hit)"
        lords = _chart_lords(truth_ep)

        print("=" * 90)
        print(f"[{i}] {rec.get('name')}  ({kind})")
        print(f"  truth = {truth} ({td.maha}-{td.antar}-{td.pratyantar}-{td.sookshma})")
        print(f"  pred  = {pred.predicted_date} ({pd.maha}-{pd.antar}-{pd.pratyantar}-{pd.sookshma})")
        print(f"  delta = {(pred.predicted_date - truth).days:+d} days")
        print()
        print(f"  CHART LORDS:")
        print(f"    lagna={lords.get('lagna_sign','?')} (LL={lords.get('lagna_lord')})  "
              f"Sun={lords.get('sun_sign','?')}/H{lords.get('sun_house','?')}")
        print(f"    9L={lords.get('9L')}  8L={lords.get('8L')}  "
              f"12L={lords.get('12L')}  10L={lords.get('10L')}")
        print(f"    father_lagna={lords.get('father_lagna_sign')}  "
              f"F2L={lords.get('father_2L')} F7L={lords.get('father_7L')} "
              f"F8L={lords.get('father_8L')} F12L={lords.get('father_12L')}")
        print(f"    Sun-marakas: 2nd={lords.get('sun_2nd_maraka')} "
              f"7th={lords.get('sun_7th_maraka')}")
        print()
        print(f"  TRUTH EPOCH STATE  ({truth_ep.start_time.date()}..{truth_ep.end_time.date()}):")
        print(_dump_dasha_stack(truth_ep, lords))
        print(f"  Key signifiers @ truth:")
        print(_dump_key_signifiers(truth_ep, lords))
        print()
        print(f"  PICKED EPOCH STATE  ({pred_ep.start_time.date()}..{pred_ep.end_time.date()}):")
        print(_dump_dasha_stack(pred_ep, lords))
        print(f"  Key signifiers @ picked:")
        print(_dump_key_signifiers(pred_ep, lords))
        print()
        # Rule-firing comparison
        print("  RULE FIRINGS (effective base after modifiers):")
        for ep, label in ((truth_ep, "TRUTH"), (pred_ep, "PICKED")):
            print(f"   [{label}]:")
            for spec in rules:
                try:
                    if not spec.fires_when(ep): continue
                    base = spec.rule.base_cf
                    mods_fired = []
                    for j, mp in enumerate(spec.modifier_predicates):
                        try:
                            if mp(ep):
                                mods_fired.append((j, spec.rule.modifiers[j].effect_cf,
                                                   spec.rule.modifiers[j].explanation))
                        except Exception: pass
                    rid_short = spec.rule.rule_id.split('.')[-2]
                    print(f"      {rid_short:<35} base={base:+.2f}")
                    for j, eff, expl in mods_fired:
                        print(f"          mod[{j}] {eff:+.2f}  {expl[:60]}")
                except Exception: pass
        print()


if __name__ == "__main__":
    main()
