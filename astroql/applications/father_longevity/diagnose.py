"""Per-chart diagnostic dump.

For a given subject index, prints:
  * natal Sun house + sign + sign-lord
  * dasha stack at truth and at predicted-extreme epoch
  * key transits (Saturn, Mars, Rahu/Ketu, Jupiter) at truth
  * which CF rules fired at truth vs picked
  * Saturn/Jupiter aspect targets

Run:
  python -X utf8 -m astroql.applications.father_longevity.diagnose <i> [version]
where <i> is the subject index (0-based) into the verified GT and
[version] is one of v12 | v13 | v14 | v15 (default: current = v15).
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from astroql.engine import epoch_emitter
from astroql.engine import shadbala as _sb
from astroql.engine.cf_predict import predict_extreme_epoch
from astroql.schemas.birth import BirthDetails

from .rules import ALL_VERSIONS, RULES

DATA_PATH = Path(__file__).resolve().parent / "data" / "verified.json"
WINDOW_YEARS = 3.0
SPLIT_SEED = 42


def _to_birth(rec):
    return BirthDetails(
        date=date.fromisoformat(rec["birth_date"][:10]),
        time=rec["birth_time"], tz=rec["tz"],
        lat=float(rec["lat"]), lon=float(rec["lon"]),
    )


def _window_for(truth, birth_tz, rng):
    offset = rng.uniform(0.0, WINDOW_YEARS)
    days_before = int(offset * 365.25)
    days_after = int((WINDOW_YEARS - offset) * 365.25)
    truth_dt = datetime(truth.year, truth.month, truth.day,
                        tzinfo=ZoneInfo(birth_tz))
    return (
        truth_dt - timedelta(days=days_before),
        truth_dt + timedelta(days=days_after),
        offset,
        truth_dt,
    )


def _seed_rngs_to(idx):
    """Replay the seeded uniform-offset draws up to subject idx."""
    rng = random.Random(SPLIT_SEED)
    for _ in range(idx):
        rng.uniform(0.0, WINDOW_YEARS)
    return rng


def main():
    if len(sys.argv) < 2:
        print("usage: diagnose.py <subject_index> [v12|v13|v14|v15]")
        sys.exit(1)
    idx = int(sys.argv[1])
    version = (sys.argv[2] if len(sys.argv) > 2 else "current").lower()
    if version == "current":
        rules = RULES
        version_label = "current (v15)"
    elif version in ALL_VERSIONS:
        rules = ALL_VERSIONS[version]
        version_label = version
    else:
        print(f"Unknown version: {version}")
        sys.exit(1)
    print(f"(diagnostic ruleset: {version_label}, {len(rules)} rules)\n")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    rec = records[idx]
    print(f"== Subject #{idx}  id={rec.get('id')}  "
          f"name={rec.get('name')}  "
          f"birth={rec['birth_date']} {rec.get('birth_time','?')} "
          f"tz={rec.get('tz')}")
    print(f"   gender={rec.get('gender','?')}  "
          f"truth (father death) = {rec['father_death_date']}")

    truth = date.fromisoformat(rec["father_death_date"][:10])
    birth = _to_birth(rec)
    rng = _seed_rngs_to(idx)
    start, end, offset_y, truth_dt = _window_for(truth, birth.tz, rng)
    print(f"   window = {start.date()} .. {end.date()}  "
          f"(truth offset {offset_y:.3f} y into window)")

    epochs = epoch_emitter.emit_epochs(
        birth, start, end, max_window_years=WINDOW_YEARS + 0.5,
    )
    print(f"   {len(epochs)} sookshma epochs in window")

    e0 = epochs[0]
    sun = e0.planets["Sun"]
    print()
    print("== Natal Sun (father karaka)")
    print(f"   natal_sign  = {sun.natal_sign}")
    print(f"   natal_house = {sun.natal_house}  "
          f"(=> house from lagna)")
    print(f"   sign_lord(Sun's natal sign) = "
          f"{_sb.sign_lord(sun.natal_sign) if sun.natal_sign else '?'}")
    sign_order = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
                  "Libra", "Scorpio", "Sagittarius", "Capricorn",
                  "Aquarius", "Pisces"]
    if sun.natal_sign:
        si = sign_order.index(sun.natal_sign)
        s2 = sign_order[(si + 1) % 12]
        s7 = sign_order[(si + 6) % 12]
        print(f"   2nd-from-Sun  = {s2}  (lord: {_sb.sign_lord(s2)})")
        print(f"   7th-from-Sun  = {s7}  (lord: {_sb.sign_lord(s7)})")
    print()
    print("== Natal mu (shadbala) by planet")
    for p in ("Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus",
              "Saturn", "Rahu", "Ketu"):
        if p in e0.planets:
            v = e0.planets[p]
            print(f"   {p:8s} mu={v.shadbala_coefficient:.3f}  "
                  f"natal_house={v.natal_house}  "
                  f"natal_sign={v.natal_sign}")

    closest = min(
        epochs,
        key=lambda ep: abs(
            ((ep.start_time + (ep.end_time - ep.start_time) / 2)
             - truth_dt).days
        ),
    )

    pred = predict_extreme_epoch(
        birth, start, end, rules, polarity="negative",
        max_window_years=WINDOW_YEARS + 0.5,
    )
    picked_epoch = next(
        (ep for ep in epochs if ep.epoch_id == pred.extreme_epoch.epoch_id),
        None,
    ) if pred else None

    def _dump_epoch_state(label, ep):
        print()
        print(f"== Epoch state @ {label}  "
              f"({ep.start_time.date()} .. {ep.end_time.date()})")
        print(f"   dasha: MD={ep.dashas.maha} AD={ep.dashas.antar} "
              f"PD={ep.dashas.pratyantar} SD={ep.dashas.sookshma}")
        for p in ("Sun", "Saturn", "Mars", "Jupiter", "Rahu", "Ketu",
                  "Moon", "Venus", "Mercury"):
            ps = ep.planets.get(p)
            if ps is None:
                continue
            on_natal = ",".join(ps.aspects_on_natal) or "-"
            on_transit = ",".join(ps.aspects_receiving) or "-"
            print(f"   {p:8s} sign={ps.transit_sign:14s} "
                  f"house={ps.transit_house:2d}  "
                  f"retro={'Y' if ps.is_retrograde else '-'}  "
                  f"aspects_on_natal={on_natal}  "
                  f"aspects_recv={on_transit}")

    _dump_epoch_state(f"truth ({truth})", closest)
    if picked_epoch is not None:
        _dump_epoch_state(
            f"picked ({pred.predicted_date}, "
            f"cf={pred.cf:+.3f})", picked_epoch,
        )

    print()
    print("== Rule firing at truth-closest vs picked")
    for ep, lbl in ((closest, "truth"),
                    (picked_epoch, "picked")):
        if ep is None:
            continue
        fired = []
        for spec in rules:
            try:
                if spec.fires_when(ep):
                    mods = []
                    for j, mp in enumerate(spec.modifier_predicates):
                        try:
                            if mp(ep):
                                mods.append(j)
                        except Exception:
                            pass
                    fired.append((spec.rule.rule_id, mods,
                                  spec.rule.modifiers))
            except Exception:
                pass
        print(f"   [{lbl}] fired:")
        for rid, mods, mod_objs in fired:
            print(f"      {rid}  modifiers_fired={mods}")
            for m_idx in mods:
                if m_idx < len(mod_objs):
                    explanation = mod_objs[m_idx].explanation
                    print(f"         m[{m_idx}] eff_cf="
                          f"{mod_objs[m_idx].effect_cf:+.2f} :: "
                          f"{explanation}")
        if not fired:
            print(f"      <none>")


if __name__ == "__main__":
    main()
