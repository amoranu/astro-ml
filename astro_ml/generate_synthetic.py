"""Synthetic data generator for pipeline testing.

Generates structurally valid but random chart payloads. Each chart gets a random
event date, and the analysis window is built around it.

Usage:
  python -m astro_ml.generate_synthetic --n_train 200 --n_val 50 --n_test 50 --output_dir ./data
"""
import os, sys, json, random, argparse, datetime
import numpy as np

from astro_ml.config import domain_fathers_death as cfg


def _random_date(start_year=1900, end_year=2000):
    """Random date between start_year and end_year."""
    y = random.randint(start_year, end_year)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y}-{m:02d}-{d:02d}"


def _random_time():
    return f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"


def _random_lat_lon():
    lat = random.uniform(-60, 60)
    lon = random.uniform(-180, 180)
    return round(lat, 4), round(lon, 4)


def generate_synthetic_chart(chart_id):
    """Generate a single synthetic chart with valid structure."""
    lat, lon = _random_lat_lon()
    birth_date = _random_date(1900, 1990)

    # Father death: 20-80 years after birth
    bd_parts = birth_date.split("-")
    birth_year = int(bd_parts[0])
    death_year = birth_year + random.randint(20, 80)
    death_month = random.randint(1, 12)
    death_day = random.randint(1, 28)
    father_death_date = f"{death_year}-{death_month:02d}-{death_day:02d}"

    return {
        "name": f"Synthetic_{chart_id}",
        "birth_date": birth_date,
        "birth_time": _random_time(),
        "lat": lat,
        "lon": lon,
        "tz": random.choice(["UTC", "America/New_York", "Europe/London",
                             "Asia/Kolkata", "Asia/Tokyo", "Australia/Sydney"]),
        "gender": random.choice(["M", "F"]),
        "time_rating": random.choice(["A", "B", "C"]),
        "father_death_date": father_death_date,
        "source": "Synthetic",
        "id": chart_id,
    }


def generate_synthetic_payload(chart_id):
    """Generate a synthetic payload (the compute() output) with random but valid structure.

    This is for testing the feature extraction pipeline WITHOUT astro_engine.
    """
    # KP Cusps
    kp_cusps = {}
    cusp_lons = sorted(random.sample(range(0, 360), 12))
    for i in range(12):
        house = i + 1
        lon = cusp_lons[i]
        sign_num = (lon // 30) + 1
        kp_cusps[f"Cusp_{house}"] = {
            "degree": lon,
            "rashi": cfg.SIGNS[sign_num - 1],
            "degree_in_sign": lon % 30,
            "star_lord": random.choice(cfg.PLANETS),
            "sub_lord": random.choice(cfg.PLANETS),
            "sign_lord": cfg.SIGN_LORDS[sign_num],
        }

    # KP Planets
    kp_planets = {}
    for planet in cfg.PLANETS:
        lon = random.uniform(0, 360)
        sign_num = int(lon // 30) + 1
        kp_planets[planet] = {
            "degree": lon,
            "rashi": cfg.SIGNS[sign_num - 1],
            "degree_in_sign": lon % 30,
            "nakshatra": random.choice(["Ashwini", "Bharani", "Krittika", "Rohini",
                                        "Mrigashira", "Ardra", "Punarvasu", "Pushya",
                                        "Ashlesha", "Magha", "PurvaPhalguni"]),
            "star_lord": random.choice(cfg.PLANETS),
            "sub_lord": random.choice(cfg.PLANETS),
            "sub_lord_star_lord": random.choice(cfg.PLANETS),
        }

    # Functional Nature
    fn_choices = ["Benefic", "Malefic", "Neutral", "Yogakaraka", "Mixed"]
    functional_nature = {p: random.choice(fn_choices) for p in cfg.PLANETS}

    # Dignity
    dig_choices = ["Exalted", "Own", "Friendly", "Neutral", "Enemy", "Debilitated"]
    planetary_dignity = {p: random.choice(dig_choices) for p in cfg.PLANETS}

    # Navamsha
    navamsha_positions = {}
    for p in cfg.PLANETS:
        d9_sn = random.randint(1, 12)
        natal_sn = cfg.SIGN_TO_NUM.get(kp_planets[p]["rashi"], 1)
        is_vargo = (d9_sn == natal_sn)
        navamsha_positions[p] = {
            "rashi": cfg.SIGNS[d9_sn - 1],
            "sign_num": d9_sn,
            "dignity": random.choice(dig_choices),
            "is_vargottama": is_vargo,
            "nav_dignity_string": f"VARGOTTAMA+EXALTED" if is_vargo else random.choice(dig_choices).upper(),
        }

    # Natal flags
    natal_flags = {p: {"is_retrograde": random.random() < 0.3,
                       "is_combust": random.random() < 0.15}
                   for p in cfg.PLANETS}

    # Significations
    planet_sigs = {p: sorted(random.sample(range(1, 13), random.randint(2, 5)))
                   for p in cfg.PLANETS}

    # House lordships
    house_lordships = {str(h): cfg.SIGN_LORDS.get(
        cfg.SIGN_TO_NUM.get(kp_cusps.get(f"Cusp_{h}", {}).get("rashi", "Aries"), 1), "Mars")
        for h in range(1, 13)}

    # Timeline + Triggers (24 months)
    birth_date = _random_date(1950, 1990)
    bd_parts = birth_date.split("-")
    death_year = int(bd_parts[0]) + random.randint(30, 60)
    start_year = death_year - 1
    start_month = random.randint(1, 12)

    calculated_triggers = {}
    dasha_lords = random.sample(cfg.PLANETS, 3)
    for i in range(24):
        m = (start_month + i - 1) % 12 + 1
        y = start_year + (start_month + i - 1) // 12
        month_key = f"{y}-{m:02d}"

        month_data = {"_dasha": {"md": dasha_lords[0], "ad": dasha_lords[1], "pd": dasha_lords[2]}}
        for cusp_num in cfg.TARGET_CUSPS:
            hits = []
            if random.random() < 0.3:
                hits.append({
                    "planet": random.choice(["Jupiter", "Saturn", "Rahu"]),
                    "aspect": random.choice([0, 90, 120, 180, 240]),
                    "orb": round(random.uniform(0.1, 3.0), 2),
                    "target_degree": round(random.uniform(0, 360), 2),
                })
            month_data[f"Cusp_{cusp_num}"] = {
                "Macro_Hits": hits,
                "Dasha_Lock": random.random() < 0.1,
                "Dasha_Lock_Level": random.choice(["MD", "AD", "PD", ""]),
                "Cusp_SubLord": random.choice(cfg.PLANETS),
            }
        calculated_triggers[month_key] = month_data

    cross_chart = {
        "lagna_lord": random.choice(cfg.PLANETS[:7]),
        "lagna_sign": random.choice(cfg.SIGNS),
        "lagna_sign_num": random.randint(1, 12),
        "lagna_degree": random.uniform(0, 360),
        "moon_sign": random.choice(cfg.SIGNS),
        "moon_sign_num": random.randint(1, 12),
        "moon_lon": random.uniform(0, 360),
        "sun_sign": random.choice(cfg.SIGNS),
        "9th_cusp_sign": random.choice(cfg.SIGNS),
        "9th_cusp_sign_num": random.randint(1, 12),
    }

    return {
        "Vimshottari_Timeline": [],
        "KP_Cusps": kp_cusps,
        "KP_Planets": kp_planets,
        "Functional_Nature": functional_nature,
        "Planetary_Dignity": planetary_dignity,
        "Navamsha_Positions": navamsha_positions,
        "Natal_Flags": natal_flags,
        "Calculated_Triggers": calculated_triggers,
        "Planet_Significations": planet_sigs,
        "House_Lordships": house_lordships,
        "Cross_Chart_Data": cross_chart,
    }


def generate_dataset(n_charts, output_dir, split_name="train"):
    """Generate n synthetic charts and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    charts = [generate_synthetic_chart(i + 1) for i in range(n_charts)]
    path = os.path.join(output_dir, f"synthetic_{split_name}.json")
    with open(path, "w") as f:
        json.dump(charts, f, indent=2)
    print(f"Generated {n_charts} charts -> {path}")
    return charts


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic astro data")
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_val", type=int, default=50)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./astro_ml/data")
    args = parser.parse_args()

    generate_dataset(args.n_train, args.output_dir, "train")
    generate_dataset(args.n_val, args.output_dir, "val")
    generate_dataset(args.n_test, args.output_dir, "test")
    print(f"\nAll synthetic data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
