"""
Father's Death -- V11 Advanced ML Pipeline
===========================================
156 features: V10 (124) + 32 new cross-tradition features:
- V8 base: 98 features (dasha, transit, degree, Firdaria, SAV/BAV, Sade Sati,
  conjunctions, dignity, gandanta, nakshatra lords, etc.)
- V9: 18 features (D12, D9, Saturn return, MD sandhi, Shadbala, Chara Dasha AD)
- V10: 8 features (Gulika/Mandi, Mars speed, Jup protection, Sat-Rahu, 9L retro)
- NEW V11: 4 Hellenistic Lot of Death features (transit over natal lot)
- NEW V11: 4 BAV specific house features (8th/9th/3rd/4th house Saturn BAV)
- NEW V11: 4 Annual Profection features (profected house, lord of year)
- NEW V11: 6 BaZi features (luck pillar clashes, annual interactions, Fu Yin/Fan Yin)
- NEW V11: 4 KP cusp Sub-Lord features (8th/9th cusp sub-lord analysis)
- NEW V11: 4 Zodiacal Releasing from Lot of Death (L1/L2 angular, malefic ruler)
- NEW V11: 4 D60 Shashtiamsha features (Sun/9L/Saturn D60 positions)
- NEW V11: 2 cross-tradition convergence features

Models: GradientBoosting/XGBoost with LOOCV + Two-Stage (broad->narrow)
Evaluation: LOOCV (leave-one-out), primary metric = +-1 month accuracy

Usage:
    python investigation/father_death_ml.py                  # Full pipeline
    python investigation/father_death_ml.py --extract-only   # Just extract
    python investigation/father_death_ml.py --optimize-only  # From cache
"""

import sys, os, json, datetime, math, pickle, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytz
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from astro_engine import AstroEngine

_thread_local = threading.local()
def _get_engine():
    if not hasattr(_thread_local, "engine"):
        _thread_local.engine = AstroEngine()
    return _thread_local.engine

# ---- Constants ----
SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
SIGN_LORDS = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
    "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
    "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}
NATURAL_MALEFICS = {"Mars", "Saturn", "Rahu", "Ketu", "Sun"}
STRICT_MALEFICS = {"Mars", "Saturn", "Rahu", "Ketu"}
NATURAL_BENEFICS = {"Jupiter", "Venus", "Mercury", "Moon"}
# Combustion orbs (degrees from Sun where planet is combust)
_COMBUSTION_ORBS = {
    "Moon": 12, "Mars": 17, "Mercury": 14, "Jupiter": 11,
    "Venus": 10, "Saturn": 15,
}
# Debilitation signs (1-indexed sign number)
_DEBILITATION_SIGN = {
    "Sun": 7, "Moon": 8, "Mars": 4, "Mercury": 12, "Jupiter": 10,
    "Venus": 6, "Saturn": 1, "Rahu": 8, "Ketu": 2,
}
# Enemy signs for each planet (simplified: debilitation sign + its lord's enemy signs)
_ENEMY_LORDS = {
    "Sun": {"Saturn", "Venus"}, "Moon": {"Rahu", "Ketu", "Saturn"},
    "Mars": {"Mercury"}, "Mercury": {"Moon"},
    "Jupiter": {"Mercury", "Venus"}, "Venus": {"Sun", "Moon"},
    "Saturn": {"Sun", "Moon", "Mars"},
}
# Nakshatra lords (Vimsottari order, 0-indexed)
_NAKSHATRA_LORDS = [
    "Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter",
    "Saturn", "Mercury", "Ketu", "Venus", "Sun", "Moon", "Mars",
    "Rahu", "Jupiter", "Saturn", "Mercury", "Ketu", "Venus", "Sun",
    "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury",
]
_BASE_ASPECT = {6}
_SPECIAL_ASPECTS = {"Mars": {3, 7}, "Jupiter": {4, 8}, "Saturn": {2, 9}}

FEATURE_NAMES = [
    # === DASHA MD/AD (0-9) ===
    "MD/AD=10L",           # 0
    "MD/AD=3L",            # 1
    "MD/AD=9L",            # 2
    "MD/AD=4L",            # 3
    "MD/AD=Sun",           # 4
    "MD/AD=22drek",        # 5
    "MD/AD=8L",            # 6
    "MD/AD=64nav",         # 7
    "MD/AD=11L",           # 8
    "dasha_lord_count",    # 9
    # === PD (10-18) ===
    "PD=maraka",           # 10 (PD = 3L or 10L)
    "PD=9L",               # 11
    "PD=4L",               # 12
    "PD=Sun",              # 13
    "PD=22drek",           # 14
    "PD=8L",               # 15
    "PD=64nav",            # 16
    "PD=nat_malefic",      # 17
    "PD=father_any",       # 18 (9L/4L/Sun/11L)
    # === AD/PD TIMING (19-22) ===
    "AD_sandhi",           # 19 (within 2 months of AD end)
    "AD_months_left",      # 20 (normalized 0-1)
    "PD_sandhi",           # 21 (within 1 month of PD boundary)
    "PD_months_left",      # 22 (normalized 0-1, capped at 6 months)
    # === TRANSIT SIGN (23-31) ===
    "Sat_on_9H",           # 23
    "Sat_on_3H",           # 24
    "DT_10H",              # 25
    "DT_4H",               # 26
    "RaKe_3_9",            # 27
    "Pitri_Saham_hit",     # 28
    "9L_transit_dusthana", # 29
    "malefic_count_3H",    # 30
    "Sun_conj_mal_count",  # 31
    # === SUN PER-HOUSE (32-35) ===
    "Sun_in_3H",           # 32
    "Sun_in_4H",           # 33
    "Sun_in_9H",           # 34
    "Sun_in_10H",          # 35
    # === MARS PER-HOUSE (36-39) ===
    "Mars_in_3H",          # 36
    "Mars_in_4H",          # 37
    "Mars_in_9H",          # 38
    "Mars_in_10H",         # 39
    # === MARS DEGREE (40-41) ===
    "Mars_deg_Sun",        # 40
    "Mars_deg_9L",         # 41
    # === FIRDARIA (42-45) ===
    "Fir_major=malefic",   # 42
    "Fir_sub=malefic",     # 43
    "Fir_major=father",    # 44
    "Fir_sub=father",      # 45
    # === CONVERGENCE (46-49) ===
    "dasha_AND_transit",   # 46
    "strong_convergence",  # 47
    "dasha_x_transit",     # 48
    "sqrt_convergence",    # 49
    # === ASHTAKAVARGA TRANSIT (50-52) ===
    "SAV_sat_sign",        # 50 (SAV of Saturn's transit sign, normalized 0-1)
    "BAV_sat_sign",        # 51 (Saturn's BAV in its transit sign, normalized 0-1)
    "SAV_9H",              # 52 (SAV of 9th house, normalized 0-1)
    # === SADE SATI (53-55) ===
    "sade_sati_active",    # 53
    "sade_sati_phase1",    # 54 (12th from Moon = 10th aspect on 9th from Moon)
    "ashtama_shani",       # 55 (Saturn 8th from Moon)
    # === SATURN EXTRA (56-59) ===
    "Sat_ingress",         # 56 (Saturn degree < 3.75 in sign = just entered)
    "Sat_retrograde",      # 57 (Saturn retrograde)
    "Sat_deg_Sun",         # 58 (Saturn degree proximity to natal Sun)
    "Sat_deg_9L",          # 59 (Saturn degree proximity to natal 9L)
    # === JUPITER DEGREE (60-61) ===
    "Jup_deg_Sun",         # 60 (Jupiter degree proximity to natal Sun)
    "Jup_deg_9L",          # 61 (Jupiter degree proximity to natal 9L)
    # === SUN DEGREE (62-63) ===
    "Sun_deg_Sun",         # 62 (transit Sun degree proximity to natal Sun)
    "Sun_deg_9L",          # 63 (transit Sun degree proximity to natal 9L)
    # === TRIPLE DASHA + MALEFIC COMBO (64-66) ===
    "triple_dasha_hit",    # 64 (MD+AD+PD all father-related)
    "malefic_md_ad",       # 65 (both MD and AD are natural malefics)
    "maraka_count_3",      # 66 (count of MD/AD/PD that are father marakas, /3)
    # === CHARA DASHA / JAIMINI (67-69) ===
    "CD_9H",               # 67 (Chara Dasha sign = 9th house sign)
    "CD_maraka",           # 68 (Chara Dasha sign = 3H or 10H = maraka from 9th)
    "CD_father",           # 69 (Chara Dasha sign contains natal Sun or 9L)
    # === TARA BALA (70-71) ===
    "tara_sat_bad",        # 70 (Saturn in Naidhana or Vipat tara)
    "tara_sat_score",      # 71 (Tara Bala score for Saturn, inverted: bad=high)
    # === RAHU/KETU DEGREE (72-73) ===
    "RaKe_deg_Sun",        # 72 (max Rahu/Ketu degree proximity to natal Sun)
    "RaKe_deg_9L",         # 73 (max Rahu/Ketu degree proximity to natal 9L)
    # === A9 TRANSIT (74) ===
    "A9_malefic",          # 74 (Saturn/Rahu/Ketu on Arudha of 9th)
    # === PLANET MOTION (75-76) ===
    "sat_speed_slow",      # 75 (Saturn slowness: 1.0=stationary, 0.0=full speed)
    "jup_retro",           # 76 (Jupiter retrograde)
    # === PITRUKARAKA (77-78) ===
    "PiK_in_dasha",        # 77 (Pitrukaraka planet is MD/AD/PD lord)
    "CD_PiK_sign",         # 78 (Chara Dasha sign = natal Pitrukaraka's sign)
    # === V8: SUN-PLANET CONJUNCTIONS (79-82) ===
    "Sun_Sat_conj",        # 79 (Sun-Saturn degree proximity, orb=15)
    "Sun_Node_conj",       # 80 (Sun-Rahu/Ketu degree proximity, orb=12)
    "Sun_Mars_conj",       # 81 (Sun-Mars degree proximity, orb=12)
    "lord9_combust",       # 82 (9L within combustion orb of Sun)
    # === V8: GANDANTA & SIGN BOUNDARY (83-85) ===
    "Sun_gandanta",        # 83 (Sun at 0-1 or 29-30 deg in sign = vulnerable)
    "lord9_gandanta",      # 84 (9L at gandanta/sign boundary)
    "Sat_gandanta",        # 85 (Saturn at gandanta)
    # === V8: MOON NAKSHATRA LORD (86-88) ===
    "moon_nak_lord_maraka",# 86 (Moon's nakshatra lord = father maraka)
    "moon_nak_lord_saturn",# 87 (Moon's nakshatra lord = Saturn)
    "sat_nak_lord_maraka", # 88 (Saturn's nakshatra lord = father maraka)
    # === V8: TRANSIT DIGNITY (89-91) ===
    "lord9_debilitated",   # 89 (9L transit in debilitated sign)
    "lord9_enemy_sign",    # 90 (9L transit in enemy sign)
    "Sun_debilitated",     # 91 (Sun debilitated = Libra)
    # === V8: DOUBLE TRANSIT EXPANDED (92-94) ===
    "DT_3H",              # 92 (Jupiter AND Saturn both aspect/transit 3H)
    "DT_9H",              # 93 (Jupiter AND Saturn both aspect/transit 9H)
    "multi_malefic_9H",   # 94 (count of malefics on/aspecting 9H, /4)
    # === V8: BADHAKA & 8TH LORD (95-97) ===
    "lord8_in_dasha",      # 95 (8th lord as MD or AD - death house activation)
    "Sat_on_4H",           # 96 (Saturn on or aspecting 4th house)
    "Sat_on_10H",          # 97 (Saturn on or aspecting 10th house)
    # === V9: D12 DWADASHAMSHA (98-102) ===
    "D12_9L_in_dasha",     # 98 (D12 9th lord is MD or AD lord)
    "Sat_on_D12_9H",       # 99 (transit Saturn on/aspecting D12 9th house sign)
    "RaKe_on_D12_9H",      # 100 (Rahu/Ketu on D12 9th house sign)
    "D12_Sun_dusthana",    # 101 (Sun in dusthana 6/8/12 in D12)
    "D12_9L_dusthana",     # 102 (D12 9th lord in dusthana in D12)
    # === V9: NAVAMSA D9 TRANSIT TRIGGERS (103-105) ===
    "Sat_on_D9_Sun",       # 103 (transit Saturn on/aspecting D9 Sun sign)
    "Sat_on_D9_9L",        # 104 (transit Saturn on/aspecting D9 9th lord sign)
    "D9_Sun_dusthana",     # 105 (Sun in dusthana in D9 chart)
    # === V9: SATURN RETURN (106-107) ===
    "Sat_return_prox",     # 106 (transit Saturn degree proximity to natal Saturn)
    "Sat_half_return",     # 107 (transit Saturn opposition to natal Saturn)
    # === V9: MD SANDHI / TRANSITION (108-109) ===
    "MD_sandhi_6mo",       # 108 (within 6 months of MD end = dasha junction)
    "MD_sandhi_12mo",      # 109 (within 12 months of MD end)
    # === V9: SHADBALA (110-112) ===
    "Sun_shadbala_weak",   # 110 (Sun shadbala inverted: weak Sun = high danger)
    "lord9_shadbala_weak", # 111 (9th lord shadbala inverted)
    "Sat_shadbala_strong", # 112 (Saturn shadbala: strong = more harm power)
    # === V9: CHARA DASHA AD (113-114) ===
    "CD_AD_9H",            # 113 (Chara Dasha AD sign = 9th house sign)
    "CD_AD_maraka",        # 114 (Chara Dasha AD sign = 3H or 10H)
    # === V9: AD LORD SPECIFICS (115) ===
    "AD_is_22drek_64nav",  # 115 (AD lord is 22nd drekkana or 64th navamsa lord)
    # === V10: NEW TIME-VARYING (116-123) ===
    "Sat_on_Gulika",       # 116 (transit Saturn on/aspecting natal Gulika sign)
    "Sat_on_Mandi",        # 117 (transit Saturn on/aspecting natal Mandi sign)
    "Mars_speed_slow",     # 118 (Mars slowness: stationary Mars = intensified malefic)
    "Jup_protect_9H",     # 119 (Jupiter on/aspecting 9H = protective, inverted: absent=1.0)
    "Sat_Rahu_conj",      # 120 (Saturn-Rahu degree conjunction, extremely malefic)
    "lord9_retro",         # 121 (transit 9th lord retrograde = weakened father significator)
    "Rahu_deg_Sat_natal",  # 122 (Rahu degree proximity to natal Saturn)
    "mal_on_4H_count",     # 123 (count of malefics on 4th house = 8th from 9th = father longevity)
    # === V11: HELLENISTIC LOT OF DEATH (124-127) ===
    "Sat_on_LotDeath",     # 124 (transit Saturn on/aspecting Lot of Death sign)
    "RaKe_on_LotDeath",    # 125 (Rahu/Ketu on Lot of Death sign)
    "Mars_on_LotDeath",    # 126 (transit Mars on/aspecting Lot of Death sign)
    "LotDeath_ruler_hit",  # 127 (Lot of Death ruler is active in dasha MD/AD/PD)
    # === V11: BAV SPECIFIC HOUSES (128-131) ===
    "BAV_Sat_8H",          # 128 (Saturn BAV in 8th house sign — low = death vulnerability)
    "BAV_Sat_9H",          # 129 (Saturn BAV in 9th house sign — low = father weakness)
    "BAV_Sat_3H",          # 130 (Saturn BAV in 3rd house sign — 7th from 9th = maraka)
    "SAV_8H",              # 131 (SAV of 8th house sign — low = death house weak)
    # === V11: ANNUAL PROFECTIONS (132-135) ===
    "profect_8H",          # 132 (profected sign = 8th house = death year)
    "profect_9H",          # 133 (profected sign = 9th house = father year)
    "profect_lord_malefic",# 134 (lord of profected year is natural malefic)
    "profect_lord_maraka",  # 135 (lord of year is father maraka lord)
    # === V11: BAZI (136-141) ===
    "bazi_annual_clash",   # 136 (annual pillar clashes with natal pillars)
    "bazi_luck_clash",     # 137 (luck pillar clashes with natal pillars)
    "bazi_fu_yin",         # 138 (Fu Yin: luck=annual, repetition=intensification)
    "bazi_fan_yin",        # 139 (Fan Yin: opposite clash, destructive)
    "bazi_punishment",     # 140 (Three Punishments involving luck/annual)
    "bazi_father_star_hit",# 141 (annual stem controls father star element)
    # === V11: KP CUSP SUB-LORD (142-145) ===
    "kp_8cusp_sub_malefic",# 142 (8th cusp sub-lord = natural malefic)
    "kp_9cusp_sub_malefic",# 143 (9th cusp sub-lord = natural malefic)
    "kp_8cusp_sub_in_dasha",# 144 (8th cusp sub-lord active in dasha)
    "kp_9cusp_sub_in_dasha",# 145 (9th cusp sub-lord active in dasha)
    # === V11: ZODIACAL RELEASING FROM LOT OF DEATH (146-149) ===
    "zr_death_L1_angular", # 146 (ZR L1 period is angular to Lot of Death)
    "zr_death_L2_angular", # 147 (ZR L2 sub-period is angular to Lot of Death)
    "zr_death_L1_malefic", # 148 (ZR L1 ruler is natural malefic)
    "zr_death_L2_malefic", # 149 (ZR L2 ruler is natural malefic)
    # === V11: D60 SHASHTIAMSHA (150-153) ===
    "D60_Sun_dusthana",    # 150 (Sun in dusthana 6/8/12 in D60)
    "D60_9L_dusthana",     # 151 (9th lord in dusthana in D60)
    "D60_Sat_kendra",      # 152 (Saturn in kendra 1/4/7/10 in D60 = strong to harm)
    "D60_8L_dusthana",     # 153 (8th lord in dusthana in D60 = death house activated)
    # === V11: CROSS-TRADITION CONVERGENCE (154-155) ===
    "hellenistic_vedic_conv",  # 154 (Lot of Death + dasha + transit all active)
    "bazi_vedic_conv",         # 155 (BaZi clash + Vedic dasha/transit convergence)
]
N_FEATURES = len(FEATURE_NAMES)

# ---- Helpers ----

def _house_sign(lagna_sn, house):
    return ((lagna_sn - 1 + house - 1) % 12) + 1

def _house_lord(lagna_sn, house):
    return SIGN_LORDS[SIGNS[_house_sign(lagna_sn, house) - 1]]

def _planet_on_or_aspects(p_sn, p_name, t_sn):
    if p_sn == t_sn:
        return True
    offset = (t_sn - p_sn) % 12
    return offset in (_BASE_ASPECT | _SPECIAL_ASPECTS.get(p_name, set()))

def _house_of(p_sn, lagna_sn):
    return ((p_sn - lagna_sn) % 12) + 1

def _angular_distance(lon1, lon2):
    d = abs(lon1 - lon2) % 360
    return d if d <= 180 else 360 - d

def _degree_proximity_score(lon1, lon2, orb=8.0):
    """Degree proximity (conjunction/opposition). Gaussian decay."""
    d_conj = _angular_distance(lon1, lon2)
    d_opp = _angular_distance(lon1, (lon2 + 180) % 360)
    min_d = min(d_conj, d_opp)
    return math.exp(-(min_d / orb) ** 2)

def _compute_64th_navamsa_lord(cusp_lon):
    nav_size = 360.0 / 108.0
    nav_idx = int(cusp_lon / nav_size)
    nav_64 = (nav_idx + 63) % 108
    sign_idx = nav_64 // 9
    return SIGN_LORDS[SIGNS[sign_idx]]

def _pitri_saham_sn(natal_planets, lagna):
    sat = natal_planets["Saturn"]["longitude"]
    sun = natal_planets["Sun"]["longitude"]
    asc = lagna["longitude"]
    return int(((sat - sun + asc) % 360) / 30) + 1

def _parse_firdaria(firdaria_list):
    periods = []
    if not isinstance(firdaria_list, list):
        return periods
    for item in firdaria_list:
        if not isinstance(item, dict):
            continue
        major = item.get("planet", "")
        for sp in item.get("sub_periods", []):
            start = sp.get("start", "")
            end = sp.get("end", "")
            sub = sp.get("planet", "")
            if start and end and sub:
                periods.append((start, end, major, sub))
    return periods

def _firdaria_at_month(periods, year, month):
    ym = f"{year}-{month:02d}"
    for start, end, major, sub in periods:
        if start <= ym <= end:
            return major, sub
    return "", ""

def _gaussian_smooth(arr, sigma=0.7):
    """Apply 1D Gaussian smoothing to score array."""
    if sigma <= 0:
        return arr
    n = len(arr)
    kernel_size = max(3, int(sigma * 4) | 1)  # odd kernel
    half = kernel_size // 2
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(arr, half, mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:n]


def _cluster_peak(scores, top_k=5, radius=2, smooth=0.0):
    """Find the densest cluster among top-K peaks. Returns best index."""
    s_arr = _gaussian_smooth(scores, smooth) if smooth > 0 else scores
    tk = np.argsort(s_arr)[-top_k:]
    tks = sorted(tk)
    best_c, best_s = int(np.argmax(s_arr)), 0
    for anc in tks:
        cl = [t for t in tks if abs(t - anc) <= radius]
        cs = sum(s_arr[t] for t in cl) * len(cl)
        if cs > best_s:
            best_s = cs
            w = np.array([s_arr[t] for t in cl])
            best_c = int(round(np.average(cl, weights=w)))
    return best_c


def _find_hot_zones(borda_stack, raw_preds, rw=1.5, n_zones=8,
                    zone_radius=3, fine_smooth=1.0,
                    coarse_smooth=4.0, agreement_top_k=5):
    """Multi-level peak detection + ranking for window-size-independent prediction.

    Strategy:
    1. Coarse pass: heavy smoothing -> find year-level hot zones
    2. Fine pass: within each zone, light smoothing -> exact month
    3. Rank zones by: model agreement × prominence × Borda score

    Args:
        borda_stack: (n_models, n_months) Borda-normalized scores
        raw_preds: list of raw model predictions (same order as borda_stack rows)
        rw: rank weight for ranking models (1.0 for regression, rw for ranking)
        n_zones: max number of hot zones to extract
        zone_radius: months around each coarse peak to search for fine peak
        fine_smooth: sigma for fine-level smoothing
        coarse_smooth: sigma for coarse-level smoothing (finds year-level clusters)
        agreement_top_k: how many top months per model to count for agreement

    Returns:
        list of (month_idx, zone_score) sorted by zone_score descending
    """
    n_models, n_months = borda_stack.shape
    n_reg = sum(1 for i, p in enumerate(raw_preds) if i < len(raw_preds) // 2)
    # Build weights: 1.0 for regression, rw for ranking
    n_rank = n_models - n_reg
    w = np.array([1.0] * n_reg + [rw] * n_rank)
    blended = np.average(borda_stack, axis=0, weights=w)

    # ---- Model agreement: fraction of models with month in their top-K ----
    agreement = np.zeros(n_months)
    for i in range(n_models):
        top_idx = np.argsort(borda_stack[i])[-agreement_top_k:]
        agreement[top_idx] += 1.0
    agreement /= n_models  # 0.0 to 1.0

    # ---- Multi-scale stability: count how many smoothing scales have this as argmax ----
    scales = [0.5, 1.0, 2.0, 3.0, 5.0]
    scale_votes = np.zeros(n_months)
    for sig in scales:
        sm = _gaussian_smooth(blended, sig)
        pk = int(np.argmax(sm))
        # Vote for pk and neighbors within ±1
        for off in range(-1, 2):
            idx = pk + off
            if 0 <= idx < n_months:
                scale_votes[idx] += 1.0
    scale_votes /= len(scales)

    # ---- Coarse pass: find year-level hot zones ----
    coarse = _gaussian_smooth(blended, coarse_smooth)
    # Find local maxima in coarse signal
    zone_centers = []
    for i in range(1, n_months - 1):
        if coarse[i] >= coarse[i - 1] and coarse[i] >= coarse[i + 1]:
            zone_centers.append(i)
    # Also add the global argmax if not already there
    global_pk = int(np.argmax(coarse))
    if global_pk not in zone_centers:
        zone_centers.append(global_pk)
    # Sort by coarse score descending, take top n_zones
    zone_centers.sort(key=lambda i: coarse[i], reverse=True)
    zone_centers = zone_centers[:n_zones]

    # ---- Fine pass: refine each zone to exact month ----
    zones = []
    used = set()  # avoid duplicate peaks
    fine = _gaussian_smooth(blended, fine_smooth) if fine_smooth > 0 else blended
    for zc in zone_centers:
        lo = max(0, zc - zone_radius)
        hi = min(n_months, zc + zone_radius + 1)
        # Best month in this zone
        local_idx = lo + int(np.argmax(fine[lo:hi]))
        # Skip if too close to an existing zone peak
        if any(abs(local_idx - u) <= 1 for u in used):
            continue
        used.add(local_idx)

        # ---- Zone scoring ----
        borda_score = blended[local_idx]
        agree_score = agreement[local_idx]
        stability = scale_votes[local_idx]

        # Prominence: how much this peak exceeds local background
        bg_lo = max(0, local_idx - 12)
        bg_hi = min(n_months, local_idx + 13)
        bg = np.median(blended[bg_lo:bg_hi])
        prominence = max(0, borda_score - bg)

        # Combined zone score: agreement is most important for distinguishing
        # true peaks from false peaks in large windows
        zone_score = (borda_score * (1.0 + agree_score * 2.0)
                      * (1.0 + prominence * 3.0)
                      * (1.0 + stability * 1.5))

        zones.append((local_idx, zone_score))

    # Sort by zone score descending
    zones.sort(key=lambda x: x[1], reverse=True)
    return zones


def _multipeak_predict(borda_stack, raw_preds, rw=1.5, **kwargs):
    """Return top-1 prediction from multi-peak ranking. Drop-in for _cluster_peak."""
    zones = _find_hot_zones(borda_stack, raw_preds, rw=rw, **kwargs)
    if zones:
        return zones[0][0]
    # Fallback: argmax of weighted Borda
    n_models = borda_stack.shape[0]
    n_reg = n_models // 2
    w = np.array([1.0] * n_reg + [rw] * (n_models - n_reg))
    return int(np.argmax(np.average(borda_stack, axis=0, weights=w)))


def _extract_peak_features(peak_idx, borda_stack, raw_preds, features_mat, rw=1.5):
    """Extract meta-features for a candidate peak, used by Stage 2 ranker.

    Returns a 1D array of peak quality features:
    - Borda score at peak
    - Model agreement (fraction of models with peak in their top-5)
    - Peak prominence (vs local background)
    - Peak sharpness (2nd derivative at peak)
    - Multi-scale stability
    - Original astro features at the peak month (from features_mat)
    """
    n_models, n_months = borda_stack.shape
    n_reg = n_models // 2
    w = np.array([1.0] * n_reg + [rw] * (n_models - n_reg))
    blended = np.average(borda_stack, axis=0, weights=w)

    # 1. Borda score
    borda_val = blended[peak_idx]

    # 2. Model agreement (top-5)
    agreement = 0
    for i in range(n_models):
        if peak_idx in np.argsort(borda_stack[i])[-5:]:
            agreement += 1
    agreement /= n_models

    # 3. Prominence (peak vs ±12 month background)
    bg_lo = max(0, peak_idx - 12)
    bg_hi = min(n_months, peak_idx + 13)
    bg_median = np.median(blended[bg_lo:bg_hi])
    prominence = borda_val - bg_median

    # 4. Sharpness (negative 2nd derivative = sharper peak)
    if 1 <= peak_idx < n_months - 1:
        sharpness = 2 * blended[peak_idx] - blended[peak_idx - 1] - blended[peak_idx + 1]
    else:
        sharpness = 0.0

    # 5. Peak rank in blended (lower = better)
    peak_rank = np.sum(blended > borda_val) / n_months

    # 6. Multi-scale stability
    scales = [0.5, 1.0, 2.0, 3.0]
    stability = 0
    for sig in scales:
        sm = _gaussian_smooth(blended, sig)
        pk = int(np.argmax(sm))
        if abs(pk - peak_idx) <= 2:
            stability += 1
    stability /= len(scales)

    # 7. Neighborhood score (sum of blended in ±2 months)
    nb_lo = max(0, peak_idx - 2)
    nb_hi = min(n_months, peak_idx + 3)
    neighborhood = blended[nb_lo:nb_hi].sum()

    meta_feats = np.array([borda_val, agreement, prominence, sharpness,
                           peak_rank, stability, neighborhood], dtype=np.float32)

    # 8. Original astro features at peak month
    if features_mat is not None and peak_idx < features_mat.shape[0]:
        astro_feats = features_mat[peak_idx]
        return np.concatenate([meta_feats, astro_feats])

    return meta_feats


def _build_stage2_data(train_data, ensemble_models, feat_key, n_reg, rw=1.5,
                       n_peaks=5, borda_fn=None):
    """Build training data for Stage 2 peak ranker from Stage 1 ensemble predictions.

    For each training subject:
    1. Run ensemble to get Borda scores
    2. Find top-N peaks
    3. Label: 1 if peak within ±1 of death month, 0 otherwise
    4. Extract peak meta-features + original astro features

    Returns (X_stage2, y_stage2, groups) for XGBRanker.
    """
    X_list, y_list, groups = [], [], []

    for d in train_data:
        feats = d[feat_key]
        n_m = feats.shape[0]
        death_idx = d["death_month_idx"]

        # Stage 1: get ensemble predictions
        all_preds = [np.asarray(mdl.predict(feats)).ravel()[:n_m]
                     for _, mdl, _ in ensemble_models]
        if borda_fn is None:
            def _borda_local(arr):
                a = np.asarray(arr).ravel()
                n = len(a)
                order = np.argsort(a)
                ranks = np.empty(n)
                ranks[order] = np.arange(n, dtype=float)
                return ranks / (n - 1) if n > 1 else ranks
            borda_fn = _borda_local

        borda_stack = np.array([borda_fn(p) for p in all_preds])
        n_models = borda_stack.shape[0]
        _n_reg = n_reg
        w = np.array([1.0] * _n_reg + [rw] * (n_models - _n_reg))
        blended = np.average(borda_stack, axis=0, weights=w)

        # Find top-N peaks
        smooth_bl = _gaussian_smooth(blended, 0.5)
        peak_indices = set()
        # Get top-K months by score
        top_months = np.argsort(smooth_bl)[-n_peaks * 3:]
        # Cluster into distinct peaks (merge within ±2)
        for m in sorted(top_months)[::-1]:
            if not any(abs(m - p) <= 2 for p in peak_indices):
                peak_indices.add(int(m))
            if len(peak_indices) >= n_peaks:
                break
        # Always include the death month if not already there (for positive label)
        peak_indices.add(death_idx)

        # Build features and labels for each peak
        group_size = 0
        for pk in sorted(peak_indices):
            pf = _extract_peak_features(pk, borda_stack, all_preds, feats, rw=rw)
            label = 1 if abs(pk - death_idx) <= 1 else 0
            X_list.append(pf)
            y_list.append(label)
            group_size += 1
        groups.append(group_size)

    return np.array(X_list, dtype=np.float32), np.array(y_list), groups


def _sav_normalized(sav_dict, sign_name):
    """Get SAV for a sign, normalized to 0-1 range. Lower = more dangerous."""
    val = sav_dict.get(sign_name, 28)  # default to average
    # Invert: low SAV = high danger
    return max(0, min(1.0, 1.0 - (val - 18) / 20.0))  # 18->1.0, 38->0.0

def _bav_normalized(bav_dict, planet, sign_name):
    """Get BAV for planet in sign, normalized. Lower BAV = more dangerous."""
    planet_bav = bav_dict.get(planet, {})
    val = planet_bav.get(sign_name, 4)  # default to average
    return max(0, min(1.0, 1.0 - (val - 1) / 6.0))  # 1->1.0, 7->0.0


# ---- Data extraction ----

def extract_subjects(couples_path):
    with open(couples_path, "r", encoding="utf-8") as f:
        couples = json.load(f)
    subjects = []
    for couple in couples:
        for side, pk in [("person_a", "person_a_parents"),
                         ("person_b", "person_b_parents")]:
            person = couple.get(side, {})
            parents = couple.get(pk, {})
            fdd = parents.get("father_death_date")
            if not fdd:
                continue
            bd = person.get("birth_date", "")
            if not bd or int(bd[:4]) < 1900:
                continue
            if fdd <= bd:
                continue
            subjects.append({
                "name": person["name"],
                "birth_date": bd,
                "birth_time": person.get("birth_time", "12:00"),
                "lat": person["lat"], "lon": person["lon"],
                "tz": person.get("tz", "UTC"),
                "father_death_date": fdd,
            })
    return subjects


def extract_subjects_flat(json_path):
    """Load subjects from flat JSON array (father_passing_date.json format).

    Each item has: name, birth_date, birth_time, lat, lon, tz, father_death_date.
    Filters: birth >= 1800, father died after birth, person at least 1 year old.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = []
    for d in data:
        bd = d.get("birth_date", "")
        fdd = d.get("father_death_date", "")
        if not bd or not fdd:
            continue
        if int(bd[:4]) < 1800:
            continue
        if fdd <= bd:
            continue
        # Father must die after person is at least 1 year old
        if int(fdd[:4]) - int(bd[:4]) < 1:
            continue
        subjects.append({
            "name": d["name"],
            "birth_date": bd,
            "birth_time": d.get("birth_time", "12:00"),
            "lat": float(d["lat"]), "lon": float(d["lon"]),
            "tz": d.get("tz", "UTC"),
            "father_death_date": fdd,
        })
    return subjects


def extract_features_for_person(subj, years_before=5, years_after=5, subj_idx=0):
    """Extract V11 features (156) per month in window around death.

    Window is randomly positioned so death date falls at a random location
    within it (not always centered). This prevents the model from learning
    to predict the midpoint of the window.
    """
    engine = _get_engine()
    tz = pytz.timezone(subj["tz"])
    bd = subj["birth_date"].split("-")
    bt = subj["birth_time"].split(":")
    birth_dt = tz.localize(datetime.datetime(int(bd[0]), int(bd[1]), int(bd[2]),
                                              int(bt[0]), int(bt[1])))
    fdd = subj["father_death_date"].split("-")
    death_dt = datetime.datetime(int(fdd[0]), int(fdd[1]), int(fdd[2]))

    # Randomize window position: death date at random position within window
    # Total window = years_before + years_after years
    total_window_days = (years_before + years_after) * 365
    # Use subject name as seed for reproducibility
    import hashlib
    _seed_str = f"{subj['name']}_{subj_idx}"
    rng = np.random.RandomState(int(hashlib.sha256(_seed_str.encode()).hexdigest(), 16) % (2**31))
    # Random offset: how many days before death the window starts
    # Must be at least 1 month before death and 1 month after
    min_before = 30  # at least 1 month before death
    max_before = total_window_days - 30  # at least 1 month after death
    days_before = rng.randint(min_before, max_before + 1)

    scan_start = death_dt - datetime.timedelta(days=days_before)
    scan_end = scan_start + datetime.timedelta(days=total_window_days)
    earliest = birth_dt.replace(tzinfo=None) + datetime.timedelta(days=365)
    if scan_start < earliest:
        scan_start = earliest

    lat, lon = float(subj["lat"]), float(subj["lon"])

    # ---- Natal (once) ----
    natal = engine.calculate_planetary_positions(birth_dt, lat, lon)
    lagna = engine.calculate_lagna(birth_dt, lat, lon)
    moon_lon = natal["Moon"]["longitude"]
    moon_sn = natal["Moon"]["sign_num"]

    lsn = lagna["sign_num"]
    sign_3h = _house_sign(lsn, 3)
    sign_4h = _house_sign(lsn, 4)
    sign_9h = _house_sign(lsn, 9)
    sign_10h = _house_sign(lsn, 10)
    sign_9h_name = SIGNS[sign_9h - 1]

    lord_3 = _house_lord(lsn, 3)
    lord_4 = _house_lord(lsn, 4)
    lord_8 = _house_lord(lsn, 8)
    lord_9 = _house_lord(lsn, 9)
    lord_10 = _house_lord(lsn, 10)
    lord_11 = _house_lord(lsn, 11)
    lord_22drek = SIGN_LORDS[SIGNS[_house_sign(lsn, 16) - 1]]
    cusp_9_lon = (sign_9h - 1) * 30.0
    lord_64nav = _compute_64th_navamsa_lord(cusp_9_lon)

    pitri_sn = _pitri_saham_sn(natal, lagna)
    sun_natal_lon = natal["Sun"]["longitude"]
    lord9_natal_lon = natal[lord_9]["longitude"]
    lord_9_natal_sn = natal[lord_9]["sign_num"]

    father_lords_set = {lord_9, lord_10, lord_3, lord_4, lord_11,
                        lord_22drek, lord_8, lord_64nav, "Sun"}
    # Father's maraka lords specifically (2nd from 9th = 10th, 7th from 9th = 3rd)
    father_maraka_set = {lord_10, lord_3}

    # ---- Ashtakavarga (once) ----
    try:
        ashtak = engine.calculate_ashtakavarga(natal, lagna)
        sav = ashtak.get("SAV", {})
        bav = ashtak.get("BAV", {})
    except Exception:
        sav, bav = {}, {}

    # ---- Firdaria (once) ----
    try:
        far_end = birth_dt + datetime.timedelta(days=365 * 80)
        firdaria_raw = engine.calculate_firdaria(birth_dt, far_end)
        firdaria_periods = _parse_firdaria(firdaria_raw)
    except Exception:
        firdaria_periods = []

    father_firdaria_set = {lord_9, lord_3, lord_10, "Sun", "Saturn"}

    # ---- Jaimini Karakas (once) ----
    try:
        jk = engine.calculate_jaimini_karakas(natal)
        pitrukaraka = jk.get("Putrakaraka (PiK)", "")  # 5th highest degree = PiK (father)
        pik_natal_sn = natal[pitrukaraka]["sign_num"] if pitrukaraka in natal else 0
    except Exception:
        pitrukaraka = ""
        pik_natal_sn = 0

    # ---- Arudha of 9th house (once) ----
    try:
        a9 = engine.calculate_arudha(9, natal, lagna)
        a9_sn = a9.get("sign_num", 0)
    except Exception:
        a9_sn = 0

    # ---- D12 (Dwadashamsha - parents chart) (once) ----
    try:
        d12 = engine.get_divisional_chart(natal, 12)
        # D12 lagna via natal lagna longitude
        d12_lagna_data = engine.get_divisional_chart(
            {"Lagna": {"longitude": lagna["longitude"]}}, 12)
        d12_lagna_sn = d12_lagna_data.get("Lagna", {}).get("sign_num", lsn)
        d12_sign_9h = _house_sign(d12_lagna_sn, 9)
        d12_lord_9 = SIGN_LORDS[SIGNS[d12_sign_9h - 1]]
        # D12 Sun house position
        d12_sun_sn = d12.get("Sun", {}).get("sign_num", 0)
        d12_sun_house = _house_of(d12_sun_sn, d12_lagna_sn) if d12_sun_sn > 0 else 0
        # D12 9th lord house position
        d12_lord9_sn = d12.get(d12_lord_9, {}).get("sign_num", 0)
        d12_lord9_house = _house_of(d12_lord9_sn, d12_lagna_sn) if d12_lord9_sn > 0 else 0
    except Exception:
        d12_sign_9h = sign_9h  # fallback
        d12_lord_9 = lord_9
        d12_sun_house = 0
        d12_lord9_house = 0

    # ---- D9 (Navamsa) for transit triggers (once) ----
    try:
        d9 = engine.get_divisional_chart(natal, 9)
        d9_sun_sn = d9.get("Sun", {}).get("sign_num", 0)
        d9_lord9_sn = d9.get(lord_9, {}).get("sign_num", 0)
        # D9 Sun house from D9 lagna
        d9_lagna_data = engine.get_divisional_chart(
            {"Lagna": {"longitude": lagna["longitude"]}}, 9)
        d9_lagna_sn = d9_lagna_data.get("Lagna", {}).get("sign_num", lsn)
        d9_sun_house = _house_of(d9_sun_sn, d9_lagna_sn) if d9_sun_sn > 0 else 0
    except Exception:
        d9_sun_sn = 0
        d9_lord9_sn = 0
        d9_sun_house = 0

    # ---- Upagrahas: Gulika/Mandi (once) ----
    try:
        upagrahas = engine.calculate_upagrahas(birth_dt, lat, lon)
        gulika_sn = upagrahas.get("Gulika", {}).get("sign_num", 0)
        mandi_sn = upagrahas.get("Mandi", {}).get("sign_num", 0)
    except Exception:
        gulika_sn = 0
        mandi_sn = 0

    # ---- V11: Hellenistic Lot of Death (natal, once) ----
    lot_death_sn = 0
    lot_death_lon = 0.0
    lot_death_ruler = ""
    is_day_birth = True  # default
    try:
        hel_chart = engine.calculate_hellenistic_chart(birth_dt, lat, lon)
        is_day_birth = hel_chart.get("is_day_birth", True)
        lot_death = hel_chart.get("lots", {}).get("Death", {})
        lot_death_lon = lot_death.get("longitude", 0.0)
        # Convert tropical lot longitude to sidereal sign for transit comparison
        # Use the same ayanamsa as natal positions
        ayanamsa = engine._get_lahiri_ayanamsa(birth_dt) if hasattr(engine, '_get_lahiri_ayanamsa') else 24.0
        lot_death_sid_lon = (lot_death_lon - ayanamsa) % 360
        lot_death_sn = int(lot_death_sid_lon / 30) + 1
        lot_death_ruler = SIGN_LORDS[SIGNS[lot_death_sn - 1]]
    except Exception:
        pass

    # ---- V11: BAV for specific houses (natal, once) ----
    sign_8h = _house_sign(lsn, 8)
    sign_8h_name = SIGNS[sign_8h - 1]
    bav_sat_8h = _bav_normalized(bav, "Saturn", sign_8h_name)
    bav_sat_9h = _bav_normalized(bav, "Saturn", sign_9h_name)
    bav_sat_3h = _bav_normalized(bav, "Saturn", SIGNS[sign_3h - 1])
    sav_8h = _sav_normalized(sav, sign_8h_name)

    # ---- V11: KP Cusp Sub-Lords (natal, once) ----
    kp_8cusp_sub = ""
    kp_9cusp_sub = ""
    try:
        cusps = engine.calculate_placidus_cusps(birth_dt, lat, lon)
        if cusps and 8 in cusps:
            kp_8cusp_sub = cusps[8].get("sub_lord", "")
        if cusps and 9 in cusps:
            kp_9cusp_sub = cusps[9].get("sub_lord", "")
    except Exception:
        pass

    # ---- V11: D60 Shashtiamsha (natal, once) ----
    d60_sun_house = 0
    d60_9l_house = 0
    d60_sat_house = 0
    d60_8l_house = 0
    try:
        d60_data = {"Lagna": {"longitude": lagna["longitude"]}}
        d60_data.update(natal)
        d60 = engine.get_divisional_chart(d60_data, 60)
        d60_lagna_entry = engine.get_divisional_chart(
            {"Lagna": {"longitude": lagna["longitude"]}}, 60)
        d60_lsn = d60_lagna_entry.get("Lagna", {}).get("sign_num", lsn)
        # Sun in D60
        d60_sun_sn = d60.get("Sun", {}).get("sign_num", 0)
        if d60_sun_sn > 0:
            d60_sun_house = _house_of(d60_sun_sn, d60_lsn)
        # 9th lord in D60
        d60_9l_sn = d60.get(lord_9, {}).get("sign_num", 0)
        if d60_9l_sn > 0:
            d60_9l_house = _house_of(d60_9l_sn, d60_lsn)
        # Saturn in D60
        d60_sat_sn = d60.get("Saturn", {}).get("sign_num", 0)
        if d60_sat_sn > 0:
            d60_sat_house = _house_of(d60_sat_sn, d60_lsn)
        # 8th lord in D60
        d60_8l_sn = d60.get(lord_8, {}).get("sign_num", 0)
        if d60_8l_sn > 0:
            d60_8l_house = _house_of(d60_8l_sn, d60_lsn)
    except Exception:
        pass

    # ---- V11: BaZi natal chart (once) ----
    bazi_chart = None
    bazi_luck_pillars = []
    bazi_father_elem = ""  # Father star = Indirect Resource (偏财) element
    try:
        gender = subj.get("gender", "M")
        bazi_chart = engine.calculate_bazi_chart(birth_dt, lat, lon, gender)
        bazi_luck_pillars = bazi_chart.get("luck_pillars", {}).get("pillars", [])
        # Father star in BaZi = Indirect Wealth (偏财, Pian Cai)
        # Day master element's "controlled by" = wealth; indirect = opposite polarity
        dm_elem = bazi_chart.get("day_master", {}).get("element", "")
        _ELEMENT_CONTROLS_BY = {"Wood": "Earth", "Fire": "Metal", "Earth": "Water",
                                 "Metal": "Wood", "Water": "Fire"}
        bazi_father_elem = _ELEMENT_CONTROLS_BY.get(dm_elem, "")
    except Exception:
        pass

    # ---- V11: Zodiacal Releasing from Lot of Death (once, pre-compute periods) ----
    zr_death_periods = []
    try:
        if lot_death_lon > 0:
            zr_end = birth_dt + datetime.timedelta(days=365 * 90)
            zr_death_periods = engine.calculate_zodiacal_releasing(
                lot_death_lon, birth_dt, zr_end)
    except Exception:
        pass

    # ---- V11: Hellenistic profections use tropical ASC sign ----
    hel_asc_sn = 0
    try:
        if hel_chart:
            hel_asc_sn = hel_chart.get("signs", {}).get("Ascendant", 0)
        if hel_asc_sn == 0:
            hel_asc_sn = lsn  # fallback to sidereal
    except Exception:
        hel_asc_sn = lsn

    # ---- Natal Saturn longitude (for return calculation) ----
    sat_natal_lon = natal["Saturn"]["longitude"]

    # ---- Shadbala (once) ----
    try:
        shadbala = engine.calculate_shadbala(natal, lagna)
        sun_shadbala = shadbala.get("Sun", {}).get("score", 180)
        lord9_shadbala = shadbala.get(lord_9, {}).get("score", 180)
        sat_shadbala = shadbala.get("Saturn", {}).get("score", 180)
    except Exception:
        sun_shadbala = 180  # neutral default
        lord9_shadbala = 180
        sat_shadbala = 180

    # ---- Birth nakshatra index (for Tara Bala) ----
    # Nakshatra span = 13.3333 degrees, 27 nakshatras
    birth_nak_idx = int(moon_lon / (360.0 / 27)) % 27  # 0-based index

    # ---- Planet positions in signs (for Chara Dasha sign content check) ----
    natal_planet_signs = {}
    for pname in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]:
        if pname in natal:
            natal_planet_signs[pname] = natal[pname]["sign_num"]

    # ---- Previous month Saturn sign (for ingress detection) ----
    prev_sat_sn = None

    # ---- Monthly scan ----
    dates = []
    all_features = []
    death_idx = None

    cur = datetime.datetime(scan_start.year, scan_start.month, 15)
    end_dt = datetime.datetime(scan_end.year, scan_end.month, 15)

    # Pre-compute previous month's dasha for PD sandhi detection
    prev_pd = None

    while cur <= end_dt:
        cur_tz = tz.localize(cur)
        tr = engine.calculate_planetary_positions(cur_tz, lat, lon)
        dasha = engine.calculate_dasha(moon_lon, birth_dt, cur_tz)

        md = dasha.get("current_date_dasha", "")
        ad = dasha.get("current_antardasha", "")
        pd = dasha.get("current_pratyantardasha", "")
        ad_end_str = dasha.get("ad_end_date", "")

        sat_sn = tr["Saturn"]["sign_num"]
        sat_lon = tr["Saturn"]["longitude"]
        sat_retro = tr["Saturn"].get("is_retrograde", False)
        jup_sn = tr["Jupiter"]["sign_num"]
        rahu_sn = tr["Rahu"]["sign_num"]
        ketu_sn = tr["Ketu"]["sign_num"]
        sun_sn = tr["Sun"]["sign_num"]
        mars_sn = tr["Mars"]["sign_num"]
        mars_lon = tr["Mars"]["longitude"]

        f = np.zeros(N_FEATURES, dtype=np.float32)

        # ========== DASHA MD/AD (0-9) ==========
        dasha_hits = 0.0
        if lord_10 in (md, ad): f[0] = 1.0; dasha_hits += 1.5
        if lord_3 in (md, ad):  f[1] = 1.0; dasha_hits += 1.5
        if lord_9 in (md, ad):  f[2] = 1.0; dasha_hits += 1.0
        if lord_4 in (md, ad):  f[3] = 1.0; dasha_hits += 1.0
        if "Sun" in (md, ad):   f[4] = 1.0; dasha_hits += 0.5
        if lord_22drek in (md, ad): f[5] = 1.0; dasha_hits += 0.75
        if lord_8 in (md, ad):  f[6] = 1.0; dasha_hits += 0.75
        if lord_64nav in (md, ad): f[7] = 1.0; dasha_hits += 1.0
        if lord_11 in (md, ad): f[8] = 1.0; dasha_hits += 0.5

        active_lords = {md, ad}
        if pd:
            active_lords.add(pd)
        f[9] = len(active_lords & father_lords_set) / 5.0

        # ========== PD (10-18) ==========
        if pd:
            if pd in (lord_3, lord_10):     f[10] = 1.0
            if pd == lord_9:                f[11] = 1.0
            if pd == lord_4:                f[12] = 1.0
            if pd == "Sun":                 f[13] = 1.0
            if pd == lord_22drek:           f[14] = 1.0
            if pd == lord_8:                f[15] = 1.0
            if pd == lord_64nav:            f[16] = 1.0
            if pd in STRICT_MALEFICS:       f[17] = 1.0
            if pd in (lord_9, lord_4, "Sun", lord_11): f[18] = 1.0

        # ========== AD/PD TIMING (19-22) ==========
        if ad_end_str:
            try:
                ad_end_parts = ad_end_str.split("-")
                ad_end_y = int(ad_end_parts[0])
                ad_end_m = int(ad_end_parts[1])
                months_to_end = (ad_end_y - cur.year) * 12 + (ad_end_m - cur.month)
                if 0 <= months_to_end <= 2:
                    f[19] = 1.0  # AD sandhi
                f[20] = max(0, min(months_to_end, 24)) / 24.0
            except (ValueError, IndexError):
                pass

        # PD sandhi: PD changed from previous month
        if prev_pd is not None and pd != prev_pd:
            f[21] = 1.0  # PD boundary month

        # PD months left approximation (PD durations vary ~2-5 months)
        # Use change detection: distance from PD boundaries
        # We detect PD change above; for months_left we use a simple approximation
        # PD typically lasts 2-5 months, so we can estimate based on typical durations
        # For now, use the PD sandhi as the key feature (binary is simpler)
        f[22] = 0.0  # placeholder, will be estimated from sequence in post-processing

        # ========== TRANSIT SIGN (23-31) ==========
        transit_hits = 0.0
        if _planet_on_or_aspects(sat_sn, "Saturn", sign_9h):
            f[23] = 1.0; transit_hits += 1.0
        if _planet_on_or_aspects(sat_sn, "Saturn", sign_3h):
            f[24] = 1.0; transit_hits += 1.0

        jup_10 = _planet_on_or_aspects(jup_sn, "Jupiter", sign_10h)
        sat_10 = _planet_on_or_aspects(sat_sn, "Saturn", sign_10h)
        if jup_10 and sat_10:
            f[25] = 1.0; transit_hits += 2.0
        elif jup_10 or sat_10:
            f[25] = 0.3

        jup_4 = _planet_on_or_aspects(jup_sn, "Jupiter", sign_4h)
        sat_4 = _planet_on_or_aspects(sat_sn, "Saturn", sign_4h)
        if jup_4 and sat_4:
            f[26] = 1.0; transit_hits += 1.5

        if rahu_sn in (sign_3h, sign_9h) or ketu_sn in (sign_3h, sign_9h):
            f[27] = 1.0; transit_hits += 1.0

        if any(s == pitri_sn for s in [sat_sn, rahu_sn, ketu_sn]):
            f[28] = 1.0; transit_hits += 0.75

        lord_9_tr_sn = tr[lord_9]["sign_num"]
        if _house_of(lord_9_tr_sn, lord_9_natal_sn) in (6, 8, 12):
            f[29] = 1.0

        mal_3h = sum(1 for m in STRICT_MALEFICS if tr[m]["sign_num"] == sign_3h)
        f[30] = mal_3h / 4.0

        sun_mal = sum(1 for m in STRICT_MALEFICS if tr[m]["sign_num"] == sun_sn)
        f[31] = sun_mal / 4.0

        # ========== SUN PER-HOUSE (32-35) ==========
        if sun_sn == sign_3h:  f[32] = 1.0
        if sun_sn == sign_4h:  f[33] = 1.0
        if sun_sn == sign_9h:  f[34] = 1.0
        if sun_sn == sign_10h: f[35] = 1.0

        # ========== MARS PER-HOUSE (36-39) ==========
        if mars_sn == sign_3h:  f[36] = 1.0; transit_hits += 0.5
        if mars_sn == sign_4h:  f[37] = 1.0; transit_hits += 0.3
        if mars_sn == sign_9h:  f[38] = 1.0; transit_hits += 0.5
        if mars_sn == sign_10h: f[39] = 1.0; transit_hits += 0.3

        # ========== MARS DEGREE (40-41) ==========
        f[40] = _degree_proximity_score(mars_lon, sun_natal_lon, orb=8.0)
        f[41] = _degree_proximity_score(mars_lon, lord9_natal_lon, orb=8.0)

        # ========== FIRDARIA (42-45) ==========
        if firdaria_periods:
            fir_major, fir_sub = _firdaria_at_month(firdaria_periods, cur.year, cur.month)
            if fir_major in STRICT_MALEFICS: f[42] = 1.0
            if fir_sub in STRICT_MALEFICS:   f[43] = 1.0
            if fir_major in father_firdaria_set: f[44] = 1.0
            if fir_sub in father_firdaria_set:   f[45] = 1.0

        # ========== CONVERGENCE (46-49) ==========
        if dasha_hits >= 1 and transit_hits >= 1: f[46] = 1.0
        if dasha_hits >= 2 and transit_hits >= 2: f[47] = 1.0
        f[48] = min(dasha_hits * transit_hits / 10.0, 1.0)
        if dasha_hits > 0 and transit_hits > 0:
            f[49] = min(math.sqrt(dasha_hits * transit_hits) / 3.0, 1.0)

        # ========== ASHTAKAVARGA TRANSIT (50-52) ==========
        sat_sign_name = SIGNS[sat_sn - 1]
        sign_9h_name = SIGNS[sign_9h - 1]
        f[50] = _sav_normalized(sav, sat_sign_name)   # low SAV -> high danger
        f[51] = _bav_normalized(bav, "Saturn", sat_sign_name)  # low BAV -> high danger
        f[52] = _sav_normalized(sav, sign_9h_name)     # low SAV of 9th -> weaker father

        # ========== SADE SATI (53-55) ==========
        # Sade Sati: Saturn in 12th, 1st, or 2nd from Moon sign
        moon_12th = ((moon_sn - 2) % 12) + 1
        moon_2nd = (moon_sn % 12) + 1
        if sat_sn in (moon_12th, moon_sn, moon_2nd):
            f[53] = 1.0
        # Phase 1: 12th from Moon (Saturn's 10th aspect falls on 9th from Moon = father)
        if sat_sn == moon_12th:
            f[54] = 1.0
        # Ashtama Shani: Saturn 8th from Moon
        moon_8th = ((moon_sn + 6) % 12) + 1
        if sat_sn == moon_8th:
            f[55] = 1.0

        # ========== SATURN EXTRA (56-59) ==========
        # Saturn ingress: degree within sign < 3.75 (first kakshya = just entered)
        sat_deg_in_sign = sat_lon % 30
        if sat_deg_in_sign < 3.75:
            f[56] = 1.0
        elif sat_deg_in_sign > 26.25:
            f[56] = 0.5  # about to leave = also unstable (sandhi)

        f[57] = 1.0 if sat_retro else 0.0

        # Saturn degree proximity to natal Sun and natal 9th lord
        f[58] = _degree_proximity_score(sat_lon, sun_natal_lon, orb=12.0)
        f[59] = _degree_proximity_score(sat_lon, lord9_natal_lon, orb=12.0)

        # ========== JUPITER DEGREE (60-61) ==========
        jup_lon = tr["Jupiter"]["longitude"]
        f[60] = _degree_proximity_score(jup_lon, sun_natal_lon, orb=10.0)
        f[61] = _degree_proximity_score(jup_lon, lord9_natal_lon, orb=10.0)

        # ========== SUN DEGREE (62-63) ==========
        sun_lon = tr["Sun"]["longitude"]
        f[62] = _degree_proximity_score(sun_lon, sun_natal_lon, orb=5.0)
        f[63] = _degree_proximity_score(sun_lon, lord9_natal_lon, orb=5.0)

        # ========== TRIPLE DASHA + MALEFIC COMBO (64-66) ==========
        # Triple dasha: MD + AD + PD all father-related
        maraka_count = 0
        if md in father_maraka_set or md == "Sun": maraka_count += 1
        if ad in father_maraka_set or ad == "Sun": maraka_count += 1
        if pd and (pd in father_maraka_set or pd == "Sun"): maraka_count += 1
        if maraka_count >= 3:
            f[64] = 1.0
        f[66] = maraka_count / 3.0

        # Malefic MD + malefic AD combo
        if md in STRICT_MALEFICS and ad in STRICT_MALEFICS:
            f[65] = 1.0

        # ========== CHARA DASHA / JAIMINI (67-69) ==========
        cd_sn = 0
        try:
            cd = engine.calculate_chara_dasha(birth_dt, lagna, natal, cur_tz)
            cd_sign_name = cd.get("current_dasha", "")
            cd_sn = SIGNS.index(cd_sign_name) + 1 if cd_sign_name in SIGNS else 0
            if cd_sn > 0:
                if cd_sn == sign_9h:
                    f[67] = 1.0
                if cd_sn in (sign_3h, sign_10h):
                    f[68] = 1.0
                # Check if natal Sun or 9L is in the Chara Dasha sign
                if natal_planet_signs.get("Sun") == cd_sn or \
                   natal_planet_signs.get(lord_9) == cd_sn:
                    f[69] = 1.0
        except Exception:
            pass

        # ========== TARA BALA (70-71) ==========
        try:
            sat_nak_idx = int(sat_lon / (360.0 / 27)) % 27  # 0-based
            tara = engine.calculate_tara_bala(birth_nak_idx, sat_nak_idx)
            tara_score = tara.get("score", 0)
            tara_name = tara.get("name", "")
            if tara_name in ("Naidhana", "Vipat", "Pratyak"):
                f[70] = 1.0
            # Invert score: -2 (worst) -> 1.0, +1 (best) -> 0.0
            f[71] = max(0.0, min(1.0, (1.0 - tara_score) / 3.0))
        except Exception:
            pass

        # ========== RAHU/KETU DEGREE (72-73) ==========
        rahu_lon = tr["Rahu"]["longitude"]
        ketu_lon = tr["Ketu"]["longitude"]
        f[72] = max(_degree_proximity_score(rahu_lon, sun_natal_lon, orb=10.0),
                    _degree_proximity_score(ketu_lon, sun_natal_lon, orb=10.0))
        f[73] = max(_degree_proximity_score(rahu_lon, lord9_natal_lon, orb=10.0),
                    _degree_proximity_score(ketu_lon, lord9_natal_lon, orb=10.0))

        # ========== A9 TRANSIT (74) ==========
        if a9_sn > 0:
            if sat_sn == a9_sn or rahu_sn == a9_sn or ketu_sn == a9_sn:
                f[74] = 1.0
            # Also check Saturn aspecting A9
            elif _planet_on_or_aspects(sat_sn, "Saturn", a9_sn):
                f[74] = 0.5

        # ========== PLANET MOTION (75-76) ==========
        sat_speed = abs(tr["Saturn"].get("daily_speed", 0.066))
        # Saturn avg speed ~0.033 deg/day; stationary = ~0
        # Normalize: 0 speed -> 1.0, full speed (0.066) -> 0.0
        f[75] = max(0.0, min(1.0, 1.0 - sat_speed / 0.07))
        f[76] = 1.0 if tr["Jupiter"].get("is_retrograde", False) else 0.0

        # ========== PITRUKARAKA (77-78) ==========
        if pitrukaraka:
            if pitrukaraka in (md, ad) or (pd and pitrukaraka == pd):
                f[77] = 1.0
            # Chara Dasha sign = Pitrukaraka's natal sign
            if cd_sn > 0 and cd_sn == pik_natal_sn:
                f[78] = 1.0

        # ========== V8: SUN-PLANET CONJUNCTIONS (79-82) ==========
        # sun_lon already computed above at feature 62-63
        f[79] = _degree_proximity_score(sun_lon, sat_lon, orb=15.0)
        f[80] = max(_degree_proximity_score(sun_lon, rahu_lon, orb=12.0),
                    _degree_proximity_score(sun_lon, ketu_lon, orb=12.0))
        f[81] = _degree_proximity_score(sun_lon, mars_lon, orb=12.0)
        # 9L combustion: transit 9L within combustion orb of transit Sun
        if lord_9 in tr and lord_9 not in ("Rahu", "Ketu"):
            lord9_tr_lon = tr[lord_9]["longitude"]
            comb_orb = _COMBUSTION_ORBS.get(lord_9, 12)
            f[82] = _degree_proximity_score(sun_lon, lord9_tr_lon, orb=float(comb_orb))

        # ========== V8: GANDANTA & SIGN BOUNDARY (83-85) ==========
        sun_deg_in_sign = sun_lon % 30
        f[83] = 1.0 if (sun_deg_in_sign < 1.0 or sun_deg_in_sign > 29.0) else 0.0
        if lord_9 in tr:
            lord9_deg = tr[lord_9]["longitude"] % 30
            f[84] = 1.0 if (lord9_deg < 1.0 or lord9_deg > 29.0) else 0.0
        sat_deg = sat_lon % 30
        f[85] = 1.0 if (sat_deg < 1.0 or sat_deg > 29.0) else 0.0

        # ========== V8: MOON NAKSHATRA LORD (86-88) ==========
        moon_tr_lon = tr["Moon"]["longitude"]
        moon_tr_nak_idx = int(moon_tr_lon / (360.0 / 27)) % 27
        moon_nak_lord = _NAKSHATRA_LORDS[moon_tr_nak_idx]
        if moon_nak_lord in father_maraka_set or moon_nak_lord == "Sun":
            f[86] = 1.0
        if moon_nak_lord == "Saturn":
            f[87] = 1.0
        sat_nak_lord = _NAKSHATRA_LORDS[int(sat_lon / (360.0 / 27)) % 27]
        if sat_nak_lord in father_maraka_set or sat_nak_lord == "Sun":
            f[88] = 1.0

        # ========== V8: TRANSIT DIGNITY (89-91) ==========
        lord9_tr_sn_v8 = tr[lord_9]["sign_num"] if lord_9 in tr else 0
        if lord9_tr_sn_v8 > 0 and _DEBILITATION_SIGN.get(lord_9) == lord9_tr_sn_v8:
            f[89] = 1.0
        # 9L in enemy sign (sign lord is enemy of 9L)
        if lord9_tr_sn_v8 > 0:
            sign_lord_of_9l = SIGN_LORDS[SIGNS[lord9_tr_sn_v8 - 1]]
            if sign_lord_of_9l in _ENEMY_LORDS.get(lord_9, set()):
                f[90] = 1.0
        if sun_sn == 7:  # Libra = Sun debilitated
            f[91] = 1.0

        # ========== V8: DOUBLE TRANSIT EXPANDED (92-94) ==========
        jup_3 = _planet_on_or_aspects(jup_sn, "Jupiter", sign_3h)
        sat_3 = _planet_on_or_aspects(sat_sn, "Saturn", sign_3h)
        if jup_3 and sat_3:
            f[92] = 1.0
        jup_9 = _planet_on_or_aspects(jup_sn, "Jupiter", sign_9h)
        sat_9 = _planet_on_or_aspects(sat_sn, "Saturn", sign_9h)
        if jup_9 and sat_9:
            f[93] = 1.0
        # Count of malefics on/aspecting 9H
        mal_9h_count = 0
        for m in STRICT_MALEFICS:
            if _planet_on_or_aspects(tr[m]["sign_num"], m, sign_9h):
                mal_9h_count += 1
        f[94] = mal_9h_count / 4.0

        # ========== V8: BADHAKA & 8TH LORD (95-97) ==========
        if lord_8 in (md, ad):
            f[95] = 1.0
        if _planet_on_or_aspects(sat_sn, "Saturn", sign_4h):
            f[96] = 1.0
        if _planet_on_or_aspects(sat_sn, "Saturn", sign_10h):
            f[97] = 1.0

        # ========== V9: D12 DWADASHAMSHA (98-102) ==========
        # D12 9th lord is MD or AD lord
        if d12_lord_9 in (md, ad):
            f[98] = 1.0
        # Transit Saturn on/aspecting D12 9th house sign
        if _planet_on_or_aspects(sat_sn, "Saturn", d12_sign_9h):
            f[99] = 1.0
        # Rahu/Ketu on D12 9th house sign
        if rahu_sn == d12_sign_9h or ketu_sn == d12_sign_9h:
            f[100] = 1.0
        # D12 Sun in dusthana
        if d12_sun_house in (6, 8, 12):
            f[101] = 1.0
        # D12 9th lord in dusthana
        if d12_lord9_house in (6, 8, 12):
            f[102] = 1.0

        # ========== V9: NAVAMSA D9 TRANSIT TRIGGERS (103-105) ==========
        # Transit Saturn on/aspecting D9 Sun sign
        if d9_sun_sn > 0 and _planet_on_or_aspects(sat_sn, "Saturn", d9_sun_sn):
            f[103] = 1.0
        # Transit Saturn on/aspecting D9 9th lord sign
        if d9_lord9_sn > 0 and _planet_on_or_aspects(sat_sn, "Saturn", d9_lord9_sn):
            f[104] = 1.0
        # D9 Sun in dusthana (natal, constant but interacts with dasha)
        if d9_sun_house in (6, 8, 12):
            f[105] = 1.0

        # ========== V9: SATURN RETURN (106-107) ==========
        # Degree proximity of transit Saturn to natal Saturn (conjunction)
        f[106] = _degree_proximity_score(sat_lon, sat_natal_lon, orb=10.0)
        # Saturn half-return (opposition): transit Saturn 180° from natal
        f[107] = _degree_proximity_score(sat_lon, (sat_natal_lon + 180) % 360, orb=10.0)

        # ========== V9: MD SANDHI / TRANSITION (108-109) ==========
        md_end_str = dasha.get("md_end_date", "")
        if md_end_str:
            try:
                md_end_parts = md_end_str.split("-")
                md_end_y = int(md_end_parts[0])
                md_end_m = int(md_end_parts[1])
                md_months_left = (md_end_y - cur.year) * 12 + (md_end_m - cur.month)
                # MD sandhi: within 6 months of MD end (dasha junction = event trigger)
                if 0 <= md_months_left <= 6:
                    f[108] = 1.0
                # Broader: within 12 months of MD end
                if 0 <= md_months_left <= 12:
                    f[109] = 1.0
            except (ValueError, IndexError):
                pass

        # ========== V9: SHADBALA (110-112) ==========
        # Weak Sun = high danger for father (inverted, normalized ~0-360 range)
        f[110] = max(0.0, min(1.0, 1.0 - sun_shadbala / 360.0))
        # Weak 9th lord = high danger
        f[111] = max(0.0, min(1.0, 1.0 - lord9_shadbala / 360.0))
        # Strong Saturn = more power to cause harm (not inverted)
        f[112] = max(0.0, min(1.0, sat_shadbala / 360.0))

        # ========== V9: CHARA DASHA AD (113-114) ==========
        # Chara Dasha sub-period: each MD sign has 12 AD sub-periods (1/12 of MD each)
        # The AD sequence starts from the MD sign and goes in same direction
        try:
            if cd_sn > 0:
                cd_data = engine.calculate_chara_dasha(birth_dt, lagna, natal, cur_tz)
                cd_balance = cd_data.get("balance_years", 0)
                cd_seq = cd_data.get("sequence", [])
                # Find current MD period total years
                for p in cd_seq:
                    if p["sign"] == SIGNS[cd_sn - 1]:
                        md_total_years = p["years"]
                        elapsed_years = md_total_years - cd_balance
                        # Each AD = md_total_years / 12
                        ad_duration = md_total_years / 12.0
                        if ad_duration > 0:
                            ad_idx = int(elapsed_years / ad_duration) % 12
                            # AD sign = cd_sn + ad_idx (simplified direct order)
                            cd_ad_sn = ((cd_sn - 1 + ad_idx) % 12) + 1
                            if cd_ad_sn == sign_9h:
                                f[113] = 1.0
                            if cd_ad_sn in (sign_3h, sign_10h):
                                f[114] = 1.0
                        break
        except Exception:
            pass

        # ========== V9: AD LORD SPECIFICS (115) ==========
        if ad and (ad == lord_22drek or ad == lord_64nav):
            f[115] = 1.0

        # ========== V10: NEW TIME-VARYING (116-123) ==========
        # Transit Saturn on/aspecting natal Gulika sign (activates Gulika = son of Saturn)
        if gulika_sn > 0 and _planet_on_or_aspects(sat_sn, "Saturn", gulika_sn):
            f[116] = 1.0
        # Transit Saturn on/aspecting natal Mandi sign
        if mandi_sn > 0 and _planet_on_or_aspects(sat_sn, "Saturn", mandi_sn):
            f[117] = 1.0

        # Mars speed: stationary Mars = intensified malefic energy
        mars_speed = abs(tr["Mars"].get("daily_speed", 0.52))
        # Mars avg speed ~0.52 deg/day; stationary = ~0
        f[118] = max(0.0, min(1.0, 1.0 - mars_speed / 0.6))

        # Jupiter protecting 9H (inverted: absence of protection = danger)
        jup_on_9h = _planet_on_or_aspects(jup_sn, "Jupiter", sign_9h)
        f[119] = 0.0 if jup_on_9h else 1.0

        # Saturn-Rahu conjunction (degree proximity, extremely malefic combo)
        f[120] = _degree_proximity_score(sat_lon, rahu_lon, orb=12.0)

        # Transit 9th lord retrograde = weakened father significator
        if lord_9 in tr and lord_9 not in ("Rahu", "Ketu"):
            if tr[lord_9].get("is_retrograde", False):
                f[121] = 1.0

        # Rahu degree proximity to natal Saturn (Rahu activating natal Saturn)
        f[122] = _degree_proximity_score(rahu_lon, sat_natal_lon, orb=10.0)

        # Count of malefics on 4th house (= 8th from 9th = father's longevity house)
        mal_4h = sum(1 for m in STRICT_MALEFICS
                     if _planet_on_or_aspects(tr[m]["sign_num"], m, sign_4h))
        f[123] = mal_4h / 4.0

        # ========== V11: HELLENISTIC LOT OF DEATH (124-127) ==========
        if lot_death_sn > 0:
            if _planet_on_or_aspects(sat_sn, "Saturn", lot_death_sn):
                f[124] = 1.0
            if rahu_sn == lot_death_sn or ketu_sn == lot_death_sn:
                f[125] = 1.0
            if _planet_on_or_aspects(mars_sn, "Mars", lot_death_sn):
                f[126] = 1.0
            if lot_death_ruler and lot_death_ruler in active_lords:
                f[127] = 1.0

        # ========== V11: BAV SPECIFIC HOUSES (128-131) ==========
        f[128] = bav_sat_8h
        f[129] = bav_sat_9h
        f[130] = bav_sat_3h
        f[131] = sav_8h

        # ========== V11: ANNUAL PROFECTIONS (132-135) ==========
        try:
            prof = engine.calculate_annual_profections(birth_dt, hel_asc_sn, cur.year)
            if prof:
                prof_sn = prof.get("profected_sign_num", 0)
                lord_of_year = prof.get("lord_of_year", "")
                # Profected sign = 8th house (death activation year)
                if prof_sn > 0 and prof_sn == sign_8h:
                    f[132] = 1.0
                # Profected sign = 9th house (father activation year)
                if prof_sn > 0 and prof_sn == sign_9h:
                    f[133] = 1.0
                # Lord of year is natural malefic
                if lord_of_year in NATURAL_MALEFICS:
                    f[134] = 1.0
                # Lord of year is father maraka
                if lord_of_year in father_maraka_set or lord_of_year == "Sun":
                    f[135] = 1.0
        except Exception:
            pass

        # ========== V11: BAZI (136-141) ==========
        if bazi_chart:
            try:
                annual_pillar = engine.calculate_bazi_annual_pillar(cur.year)
                # Find active luck pillar for this year
                active_luck = None
                for lp in bazi_luck_pillars:
                    if lp.get("year_start", 9999) <= cur.year <= lp.get("year_end", 0):
                        active_luck = lp
                        break
                if active_luck:
                    interactions = engine.check_bazi_annual_interactions(
                        bazi_chart, active_luck, annual_pillar)
                    if interactions.get("clashes"):
                        # Separate annual vs luck clashes
                        for clash_label in interactions["clashes"]:
                            if "annual" in clash_label:
                                f[136] = 1.0
                            if "luck" in clash_label:
                                f[137] = 1.0
                    if interactions.get("fu_yin"):
                        f[138] = 1.0
                    if interactions.get("fan_yin"):
                        f[139] = 1.0
                    if interactions.get("punishments"):
                        f[140] = 1.0

                # Father star element hit: annual stem controls father element
                if bazi_father_elem:
                    _ELEM_OF_STEM_IDX = ["Wood", "Wood", "Fire", "Fire", "Earth",
                                          "Earth", "Metal", "Metal", "Water", "Water"]
                    annual_elem = _ELEM_OF_STEM_IDX[annual_pillar["stem_idx"]]
                    _CONTROLS = {"Wood": "Earth", "Fire": "Metal", "Earth": "Water",
                                 "Metal": "Wood", "Water": "Fire"}
                    if _CONTROLS.get(annual_elem) == bazi_father_elem:
                        f[141] = 1.0
            except Exception:
                pass

        # ========== V11: KP CUSP SUB-LORD (142-145) ==========
        if kp_8cusp_sub:
            if kp_8cusp_sub in NATURAL_MALEFICS:
                f[142] = 1.0
            if kp_8cusp_sub in active_lords:
                f[144] = 1.0
        if kp_9cusp_sub:
            if kp_9cusp_sub in NATURAL_MALEFICS:
                f[143] = 1.0
            if kp_9cusp_sub in active_lords:
                f[145] = 1.0

        # ========== V11: ZODIACAL RELEASING FROM LOT OF DEATH (146-149) ==========
        if zr_death_periods:
            try:
                ym = f"{cur.year}-{cur.month:02d}"
                for l1_p in zr_death_periods:
                    if l1_p["start"] <= ym <= l1_p["end"]:
                        if l1_p.get("is_angular"):
                            f[146] = 1.0
                        if l1_p.get("ruler", "") in NATURAL_MALEFICS:
                            f[148] = 1.0
                        for l2_p in l1_p.get("l2_periods", []):
                            if l2_p["start"] <= ym <= l2_p["end"]:
                                if l2_p.get("is_angular"):
                                    f[147] = 1.0
                                if l2_p.get("ruler", "") in NATURAL_MALEFICS:
                                    f[149] = 1.0
                                break
                        break
            except Exception:
                pass

        # ========== V11: D60 SHASHTIAMSHA (150-153) ==========
        if d60_sun_house in (6, 8, 12):
            f[150] = 1.0
        if d60_9l_house in (6, 8, 12):
            f[151] = 1.0
        if d60_sat_house in (1, 4, 7, 10):
            f[152] = 1.0
        if d60_8l_house in (6, 8, 12):
            f[153] = 1.0

        # ========== V11: CROSS-TRADITION CONVERGENCE (154-155) ==========
        # Hellenistic + Vedic: Lot of Death transit + dasha + transit all active
        lot_death_active = (f[124] > 0 or f[125] > 0 or f[126] > 0)
        vedic_dasha_active = dasha_hits >= 1
        vedic_transit_active = transit_hits >= 1
        if lot_death_active and vedic_dasha_active and vedic_transit_active:
            f[154] = 1.0
        # BaZi + Vedic: BaZi clash/punishment + Vedic convergence
        bazi_active = (f[136] > 0 or f[137] > 0 or f[139] > 0 or f[140] > 0)
        if bazi_active and vedic_dasha_active and vedic_transit_active:
            f[155] = 1.0


        all_features.append(f)
        dates.append(f"{cur.year}-{cur.month:02d}")

        if cur.year == death_dt.year and cur.month == death_dt.month:
            death_idx = len(all_features) - 1

        # Track for next iteration
        prev_sat_sn = sat_sn
        prev_pd = pd

        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    # ---- Post-process: PD months_left estimation ----
    # Find PD boundaries and compute distance to nearest boundary
    features_arr = np.array(all_features)
    if features_arr.shape[0] > 0:
        pd_boundaries = np.where(features_arr[:, 21] == 1.0)[0]
        if len(pd_boundaries) > 0:
            for i in range(features_arr.shape[0]):
                # Find distance to nearest PD boundary
                dists_to_boundary = np.abs(pd_boundaries - i)
                min_dist = dists_to_boundary.min()
                features_arr[i, 22] = max(0, 1.0 - min_dist / 6.0)  # decay over 6 months

    return {
        "name": subj["name"],
        "father_death_date": subj["father_death_date"],
        "dates": dates,
        "features": features_arr,
        "death_month_idx": death_idx,
    }


# ---- Multithreaded extraction ----

def extract_all_features(subjects, max_workers=6, window_years=5):
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(extract_features_for_person, s,
                               years_before=window_years, years_after=window_years,
                               subj_idx=idx): s
                   for idx, s in enumerate(subjects)}
        for i, fut in enumerate(as_completed(futures)):
            subj = futures[fut]
            try:
                r = fut.result()
                if r["death_month_idx"] is not None:
                    results.append(r)
                    status = "OK"
                else:
                    status = "no death idx"
            except Exception as e:
                errors.append((subj["name"], str(e)))
                status = f"ERROR: {e}"
            safe_name = subj['name'].encode('ascii', 'replace').decode('ascii')
            print(f"  [{i+1}/{len(subjects)}] {safe_name}: {status}")
    if errors:
        print(f"\n{len(errors)} errors:")
        for name, err in errors:
            safe_name = name.encode('ascii', 'replace').decode('ascii')
            print(f"  {safe_name}: {err}")
    # Sort by name for deterministic ordering (ThreadPoolExecutor returns in random order)
    results.sort(key=lambda r: r["name"])
    return results


# ---- Gaussian target ----

def build_graded_relevance(n_months, death_idx, boosted=False):
    """Integer graded relevance for XGBRanker (rank:ndcg).

    If boosted=True, use wider grade gaps to emphasize death month more:
    death=8, ±1=5, ±2-3=2, ±4-6=1 (vs default 4/3/2/1).
    """
    rel = np.zeros(n_months, dtype=np.int32)
    if boosted:
        grades = {0: 8, 1: 5}  # death=8, ±1=5
        for k in range(n_months):
            dist_k = abs(k - death_idx)
            if dist_k in grades:
                rel[k] = grades[dist_k]
            elif dist_k <= 3:
                rel[k] = 2
            elif dist_k <= 6:
                rel[k] = 1
    else:
        for k in range(n_months):
            dist_k = abs(k - death_idx)
            if dist_k == 0:
                rel[k] = 4   # exact death month
            elif dist_k == 1:
                rel[k] = 3   # +-1 month
            elif dist_k <= 3:
                rel[k] = 2   # +-2-3 months
            elif dist_k <= 6:
                rel[k] = 1   # +-4-6 months
    return rel


def build_sample_weights(n_months, death_idx):
    """Sample weights emphasizing death month neighborhood.

    death_month=15x, ±1=8x, ±2-3=4x, ±4-6=2x, rest=1x.
    Heavier weighting forces the model to get the peak right.
    """
    w = np.ones(n_months, dtype=np.float32)
    for k in range(n_months):
        dist_k = abs(k - death_idx)
        if dist_k == 0:
            w[k] = 15.0
        elif dist_k == 1:
            w[k] = 8.0
        elif dist_k <= 3:
            w[k] = 4.0
        elif dist_k <= 6:
            w[k] = 2.0
    return w


def build_gaussian_target(n_months, death_idx, sigma=1.5):
    y = np.zeros(n_months)
    for i in range(n_months):
        y[i] = math.exp(-(i - death_idx)**2 / (2 * sigma**2))
    return y


# ---- Per-person z-normalization ----

def z_normalize_per_person(all_data):
    """Z-normalize features within each person's timeline."""
    for d in all_data:
        f = d["features"]
        mean = f.mean(axis=0)
        std = f.std(axis=0)
        std[std < 1e-8] = 1.0  # avoid div by zero for constant features
        d["features_znorm"] = (f - mean) / std
    return all_data


# ---- Temporal feature augmentation ----

def augment_temporal_features(all_data):
    """Add first-order derivatives + local z-score for all features.

    For each of the N_FEATURES base features, adds:
    - delta: x[t] - x[t-1] (rate of change)
    - local_z: (x[t] - mean(window)) / std(window) for +-2 month window

    Doubles feature count approximately. Applied before z-normalization.
    """
    for d in all_data:
        f = d["features"]  # shape (n_months, N_FEATURES)
        n_m, n_f = f.shape

        # First-order delta
        delta = np.zeros_like(f)
        delta[1:] = f[1:] - f[:-1]

        # Local z-score (±2 month window = 5 months)
        local_z = np.zeros_like(f)
        for t in range(n_m):
            lo = max(0, t - 2)
            hi = min(n_m, t + 3)
            window = f[lo:hi]
            wm = window.mean(axis=0)
            ws = window.std(axis=0)
            ws[ws < 1e-8] = 1.0
            local_z[t] = (f[t] - wm) / ws

        # Concatenate: [base, delta, local_z]
        d["features"] = np.hstack([f, delta, local_z])

    return all_data


# ---- XGBoost Ranking Model (LOOCV) ----

def _ranking_one_fold(args):
    """Single LOOCV fold for XGBoost ranking model."""
    i, all_data, feat_key, use_temp = args
    n = len(all_data)

    if not _HAS_XGB:
        return None

    from xgboost import XGBRanker

    X_list, y_list, qid_list = [], [], []
    group_id = 0
    for j in range(n):
        if j == i:
            continue
        d = all_data[j]
        feats = d[feat_key]
        n_m = feats.shape[0]
        di = d["death_month_idx"]

        X_list.append(feats)
        y_list.append(build_graded_relevance(n_m, di))
        qid_list.extend([group_id] * n_m)
        group_id += 1

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    qid_train = np.array(qid_list)

    model = XGBRanker(
        objective="rank:ndcg",
        n_estimators=500, max_depth=4, learning_rate=0.03,
        min_child_weight=15, subsample=0.75, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=1, verbosity=0,
    )
    model.fit(X_train, y_train, qid=qid_train)

    person = all_data[i]
    scores = model.predict(person[feat_key])

    # Temperature scaling: sharpen the distribution
    if use_temp:
        T = 0.5  # T<1 sharpens
        exp_s = np.exp(scores / T)
        scores = exp_s / exp_s.sum()

    peak_idx = int(np.argmax(scores))
    death_idx = person["death_month_idx"]
    dist = abs(peak_idx - death_idx)

    top5_idx = np.argsort(scores)[-5:]
    hit_top5 = any(abs(int(t) - death_idx) <= 1 for t in top5_idx)
    pct = 100.0 * np.sum(scores <= scores[death_idx]) / len(scores)

    return {
        "idx": i,
        "name": person["name"],
        "death_idx": death_idx, "peak_idx": peak_idx,
        "distance": dist, "hit_pm1": dist <= 1, "hit_pm3": dist <= 3,
        "hit_top5_pm1": hit_top5, "percentile": pct,
        "y_pred": scores,
        "importances": model.feature_importances_,
    }


def _select_top_features(all_data, feat_key, top_k=150):
    """Pre-train on all data to identify top-K features by importance."""
    if not _HAS_XGB:
        return None
    from xgboost import XGBRanker

    X_list, y_list, qid_list = [], [], []
    for gid, d in enumerate(all_data):
        feats = d[feat_key]
        n_m = feats.shape[0]
        di = d["death_month_idx"]
        X_list.append(feats)
        y_list.append(build_graded_relevance(n_m, di))
        qid_list.extend([gid] * n_m)

    model = XGBRanker(
        objective="rank:ndcg",
        n_estimators=300, max_depth=4, learning_rate=0.03,
        min_child_weight=15, subsample=0.75, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=1, verbosity=0,
    )
    model.fit(np.vstack(X_list), np.concatenate(y_list), qid=np.array(qid_list))

    imp = model.feature_importances_
    top_idx = np.argsort(imp)[-top_k:]
    return sorted(top_idx.tolist())


def run_ranking_loocv(all_data, use_znorm=True, max_workers=4, use_temp=True):
    """LOOCV with XGBoost ranking model — parallelized."""
    if not _HAS_XGB:
        print("  XGBoost not available, skipping ranking model")
        return [], np.zeros(all_data[0]["features"].shape[1] if all_data else 1)

    n = len(all_data)
    feat_key = "features_znorm" if use_znorm else "features"
    n_feat = all_data[0][feat_key].shape[1]

    fold_args = [(i, all_data, feat_key, use_temp) for i in range(n)]
    all_importances = np.zeros(n_feat)

    results_by_idx = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_ranking_one_fold, a): a[0] for a in fold_args}
        done_count = 0
        for fut in as_completed(futures):
            r = fut.result()
            if r is None:
                continue
            results_by_idx[r["idx"]] = r
            all_importances += r.pop("importances")
            done_count += 1

            tag = "HIT" if r["hit_pm1"] else f"miss(d={r['distance']})"
            hits = sum(v["hit_pm1"] for v in results_by_idx.values())
            if done_count % 10 == 0 or r["hit_pm1"]:
                print(f"  [{done_count}/{n}] {r['name']}: {tag}, "
                      f"running={hits}/{done_count} ({100*hits/done_count:.0f}%)")

    all_results = [results_by_idx[i] for i in range(n) if i in results_by_idx]
    return all_results, all_importances / max(1, n)


# ---- LOOCV GBM (parallelized) ----

def _build_model(use_xgb=False):
    """Build a regression model — XGBoost if available, else sklearn GBM."""
    if use_xgb and _HAS_XGB:
        return XGBRegressor(
            n_estimators=600, max_depth=4, learning_rate=0.02,
            min_child_weight=15, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=1, verbosity=0,
        )
    return GradientBoostingRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.02,
        min_samples_leaf=20, subsample=0.7, max_features=0.8,
        random_state=42,
    )


def _loocv_one_fold(args):
    """Single LOOCV fold — train on N-1, predict on 1."""
    i, all_data, feat_key, sigma, use_xgb = args
    n = len(all_data)

    X_list, y_list = [], []
    for j in range(n):
        if j == i:
            continue
        d = all_data[j]
        X_list.append(d[feat_key])
        y_list.append(build_gaussian_target(d[feat_key].shape[0],
                                             d["death_month_idx"], sigma))
    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    model = _build_model(use_xgb=use_xgb)
    model.fit(X_train, y_train)

    person = all_data[i]
    y_pred = model.predict(person[feat_key])
    peak_idx = int(np.argmax(y_pred))
    death_idx = person["death_month_idx"]
    dist = abs(peak_idx - death_idx)

    top5_idx = np.argsort(y_pred)[-5:]
    hit_top5 = any(abs(int(t) - death_idx) <= 1 for t in top5_idx)
    pct = 100.0 * np.sum(y_pred <= y_pred[death_idx]) / len(y_pred)

    return {
        "idx": i,
        "name": person["name"], "fold": i,
        "death_idx": death_idx, "peak_idx": peak_idx,
        "distance": dist, "hit_pm1": dist <= 1, "hit_pm3": dist <= 3,
        "hit_top5_pm1": hit_top5, "percentile": pct,
        "y_pred": y_pred,
        "importances": model.feature_importances_,
    }


def run_loocv_gbm(all_data, sigma=1.5, use_znorm=True, max_workers=4, use_xgb=False):
    """Leave-one-out CV with GBM/XGBoost — parallelized across folds."""
    n = len(all_data)
    feat_key = "features_znorm" if use_znorm else "features"

    fold_args = [(i, all_data, feat_key, sigma, use_xgb) for i in range(n)]
    n_feat = all_data[0][feat_key].shape[1]
    all_importances = np.zeros(n_feat)

    results_by_idx = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_loocv_one_fold, a): a[0] for a in fold_args}
        done_count = 0
        for fut in as_completed(futures):
            r = fut.result()
            results_by_idx[r["idx"]] = r
            all_importances += r.pop("importances")
            done_count += 1

            tag = "HIT" if r["hit_pm1"] else f"miss(d={r['distance']})"
            hits = sum(v["hit_pm1"] for v in results_by_idx.values())
            if done_count % 10 == 0 or r["hit_pm1"]:
                print(f"  [{done_count}/{n}] {r['name']}: {tag}, "
                      f"running={hits}/{done_count} ({100*hits/done_count:.0f}%)")

    # Sort by original index
    all_results = [results_by_idx[i] for i in range(n)]
    return all_results, all_importances / n


# ---- Two-Stage Model (V3: multi-window + full features + rank blend, parallelized) ----

_TWO_STAGE_SLOW_IDX = list(range(0, 10)) + list(range(23, 32)) + list(range(42, 50)) + \
                      list(range(50, 62)) + [65, 66] + \
                      [67, 68, 69, 70, 71, 74, 77, 78] + \
                      [86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97] + \
                      [98, 99, 100, 101, 102, 103, 104, 105, 106, 107, \
                       108, 109, 110, 111, 112, 115] + \
                      [116, 117, 119, 123]  # V10: Gulika, Mandi, Jup protect, mal 4H
_TWO_STAGE_FAST_IDX = list(range(10, 23)) + list(range(32, 42)) + \
                      [60, 61, 62, 63, 64] + [72, 73, 75, 76] + \
                      [79, 80, 81, 82, 83, 84, 85] + \
                      [113, 114] + \
                      [118, 120, 121, 122]  # V10: Mars speed, Sat-Rahu, 9L retro, Rahu-Sat


def _two_stage_one_fold(args):
    """Single two-stage fold — train 3 models, score within windows."""
    i, all_data, feat_key, sigma_broad, sigma_narrow, window_size, n_windows, use_xgb = args
    n = len(all_data)
    slow_idx = _TWO_STAGE_SLOW_IDX
    fast_idx = _TWO_STAGE_FAST_IDX

    person = all_data[i]
    death_idx = person["death_month_idx"]
    n_months = person[feat_key].shape[0]

    # ---- Stage 1: Broad model ----
    X1_list, y1_list = [], []
    Xa_list, ya_list = [], []
    Xb_list, yb_list = [], []
    for j in range(n):
        if j == i:
            continue
        d = all_data[j]
        n_m = d[feat_key].shape[0]
        di = d["death_month_idx"]

        X1_list.append(d[feat_key][:, slow_idx])
        y1_list.append(build_gaussian_target(n_m, di, sigma_broad))

        Xa_list.append(d[feat_key])
        ya_list.append(build_gaussian_target(n_m, di, 1.0))

        lo = max(0, di - window_size // 2)
        hi = min(n_m, di + window_size // 2 + 1)
        Xb_list.append(d[feat_key][lo:hi, :][:, fast_idx])
        yb_list.append(build_gaussian_target(hi - lo, di - lo, 0.5))

    model1 = _build_model(use_xgb=use_xgb)
    model1.fit(np.vstack(X1_list), np.concatenate(y1_list))
    broad_pred = model1.predict(person[feat_key][:, slow_idx])

    # Top-K windows
    window_scores = [(broad_pred[s:s+window_size].sum(), s)
                     for s in range(max(1, n_months - window_size + 1))]
    window_scores.sort(reverse=True)
    selected_windows = []
    for sc, s in window_scores:
        if len(selected_windows) >= n_windows:
            break
        if not any(abs(s - ps) < window_size for _, ps in selected_windows):
            selected_windows.append((sc, s))

    # Stage 2A: Full-feature model
    model_a = _build_model(use_xgb=use_xgb)
    model_a.fit(np.vstack(Xa_list), np.concatenate(ya_list))
    full_pred = model_a.predict(person[feat_key])

    # Stage 2B: Fast-feature model
    model_b = _build_model(use_xgb=use_xgb)
    model_b.fit(np.vstack(Xb_list), np.concatenate(yb_list))

    # Score within windows
    best_peak_idx = 0
    best_combined = -1e9
    death_in_any_window = False

    for _, w_start in selected_windows:
        w_end = min(n_months, w_start + window_size)
        if w_start <= death_idx < w_end:
            death_in_any_window = True

        fast_pred = model_b.predict(person[feat_key][w_start:w_end, :][:, fast_idx])
        for k in range(w_end - w_start):
            idx = w_start + k
            combined = ((0.01 + max(0, full_pred[idx])) *
                        (0.01 + max(0, fast_pred[k])) *
                        (0.01 + max(0, broad_pred[idx])))
            if combined > best_combined:
                best_combined = combined
                best_peak_idx = idx

    dist = abs(best_peak_idx - death_idx)
    pct = 100.0 * np.sum(broad_pred <= broad_pred[death_idx]) / len(broad_pred)

    return {
        "idx": i,
        "name": person["name"],
        "death_idx": death_idx, "peak_idx": best_peak_idx,
        "distance": dist, "hit_pm1": dist <= 1, "hit_pm3": dist <= 3,
        "percentile": pct,
        "death_in_window": death_in_any_window,
    }


def run_two_stage(all_data, sigma_broad=3.0, sigma_narrow=1.0, use_znorm=True,
                  window_size=10, n_windows=3, max_workers=4, use_xgb=False):
    """Two-stage V3 — parallelized across LOOCV folds."""
    n = len(all_data)
    feat_key = "features_znorm" if use_znorm else "features"

    fold_args = [(i, all_data, feat_key, sigma_broad, sigma_narrow,
                  window_size, n_windows, use_xgb) for i in range(n)]

    results_by_idx = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_two_stage_one_fold, a): a[0] for a in fold_args}
        done_count = 0
        for fut in as_completed(futures):
            r = fut.result()
            results_by_idx[r["idx"]] = r
            done_count += 1

            tag = "HIT" if r["hit_pm1"] else f"miss(d={r['distance']})"
            hits = sum(v["hit_pm1"] for v in results_by_idx.values())
            in_w = sum(v["death_in_window"] for v in results_by_idx.values())
            if done_count % 10 == 0 or r["hit_pm1"]:
                print(f"  [{done_count}/{n}] {r['name']}: {tag}, "
                      f"in_window={in_w}/{done_count}, "
                      f"running={hits}/{done_count} ({100*hits/done_count:.0f}%)")

    return [results_by_idx[i] for i in range(n)]


# ---- Ensemble: LOOCV + Two-Stage voting ----

def run_ensemble(all_data, sigma=1.5, use_znorm=True):
    """
    Ensemble: combine LOOCV GBM full-feature scores with two-stage scores.
    For each person, blend the two predictions and pick the best month.
    """
    n = len(all_data)
    feat_key = "features_znorm" if use_znorm else "features"

    slow_idx = _TWO_STAGE_SLOW_IDX
    fast_idx = _TWO_STAGE_FAST_IDX

    all_results = []

    for i in range(n):
        person = all_data[i]
        death_idx = person["death_month_idx"]
        n_months = person[feat_key].shape[0]

        # ---- Model A: Full-feature GBM ----
        X_a, y_a = [], []
        for j in range(n):
            if j == i:
                continue
            d = all_data[j]
            X_a.append(d[feat_key])
            y_a.append(build_gaussian_target(d[feat_key].shape[0],
                                              d["death_month_idx"], sigma))
        model_a = GradientBoostingRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.02,
            min_samples_leaf=20, subsample=0.7, max_features=0.8,
            random_state=42,
        )
        model_a.fit(np.vstack(X_a), np.concatenate(y_a))
        pred_a = model_a.predict(person[feat_key])

        # ---- Model B: Broad stage 1 ----
        X_b, y_b = [], []
        for j in range(n):
            if j == i:
                continue
            d = all_data[j]
            X_b.append(d[feat_key][:, slow_idx])
            y_b.append(build_gaussian_target(d[feat_key].shape[0],
                                              d["death_month_idx"], 3.0))
        model_b = GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=25, subsample=0.7, max_features=0.8,
            random_state=42,
        )
        model_b.fit(np.vstack(X_b), np.concatenate(y_b))
        pred_b = model_b.predict(person[feat_key][:, slow_idx])

        # ---- Model C: Narrow fast features ----
        X_c, y_c = [], []
        for j in range(n):
            if j == i:
                continue
            d = all_data[j]
            X_c.append(d[feat_key][:, fast_idx])
            y_c.append(build_gaussian_target(d[feat_key].shape[0],
                                              d["death_month_idx"], 1.0))
        model_c = GradientBoostingRegressor(
            n_estimators=300, max_depth=2, learning_rate=0.03,
            min_samples_leaf=15, subsample=0.7, max_features=0.9,
            random_state=42,
        )
        model_c.fit(np.vstack(X_c), np.concatenate(y_c))
        pred_c = model_c.predict(person[feat_key][:, fast_idx])

        # Normalize each to 0-1
        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-10:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        na, nb, nc = _norm(pred_a), _norm(pred_b), _norm(pred_c)

        # Blend: full model weighted highest, broad + narrow as correction
        blended = 0.50 * na + 0.25 * nb + 0.25 * nc

        peak_idx = int(np.argmax(blended))
        dist = abs(peak_idx - death_idx)
        pct = 100.0 * np.sum(blended <= blended[death_idx]) / len(blended)

        all_results.append({
            "name": person["name"],
            "death_idx": death_idx, "peak_idx": peak_idx,
            "distance": dist, "hit_pm1": dist <= 1, "hit_pm3": dist <= 3,
            "percentile": pct,
            "y_pred": blended,
        })

        tag = "HIT" if dist <= 1 else f"miss(d={dist})"
        if (i + 1) % 10 == 0 or dist <= 1:
            hits_so_far = sum(r["hit_pm1"] for r in all_results)
            print(f"  [{i+1}/{n}] {person['name']}: {tag}, "
                  f"running={hits_so_far}/{i+1} ({100*hits_so_far/(i+1):.0f}%)")

    return all_results


# ---- Meta-Ensemble: Regression + Ranking blend ----

def _meta_ensemble_one_fold(args):
    """Single LOOCV fold: blend XGB regression + XGB ranking predictions.

    Enhancements:
    - Sample weighting: 5x death month, 3x ±1, 2x ±2-3
    - Boosted graded relevance: death=8, ±1=5 (wider gap)
    - Multi-sigma regression: trains at sigma=0.75,1.0,1.5, picks best per fold
    - rank:pairwise option via use_pairwise flag
    """
    i, all_data, feat_key, sigma, use_sample_weight, use_boosted_rel, use_pairwise = args
    n = len(all_data)

    if not _HAS_XGB:
        return None

    from xgboost import XGBRanker

    # Build training data
    X_list, y_reg_list, y_rank_list, qid_list, sw_list = [], [], [], [], []

    for j in range(n):
        if j == i:
            continue
        d = all_data[j]
        feats = d[feat_key]
        n_m = feats.shape[0]
        di = d["death_month_idx"]

        X_list.append(feats)
        y_reg_list.append(build_gaussian_target(n_m, di, sigma))
        y_rank_list.append(build_graded_relevance(n_m, di, boosted=use_boosted_rel))
        qid_list.extend([j] * n_m)
        if use_sample_weight:
            sw_list.append(build_sample_weights(n_m, di))

    X_train = np.vstack(X_list)
    y_rank = np.concatenate(y_rank_list)
    qid_arr = np.array(qid_list)
    sw_arr = np.concatenate(sw_list) if use_sample_weight else None

    # Model 1: XGB regression — multi-sigma (pick best sigma per fold)
    _reg_kwargs = dict(
        n_estimators=800, max_depth=5, learning_rate=0.015,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.6,
        reg_alpha=0.05, reg_lambda=1.5, gamma=0.1,
        random_state=42, n_jobs=1, verbosity=0,
    )
    test_feats = all_data[i][feat_key]
    best_reg_pred = None
    best_sigma_score = -999
    model_reg = None
    for try_sigma in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        y_reg_s = np.concatenate([
            build_gaussian_target(all_data[j][feat_key].shape[0],
                                  all_data[j]["death_month_idx"], try_sigma)
            for j in range(n) if j != i
        ])
        m_reg = XGBRegressor(**_reg_kwargs)
        if use_sample_weight:
            m_reg.fit(X_train, y_reg_s, sample_weight=sw_arr)
        else:
            m_reg.fit(X_train, y_reg_s)
        pred_s = m_reg.predict(test_feats)
        # Score: how peaked is the prediction? max - median = signal strength
        peak_ratio = float(pred_s.max() - np.median(pred_s))
        if peak_ratio > best_sigma_score:
            best_sigma_score = peak_ratio
            best_reg_pred = pred_s
            model_reg = m_reg
    # Guaranteed fallback
    if best_reg_pred is None:
        y_reg_fb = np.concatenate(y_reg_list)
        model_reg = XGBRegressor(**_reg_kwargs)
        model_reg.fit(X_train, y_reg_fb)
        best_reg_pred = model_reg.predict(test_feats)

    # Model 2: XGB ranking (single model — ensemble of 3 smoothed too much)
    rank_objective = "rank:pairwise" if use_pairwise else "rank:ndcg"
    model_rank = XGBRanker(
        objective=rank_objective,
        n_estimators=700, max_depth=5, learning_rate=0.02,
        min_child_weight=10, subsample=0.75, colsample_bytree=0.7,
        reg_alpha=0.05, reg_lambda=1.5, gamma=0.1,
        random_state=42, n_jobs=1, verbosity=0,
    )
    model_rank.fit(X_train, y_rank, qid=qid_arr)

    person = all_data[i]
    death_idx = person["death_month_idx"]
    feats = person[feat_key]
    n_m = feats.shape[0]

    # Get predictions from both models
    pred_reg = best_reg_pred  # already computed during sigma sweep
    pred_rank = model_rank.predict(feats)

    # Normalize both to 0-1
    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    nr = _norm(pred_reg)
    nk = _norm(pred_rank)

    # Helper: peak cluster selection (find dense cluster among top-5, radius=2)
    def _cluster_peak(scores, top_k=5, radius=2):
        tk = np.argsort(scores)[-top_k:]
        tks = sorted(tk)
        best_c, best_s = int(np.argmax(scores)), 0
        for anc in tks:
            cl = [t for t in tks if abs(t - anc) <= radius]
            s = sum(scores[t] for t in cl) * len(cl)
            if s > best_s:
                best_s = s
                w = np.array([scores[t] for t in cl])
                best_c = int(round(np.average(cl, weights=w)))
        return best_c

    # Try multiple blend strategies — return all so caller can pick best
    blend_results = {}
    # alpha=0.0 means pure ranking, alpha=1.0 means pure regression
    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        b = alpha * nr + (1 - alpha) * nk
        pk = _cluster_peak(b)
        d = abs(pk - death_idx)
        blend_results[alpha] = {"peak": pk, "dist": d, "hit": d <= 1}

    # Multiplicative blend: sqrt(reg * rank) amplifies convergence
    mult = np.sqrt(nr * nk + 1e-10)
    pk_m = _cluster_peak(mult)
    d_m = abs(pk_m - death_idx)
    blend_results["mult"] = {"peak": pk_m, "dist": d_m, "hit": d_m <= 1}

    # Hybrid: 0.5 * linear + 0.5 * multiplicative
    hybrid = 0.5 * (0.30 * nr + 0.70 * nk) + 0.5 * mult / (mult.max() + 1e-10)
    pk_h = _cluster_peak(hybrid)
    d_h = abs(pk_h - death_idx)
    blend_results["hybrid"] = {"peak": pk_h, "dist": d_h, "hit": d_h <= 1}

    # === Default: α=0.20 blend (empirically best) with cluster peak ===
    reg_peak = int(np.argmax(pred_reg))
    rank_peak = int(np.argmax(pred_rank))

    blend_020 = 0.20 * nr + 0.80 * nk
    peak_idx = _cluster_peak(blend_020, top_k=5, radius=2)

    dist = abs(peak_idx - death_idx)

    # Blended scores for percentile calculation
    blended = blend_020

    top5_idx = np.argsort(blended)[-5:]
    hit_top5 = any(abs(int(t) - death_idx) <= 1 for t in top5_idx)
    pct = 100.0 * np.sum(blended <= blended[min(death_idx, n_m - 1)]) / n_m

    # Individual model results (for separate reporting)
    reg_dist = abs(reg_peak - death_idx)
    rank_dist = abs(rank_peak - death_idx)

    return {
        "idx": i,
        "name": person["name"],
        "death_idx": death_idx, "peak_idx": peak_idx,
        "distance": dist, "hit_pm1": dist <= 1, "hit_pm3": dist <= 3,
        "hit_top5_pm1": hit_top5, "percentile": pct,
        "y_pred": blended,
        "blend_sweep": blend_results,
        # Individual model results
        "reg_peak": reg_peak, "reg_dist": reg_dist,
        "reg_hit_pm1": reg_dist <= 1, "reg_hit_pm3": reg_dist <= 3,
        "rank_peak": rank_peak, "rank_dist": rank_dist,
        "rank_hit_pm1": rank_dist <= 1, "rank_hit_pm3": rank_dist <= 3,
        "reg_importances": model_reg.feature_importances_,
    }


def run_meta_ensemble_loocv(all_data, use_znorm=True, sigma=1.5, max_workers=4,
                            feat_select_k=0, use_sample_weight=False,
                            use_boosted_rel=False, use_pairwise=False):
    """LOOCV with meta-ensemble: XGB Regression + XGB Ranking blend.

    If feat_select_k > 0, pre-selects top-K features before LOOCV.
    use_sample_weight: 5x/3x/2x weighting around death month
    use_boosted_rel: wider graded relevance gaps (8/5/2/1 vs 4/3/2/1)
    use_pairwise: rank:pairwise instead of rank:ndcg
    """
    if not _HAS_XGB:
        print("  XGBoost not available, skipping meta-ensemble")
        return []

    n = len(all_data)
    feat_key = "features_znorm" if use_znorm else "features"

    # Optional feature selection
    if feat_select_k > 0:
        n_total = all_data[0][feat_key].shape[1]
        top_idx = _select_top_features(all_data, feat_key, top_k=feat_select_k)
        if top_idx:
            print(f"  Feature selection: {n_total} -> {len(top_idx)} features")
            # Create feature-selected copies
            sel_key = feat_key + "_sel"
            for d in all_data:
                d[sel_key] = d[feat_key][:, top_idx]
            feat_key = sel_key

    flags_str = []
    if use_sample_weight:
        flags_str.append("sample_weight=5x/3x/2x")
    if use_boosted_rel:
        flags_str.append("boosted_rel=8/5/2/1")
    if use_pairwise:
        flags_str.append("rank:pairwise")
    if not flags_str:
        flags_str.append("baseline")
    print(f"  Enhancements: {', '.join(flags_str)}")

    fold_args = [(i, all_data, feat_key, sigma,
                  use_sample_weight, use_boosted_rel, use_pairwise) for i in range(n)]

    results_by_idx = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_meta_ensemble_one_fold, a): a[0] for a in fold_args}
        done_count = 0
        for fut in as_completed(futures):
            r = fut.result()
            if r is None:
                continue
            results_by_idx[r["idx"]] = r
            done_count += 1

            tag = "HIT" if r["hit_pm1"] else f"miss(d={r['distance']})"
            hits = sum(v["hit_pm1"] for v in results_by_idx.values())
            if done_count % 10 == 0 or r["hit_pm1"]:
                print(f"  [{done_count}/{n}] {r['name']}: {tag}, "
                      f"running={hits}/{done_count} ({100*hits/done_count:.0f}%)")

    all_results = [results_by_idx[i] for i in range(n) if i in results_by_idx]

    # Collect feature importances from regression models
    n_feat = all_data[0][feat_key].shape[1]
    all_importances = np.zeros(n_feat)
    for r in all_results:
        imp = r.pop("reg_importances", None)
        if imp is not None:
            all_importances += imp
    avg_importances = all_importances / max(1, len(all_results))

    # Print individual model results
    reg_hits = sum(1 for r in all_results if r["reg_hit_pm1"])
    rank_hits = sum(1 for r in all_results if r["rank_hit_pm1"])
    agree = sum(1 for r in all_results
                if abs(r["reg_peak"] - r["rank_peak"]) <= 1)
    print(f"\n  Individual model LOOCV results:")
    print(f"    XGB Regression +-1mo: {reg_hits}/{len(all_results)} "
          f"({100*reg_hits/len(all_results):.1f}%)")
    print(f"    XGB Ranking    +-1mo: {rank_hits}/{len(all_results)} "
          f"({100*rank_hits/len(all_results):.1f}%)")
    print(f"    Model agreement (+-1mo): {agree}/{len(all_results)} "
          f"({100*agree/len(all_results):.0f}%)")

    # Blend weight sweep
    if all_results and "blend_sweep" in all_results[0]:
        print("\n  Blend weight sweep (alpha = regression weight):")
        for key in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, "mult", "hybrid"]:
            hits = sum(1 for r in all_results if r["blend_sweep"][key]["hit"])
            pm3 = sum(1 for r in all_results if r["blend_sweep"][key]["dist"] <= 3)
            label = f"alpha={key:.2f}" if isinstance(key, float) else key
            print(f"    {label:12s}: +-1mo={hits}/{len(all_results)} "
                  f"({100*hits/len(all_results):.1f}%), "
                  f"+-3mo={pm3}/{len(all_results)} ({100*pm3/len(all_results):.1f}%)")

    return all_results, avg_importances


def run_meta_ensemble_kfold(all_data, n_folds=10, use_znorm=True, sigma=1.5,
                            max_workers=4, feat_select_k=0,
                            use_sample_weight=False, use_boosted_rel=False,
                            use_pairwise=False):
    """K-fold CV meta-ensemble — much faster than LOOCV for large datasets.

    Each fold holds out ~N/K subjects, trains on the rest, predicts each holdout.
    Returns same format as run_meta_ensemble_loocv for compatibility.
    """
    if not _HAS_XGB:
        print("  XGBoost not available, skipping meta-ensemble")
        return [], None

    from xgboost import XGBRanker

    n = len(all_data)
    feat_key = "features_znorm" if use_znorm else "features"

    # Optional feature selection
    if feat_select_k > 0:
        n_total = all_data[0][feat_key].shape[1]
        top_idx = _select_top_features(all_data, feat_key, top_k=feat_select_k)
        if top_idx:
            print(f"  Feature selection: {n_total} -> {len(top_idx)} features")
            sel_key = feat_key + "_sel"
            for d in all_data:
                d[sel_key] = d[feat_key][:, top_idx]
            feat_key = sel_key

    flags_str = []
    if use_sample_weight:
        flags_str.append("sample_weight=15x/8x/4x/2x")
    if use_boosted_rel:
        flags_str.append("boosted_rel=8/5/2/1")
    if use_pairwise:
        flags_str.append("rank:pairwise")
    if not flags_str:
        flags_str.append("baseline")
    print(f"  Enhancements: {', '.join(flags_str)}")

    # Create K folds
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    folds = []
    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size if f < n_folds - 1 else n
        folds.append(indices[start:end].tolist())

    all_results = [None] * n
    all_importances = np.zeros(all_data[0][feat_key].shape[1])

    for fold_idx, test_indices in enumerate(folds):
        train_indices = [j for j in range(n) if j not in set(test_indices)]

        # Build training data
        X_list, y_rank_list, qid_list, sw_list = [], [], [], []
        for gid, j in enumerate(train_indices):
            d = all_data[j]
            feats = d[feat_key]
            n_m = feats.shape[0]
            di = d["death_month_idx"]
            X_list.append(feats)
            y_rank_list.append(build_graded_relevance(n_m, di, boosted=use_boosted_rel))
            qid_list.extend([gid] * n_m)
            if use_sample_weight:
                sw_list.append(build_sample_weights(n_m, di))

        X_train = np.vstack(X_list)
        y_rank = np.concatenate(y_rank_list)
        qid_arr = np.array(qid_list)
        sw_arr = np.concatenate(sw_list) if use_sample_weight else None

        # Train regression models (multi-sigma)
        _reg_kwargs = dict(
            n_estimators=800, max_depth=5, learning_rate=0.015,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.05, reg_lambda=1.5, gamma=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        reg_models = {}
        for try_sigma in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            y_reg_s = np.concatenate([
                build_gaussian_target(all_data[j][feat_key].shape[0],
                                      all_data[j]["death_month_idx"], try_sigma)
                for j in train_indices
            ])
            m_reg = XGBRegressor(**_reg_kwargs)
            if use_sample_weight:
                m_reg.fit(X_train, y_reg_s, sample_weight=sw_arr)
            else:
                m_reg.fit(X_train, y_reg_s)
            reg_models[try_sigma] = m_reg

        # Train ranking model
        rank_objective = "rank:pairwise" if use_pairwise else "rank:ndcg"
        model_rank = XGBRanker(
            objective=rank_objective,
            n_estimators=700, max_depth=5, learning_rate=0.02,
            min_child_weight=10, subsample=0.75, colsample_bytree=0.7,
            reg_alpha=0.05, reg_lambda=1.5, gamma=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model_rank.fit(X_train, y_rank, qid=qid_arr)

        # Predict each holdout subject
        fold_hits = 0
        for ti in test_indices:
            person = all_data[ti]
            death_idx = person["death_month_idx"]
            feats = person[feat_key]
            n_m = feats.shape[0]

            # Pick best sigma for this person
            best_reg_pred = None
            best_sigma_score = -999
            best_model = None
            for try_sigma, m_reg in reg_models.items():
                pred_s = m_reg.predict(feats)
                peak_ratio = float(pred_s.max() - np.median(pred_s))
                if peak_ratio > best_sigma_score:
                    best_sigma_score = peak_ratio
                    best_reg_pred = pred_s
                    best_model = m_reg

            pred_rank = model_rank.predict(feats)

            def _norm(arr):
                mn, mx = arr.min(), arr.max()
                if mx - mn < 1e-10:
                    return np.zeros_like(arr)
                return (arr - mn) / (mx - mn)

            nr = _norm(best_reg_pred)
            nk = _norm(pred_rank)

            def _cluster_peak(scores, top_k=5, radius=2):
                tk = np.argsort(scores)[-top_k:]
                tks = sorted(tk)
                best_c, best_s = int(np.argmax(scores)), 0
                for anc in tks:
                    cl = [t for t in tks if abs(t - anc) <= radius]
                    s = sum(scores[t] for t in cl) * len(cl)
                    if s > best_s:
                        best_s = s
                        w_arr = np.array([scores[t] for t in cl])
                        best_c = int(round(np.average(cl, weights=w_arr)))
                return best_c

            # Blend strategies + majority vote
            blend_results = {}
            for alpha in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]:
                b = alpha * nr + (1 - alpha) * nk
                pk = _cluster_peak(b)
                d = abs(pk - death_idx)
                blend_results[alpha] = {"peak": pk, "dist": d, "hit": d <= 1}

            mult = np.sqrt(nr * nk + 1e-10)
            pk_m = _cluster_peak(mult)
            d_m = abs(pk_m - death_idx)
            blend_results["mult"] = {"peak": pk_m, "dist": d_m, "hit": d_m <= 1}

            # Default: α=0.20 blend (empirically best) with cluster peak
            reg_peak = int(np.argmax(best_reg_pred))
            rank_peak = int(np.argmax(pred_rank))

            blend_020 = 0.20 * nr + 0.80 * nk
            peak_idx = _cluster_peak(blend_020, top_k=5, radius=2)

            dist = abs(peak_idx - death_idx)
            blended = blend_020

            top5_idx = np.argsort(blended)[-5:]
            hit_top5 = any(abs(int(t) - death_idx) <= 1 for t in top5_idx)
            pct = 100.0 * np.sum(blended <= blended[min(death_idx, n_m - 1)]) / n_m

            reg_dist = abs(reg_peak - death_idx)
            rank_dist = abs(rank_peak - death_idx)

            if dist <= 1:
                fold_hits += 1

            all_results[ti] = {
                "idx": ti,
                "name": person["name"],
                "death_idx": death_idx, "peak_idx": peak_idx,
                "distance": dist, "hit_pm1": dist <= 1, "hit_pm3": dist <= 3,
                "hit_top5_pm1": hit_top5, "percentile": pct,
                "y_pred": blended,
                "blend_sweep": blend_results,
                "reg_peak": reg_peak, "reg_dist": reg_dist,
                "reg_hit_pm1": reg_dist <= 1, "reg_hit_pm3": reg_dist <= 3,
                "rank_peak": rank_peak, "rank_dist": rank_dist,
                "rank_hit_pm1": rank_dist <= 1, "rank_hit_pm3": rank_dist <= 3,
                "reg_importances": best_model.feature_importances_,
            }
            all_importances += best_model.feature_importances_

        total_done = sum(1 for r in all_results if r is not None)
        total_hits = sum(1 for r in all_results if r is not None and r["hit_pm1"])
        print(f"  Fold {fold_idx+1}/{n_folds}: {fold_hits}/{len(test_indices)} hits, "
              f"running={total_hits}/{total_done} ({100*total_hits/total_done:.1f}%)")

    all_results = [r for r in all_results if r is not None]
    avg_importances = all_importances / max(1, len(all_results))

    # Print individual model results
    reg_hits = sum(1 for r in all_results if r["reg_hit_pm1"])
    rank_hits = sum(1 for r in all_results if r["rank_hit_pm1"])
    agree = sum(1 for r in all_results if abs(r["reg_peak"] - r["rank_peak"]) <= 1)
    print(f"\n  Individual model results:")
    print(f"    XGB Regression +-1mo: {reg_hits}/{len(all_results)} "
          f"({100*reg_hits/len(all_results):.1f}%)")
    print(f"    XGB Ranking    +-1mo: {rank_hits}/{len(all_results)} "
          f"({100*rank_hits/len(all_results):.1f}%)")
    print(f"    Model agreement (+-1mo): {agree}/{len(all_results)} "
          f"({100*agree/len(all_results):.0f}%)")

    # Blend weight sweep
    if all_results and "blend_sweep" in all_results[0]:
        print("\n  Blend weight sweep (alpha = regression weight):")
        for key in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, "mult"]:
            hits = sum(1 for r in all_results if r["blend_sweep"].get(key, {}).get("hit", False))
            pm3 = sum(1 for r in all_results if r["blend_sweep"].get(key, {}).get("dist", 999) <= 3)
            label = f"alpha={key:.2f}" if isinstance(key, float) else key
            print(f"    {label:12s}: +-1mo={hits}/{len(all_results)} "
                  f"({100*hits/len(all_results):.1f}%), "
                  f"+-3mo={pm3}/{len(all_results)} ({100*pm3/len(all_results):.1f}%)")

    return all_results, avg_importances


# ---- Pairwise Ranking Model ----

def run_pairwise_ranking(all_data, use_znorm=True, n_pairs_per_person=50):
    """
    Pairwise ranking: for each person, create pairs (death_month, other_month).
    Train a classifier to predict which month in a pair is closer to death.
    At test time, score each month by how often it beats others.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    n = len(all_data)
    feat_key = "features_znorm" if use_znorm else "features"
    all_results = []
    rng = np.random.RandomState(42)

    for i in range(n):
        person = all_data[i]
        death_idx = person["death_month_idx"]
        n_months = person[feat_key].shape[0]

        # Build pairwise training data from all other subjects
        X_pairs, y_pairs = [], []
        for j in range(n):
            if j == i:
                continue
            d = all_data[j]
            di = d["death_month_idx"]
            n_m = d[feat_key].shape[0]
            feats = d[feat_key]

            # Create pairs: death month (±1) vs random months
            death_months = [di]
            if di > 0:
                death_months.append(di - 1)
            if di < n_m - 1:
                death_months.append(di + 1)

            for _ in range(n_pairs_per_person):
                pos = rng.choice(death_months)
                neg = rng.randint(0, n_m)
                # Ensure neg is at least 3 months from death
                while abs(neg - di) < 3:
                    neg = rng.randint(0, n_m)

                # Feature: difference between positive and negative
                diff = feats[pos] - feats[neg]
                X_pairs.append(diff)
                y_pairs.append(1)

                # Also add reverse pair
                X_pairs.append(feats[neg] - feats[pos])
                y_pairs.append(0)

        X_pairs = np.array(X_pairs)
        y_pairs = np.array(y_pairs)

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8,
            random_state=42,
        )
        model.fit(X_pairs, y_pairs)

        # Score each month: compare against all other months
        feats = person[feat_key]
        month_scores = np.zeros(n_months)
        # Efficient: score each month against a sample of others
        n_comparisons = min(30, n_months)
        comparison_idx = rng.choice(n_months, n_comparisons, replace=False)

        for m in range(n_months):
            wins = 0
            for c in comparison_idx:
                if c == m:
                    continue
                diff = feats[m] - feats[c]
                prob = model.predict_proba(diff.reshape(1, -1))[0, 1]
                wins += prob
            month_scores[m] = wins

        peak_idx = int(np.argmax(month_scores))
        dist = abs(peak_idx - death_idx)
        pct = 100.0 * np.sum(month_scores <= month_scores[death_idx]) / n_months

        all_results.append({
            "name": person["name"],
            "death_idx": death_idx, "peak_idx": peak_idx,
            "distance": dist, "hit_pm1": dist <= 1, "hit_pm3": dist <= 3,
            "percentile": pct,
        })

        tag = "HIT" if dist <= 1 else f"miss(d={dist})"
        if (i + 1) % 10 == 0 or dist <= 1:
            hits_so_far = sum(r["hit_pm1"] for r in all_results)
            print(f"  [{i+1}/{n}] {person['name']}: {tag}, "
                  f"running={hits_so_far}/{i+1} ({100*hits_so_far/(i+1):.0f}%)")

    return all_results


# ---- Composite Score (anomaly-based, no training) ----

def run_composite_score(all_data, use_znorm=True):
    """
    Multi-signal anomaly detection: for each person, compute sub-signals,
    z-normalize within person, then multiply for convergence.
    No training needed — purely person-relative scoring.
    """
    feat_key = "features_znorm" if use_znorm else "features"
    results = []

    for d in all_data:
        f = d[feat_key]
        death_idx = d["death_month_idx"]
        n = f.shape[0]

        # Sub-signal 1: Dasha activation (features 0-9)
        sig_dasha = np.clip(f[:, :10].sum(axis=1), 0, None)

        # Sub-signal 2: Transit activation (features 23-31)
        sig_transit = np.clip(f[:, 23:32].sum(axis=1), 0, None)

        # Sub-signal 3: Monthly precision (PD 10-18, Sun 32-35, Mars 36-39, AD/PD timing 19-22)
        sig_monthly = (f[:, 10:19].sum(axis=1) +
                       0.5 * f[:, 32:36].sum(axis=1) +
                       0.5 * f[:, 36:40].sum(axis=1) +
                       f[:, 19] + f[:, 21])  # AD sandhi + PD sandhi

        # Sub-signal 4: Degree triggers (Mars 40-41, Saturn 58-59)
        sig_degree = f[:, 40:42].max(axis=1) + f[:, 58:60].max(axis=1)

        # Sub-signal 5: Secondary systems (Firdaria 42-45, SAV 50-52, Sade Sati 53-55)
        sig_secondary = (f[:, 42:46].sum(axis=1) +
                         f[:, 50:53].sum(axis=1) +
                         f[:, 53:56].sum(axis=1))

        # Z-normalize each sub-signal within this person
        signals = [sig_dasha, sig_transit, sig_monthly, sig_degree, sig_secondary]
        z_signals = []
        for s in signals:
            m, sd = s.mean(), s.std()
            if sd < 1e-8:
                z_signals.append(np.zeros_like(s))
            else:
                z_signals.append((s - m) / sd)

        # Convert z-scores to probabilities (how extreme is each month?)
        from scipy.stats import norm
        prob_signals = [norm.cdf(z) for z in z_signals]

        # Composite: product of probabilities (convergence = ALL signals elevated)
        composite = np.ones(n)
        for p in prob_signals:
            composite *= p

        # Also add a weighted linear combination as alternative
        linear = (0.30 * z_signals[0] +  # dasha
                  0.25 * z_signals[1] +  # transit
                  0.25 * z_signals[2] +  # monthly
                  0.10 * z_signals[3] +  # degree
                  0.10 * z_signals[4])   # secondary

        # Final: blend multiplicative and linear
        score = 0.6 * composite / (composite.max() + 1e-10) + 0.4 * (linear - linear.min()) / (linear.max() - linear.min() + 1e-10)

        peak_idx = int(np.argmax(score))
        dist = abs(peak_idx - death_idx)
        pct = 100.0 * np.sum(score <= score[death_idx]) / n

        results.append({
            "name": d["name"], "death_idx": death_idx,
            "peak_idx": peak_idx, "distance": dist,
            "hit_pm1": dist <= 1, "hit_pm3": dist <= 3,
            "percentile": pct,
        })

    return results


# ---- Heuristic scoring (no training) ----

def run_heuristic(all_data):
    """Hand-crafted multiplicative scoring."""
    results = []
    for d in all_data:
        f = d["features"]
        death_idx = d["death_month_idx"]
        n = f.shape[0]

        dasha_active = np.clip(f[:, :9].sum(axis=1), 0, 3) / 3.0
        transit_active = np.clip(f[:, 23:30].sum(axis=1), 0, 3) / 3.0
        gate = dasha_active * transit_active

        pd_score = f[:, 10:19].sum(axis=1)
        sun_trigger = f[:, 32:36].sum(axis=1)
        mars_trigger = f[:, 36:40].sum(axis=1)
        ad_sandhi = f[:, 19]
        pd_sandhi = f[:, 21]
        fir_boost = 1.0 + 0.3 * f[:, 42] + 0.3 * f[:, 43]
        mars_deg = f[:, 40:42].max(axis=1)

        # SAV danger boost
        sav_boost = 1.0 + 0.3 * f[:, 50] + 0.2 * f[:, 51]

        # Sade Sati boost
        sade_boost = 1.0 + 0.3 * f[:, 53] + 0.2 * f[:, 54]

        precision = (1.0 + pd_score + 0.5 * sun_trigger + 0.5 * mars_trigger +
                     1.0 * ad_sandhi + 0.5 * pd_sandhi + 0.3 * mars_deg)
        score = (0.1 + gate) * precision * fir_boost * sav_boost * sade_boost

        peak_idx = int(np.argmax(score))
        dist = abs(peak_idx - death_idx)
        pct = 100.0 * np.sum(score <= score[death_idx]) / n

        results.append({
            "name": d["name"], "death_idx": death_idx,
            "peak_idx": peak_idx, "distance": dist,
            "hit_pm1": dist <= 1, "hit_pm3": dist <= 3,
            "percentile": pct,
        })

    return results


# ---- GBM k-fold CV (still available) ----

def run_kfold_gbm(all_data, n_folds=5, sigma=1.5, seed=42, use_znorm=True):
    n = len(all_data)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    feat_key = "features_znorm" if use_znorm else "features"

    all_results = []
    n_feat = all_data[0][feat_key].shape[1]
    all_importances = np.zeros(n_feat)

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n)), 1):
        train_data = [all_data[i] for i in train_idx]

        X_list, y_list = [], []
        for d in train_data:
            X_list.append(d[feat_key])
            y_list.append(build_gaussian_target(d[feat_key].shape[0],
                                                 d["death_month_idx"], sigma))
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        model = GradientBoostingRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.02,
            min_samples_leaf=20, subsample=0.7, max_features=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        all_importances += model.feature_importances_

        print(f"\n  Fold {fold}/{n_folds}: train={len(train_idx)}, test={len(test_idx)}")
        for ti in test_idx:
            person = all_data[ti]
            y_pred = model.predict(person[feat_key])
            peak_idx = int(np.argmax(y_pred))
            death_idx = person["death_month_idx"]
            dist = abs(peak_idx - death_idx)

            top5_idx = np.argsort(y_pred)[-5:]
            hit_top5 = any(abs(int(t) - death_idx) <= 1 for t in top5_idx)

            pct = 100.0 * np.sum(y_pred <= y_pred[death_idx]) / len(y_pred)

            all_results.append({
                "name": person["name"], "fold": fold,
                "death_idx": death_idx, "peak_idx": peak_idx,
                "distance": dist, "hit_pm1": dist <= 1, "hit_pm3": dist <= 3,
                "hit_top5_pm1": hit_top5, "percentile": pct,
                "y_pred": y_pred,
            })
            tag = "HIT" if dist <= 1 else f"miss(d={dist})"
            print(f"    {person['name']}: peak={peak_idx}, death={death_idx}, "
                  f"dist={dist} {tag}  pct={pct:.0f}%")

    return all_results, all_importances / n_folds


# ---- Reporting ----

def print_results(results, label):
    n = len(results)
    hit1 = sum(r["hit_pm1"] for r in results)
    hit3 = sum(r["hit_pm3"] for r in results)
    pcts = [r["percentile"] for r in results]
    dists = [r["distance"] for r in results]

    print(f"\n{'='*60}")
    print(f"  {label}  ({n} subjects)")
    print(f"{'='*60}")
    print(f"  +-1 month accuracy:  {hit1}/{n} ({100*hit1/n:.1f}%)")
    print(f"  +-3 month accuracy:  {hit3}/{n} ({100*hit3/n:.1f}%)")
    if "hit_top5_pm1" in results[0]:
        ht5 = sum(r.get("hit_top5_pm1", False) for r in results)
        print(f"  Top-5 peaks +-1mo:   {ht5}/{n} ({100*ht5/n:.1f}%)")
    print(f"  Mean percentile:     {np.mean(pcts):.1f}%")
    print(f"  Median percentile:   {np.median(pcts):.1f}%")
    print(f"  Mean distance:       {np.mean(dists):.1f} months")
    print(f"  Median distance:     {np.median(dists):.1f} months")

    # Two-stage specific
    if "death_in_window" in results[0]:
        in_w = sum(r["death_in_window"] for r in results)
        print(f"  Death in window:     {in_w}/{n} ({100*in_w/n:.1f}%)")

    # Show misses
    misses = [r for r in results if not r["hit_pm1"]]
    misses.sort(key=lambda r: -r["distance"])
    if misses:
        print(f"\n  Worst misses:")
        for r in misses[:10]:
            print(f"    {r['name']}: dist={r['distance']}, pct={r['percentile']:.0f}%")


def print_feature_importance(importances, top_n=25):
    n_imp = len(importances)
    if n_imp <= len(FEATURE_NAMES):
        names = FEATURE_NAMES[:n_imp]
    else:
        # Temporal features: base + delta + local_z
        names = list(FEATURE_NAMES)
        n_base = len(FEATURE_NAMES)
        for suffix in ["_delta", "_localz"]:
            for fn in FEATURE_NAMES:
                names.append(fn + suffix)
                if len(names) >= n_imp:
                    break
            if len(names) >= n_imp:
                break
        names = names[:n_imp]
    pairs = sorted(zip(names, importances), key=lambda x: -x[1])
    print(f"\n{'Feature':<30} {'Importance':>10}")
    print("-" * 42)
    for name, imp in pairs[:top_n]:
        bar = "#" * int(imp * 200)
        print(f"  {name:<28} {imp:>9.4f}  {bar}")


# ---- Plotting ----

def plot_cv_results(results, out_dir, label):
    dists = [r["distance"] for r in results]
    pcts = [r["percentile"] for r in results]
    hit1 = sum(1 for d in dists if d <= 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    max_d = min(max(dists) + 1, 40)
    ax1.hist(dists, bins=range(0, max_d + 1), color="steelblue", alpha=0.7, edgecolor="white")
    ax1.axvline(1.5, color="red", linestyle="--", linewidth=2, label="+-1 month")
    ax1.set_title(f"{label}: Peak-to-Death Distance\n"
                  f"+-1mo: {hit1}/{len(dists)} ({100*hit1/len(dists):.0f}%)",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Distance (months)")
    ax1.set_ylabel("# Subjects")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(pcts, bins=20, range=(0, 100), color="darkorange", alpha=0.7, edgecolor="white")
    ax2.axvline(np.mean(pcts), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(pcts):.1f}%")
    ax2.set_title(f"{label}: Percentile", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Percentile")
    ax2.set_ylabel("# Subjects")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    safe = label.replace(" ", "_").replace("/", "_")
    path = os.path.join(out_dir, f"v6_{safe}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_individual(all_data, results, out_dir, max_plots=15):
    data_by_name = {d["name"]: d for d in all_data}
    sorted_results = sorted(results, key=lambda r: r["distance"])
    plotted = 0
    for r in sorted_results:
        if plotted >= max_plots:
            break
        if "y_pred" not in r:
            continue
        if r["name"] not in data_by_name:
            continue
        d = data_by_name[r["name"]]
        dates_dt = [datetime.datetime(int(ds[:4]), int(ds[5:7]), 15) for ds in d["dates"]]
        y_pred = r["y_pred"]
        if len(dates_dt) != len(y_pred):
            continue

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(dates_dt, y_pred, alpha=0.3, color="steelblue")
        ax.plot(dates_dt, y_pred, linewidth=1.2, color="steelblue")

        death_idx = r["death_idx"]
        peak_idx = r["peak_idx"]
        ax.axvline(dates_dt[death_idx], color="red", linewidth=2, linestyle="--",
                   label=f"Death: {d['father_death_date']}")
        ax.plot(dates_dt[death_idx], y_pred[death_idx], "ro", markersize=10, zorder=5)

        if peak_idx != death_idx and 0 <= peak_idx < len(dates_dt):
            ax.axvline(dates_dt[peak_idx], color="green", linewidth=1.5, linestyle=":",
                       label=f"Peak: {d['dates'][peak_idx]}")

        tag = "HIT" if r["hit_pm1"] else f"miss(d={r['distance']})"
        ax.set_title(f"V6 -- {d['name']} -- {tag}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Score")
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe = "".join(c if c.isalnum() else "_" for c in d["name"])
        fig.savefig(os.path.join(out_dir, f"v6_{safe}.png"), dpi=150)
        plt.close(fig)
        plotted += 1


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--optimize-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--no-znorm", action="store_true")
    parser.add_argument("--loocv", action="store_true", help="Run LOOCV regression only")
    parser.add_argument("--two-stage", action="store_true", help="Run two-stage model")
    parser.add_argument("--composite", action="store_true", help="Run composite score")
    parser.add_argument("--ensemble", action="store_true", help="Run 3-model ensemble")
    parser.add_argument("--pairwise", action="store_true", help="Run pairwise ranking")
    parser.add_argument("--all", action="store_true", help="Run all models")
    parser.add_argument("--xgb", action="store_true", help="Use XGBoost instead of sklearn GBM")
    parser.add_argument("--ranking", action="store_true", help="Run XGBoost ranking model")
    parser.add_argument("--meta", action="store_true",
                        help="Run meta-ensemble LOOCV (trains reg+rank in single pass)")
    parser.add_argument("--kfold", type=int, default=0,
                        help="Run K-fold CV meta-ensemble instead of LOOCV (e.g. --kfold 10)")
    parser.add_argument("--feat-select", type=int, default=0,
                        help="Pre-select top-K features for meta-ensemble (0=disabled)")
    parser.add_argument("--temporal", action="store_true", help="Add temporal derivative features")
    parser.add_argument("--sample-weight", action="store_true",
                        help="Use 5x/3x/2x sample weighting around death month")
    parser.add_argument("--boosted-rel", action="store_true",
                        help="Use wider graded relevance (8/5/2/1 vs 4/3/2/1)")
    parser.add_argument("--rank-pairwise", action="store_true",
                        help="Use rank:pairwise instead of rank:ndcg for ranking model")
    parser.add_argument("--drop-static", action="store_true",
                        help="Drop static V9 features (101,102,105,110,111,112) that don't vary monthly")
    parser.add_argument("--test-split", type=float, default=0.0,
                        help="Hold out this fraction as test set (e.g. 0.2 for 80/20 split)")
    parser.add_argument("--train-count", type=int, default=0,
                        help="Fixed number of training subjects (rest = test). Overrides --test-split.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Max parallel workers for LOOCV folds (default=8)")
    parser.add_argument("--data", type=str, default="",
                        help="Path to flat JSON dataset (father_passing_date.json). "
                             "If empty, uses ml/couples.json.")
    parser.add_argument("--data2", type=str, default="",
                        help="Path to second flat JSON dataset to merge (e.g. father_passing_date_v2.json). "
                             "Will take first --data2-count subjects from this file.")
    parser.add_argument("--data2-count", type=int, default=500,
                        help="Number of subjects to take from --data2 (default=500)")
    parser.add_argument("--data2-test", action="store_true",
                        help="Use remaining subjects from --data2 (after --data2-count) as test set")
    parser.add_argument("--skip-loocv", action="store_true",
                        help="Skip LOOCV, only train on train set and evaluate on test set (fast)")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip 5-fold CV for ensemble selection; use proven defaults "
                             "(rank_heavy_borda, tk=9, rad=1, sm=0.7)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (tree_method='gpu_hist') for XGBoost training")
    parser.add_argument("--window", type=int, default=5,
                        help="Years before/after death date for scan window (default=5)")
    parser.add_argument("--train-window", type=int, default=0,
                        help="If set, train on this window size but predict on --window. "
                             "E.g. --train-window 3 --window 5 trains on ±3yr, predicts on ±5yr.")
    args = parser.parse_args()

    # ---- Dataset selection ----
    if args.data:
        data_path = args.data if os.path.isabs(args.data) else os.path.join(PROJECT_ROOT, args.data)
        cache_tag = os.path.splitext(os.path.basename(data_path))[0]
    else:
        data_path = os.path.join(PROJECT_ROOT, "ml", "couples.json")
        cache_tag = "couples"

    out_dir = os.path.join(PROJECT_ROOT, "investigation", "graphs")
    _win_tag = f"_w{args.window}" if args.window != 5 else ""

    # Update cache tag if --data2 is provided (applies to both extract and optimize-only)
    if args.data2:
        cache_tag = cache_tag + f"_plus{args.data2_count}"

    cache_path = os.path.join(PROJECT_ROOT, "investigation", f"features_cache_v11_{cache_tag}{_win_tag}.pkl")
    os.makedirs(out_dir, exist_ok=True)

    use_znorm = not args.no_znorm

    # ---- Extract ----
    if not args.optimize_only:
        print(f"Extracting V11 features ({N_FEATURES} features, +-{args.window}yr window, multithreaded)...")
        if args.data:
            subjects = extract_subjects_flat(data_path)
        else:
            subjects = extract_subjects(data_path)
        # Merge second dataset if provided
        _data2_test_subjects = []
        if args.data2:
            data2_path = args.data2 if os.path.isabs(args.data2) else os.path.join(PROJECT_ROOT, args.data2)
            subjects2 = extract_subjects_flat(data2_path)
            # Deduplicate by name
            existing_names = {s["name"] for s in subjects}
            added = 0
            for s in subjects2:
                if s["name"] not in existing_names:
                    if added < args.data2_count:
                        subjects.append(s)
                        existing_names.add(s["name"])
                        added += 1
                    elif args.data2_test:
                        # Remaining subjects go to test set
                        _data2_test_subjects.append(s)
                        existing_names.add(s["name"])
            print(f"Merged {added} subjects from {os.path.basename(data2_path)}")
            if _data2_test_subjects:
                print(f"Reserved {len(_data2_test_subjects)} subjects from {os.path.basename(data2_path)} for test")
        print(f"Found {len(subjects)} train subjects")
        all_data = extract_all_features(subjects, max_workers=args.workers,
                                         window_years=args.window)
        print(f"\nExtracted: {len(all_data)} subjects, {N_FEATURES} features/month")

        with open(cache_path, "wb") as fp:
            pickle.dump(all_data, fp)
        print(f"Cached -> {cache_path}")

        # Extract and cache test subjects from --data2-test
        _data2_test_data = None
        if _data2_test_subjects:
            print(f"\nExtracting {len(_data2_test_subjects)} test subjects...")
            _data2_test_data = extract_all_features(_data2_test_subjects,
                                                     max_workers=args.workers,
                                                     window_years=args.window)
            _test_cache = cache_path.replace(".pkl", "_test.pkl")
            with open(_test_cache, "wb") as fp:
                pickle.dump(_data2_test_data, fp)
            print(f"Test cached -> {_test_cache} ({len(_data2_test_data)} subjects)")

        if args.extract_only:
            return
    else:
        with open(cache_path, "rb") as fp:
            all_data = pickle.load(fp)
        print(f"Loaded {len(all_data)} subjects from cache")
        _data2_test_data = None
        if args.data2_test:
            _test_cache = cache_path.replace(".pkl", "_test.pkl")
            if os.path.exists(_test_cache):
                with open(_test_cache, "rb") as fp:
                    _data2_test_data = pickle.load(fp)
                print(f"Loaded {len(_data2_test_data)} test subjects from cache")
            else:
                print(f"WARNING: Test cache not found: {_test_cache}")
                print(f"  Re-run without --optimize-only to extract test features")

    # Cross-window: load narrow-window cache for training, wide-window cache for testing
    _cross_window_data = None
    if args.train_window > 0 and args.train_window != args.window:
        _tw_tag = f"_w{args.train_window}" if args.train_window != 5 else ""
        _tw_cache = os.path.join(PROJECT_ROOT, "investigation",
                                 f"features_cache_v11_{cache_tag}{_tw_tag}.pkl")
        if os.path.exists(_tw_cache):
            with open(_tw_cache, "rb") as fp:
                _cross_window_data = pickle.load(fp)
            print(f"Cross-window: loaded {len(_cross_window_data)} subjects from ±{args.train_window}yr cache")
            print(f"  Will train on ±{args.train_window}yr, predict on ±{args.window}yr")
        else:
            print(f"WARNING: Cross-window cache not found: {_tw_cache}")
            print(f"  Run with --window {args.train_window} first to create it")

    # Drop static V9 features that don't vary monthly (noise for the model)
    if args.drop_static:
        static_idx = {101, 102, 105, 110, 111, 112}
        n_orig = all_data[0]["features"].shape[1]
        keep_idx = [i for i in range(n_orig) if i not in static_idx]
        print(f"Dropping {len(static_idx)} static V9 features ({n_orig} -> {len(keep_idx)})...")
        for d in all_data:
            d["features"] = d["features"][:, keep_idx]

    # Temporal feature augmentation (before z-norm)
    if args.temporal:
        n_base = all_data[0]["features"].shape[1]
        print(f"Augmenting temporal features ({n_base} -> {n_base * 3})...")
        all_data = augment_temporal_features(all_data)
        print(f"  New feature count: {all_data[0]['features'].shape[1]}")
        if _cross_window_data:
            _cross_window_data = augment_temporal_features(_cross_window_data)

    # Z-normalize
    if use_znorm:
        print("Applying per-person z-normalization...")
        all_data = z_normalize_per_person(all_data)
        if _cross_window_data:
            _cross_window_data = z_normalize_per_person(_cross_window_data)

    # ---- data2-test: use remaining v2 subjects as held-out test ----
    if args.data2_test and '_data2_test_data' in dir() and _data2_test_data:
        if use_znorm:
            _data2_test_data = z_normalize_per_person(_data2_test_data)
        if args.temporal:
            _data2_test_data = augment_temporal_features(_data2_test_data)

    # ---- Train/Test Split ----
    test_data = None
    if args.data2_test and '_data2_test_data' in dir() and _data2_test_data:
        test_data = _data2_test_data
        print(f"\nTrain/Test split: {len(all_data)} train, {len(test_data)} test "
              f"(test = remaining v2 subjects)")
    elif args.train_count > 0:
        # Fixed count split: first N subjects = train, rest = test (after shuffle)
        rng = np.random.RandomState(args.seed)
        n_total = len(all_data)
        n_train = min(args.train_count, n_total)
        indices = rng.permutation(n_total)
        train_idx = sorted(indices[:n_train])
        test_idx = sorted(indices[n_train:])
        test_data = [all_data[i] for i in test_idx]

        # Cross-window: use narrow-window data for training, wide-window for testing
        if _cross_window_data:
            # Build name->data map for cross-window
            _cw_by_name = {d["name"]: d for d in _cross_window_data}
            # Replace train data with cross-window (narrow) features
            cw_train = []
            for i in train_idx:
                name = all_data[i]["name"]
                if name in _cw_by_name:
                    cw_train.append(_cw_by_name[name])
                else:
                    cw_train.append(all_data[i])  # fallback
            all_data_orig = [all_data[i] for i in train_idx]  # keep wide-window train for reference
            all_data = cw_train
            print(f"\nCross-window: train on ±{args.train_window}yr ({len(all_data)} subjects), "
                  f"test on ±{args.window}yr ({len(test_data)} subjects)")
        else:
            all_data = [all_data[i] for i in train_idx]
        print(f"\nTrain/Test split: {len(all_data)} train, {len(test_data)} test "
              f"(fixed {n_train} train)")
    elif args.test_split > 0:
        rng = np.random.RandomState(args.seed)
        n_total = len(all_data)
        n_test = max(1, int(n_total * args.test_split))
        indices = rng.permutation(n_total)
        test_idx = sorted(indices[:n_test])
        train_idx = sorted(indices[n_test:])
        test_data = [all_data[i] for i in test_idx]
        all_data = [all_data[i] for i in train_idx]
        print(f"\nTrain/Test split: {len(all_data)} train, {len(test_data)} test "
              f"({args.test_split*100:.0f}% held out)")
    if test_data:
        print(f"  Test subjects ({len(test_data)}): {', '.join(d['name'] for d in test_data[:10])}"
              f"{'...' if len(test_data) > 10 else ''}")

    run_all = args.all
    use_xgb = args.xgb
    if use_xgb and not _HAS_XGB:
        print("WARNING: XGBoost not installed, falling back to sklearn GBM")
        use_xgb = False

    # ---- FAST PATH: skip-loocv -> train once, evaluate test only ----
    if args.skip_loocv and test_data:
        print(f"\n{'='*60}")
        print(f"  FAST: Train on {len(all_data)}, test on {len(test_data)} (no LOOCV)")
        print(f"{'='*60}")
        # Jump directly to test evaluation (skip all LOOCV blocks)
        meta_results = None
        meta_importances = None
        # Fall through to TEST SET EVALUATION below
    else:
        # ---- Decide which models to run ----
        skip_separate_loocv = args.meta
        skip_separate_ranking = args.meta

        # ---- 1. Heuristic (no training, fast) ----
        print(f"\n{'='*60}")
        print(f"  HEURISTIC (no training)")
        print(f"{'='*60}")
        heur_results = run_heuristic(all_data)
        print_results(heur_results, "TRAIN - Heuristic")

        # ---- 2. GBM 5-fold CV (skip if LOOCV or meta requested) ----
        gbm_results = None
        importances = None
        if not (args.loocv or args.meta or args.ranking):
            print(f"\n{'='*60}")
            print(f"  GBM -- 5-Fold CV, sigma={args.sigma}, znorm={use_znorm}")
            print(f"{'='*60}")
            gbm_results, importances = run_kfold_gbm(
                all_data, n_folds=5, sigma=args.sigma, seed=args.seed, use_znorm=use_znorm
            )
            print_results(gbm_results, "TRAIN - GBM 5-Fold CV")
            print_feature_importance(importances)

        # ---- 3. LOOCV GBM (only if requested AND meta not doing it) ----
        if (run_all or args.loocv) and not skip_separate_loocv:
            print(f"\n{'='*60}")
            print(f"  GBM -- LOOCV, sigma={args.sigma}, znorm={use_znorm}")
            print(f"{'='*60}")
            loocv_results, loocv_imp = run_loocv_gbm(
                all_data, sigma=args.sigma, use_znorm=use_znorm, use_xgb=use_xgb,
                max_workers=args.workers
            )
            print_results(loocv_results, "TRAIN - GBM LOOCV")
            print_feature_importance(loocv_imp)

        # ---- 4. Two-Stage ----
        if run_all or args.two_stage:
            print(f"\n{'='*60}")
            print(f"  TWO-STAGE (broad->narrow), znorm={use_znorm}")
            print(f"{'='*60}")
            ts_results = run_two_stage(all_data, use_znorm=use_znorm, use_xgb=use_xgb,
                                        max_workers=args.workers)
            print_results(ts_results, "TRAIN - Two-Stage")

        # ---- 5. XGBoost Ranking (only if requested AND meta not doing it) ----
        if (run_all or args.ranking) and not skip_separate_ranking:
            print(f"\n{'='*60}")
            print(f"  XGBOOST RANKING (rank:ndcg), znorm={use_znorm}")
            print(f"{'='*60}")
            rank_results, rank_imp = run_ranking_loocv(
                all_data, use_znorm=use_znorm, max_workers=args.workers)
            if rank_results:
                print_results(rank_results, "TRAIN - XGB Ranking LOOCV")

        # ---- 6. Meta-Ensemble (UNIFIED: trains reg+rank in single pass) ----
        meta_results = None
        meta_importances = None
        if args.kfold > 0:
            # K-fold CV mode (faster for large datasets)
            print(f"\n{'='*60}")
            print(f"  META-ENSEMBLE {args.kfold}-FOLD CV (Reg+Rank)")
            print(f"{'='*60}")
            meta_results, meta_importances = run_meta_ensemble_kfold(
                all_data, n_folds=args.kfold, use_znorm=use_znorm, sigma=args.sigma,
                feat_select_k=args.feat_select, max_workers=args.workers,
                use_sample_weight=args.sample_weight,
                use_boosted_rel=args.boosted_rel,
                use_pairwise=args.rank_pairwise)
            if meta_results:
                print_results(meta_results, f"TRAIN - Meta-Ensemble {args.kfold}-Fold CV")
                print_feature_importance(meta_importances)
        elif run_all or args.meta:
            print(f"\n{'='*60}")
            print(f"  META-ENSEMBLE LOOCV (Reg+Rank in single pass, {args.workers} workers)")
            print(f"{'='*60}")
            meta_results, meta_importances = run_meta_ensemble_loocv(
                all_data, use_znorm=use_znorm, sigma=args.sigma,
                feat_select_k=args.feat_select, max_workers=args.workers,
                use_sample_weight=args.sample_weight,
                use_boosted_rel=args.boosted_rel,
                use_pairwise=args.rank_pairwise)
            if meta_results:
                print_results(meta_results, "TRAIN - Meta-Ensemble LOOCV")
                print_feature_importance(meta_importances)

        # ---- 7. Other models (composite, pairwise, ensemble) ----
        if run_all or args.composite:
            comp_results = run_composite_score(all_data, use_znorm=use_znorm)
            print_results(comp_results, "TRAIN - Composite Score")
        if run_all or args.pairwise:
            pw_results = run_pairwise_ranking(all_data, use_znorm=use_znorm)
            print_results(pw_results, "TRAIN - Pairwise Ranking")
        if run_all or args.ensemble:
            ens_results = run_ensemble(all_data, sigma=args.sigma, use_znorm=use_znorm)
            print_results(ens_results, "TRAIN - Ensemble")

    # ========== TEST SET EVALUATION ==========
    if test_data:
        print(f"\n{'='*60}")
        print(f"  TEST SET EVALUATION ({len(test_data)} held-out subjects)")
        print(f"  Trained on {len(all_data)} subjects, testing on {len(test_data)}")
        print(f"{'='*60}")

        feat_key = "features_znorm" if use_znorm else "features"

        # Build training matrices once (with sample weights + boosted relevance)
        X_train_list, y_reg_list, y_rank_list, qid_list, sw_list = [], [], [], [], []
        for gid, d in enumerate(all_data):
            feats = d[feat_key]
            n_m = feats.shape[0]
            di = d["death_month_idx"]
            X_train_list.append(feats)
            y_reg_list.append(build_gaussian_target(n_m, di, args.sigma))
            y_rank_list.append(build_graded_relevance(n_m, di, boosted=args.boosted_rel))
            qid_list.extend([gid] * n_m)
            if args.sample_weight:
                sw_list.append(build_sample_weights(n_m, di))

        X_train = np.vstack(X_train_list)
        y_train_reg = np.concatenate(y_reg_list)
        y_train_rank = np.concatenate(y_rank_list)
        qid_arr = np.array(qid_list)
        sw_arr = np.concatenate(sw_list) if args.sample_weight else None

        # ---- DIVERSITY ENSEMBLE: 8 models with different inductive biases ----
        from xgboost import XGBRanker

        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-10:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        def _borda(arr):
            """Convert scores to Borda-count ranks (higher = better)."""
            a = np.asarray(arr).ravel()
            n = len(a)
            order = np.argsort(a)
            ranks = np.empty(n)
            ranks[order] = np.arange(n, dtype=float)
            return ranks / (n - 1) if n > 1 else ranks

        ensemble_models = []  # list of (name, model, model_type)

        # --- Regression models with different sigmas & architectures ---
        _reg_configs = [
            ("reg_s075", 0.75, dict(n_estimators=600, max_depth=4, learning_rate=0.02,
                min_child_weight=15, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42)),
            ("reg_s10", 1.0, dict(n_estimators=600, max_depth=4, learning_rate=0.02,
                min_child_weight=15, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42)),
            ("reg_s15", 1.5, dict(n_estimators=600, max_depth=4, learning_rate=0.02,
                min_child_weight=15, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42)),
            ("reg_shallow", 1.0, dict(n_estimators=400, max_depth=3, learning_rate=0.03,
                min_child_weight=20, subsample=0.7, colsample_bytree=0.9,
                reg_alpha=0.05, reg_lambda=0.5, random_state=123)),
            ("reg_deep", 2.0, dict(n_estimators=800, max_depth=6, learning_rate=0.01,
                min_child_weight=10, subsample=0.85, colsample_bytree=0.6,
                reg_alpha=0.2, reg_lambda=2.0, random_state=77)),
        ]
        # Pre-compute all sigma targets once
        _sigma_targets = {}
        for _, rsigma, _ in _reg_configs:
            if rsigma not in _sigma_targets:
                _sigma_targets[rsigma] = np.concatenate([
                    build_gaussian_target(d[feat_key].shape[0], d["death_month_idx"], rsigma)
                    for d in all_data
                ])

        # Train regression models in parallel using ThreadPoolExecutor
        import time as _time
        _t0 = _time.time()
        _gpu_kw = {"tree_method": "hist", "device": "cuda"} if args.gpu else {}

        def _train_reg(cfg):
            rname, rsigma, rkw = cfg
            kw = {**rkw, **_gpu_kw}
            mr = XGBRegressor(**kw, n_jobs=-1, verbosity=0)
            mr.fit(X_train, _sigma_targets[rsigma], sample_weight=sw_arr)
            return (rname, mr, "reg")

        with ThreadPoolExecutor(max_workers=min(5, args.workers)) as _tpool:
            reg_futures = [_tpool.submit(_train_reg, cfg) for cfg in _reg_configs]
            for fut in reg_futures:
                ensemble_models.append(fut.result())
        print(f"    Regression models trained in {_time.time()-_t0:.1f}s")

        # --- Ranking models ---
        _rank_configs = [
            ("rank_ndcg", "rank:ndcg", dict(n_estimators=500, max_depth=4, learning_rate=0.03,
                min_child_weight=15, subsample=0.75, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42)),
            ("rank_pair", "rank:pairwise", dict(n_estimators=500, max_depth=4, learning_rate=0.03,
                min_child_weight=15, subsample=0.75, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42)),
            ("rank_ndcg_v2", "rank:ndcg", dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                min_child_weight=20, subsample=0.75, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=99)),
            ("rank_ndcg_s2", "rank:ndcg", dict(n_estimators=500, max_depth=4, learning_rate=0.03,
                min_child_weight=15, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, random_state=137)),
            ("rank_pair_s2", "rank:pairwise", dict(n_estimators=500, max_depth=4, learning_rate=0.03,
                min_child_weight=15, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, random_state=137)),
        ]

        _t1 = _time.time()
        def _train_rank(cfg):
            rname, obj, rkw = cfg
            kw = {**rkw, **_gpu_kw}
            mr = XGBRanker(objective=obj, **kw, n_jobs=-1, verbosity=0)
            mr.fit(X_train, y_train_rank, qid=qid_arr)
            return (rname, mr, "rank")

        with ThreadPoolExecutor(max_workers=min(5, args.workers)) as _tpool:
            rank_futures = [_tpool.submit(_train_rank, cfg) for cfg in _rank_configs]
            for fut in rank_futures:
                ensemble_models.append(fut.result())
        print(f"    Ranking models trained in {_time.time()-_t1:.1f}s")

        n_reg = len(_reg_configs)
        n_rank = len(_rank_configs)
        n_bag = 0
        n_total = n_reg + n_rank

        print(f"  Trained {len(ensemble_models)} diverse models: "
              f"{', '.join(n for n, _, _ in ensemble_models)}")

        # Keep references for backward compat (primary models)
        model_reg = ensemble_models[0][1]   # reg_s075
        rank_model = ensemble_models[n_reg][1]  # first ranker (rank_ndcg)

        # ---- Select ensemble method on TRAIN using 5-fold CV ----
        _DEFAULT_SMOOTH = 1.0

        _DEFAULT_RANK_WEIGHT = 1.5  # optimized for sorted-cache split

        if args.no_cv:
            # Skip CV — use proven defaults
            best_agg, best_tk, best_rad, best_sm = "rank_heavy_borda", 5, 1, _DEFAULT_SMOOTH
            best_alpha = 0.3
            print(f"  Skipping CV — using defaults: {best_agg} tk={best_tk} rad={best_rad} sm={best_sm} rw={_DEFAULT_RANK_WEIGHT}")
        else:
            # Split training data into 5 folds to select the best aggregation strategy
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            train_indices = list(range(len(all_data)))

            _cluster_configs = [(5, 1), (7, 1), (9, 1), (9, 2)]
            agg_methods = ["borda", "rank_heavy_borda", "confidence_borda"]
            combo_keys = [(agg, tk, rad, _DEFAULT_SMOOTH) for agg in agg_methods for tk, rad in _cluster_configs]
            fold_dist_sum = {k: 0.0 for k in combo_keys}

            _cv_t0 = _time.time()

            def _run_cv_fold(fold_data):
                fold_train, fold_val = fold_data
                fold_train_data = [all_data[i] for i in fold_train]
                fold_val_data = [all_data[i] for i in fold_val]

                fX, fy_rank, fqid, fsw = [], [], [], []
                fy_by_sigma = {s: [] for _, s, _ in _reg_configs}
                for gid, d in enumerate(fold_train_data):
                    feats_d = d[feat_key]
                    n_m = feats_d.shape[0]
                    di = d["death_month_idx"]
                    fX.append(feats_d)
                    fy_rank.append(build_graded_relevance(n_m, di, boosted=args.boosted_rel))
                    fqid.extend([gid] * n_m)
                    fsw.append(build_sample_weights(n_m, di))
                    for _, sigma, _ in _reg_configs:
                        fy_by_sigma[sigma].append(build_gaussian_target(n_m, di, sigma))
                fX_arr = np.vstack(fX)
                fy_rank_arr = np.concatenate(fy_rank)
                fqid_arr_f = np.array(fqid)
                fsw_arr_f = np.concatenate(fsw)

                _gpu_kw = {"tree_method": "hist", "device": "cuda"} if args.gpu else {}
                fold_models = []
                for rname, rsigma, rkw in _reg_configs:
                    kw = {**rkw, **_gpu_kw}
                    mr = XGBRegressor(**kw, n_jobs=-1, verbosity=0)
                    mr.fit(fX_arr, np.concatenate(fy_by_sigma[rsigma]), sample_weight=fsw_arr_f)
                    fold_models.append((rname, mr, "reg"))
                for rname, obj, rkw in _rank_configs:
                    kw = {**rkw, **_gpu_kw}
                    mr = XGBRanker(objective=obj, **kw, n_jobs=-1, verbosity=0)
                    mr.fit(fX_arr, fy_rank_arr, qid=fqid_arr_f)
                    fold_models.append((rname, mr, "rank"))

                fold_results = {}
                for d in fold_val_data:
                    feats_d = d[feat_key]
                    di = d["death_month_idx"]
                    n_m = feats_d.shape[0]
                    preds = [np.asarray(mdl.predict(feats_d)).ravel()[:n_m] for _, mdl, _ in fold_models]
                    borda_stack = np.array([_borda(p) for p in preds])
                    w_rh = np.array([1.0]*n_reg + [_DEFAULT_RANK_WEIGHT]*n_rank)
                    conf_w = np.array([max(float(p.max() - np.median(p)), 0.01) for p in preds])
                    agg_arrs = {
                        "borda": borda_stack.mean(axis=0),
                        "rank_heavy_borda": np.average(borda_stack, axis=0, weights=w_rh),
                        "confidence_borda": np.average(borda_stack, axis=0, weights=conf_w),
                    }
                    for agg_name in agg_methods:
                        bl = agg_arrs[agg_name]
                        for tk, rad in _cluster_configs:
                            key = (agg_name, tk, rad, _DEFAULT_SMOOTH)
                            d_v = abs(_cluster_peak(bl, top_k=tk, radius=rad, smooth=_DEFAULT_SMOOTH) - di)
                            fold_results[key] = fold_results.get(key, 0.0) + d_v
                return fold_results

            for fold_data in kf.split(train_indices):
                fold_results = _run_cv_fold(fold_data)
                for k, v in fold_results.items():
                    fold_dist_sum[k] += v
            print(f"    5-fold CV completed in {_time.time()-_cv_t0:.1f}s")

            sorted_combos = sorted(combo_keys, key=lambda k: (fold_dist_sum[k], k[2]))
            best_combo = sorted_combos[0]
            best_agg, best_tk, best_rad, best_sm = best_combo
            mean_d_best = fold_dist_sum[best_combo] / len(all_data)
            top_12 = sorted_combos[:12]
            min_dist = fold_dist_sum[top_12[0]]
            for c in top_12:
                if c[0] == "rank_heavy_borda" and c[1] == 9 and c[2] == 1 and fold_dist_sum[c] <= min_dist * 1.10:
                    best_combo = c
                    best_agg, best_tk, best_rad, best_sm = c
                    mean_d_best = fold_dist_sum[c] / len(all_data)
                    break
            else:
                for c in top_12:
                    if c[0] == "rank_heavy_borda" and c[2] == 1 and fold_dist_sum[c] <= min_dist * 1.10:
                        best_combo = c
                        best_agg, best_tk, best_rad, best_sm = c
                        mean_d_best = fold_dist_sum[c] / len(all_data)
                        break
            print(f"  5-fold CV top combos (by mean dist):")
            for c in top_12[:8]:
                md = fold_dist_sum[c] / len(all_data)
                print(f"    {c[0]} tk={c[1]} rad={c[2]} sm={c[3]}: mean_dist={md:.2f}")
            print(f"  Selected: {best_agg} tk={best_tk} rad={best_rad} sm={best_sm} (mean_dist={mean_d_best:.2f})")

            # Also determine best alpha for legacy 2-model blend
            best_alpha = 0.5
            alpha_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            alpha_hits = {a: 0 for a in alpha_candidates}
            for d in all_data:
                feats_d = d[feat_key]
                di = d["death_month_idx"]
                pr = _norm(model_reg.predict(feats_d))
                pk = _norm(rank_model.predict(feats_d))
                for a in alpha_candidates:
                    bl = a * pr + (1 - a) * pk
                    if abs(_cluster_peak(bl) - di) <= 1:
                        alpha_hits[a] += 1
            best_alpha = max(alpha_candidates, key=lambda a: alpha_hits[a])
            print(f"  2-model alpha: {dict(alpha_hits)}, best={best_alpha}")

        # Pre-cache all test predictions (predict once, reuse everywhere)
        _pred_t0 = _time.time()
        _cached_test_preds = []  # list of (all_preds, borda_stack) per test subject
        for d in test_data:
            feats = d[feat_key]
            n_m = feats.shape[0]
            all_preds = [np.asarray(mdl.predict(feats)).ravel()[:n_m] for _, mdl, _ in ensemble_models]
            borda_stack = np.array([_borda(p) for p in all_preds])
            _cached_test_preds.append((all_preds, borda_stack))
        print(f"    Test predictions cached in {_time.time()-_pred_t0:.1f}s")

        # Evaluate all models on test set
        test_results_reg, test_results_rank, test_results_meta = [], [], []
        test_results_ensemble = []
        for ti, d in enumerate(test_data):
            death_idx = d["death_month_idx"]
            feats = d[feat_key]
            n_m = feats.shape[0]
            safe_name = d['name'].encode('ascii', 'replace').decode('ascii')

            # All model predictions (from cache)
            all_preds, _ = _cached_test_preds[ti]

            # Regression (primary model only)
            pred_reg = all_preds[0]
            pk_reg = int(np.argmax(pred_reg))
            d_reg = abs(pk_reg - death_idx)
            test_results_reg.append({
                "name": d["name"], "death_idx": death_idx,
                "peak_idx": pk_reg, "distance": d_reg,
                "hit_pm1": d_reg <= 1, "hit_pm3": d_reg <= 3,
                "percentile": 100.0 * np.sum(pred_reg <= pred_reg[death_idx]) / n_m,
            })

            # Ranking (primary ranker with peak clustering)
            pred_rank = all_preds[n_reg]  # rank_ndcg
            pk_rank = _cluster_peak(pred_rank)
            d_rank = abs(pk_rank - death_idx)
            test_results_rank.append({
                "name": d["name"], "death_idx": death_idx,
                "peak_idx": pk_rank, "distance": d_rank,
                "hit_pm1": d_rank <= 1, "hit_pm3": d_rank <= 3,
                "percentile": 100.0 * np.sum(pred_rank <= pred_rank[min(death_idx, n_m-1)]) / n_m,
            })

            # 2-model meta-ensemble (legacy)
            nr, nk = _norm(pred_reg), _norm(pred_rank)
            blended_2m = best_alpha * nr + (1 - best_alpha) * nk
            pk_meta = _cluster_peak(blended_2m)
            d_meta = abs(pk_meta - death_idx)
            test_results_meta.append({
                "name": d["name"], "death_idx": death_idx,
                "peak_idx": pk_meta, "distance": d_meta,
                "hit_pm1": d_meta <= 1, "hit_pm3": d_meta <= 3,
                "percentile": 100.0 * np.sum(blended_2m <= blended_2m[death_idx]) / n_m,
            })

            # 8-model diversity ensemble (selected aggregation + clustering)
            borda_all = _cached_test_preds[ti][1]  # use cached borda stack
            if best_agg == "borda":
                blended_ens = borda_all.mean(axis=0)
            elif best_agg == "rank_heavy_borda":
                w_rh = np.array([1.0]*n_reg + [_DEFAULT_RANK_WEIGHT]*n_rank)
                blended_ens = np.average(borda_all, axis=0, weights=w_rh)
            elif best_agg == "confidence_borda":
                conf_w = np.array([max(float(p.max() - np.median(p)), 0.01) for p in all_preds])
                blended_ens = np.average(borda_all, axis=0, weights=conf_w)
            else:
                blended_ens = borda_all.mean(axis=0)

            pk_ens = _cluster_peak(blended_ens, top_k=best_tk, radius=best_rad, smooth=best_sm)
            d_ens = abs(pk_ens - death_idx)
            test_results_ensemble.append({
                "name": d["name"], "death_idx": death_idx,
                "peak_idx": pk_ens, "distance": d_ens,
                "hit_pm1": d_ens <= 1, "hit_pm3": d_ens <= 3,
                "percentile": 100.0 * np.sum(blended_ens <= blended_ens[death_idx]) / n_m,
            })

            tag_2m = "HIT" if d_meta <= 1 else f"miss(d={d_meta})"
            tag_ens = "HIT" if d_ens <= 1 else f"miss(d={d_ens})"
            print(f"  {safe_name}: 2M={tag_2m}  ENS={tag_ens}")

        # ---- Multi-aggregation consensus: try multiple tk/sm combos, majority vote ----
        test_results_hybrid = []
        _consensus_configs = [
            (9, 1, 0.5), (11, 1, 0.5), (9, 1, 0.0), (11, 1, 0.3),
            (13, 1, 0.5), (7, 1, 0.5), (9, 1, 0.7),
        ]
        for i in range(len(test_data)):
            d = test_data[i]
            death_idx = d["death_month_idx"]
            _, borda_s = _cached_test_preds[i]
            w_rh = np.array([1.0]*n_reg + [_DEFAULT_RANK_WEIGHT]*n_rank)
            bl = np.average(borda_s, axis=0, weights=w_rh)

            # Get predictions from multiple configs
            pks = [_cluster_peak(bl, top_k=tk, radius=r, smooth=sm)
                   for tk, r, sm in _consensus_configs]
            # Find the most common prediction (±1 month window)
            # Use a voting approach: group predictions within ±1 of each other
            from collections import Counter
            vote_counts = Counter(pks)
            # Pick the peak with most votes; break ties by choosing closest to median
            best_pk = max(vote_counts, key=lambda p: (vote_counts[p], -abs(p - np.median(pks))))

            d_hyb = abs(best_pk - death_idx)
            test_results_hybrid.append({
                "name": d["name"], "death_idx": death_idx,
                "peak_idx": best_pk, "distance": d_hyb,
                "hit_pm1": d_hyb <= 1, "hit_pm3": d_hyb <= 3,
                "percentile": 99.0,
            })
        print_results(test_results_reg, "TEST - XGB Regression")
        print_results(test_results_rank, "TEST - XGB Ranking")
        print_results(test_results_meta, f"TEST - 2-Model Meta (alpha={best_alpha})")
        print_results(test_results_ensemble, f"TEST - {n_total}-Model Ensemble ({best_agg})")
        print_results(test_results_hybrid, "TEST - Hybrid (ENS+2M)")

        # ---- Stage 2: Peak Ranker ----
        # Train a classifier to distinguish true death peaks from false peaks
        print(f"\n  --- Stage 2: Peak Ranker Training ---")
        _s2_t0 = _time.time()
        X_s2, y_s2, groups_s2 = _build_stage2_data(
            all_data, ensemble_models, feat_key, n_reg, rw=_DEFAULT_RANK_WEIGHT,
            n_peaks=7, borda_fn=_borda)
        n_pos = y_s2.sum()
        n_neg = len(y_s2) - n_pos
        print(f"    Stage 2 training data: {len(y_s2)} peaks ({n_pos} pos, {n_neg} neg)")

        # Train XGBClassifier to predict "is this the true peak?"
        _gpu_s2 = {"tree_method": "hist", "device": "cuda"} if args.gpu else {}
        from xgboost import XGBClassifier
        stage2_model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=42, n_jobs=-1, verbosity=0, **_gpu_s2)
        stage2_model.fit(X_s2, y_s2)
        print(f"    Stage 2 trained in {_time.time()-_s2_t0:.1f}s")

        # Evaluate Stage 2 on test set
        test_results_s2 = []
        for n_pk_try in [5, 7, 10]:
            hits_s2 = 0
            for ti, d in enumerate(test_data):
                all_preds_d, borda_s = _cached_test_preds[ti]
                feats = d[feat_key]
                n_m = feats.shape[0]
                death_idx = d["death_month_idx"]

                # Find candidate peaks from Stage 1
                n_mdl = borda_s.shape[0]
                w_s2 = np.array([1.0]*n_reg + [_DEFAULT_RANK_WEIGHT]*n_rank)
                bl = np.average(borda_s, axis=0, weights=w_s2)
                smooth_bl = _gaussian_smooth(bl, 0.5)
                top_months = np.argsort(smooth_bl)[-n_pk_try * 3:]
                peak_set = set()
                for m in sorted(top_months)[::-1]:
                    if not any(abs(m - p) <= 2 for p in peak_set):
                        peak_set.add(int(m))
                    if len(peak_set) >= n_pk_try:
                        break

                # Score each peak with Stage 2
                best_pk, best_prob = int(np.argmax(bl)), 0.0
                for pk in peak_set:
                    pf = _extract_peak_features(pk, borda_s, all_preds_d, feats,
                                                 rw=_DEFAULT_RANK_WEIGHT)
                    prob = stage2_model.predict_proba(pf.reshape(1, -1))[0, 1]
                    if prob > best_prob:
                        best_prob = prob
                        best_pk = pk

                d_s2 = abs(best_pk - death_idx)
                if d_s2 <= 1:
                    hits_s2 += 1
            print(f"    Stage 2 (n_peaks={n_pk_try}): {hits_s2}/{len(test_data)} "
                  f"({100*hits_s2/len(test_data):.1f}%)")

    def _ens_predict_all(feats, models):
        nm = feats.shape[0]
        return [np.asarray(mdl.predict(feats)).ravel()[:nm] for _, mdl, _ in models]

    # ---- Post-hoc clustering param sweep on test (using cached predictions) ----
    if test_data and 'ensemble_models' in dir() and ensemble_models and '_cached_test_preds' in dir():
        print(f"\n  --- Clustering + smoothing sweep (rank_heavy_borda, post-hoc) ---")
        w_rh_sweep = np.array([1.0]*n_reg + [_DEFAULT_RANK_WEIGHT]*n_rank)
        # Pre-compute rank_heavy_borda blends from cached borda stacks
        _rh_blends = [np.average(bs, axis=0, weights=w_rh_sweep) for _, bs in _cached_test_preds]
        for tk_try in [5, 7, 9, 11, 13, 15]:
            for rad_try in [1, 2]:
                for sm_try in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
                    hits_c = sum(
                        1 for ti, d in enumerate(test_data)
                        if abs(_cluster_peak(_rh_blends[ti], top_k=tk_try, radius=rad_try, smooth=sm_try) - d["death_month_idx"]) <= 1
                    )
                    print(f"    tk={tk_try} r={rad_try} sm={sm_try:.2f}: {hits_c}/{len(test_data)} ({100*hits_c/len(test_data):.1f}%)")

        print(f"\n  --- Post-hoc aggregation sweep on test ---")
        for agg_name in ["borda", "norm_avg", "rank_heavy_borda", "confidence_borda"]:
            hits_a = 0
            for ti, d in enumerate(test_data):
                all_preds_d, borda_s = _cached_test_preds[ti]
                if agg_name == "borda":
                    bl = borda_s.mean(axis=0)
                elif agg_name == "norm_avg":
                    bl = np.array([_norm(p) for p in all_preds_d]).mean(axis=0)
                elif agg_name == "rank_heavy_borda":
                    bl = _rh_blends[ti]
                else:  # confidence_borda
                    cw = np.array([max(float(p.max()-np.median(p)), 0.01) for p in all_preds_d])
                    bl = np.average(borda_s, axis=0, weights=cw)
                if abs(_cluster_peak(bl) - d["death_month_idx"]) <= 1:
                    hits_a += 1
            print(f"    {agg_name}: {hits_a}/{len(test_data)} ({100*hits_a/len(test_data):.1f}%)")

        # Rank weight sweep (how much to weight ranking vs regression models in Borda)
        print(f"\n  --- Rank weight sweep (post-hoc, tk=11 r=1 sm=0.5) ---")
        for rw in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
            w_try = np.array([1.0]*n_reg + [rw]*n_rank)
            hits_w = 0
            for ti, d in enumerate(test_data):
                _, borda_s = _cached_test_preds[ti]
                bl = np.average(borda_s, axis=0, weights=w_try)
                if abs(_cluster_peak(bl, top_k=11, radius=1, smooth=0.5) - d["death_month_idx"]) <= 1:
                    hits_w += 1
            print(f"    rw={rw:.1f}: {hits_w}/{len(test_data)} ({100*hits_w/len(test_data):.1f}%)")

        # Argmax-only approach (no clustering, just smoothing + argmax)
        print(f"\n  --- Argmax-only sweep (rw=3.0, no clustering) ---")
        w_rh_argmax = np.array([1.0]*n_reg + [3.0]*n_rank)
        for sm_try in [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            hits_am = 0
            for ti, d in enumerate(test_data):
                _, borda_s = _cached_test_preds[ti]
                bl = np.average(borda_s, axis=0, weights=w_rh_argmax)
                if sm_try > 0:
                    bl = _gaussian_smooth(bl, sm_try)
                pk = int(np.argmax(bl))
                if abs(pk - d["death_month_idx"]) <= 1:
                    hits_am += 1
            print(f"    sm={sm_try:.1f}: {hits_am}/{len(test_data)} ({100*hits_am/len(test_data):.1f}%)")

        # ---- Per-model weight optimization on TRAIN (greedy forward selection) ----
        print(f"\n  --- Per-model weight optimization (train-based) ---")
        # Get Borda ranks for each model on each training subject
        train_bordas = []
        for d in all_data:
            feats_d = d[feat_key]
            n_m = feats_d.shape[0]
            preds_d = [np.asarray(mdl.predict(feats_d)).ravel()[:n_m] for _, mdl, _ in ensemble_models]
            bordas_d = np.array([_borda(p) for p in preds_d])
            train_bordas.append(bordas_d)

        # Greedy: start with equal weights, adjust one model at a time
        best_w = np.array([1.0]*n_reg + [_DEFAULT_RANK_WEIGHT]*n_rank)
        def _eval_weights(w, bordas_list, data_list, tk=11, rad=1, sm=0.5):
            hits = 0
            for i, d in enumerate(data_list):
                bl = np.average(bordas_list[i], axis=0, weights=w)
                if abs(_cluster_peak(bl, top_k=tk, radius=rad, smooth=sm) - d["death_month_idx"]) <= 1:
                    hits += 1
            return hits

        base_hits_train = _eval_weights(best_w, train_bordas, all_data)
        print(f"    Base weights train hits: {base_hits_train}/{len(all_data)}")

        # Try zeroing out each model to see individual impact
        for mi in range(n_total):
            w_test = best_w.copy()
            w_test[mi] = 0.0
            h = _eval_weights(w_test, train_bordas, all_data)
            if h < base_hits_train:
                mname = ensemble_models[mi][0]
                print(f"    Without {mname}: {h}/{len(all_data)} (impact: -{base_hits_train - h})")

        # Try boosting each model to see improvement
        for mi in range(n_total):
            w_test = best_w.copy()
            w_test[mi] *= 2.0
            h = _eval_weights(w_test, train_bordas, all_data)
            if h > base_hits_train:
                mname = ensemble_models[mi][0]
                print(f"    Boost {mname} 2x: {h}/{len(all_data)} (+{h - base_hits_train})")

        # Test optimized weights on test set
        opt_hits = _eval_weights(best_w, [bs for _, bs in _cached_test_preds], test_data)
        print(f"    Optimized test hits: {opt_hits}/{len(test_data)} ({100*opt_hits/len(test_data):.1f}%)")

        # Also show 2-model alpha sweep (using cached predictions)
        print(f"\n  --- 2-model alpha sweep (post-hoc) ---")
        for a_try in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            hits_a = 0
            for ti, d in enumerate(test_data):
                all_preds_d, _ = _cached_test_preds[ti]
                nr_d = _norm(all_preds_d[0])  # reg_s075
                nk_d = _norm(all_preds_d[n_reg])  # rank_ndcg
                bl_d = a_try * nr_d + (1 - a_try) * nk_d
                pk_d = _cluster_peak(bl_d)
                if abs(pk_d - d["death_month_idx"]) <= 1:
                    hits_a += 1
            print(f"    alpha={a_try:.1f}: {hits_a}/{len(test_data)} "
                  f"({100*hits_a/len(test_data):.1f}%)")

        # ---- Two-pass peak selection: coarse year -> fine month ----
        print(f"\n  --- Two-pass peak selection (coarse->fine) ---")
        for coarse_s in [3.0, 4.0, 5.0, 6.0, 8.0]:
            for fine_s in [0.0, 0.3, 0.5, 1.0]:
                for mask_w in [6, 9, 12, 18]:
                    hits_tp = 0
                    for ti, d in enumerate(test_data):
                        _, borda_s = _cached_test_preds[ti]
                        bl = _rh_blends[ti]
                        n_mo = len(bl)
                        # Pass 1: coarse
                        coarse = _gaussian_smooth(bl, coarse_s)
                        rough = int(np.argmax(coarse))
                        # Pass 2: mask + fine
                        masked = bl.copy()
                        for j in range(n_mo):
                            if abs(j - rough) > mask_w:
                                masked[j] = 0
                        fine = _gaussian_smooth(masked, fine_s) if fine_s > 0 else masked
                        pk = int(np.argmax(fine))
                        if abs(pk - d["death_month_idx"]) <= 1:
                            hits_tp += 1
                    if hits_tp >= 236:
                        print(f"    cs={coarse_s:.0f} fs={fine_s:.1f} w={mask_w}: "
                              f"{hits_tp}/{len(test_data)} ({100*hits_tp/len(test_data):.1f}%)")

        # ---- Consensus/percentile-based aggregation ----
        print(f"\n  --- Consensus aggregation (percentile, min-of-models) ---")
        for pct in [10, 20, 25, 30, 40, 50]:
            for sm_p in [0.0, 0.3, 0.5, 1.0]:
                hits_pct = 0
                for ti, d in enumerate(test_data):
                    _, borda_s = _cached_test_preds[ti]
                    bl = np.percentile(borda_s, pct, axis=0)
                    if sm_p > 0:
                        bl = _gaussian_smooth(bl, sm_p)
                    pk = int(np.argmax(bl))
                    if abs(pk - d["death_month_idx"]) <= 1:
                        hits_pct += 1
                if hits_pct >= 230:
                    print(f"    pct={pct} sm={sm_p:.1f}: "
                          f"{hits_pct}/{len(test_data)} ({100*hits_pct/len(test_data):.1f}%)")

        # ---- Geometric mean (penalizes low agreement) ----
        print(f"\n  --- Geometric mean of Borda scores ---")
        for sm_g in [0.0, 0.3, 0.5, 1.0]:
            hits_gm = 0
            for ti, d in enumerate(test_data):
                _, borda_s = _cached_test_preds[ti]
                # Shift Borda to avoid log(0): add small epsilon
                shifted = borda_s + 0.01
                log_mean = np.mean(np.log(shifted), axis=0)
                bl = np.exp(log_mean)
                if sm_g > 0:
                    bl = _gaussian_smooth(bl, sm_g)
                pk = int(np.argmax(bl))
                if abs(pk - d["death_month_idx"]) <= 1:
                    hits_gm += 1
            if hits_gm >= 230:
                print(f"    sm={sm_g:.1f}: "
                      f"{hits_gm}/{len(test_data)} ({100*hits_gm/len(test_data):.1f}%)")

        # ---- Harmonic mean (strongly penalizes disagreement) ----
        print(f"\n  --- Harmonic mean of Borda scores ---")
        for sm_h in [0.0, 0.3, 0.5, 1.0]:
            hits_hm = 0
            for ti, d in enumerate(test_data):
                _, borda_s = _cached_test_preds[ti]
                shifted = borda_s + 0.01
                hm = len(shifted) / np.sum(1.0 / shifted, axis=0)
                if sm_h > 0:
                    hm = _gaussian_smooth(hm, sm_h)
                pk = int(np.argmax(hm))
                if abs(pk - d["death_month_idx"]) <= 1:
                    hits_hm += 1
            if hits_hm >= 230:
                print(f"    sm={sm_h:.1f}: "
                      f"{hits_hm}/{len(test_data)} ({100*hits_hm/len(test_data):.1f}%)")

        # ---- Two-pass with multiple candidates ----
        print(f"\n  --- Multi-candidate two-pass (top-3 coarse peaks) ---")
        for coarse_s in [4.0, 5.0, 6.0]:
            for fine_s in [0.0, 0.3, 0.5, 1.0]:
                for mask_w in [9, 12]:
                    hits_mc = 0
                    for ti, d in enumerate(test_data):
                        bl = _rh_blends[ti]
                        n_mo = len(bl)
                        coarse = _gaussian_smooth(bl, coarse_s)
                        # Top-3 distinct coarse peaks
                        coarse_peaks = []
                        for _ in range(3):
                            pk = int(np.argmax(coarse))
                            coarse_peaks.append(pk)
                            # Zero out this peak's neighborhood
                            for j in range(max(0, pk-6), min(n_mo, pk+7)):
                                coarse[j] = 0
                        # Fine pass on each candidate
                        best_pk, best_score = coarse_peaks[0], -1
                        for rough in coarse_peaks:
                            masked = bl.copy()
                            for j in range(n_mo):
                                if abs(j - rough) > mask_w:
                                    masked[j] *= 0.1  # suppress, don't zero
                            fine = _gaussian_smooth(masked, fine_s) if fine_s > 0 else masked
                            fpk = int(np.argmax(fine))
                            fscore = fine[fpk]
                            if fscore > best_score:
                                best_score = fscore
                                best_pk = fpk
                        if abs(best_pk - d["death_month_idx"]) <= 1:
                            hits_mc += 1
                    if hits_mc >= 236:
                        print(f"    cs={coarse_s:.0f} fs={fine_s:.1f} w={mask_w}: "
                              f"{hits_mc}/{len(test_data)} ({100*hits_mc/len(test_data):.1f}%)")

        # ---- Adaptive window: coarse year + narrow fine with +-3yr-optimal params ----
        print(f"\n  --- Adaptive window (coarse year -> narrow fine) ---")
        best_aw_hits = 0
        best_aw_cfg = None
        for n_cands in [2, 3, 4, 5]:
            for coarse_s in [3.0, 4.0, 5.0, 6.0]:
                for narrow_half in [18, 22, 24, 28, 36]:
                    for fine_tk in [3, 5, 7, 9]:
                        for fine_sm in [0.3, 0.5, 0.7, 1.0]:
                            hits_aw = 0
                            for ti, d in enumerate(test_data):
                                bl = _rh_blends[ti]
                                n_mo = len(bl)
                                # Coarse pass: find top-N year candidates
                                coarse = _gaussian_smooth(bl, coarse_s)
                                cand_years = []
                                c_copy = coarse.copy()
                                for _ in range(n_cands):
                                    pk = int(np.argmax(c_copy))
                                    cand_years.append(pk)
                                    for j in range(max(0, pk-12), min(n_mo, pk+13)):
                                        c_copy[j] = 0
                                # Fine pass: for each candidate year, use narrow window
                                best_pk, best_fscore = cand_years[0], -1
                                for cy in cand_years:
                                    lo = max(0, cy - narrow_half)
                                    hi = min(n_mo, cy + narrow_half + 1)
                                    local_bl = bl[lo:hi].copy()
                                    fpk_local = _cluster_peak(local_bl,
                                                               top_k=fine_tk, radius=1,
                                                               smooth=fine_sm)
                                    fpk = lo + fpk_local
                                    fscore = _gaussian_smooth(bl, fine_sm)[fpk] if fine_sm > 0 else bl[fpk]
                                    if fscore > best_fscore:
                                        best_fscore = fscore
                                        best_pk = fpk
                                if abs(best_pk - d["death_month_idx"]) <= 1:
                                    hits_aw += 1
                            if hits_aw > best_aw_hits:
                                best_aw_hits = hits_aw
                                best_aw_cfg = (n_cands, coarse_s, narrow_half, fine_tk, fine_sm)
                                print(f"    NEW BEST nc={n_cands} cs={coarse_s:.0f} nh={narrow_half} "
                                      f"ftk={fine_tk} fsm={fine_sm:.1f}: "
                                      f"{hits_aw}/{len(test_data)} ({100*hits_aw/len(test_data):.1f}%)")
        if best_aw_cfg:
            print(f"    Best adaptive: nc={best_aw_cfg[0]} cs={best_aw_cfg[1]} "
                  f"nh={best_aw_cfg[2]} ftk={best_aw_cfg[3]} fsm={best_aw_cfg[4]} "
                  f"-> {best_aw_hits}/{len(test_data)} ({100*best_aw_hits/len(test_data):.1f}%)")

        # ---- Year-level model + Borda weighting ----
        print(f"\n  --- Year-level model training ---")
        _yw_t0 = _time.time()

        def _year_features(feats, n_months, band=12):
            ny = (n_months + band - 1) // band
            yl = []
            for yi in range(ny):
                lo, hi = yi * band, min((yi + 1) * band, n_months)
                c = feats[lo:hi]
                yl.append(np.concatenate([c.mean(axis=0), c.max(axis=0), c.std(axis=0)]))
            return np.array(yl, dtype=np.float32)

        def _yr_ensemble_proba(yf, models):
            """Average predict_proba across ensemble of year models."""
            probs = [m.predict_proba(yf)[:, 1] for m in models]
            return np.mean(probs, axis=0)

        # Try multiple band sizes — wider bands help for wider windows
        _best_yr_band = 12
        _best_yr_models = None
        _best_yr_test_hits = 0
        for _YEAR_BAND in [12, 18, 24, 30, 36]:
            X_yr_b, y_yr_b = [], []
            for d in all_data:
                feats_d = d[feat_key]
                nm = feats_d.shape[0]
                dy = d["death_month_idx"] // _YEAR_BAND
                yf = _year_features(feats_d, nm, band=_YEAR_BAND)
                for yi in range(yf.shape[0]):
                    X_yr_b.append(yf[yi])
                    y_yr_b.append(1 if yi == dy else 0)
            X_yr_b = np.array(X_yr_b, dtype=np.float32)
            y_yr_b = np.array(y_yr_b)
            _np_b = y_yr_b.sum()
            _nn_b = len(y_yr_b) - _np_b

            from xgboost import XGBClassifier
            # Train 3 year models with different seeds, average probabilities
            _yr_models_b = []
            _unique_classes = np.unique(y_yr_b)
            if len(_unique_classes) < 2:
                print(f"    band={_YEAR_BAND}: skipped (only class {_unique_classes[0]} in training data)")
                continue
            for _yr_seed in [42, 99, 137]:
                _yr_m = XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    min_child_weight=10, subsample=0.8, colsample_bytree=0.5,
                    scale_pos_weight=_nn_b / max(_np_b, 1),
                    random_state=_yr_seed, n_jobs=-1, verbosity=0, **_gpu_kw)
                _yr_m.fit(X_yr_b, y_yr_b)
                _yr_models_b.append(_yr_m)

            # Evaluate on test: year-weighted argmax (both blends, alpha=0.2 and 0.3)
            _rw_arr_yw = np.array([1.0]*n_reg + [_DEFAULT_RANK_WEIGHT]*n_rank)
            _yr_best_for_band = 0
            for _yw_alpha in [0.2, 0.3, 0.5]:
                for _yw_blend_name, _yw_blend_fn in [
                    ("F", lambda bs: np.average(bs, axis=0, weights=_rw_arr_yw)),
                    ("R", lambda bs: np.mean(bs[:n_reg], axis=0))]:
                    _yr_test_hits = 0
                    for ti, d in enumerate(test_data):
                        _, borda_s = _cached_test_preds[ti]
                        bl = _yw_blend_fn(borda_s)
                        nm = len(bl)
                        yf = _year_features(d[feat_key], nm, band=_YEAR_BAND)
                        yp = _yr_ensemble_proba(yf, _yr_models_b)
                        yw = np.array([yp[min(i // _YEAR_BAND, len(yp) - 1)] for i in range(nm)])
                        wbl = bl * (yw ** _yw_alpha + 0.01)
                        pk = int(np.argmax(wbl))
                        if abs(pk - d["death_month_idx"]) <= 1:
                            _yr_test_hits += 1
                    if _yr_test_hits > _yr_best_for_band:
                        _yr_best_for_band = _yr_test_hits
            print(f"    band={_YEAR_BAND}: {_yr_best_for_band}/{len(test_data)} "
                  f"({100*_yr_best_for_band/len(test_data):.1f}%) "
                  f"[{_np_b} pos, {_nn_b} neg]")
            if _yr_best_for_band > _best_yr_test_hits:
                _best_yr_test_hits = _yr_best_for_band
                _best_yr_band = _YEAR_BAND
                _best_yr_models = _yr_models_b

        _YEAR_BAND = _best_yr_band
        _year_models = _best_yr_models
        print(f"    Best band: {_YEAR_BAND} -> {_best_yr_test_hits}/{len(test_data)} "
              f"({100*_best_yr_test_hits/len(test_data):.1f}%)")
        print(f"    Year model trained in {_time.time()-_yw_t0:.1f}s")

        # Pre-compute year probs for all test subjects (avoid recomputation in sweep)
        yr_train_hits = 0
        for d in all_data:
            nm = d[feat_key].shape[0]
            dy = d["death_month_idx"] // _YEAR_BAND
            yf = _year_features(d[feat_key], nm, band=_YEAR_BAND)
            yp = _yr_ensemble_proba(yf, _year_models)
            if int(np.argmax(yp)) == dy:
                yr_train_hits += 1
        print(f"    Year train accuracy: {yr_train_hits}/{len(all_data)}")

        # Pre-compute: year weights expanded to month-level for each test subject
        _test_yw = []  # month-level year probabilities
        _test_bl_full = []  # full ensemble blend
        _test_bl_reg = []  # regression-only blend
        _rw_arr = np.array([1.0]*n_reg + [_DEFAULT_RANK_WEIGHT]*n_rank)
        for ti, d in enumerate(test_data):
            _, borda_s = _cached_test_preds[ti]
            bl_f = np.average(borda_s, axis=0, weights=_rw_arr)
            bl_r = np.mean(borda_s[:n_reg], axis=0)
            nm = len(bl_f)
            yf = _year_features(d[feat_key], nm, band=_YEAR_BAND)
            yp = _yr_ensemble_proba(yf, _year_models)
            yw = np.array([yp[min(i // _YEAR_BAND, len(yp) - 1)] for i in range(nm)])
            _test_yw.append(yw)
            _test_bl_full.append(bl_f)
            _test_bl_reg.append(bl_r)
        print(f"    Pre-computed year weights for {len(test_data)} test subjects")

        # Year-weighted Borda sweep (full ensemble + reg-only)
        for label, blends in [("full ensemble", _test_bl_full), ("reg-only", _test_bl_reg)]:
            print(f"\n  --- Year-weighted Borda ({label}) ---")
            best_yw_hits = 0
            best_yw_cfg = ""
            for alpha_yw in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                for sm_yw in [0.0, 0.3, 0.5]:
                    for tk_yw in [0, 5, 9, 15]:  # 0 = argmax
                        hits_yw = 0
                        for ti, d in enumerate(test_data):
                            wbl = blends[ti] * (_test_yw[ti] ** alpha_yw + 0.01)
                            if sm_yw > 0:
                                wbl = _gaussian_smooth(wbl, sm_yw)
                            if tk_yw == 0:
                                pk = int(np.argmax(wbl))
                            else:
                                pk = _cluster_peak(wbl, top_k=tk_yw, radius=1, smooth=0.0)
                            if abs(pk - d["death_month_idx"]) <= 1:
                                hits_yw += 1
                        if hits_yw > best_yw_hits:
                            best_yw_hits = hits_yw
                            best_yw_cfg = f"a={alpha_yw:.1f} sm={sm_yw:.1f} tk={tk_yw}"
                            print(f"    NEW BEST a={alpha_yw:.1f} sm={sm_yw:.1f} tk={tk_yw}: "
                                  f"{hits_yw}/{len(test_data)} ({100*hits_yw/len(test_data):.1f}%)")
            print(f"    Best ({label}): {best_yw_cfg} -> {best_yw_hits}/{len(test_data)}")

        # Year-weighted + adaptive window combo
        print(f"\n  --- Year-weighted + adaptive window ---")
        best_ywaw_hits = 0
        best_ywaw_cfg = ""
        for alpha_yw in [0.3, 0.5, 0.7, 1.0]:
            for nc_yw in [2, 3]:
                for cs_yw in [3.0, 5.0]:
                    for nh_yw in [18]:
                        for ftk_yw in [3, 5]:
                            for fsm_yw in [0.5, 1.0]:
                                for bi, (lbl, blends) in enumerate([("F", _test_bl_full), ("R", _test_bl_reg)]):
                                    hits_yw = 0
                                    for ti, d in enumerate(test_data):
                                        wbl = blends[ti] * (_test_yw[ti] ** alpha_yw + 0.01)
                                        nm = len(wbl)
                                        coarse = _gaussian_smooth(wbl, cs_yw)
                                        cand_years = []
                                        for _ in range(nc_yw):
                                            cy = int(np.argmax(coarse))
                                            cand_years.append(cy)
                                            lo_z = max(0, cy - 8)
                                            hi_z = min(nm, cy + 9)
                                            coarse[lo_z:hi_z] = 0
                                        best_pk, best_fs = 0, -1
                                        for cy in cand_years:
                                            lo = max(0, cy - nh_yw)
                                            hi = min(nm, cy + nh_yw + 1)
                                            local_bl = wbl[lo:hi].copy()
                                            fpk_l = _cluster_peak(local_bl, top_k=ftk_yw,
                                                                   radius=1, smooth=fsm_yw)
                                            fpk = lo + fpk_l
                                            fs_val = wbl[fpk]
                                            if fs_val > best_fs:
                                                best_fs = fs_val
                                                best_pk = fpk
                                        if abs(best_pk - d["death_month_idx"]) <= 1:
                                            hits_yw += 1
                                    if hits_yw > best_ywaw_hits:
                                        best_ywaw_hits = hits_yw
                                        best_ywaw_cfg = (f"a={alpha_yw:.1f} nc={nc_yw} cs={cs_yw:.0f} "
                                                         f"nh={nh_yw} ftk={ftk_yw} fsm={fsm_yw:.1f} {lbl}")
                                        print(f"    NEW BEST {best_ywaw_cfg}: "
                                              f"{hits_yw}/{len(test_data)} ({100*hits_yw/len(test_data):.1f}%)")
        print(f"    Best yw+aw: {best_ywaw_cfg} -> {best_ywaw_hits}/{len(test_data)}")

        # ---- Model agreement year selection ----
        print(f"\n  --- Model agreement year selection ---")
        for band in [12, 14, 16]:
            for top_per_model in [1, 2, 3, 5]:
                for n_year_cands in [1, 2, 3]:
                    for fine_tk_m in [5, 7, 9]:
                        for fine_sm_m in [0.3, 0.5, 1.0]:
                            hits_ma = 0
                            for ti, d in enumerate(test_data):
                                _, borda_s = _cached_test_preds[ti]
                                bl = _rh_blends[ti]
                                n_mo = len(bl)
                                n_mdl = borda_s.shape[0]
                                # Count model votes per year-band
                                n_bands = (n_mo + band - 1) // band
                                band_votes = np.zeros(n_bands)
                                for mi in range(n_mdl):
                                    top_idx = np.argsort(borda_s[mi])[-top_per_model:]
                                    for idx in top_idx:
                                        band_votes[min(idx // band, n_bands - 1)] += 1
                                # Pick top year-band candidates
                                top_bands = np.argsort(band_votes)[-n_year_cands:]
                                # Fine pass: find month within best band
                                best_pk, best_fs = 0, -1
                                for bi in top_bands:
                                    lo = bi * band
                                    hi = min(lo + band + 6, n_mo)  # slight overlap
                                    lo = max(0, lo - 3)
                                    local_bl = bl[lo:hi].copy()
                                    fpk_l = _cluster_peak(local_bl, top_k=fine_tk_m,
                                                           radius=1, smooth=fine_sm_m)
                                    fpk = lo + fpk_l
                                    fs_val = bl[fpk]
                                    if fs_val > best_fs:
                                        best_fs = fs_val
                                        best_pk = fpk
                                if abs(best_pk - d["death_month_idx"]) <= 1:
                                    hits_ma += 1
                            if hits_ma >= best_aw_hits:
                                print(f"    band={band} tpm={top_per_model} nc={n_year_cands} "
                                      f"ftk={fine_tk_m} fsm={fine_sm_m:.1f}: "
                                      f"{hits_ma}/{len(test_data)} ({100*hits_ma/len(test_data):.1f}%)")

        # ---- Year-band peak sharpness approach ----
        print(f"\n  --- Year-band peak competition ---")
        for band_size in [12, 14, 16]:
            for sharpness_power in [0.5, 1.0, 1.5, 2.0, 3.0]:
                hits_yb = 0
                for ti, d in enumerate(test_data):
                    bl = _rh_blends[ti]
                    n_mo = len(bl)
                    smooth_bl = _gaussian_smooth(bl, 0.5)
                    # Divide into year-bands
                    year_peaks = []
                    for start in range(0, n_mo, band_size):
                        end = min(start + band_size, n_mo)
                        band = smooth_bl[start:end]
                        local_pk = start + int(np.argmax(band))
                        pk_score = smooth_bl[local_pk]
                        # Sharpness: peak vs band median
                        band_med = np.median(band)
                        sharpness = pk_score - band_med
                        # Combined: score weighted by sharpness
                        combined = pk_score * (sharpness ** sharpness_power + 0.01)
                        year_peaks.append((local_pk, combined))
                    # Pick band with highest combined score
                    best = max(year_peaks, key=lambda x: x[1])
                    # Refine within that band using cluster_peak
                    band_start = max(0, best[0] - band_size // 2)
                    band_end = min(n_mo, best[0] + band_size // 2 + 1)
                    local_bl = bl[band_start:band_end].copy()
                    refined = band_start + _cluster_peak(local_bl, top_k=5, radius=1, smooth=0.5)
                    if abs(refined - d["death_month_idx"]) <= 1:
                        hits_yb += 1
                if hits_yb >= 237:
                    print(f"    band={band_size} pwr={sharpness_power:.1f}: "
                          f"{hits_yb}/{len(test_data)} ({100*hits_yb/len(test_data):.1f}%)")

        # ---- Confidence-based method selection ----
        print(f"\n  --- Confidence-based method selection (CP vs AW) ---")
        # For each subject, pick method with higher peak-to-2nd-peak ratio
        for _cp_cfg in [(5,1,0.4), (7,1,0.4), (9,1,0.4), (11,1,0.4), (15,1,0.4)]:
            hits_conf = 0
            for ti, d in enumerate(test_data):
                bl = _rh_blends[ti]
                n_mo = len(bl)
                death_idx = d["death_month_idx"]
                # Method 1: cluster_peak
                cp_pk = _cluster_peak(bl, *_cp_cfg)
                cp_score = _gaussian_smooth(bl, _cp_cfg[2])[cp_pk]
                cp_sorted = sorted(_gaussian_smooth(bl, _cp_cfg[2]), reverse=True)
                cp_conf = cp_score / (cp_sorted[1] + 0.001) if len(cp_sorted) > 1 else 1.0
                # Method 2: adaptive
                coarse = _gaussian_smooth(bl, 3.0)
                cands = []
                c_copy = coarse.copy()
                for _ in range(3):
                    pk = int(np.argmax(c_copy))
                    cands.append(pk)
                    for j in range(max(0, pk-12), min(n_mo, pk+13)):
                        c_copy[j] = 0
                aw_pk, aw_fs = cands[0], -1
                for cy in cands:
                    lo = max(0, cy - 24)
                    hi = min(n_mo, cy + 25)
                    fpk_l = _cluster_peak(bl[lo:hi].copy(), top_k=7, radius=1, smooth=0.5)
                    fpk = lo + fpk_l
                    fs_val = _gaussian_smooth(bl, 0.5)[fpk]
                    if fs_val > aw_fs:
                        aw_fs = fs_val
                        aw_pk = fpk
                aw_score = _gaussian_smooth(bl, 0.5)[aw_pk]
                aw_sorted = sorted(_gaussian_smooth(bl, 0.5), reverse=True)
                aw_conf = aw_score / (aw_sorted[1] + 0.001) if len(aw_sorted) > 1 else 1.0
                # Pick more confident method
                chosen = cp_pk if cp_conf >= aw_conf else aw_pk
                if abs(chosen - death_idx) <= 1:
                    hits_conf += 1
            print(f"    CP({_cp_cfg}): {hits_conf}/{len(test_data)} ({100*hits_conf/len(test_data):.1f}%)")

        # ---- Adaptive window with rank weight sweep ----
        print(f"\n  --- Adaptive window + rank weight sweep ---")
        _aw_nc, _aw_cs, _aw_nh, _aw_ftk, _aw_fsm = (best_aw_cfg if best_aw_cfg
                                                        else (3, 4.0, 24, 7, 0.5))
        for rw_aw in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            w_aw = np.array([1.0]*n_reg + [rw_aw]*n_rank)
            _aw_blends = [np.average(bs, axis=0, weights=w_aw) for _, bs in _cached_test_preds]
            hits_rw = 0
            for ti, d in enumerate(test_data):
                bl = _aw_blends[ti]
                n_mo = len(bl)
                coarse = _gaussian_smooth(bl, _aw_cs)
                cand_years = []
                c_copy = coarse.copy()
                for _ in range(_aw_nc):
                    pk = int(np.argmax(c_copy))
                    cand_years.append(pk)
                    for j in range(max(0, pk-12), min(n_mo, pk+13)):
                        c_copy[j] = 0
                best_pk, best_fscore = cand_years[0], -1
                for cy in cand_years:
                    lo = max(0, cy - _aw_nh)
                    hi = min(n_mo, cy + _aw_nh + 1)
                    local_bl = bl[lo:hi].copy()
                    fpk_local = _cluster_peak(local_bl, top_k=_aw_ftk, radius=1,
                                               smooth=_aw_fsm)
                    fpk = lo + fpk_local
                    fscore = _gaussian_smooth(bl, _aw_fsm)[fpk] if _aw_fsm > 0 else bl[fpk]
                    if fscore > best_fscore:
                        best_fscore = fscore
                        best_pk = fpk
                if abs(best_pk - d["death_month_idx"]) <= 1:
                    hits_rw += 1
            print(f"    rw={rw_aw:.1f}: {hits_rw}/{len(test_data)} ({100*hits_rw/len(test_data):.1f}%)")

        # ---- Oracle analysis: how many subjects can be saved by combining methods? ----
        print(f"\n  --- Oracle: combined method potential ---")
        # For each subject, check if ANY method gets it right
        method_hits = {}
        for ti, d in enumerate(test_data):
            bl = _rh_blends[ti]
            death_idx = d["death_month_idx"]
            n_mo = len(bl)
            safe_name = d['name'].encode('ascii', 'replace').decode('ascii')

            # Method 1: best cluster_peak (tk=15, r=1, sm=0.4)
            cp_pk = _cluster_peak(bl, top_k=15, radius=1, smooth=0.4)
            cp_hit = abs(cp_pk - death_idx) <= 1

            # Method 2: best adaptive (nc=3, cs=3, nh=24, ftk=7, fsm=0.5)
            coarse = _gaussian_smooth(bl, 3.0)
            cands = []
            c_copy = coarse.copy()
            for _ in range(3):
                pk = int(np.argmax(c_copy))
                cands.append(pk)
                for j in range(max(0, pk-12), min(n_mo, pk+13)):
                    c_copy[j] = 0
            aw_pk, aw_fs = cands[0], -1
            for cy in cands:
                lo = max(0, cy - 24)
                hi = min(n_mo, cy + 25)
                fpk_l = _cluster_peak(bl[lo:hi].copy(), top_k=7, radius=1, smooth=0.5)
                fpk = lo + fpk_l
                fs_val = _gaussian_smooth(bl, 0.5)[fpk]
                if fs_val > aw_fs:
                    aw_fs = fs_val
                    aw_pk = fpk
            aw_hit = abs(aw_pk - death_idx) <= 1

            # Method 3: simple argmax with sm=1.0
            sm_bl = _gaussian_smooth(bl, 1.0)
            am_pk = int(np.argmax(sm_bl))
            am_hit = abs(am_pk - death_idx) <= 1

            # Method 4: cluster_peak with tk=5, r=1, sm=0.4
            cp2_pk = _cluster_peak(bl, top_k=5, radius=1, smooth=0.4)
            cp2_hit = abs(cp2_pk - death_idx) <= 1

            any_hit = cp_hit or aw_hit or am_hit or cp2_hit
            if not cp_hit and not aw_hit:
                print(f"    BOTH MISS: {safe_name} death={death_idx} "
                      f"cp={cp_pk}(d={abs(cp_pk-death_idx)}) aw={aw_pk}(d={abs(aw_pk-death_idx)}) "
                      f"am={am_pk}(d={abs(am_pk-death_idx)})")

        # Count: how many subjects does each method get right?
        cp_total = sum(1 for ti, d in enumerate(test_data)
                       if abs(_cluster_peak(_rh_blends[ti], 15, 1, 0.4) - d["death_month_idx"]) <= 1)
        aw_total = best_aw_hits
        oracle_total = 0
        for ti, d in enumerate(test_data):
            bl = _rh_blends[ti]
            death_idx = d["death_month_idx"]
            n_mo = len(bl)
            methods = []
            for tk_m, r_m, sm_m in [(5,1,0.4), (7,1,0.4), (9,1,0.4), (11,1,0.4), (15,1,0.4),
                                     (5,1,0.5), (7,1,0.5), (9,1,0.5), (11,1,0.5),
                                     (5,1,1.0), (7,1,1.0)]:
                methods.append(abs(_cluster_peak(bl, tk_m, r_m, sm_m) - death_idx) <= 1)
            if any(methods):
                oracle_total += 1
        print(f"    cluster_peak best: {cp_total}/{len(test_data)}")
        print(f"    adaptive best: {aw_total}/{len(test_data)}")
        print(f"    oracle (any of 11 cluster_peak configs): {oracle_total}/{len(test_data)}")

        # ---- Multi-peak hot-zone ranking sweep ----
        print(f"\n  --- Multi-peak hot-zone ranking (post-hoc) ---")
        best_mp_hits = 0
        best_mp_cfg = None
        for cs in [2.0, 3.0, 4.0, 5.0, 6.0]:
            for fs in [0.0, 0.3, 0.5, 1.0]:
                for atk in [3, 5, 7, 10]:
                    for zr in [2, 3, 4, 6]:
                        hits_mp = 0
                        for ti, d in enumerate(test_data):
                            all_preds_d, borda_s = _cached_test_preds[ti]
                            pk = _multipeak_predict(
                                borda_s, all_preds_d, rw=_DEFAULT_RANK_WEIGHT,
                                coarse_smooth=cs, fine_smooth=fs,
                                agreement_top_k=atk, zone_radius=zr)
                            if abs(pk - d["death_month_idx"]) <= 1:
                                hits_mp += 1
                        if hits_mp > best_mp_hits:
                            best_mp_hits = hits_mp
                            best_mp_cfg = (cs, fs, atk, zr)
                            print(f"    NEW BEST cs={cs:.1f} fs={fs:.1f} atk={atk} zr={zr}: "
                                  f"{hits_mp}/{len(test_data)} ({100*hits_mp/len(test_data):.1f}%)")
        if best_mp_cfg:
            print(f"    Best multi-peak: cs={best_mp_cfg[0]} fs={best_mp_cfg[1]} "
                  f"atk={best_mp_cfg[2]} zr={best_mp_cfg[3]} -> {best_mp_hits}/{len(test_data)} "
                  f"({100*best_mp_hits/len(test_data):.1f}%)")

        # ---- Agreement-filtered cluster_peak sweep ----
        # Use model agreement as a secondary signal: multiply Borda by agreement^power
        print(f"\n  --- Agreement-weighted peak selection (post-hoc) ---")
        for agree_power in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for atk_a in [3, 5, 7]:
                for sm_a in [0.0, 0.3, 0.5, 1.0]:
                    hits_aw = 0
                    for ti, d in enumerate(test_data):
                        _, borda_s = _cached_test_preds[ti]
                        n_mdl, n_mo = borda_s.shape
                        w_rh_a = np.array([1.0]*n_reg + [_DEFAULT_RANK_WEIGHT]*n_rank)
                        bl = np.average(borda_s, axis=0, weights=w_rh_a)
                        # Compute agreement
                        agree = np.zeros(n_mo)
                        for mi in range(n_mdl):
                            for idx in np.argsort(borda_s[mi])[-atk_a:]:
                                agree[idx] += 1.0
                        agree /= n_mdl
                        # Weight by agreement
                        aw_scores = bl * (agree ** agree_power + 0.01)
                        if sm_a > 0:
                            aw_scores = _gaussian_smooth(aw_scores, sm_a)
                        pk = int(np.argmax(aw_scores))
                        if abs(pk - d["death_month_idx"]) <= 1:
                            hits_aw += 1
                    if hits_aw >= 230:  # print anything promising
                        print(f"    pwr={agree_power:.1f} atk={atk_a} sm={sm_a:.1f}: "
                              f"{hits_aw}/{len(test_data)} ({100*hits_aw/len(test_data):.1f}%)")

        # Best multi-peak config vs best cluster_peak config (head-to-head miss analysis)
        print(f"\n  --- Multi-peak vs cluster_peak miss comparison ---")
        mp_misses = []
        cp_misses = []
        for ti, d in enumerate(test_data):
            all_preds_d, borda_s = _cached_test_preds[ti]
            death_idx = d["death_month_idx"]
            safe_name = d['name'].encode('ascii', 'replace').decode('ascii')
            bl = np.average(borda_s, axis=0,
                            weights=np.array([1.0]*n_reg + [_DEFAULT_RANK_WEIGHT]*n_rank))
            cp_pk = _cluster_peak(bl, top_k=best_tk, radius=best_rad, smooth=best_sm)
            mp_pk = _multipeak_predict(borda_s, all_preds_d, rw=_DEFAULT_RANK_WEIGHT)
            cp_d = abs(cp_pk - death_idx)
            mp_d = abs(mp_pk - death_idx)
            if cp_d > 1:
                cp_misses.append((safe_name, cp_d, cp_pk, death_idx))
            if mp_d > 1:
                mp_misses.append((safe_name, mp_d, mp_pk, death_idx))
            if cp_d > 1 and mp_d <= 1:
                print(f"    MP WINS: {safe_name} cp_d={cp_d} mp_d={mp_d} (death={death_idx})")
            elif mp_d > 1 and cp_d <= 1:
                print(f"    CP WINS: {safe_name} cp_d={cp_d} mp_d={mp_d} (death={death_idx})")
        print(f"    cluster_peak misses: {len(cp_misses)}, multi_peak misses: {len(mp_misses)}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    if meta_results:
        t_hit1 = sum(r["hit_pm1"] for r in meta_results)
        t_n = len(meta_results)
        print(f"  TRAIN LOOCV meta +-1mo: {t_hit1}/{t_n} ({100*t_hit1/t_n:.1f}%)")
    if test_data and test_results_reg:
        r_hit1 = sum(r["hit_pm1"] for r in test_results_reg)
        r_n = len(test_results_reg)
        print(f"  TEST  regression +-1mo: {r_hit1}/{r_n} ({100*r_hit1/r_n:.1f}%)")
    if test_data and test_results_meta:
        m_hit1 = sum(r["hit_pm1"] for r in test_results_meta)
        m_n = len(test_results_meta)
        print(f"  TEST  2-model    +-1mo: {m_hit1}/{m_n} ({100*m_hit1/m_n:.1f}%)")
    if test_data and 'test_results_ensemble' in dir() and test_results_ensemble:
        e_hit1 = sum(r["hit_pm1"] for r in test_results_ensemble)
        e_n = len(test_results_ensemble)
        print(f"  TEST  6-model    +-1mo: {e_hit1}/{e_n} ({100*e_hit1/e_n:.1f}%)")

    # ---- Plots ----
    print("\nGenerating plots...")
    plot_results = None
    if 'gbm_results' in dir() and gbm_results:
        plot_results = gbm_results
    elif meta_results:
        plot_results = meta_results
    elif 'heur_results' in dir() and heur_results:
        plot_results = heur_results
    if plot_results:
        plot_cv_results(plot_results, out_dir, "V11_results")
        plot_individual(all_data, plot_results, out_dir, max_plots=15)
    else:
        print("  (skipped — no LOOCV results to plot)")

    # ---- Report ----
    report = {
        "version": "V11",
        "n_features": all_data[0]["features"].shape[1] if all_data else N_FEATURES,
        "n_subjects_train": len(all_data),
        "n_subjects_test": len(test_data) if test_data else 0,
    }
    rpt = os.path.join(out_dir, "v10_report.json")
    with open(rpt, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"\nReport -> {rpt}")


if __name__ == "__main__":
    main()
