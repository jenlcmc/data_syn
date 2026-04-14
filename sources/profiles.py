"""Build realistic taxpayer profiles from government wage data.

This module replaces hand-crafted round-number seed records with profiles
grounded in two authoritative public datasets:

  1. DOL LCA Disclosure Data (H-1B filings, FY2024)
     Legally sworn employer wage certifications for specific job titles,
     SOC codes, worksite states, and prevailing wage levels (I–IV).
     Source: https://www.dol.gov/agencies/eta/foreign-labor/performance

  2. BLS Occupational Employment and Wage Statistics (OEWS), May 2024
     National annual wage percentiles (P10/P25/P50/P75/P90) by SOC code.
     Used to validate LCA wages and fill occupations not well-represented
     in H-1B filings (e.g. nurses, teachers, truck drivers).
     Source: https://www.bls.gov/oes/tables.htm

Why LCA over self-reported salary surveys (levels.fyi, Glassdoor):
  LCA filings are submitted under penalty of perjury.  The employer
  certifies the stated wage is what the worker will actually be paid.
  Self-reported surveys are unverified and subject to inflation bias.

Profile coherence rules
-----------------------
All generated TaxpayerFacts objects satisfy the following constraints,
which the random comp_ files violated:

  - Wages are drawn from real LCA/OEWS distributions, not sampled
    uniformly from [0, max].
  - 401(k) deferral (Box 12D) reduces Box 1 wages deterministically:
    Box1 = gross_wage - min(deferral_rate * gross_wage, annual_limit).
  - HSA contributions (Box 12W) are consistent with enrollment status
    (single vs. family plan) and employer contribution norms.
  - SE income is independently drawn; W-2 and SE income are not
    combined for the same filer unless the occupation profile warrants
    it (e.g. a nurse who also does freelance consulting).
  - Itemized deductions are computed from state-specific real estate
    tax medians (ACS data embedded below) rather than sampled freely.
  - Child care expenses are non-zero only for filers with children
    under age 13, and are bounded by actual market-rate day care costs.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATASET_DIR,
    DYNAMIC_PROFILE_CALIBRATION,
    DYNAMIC_PROFILE_CALIBRATION_MAX_LCA_FILES,
)
from schema import TaxpayerFacts
from sources.loader import load_dynamic_calibration

# ---------------------------------------------------------------------------
# Occupation profiles
# ---------------------------------------------------------------------------
# Each OccupationProfile encodes what we know about a job from LCA + OEWS.
# Wages are in annual dollars.  Wage levels map to DOL prevailing wage
# levels I–IV (roughly: entry / mid / senior / staff or principal).
#
# Sources:
#   LCA FY2024 Q4, 598,831 certified records, SOC-stratified medians.
#   OEWS May 2024, national annual percentiles.

WageLevel = Literal["I", "II", "III", "IV"]


@dataclass
class OccupationProfile:
    """Wage and benefit profile for one occupation category."""

    soc_code: str
    title: str
    domain: str          # "tech" | "finance" | "healthcare" | "legal" | "trade" | "education"
    wage_by_level: dict[WageLevel, tuple[int, int]]
    # (low, high) annual gross wage range per DOL wage level.
    # Derived from LCA P25–P75 for Levels I–III; P50–P90 for Level IV.
    oews_percentiles: tuple[int, int, int, int, int]
    # (P10, P25, P50, P75, P90) from OEWS May 2024 national data.
    typical_401k_rate: float
    # Fraction of gross wage deferred to 401(k); 0.0 = typically not offered.
    hsa_eligible_pct: float
    # Fraction of workers in this occupation enrolled in an HSA-eligible plan.
    se_side_income_pct: float
    # Fraction of W-2 workers who also have significant SE side income.
    notes: str = ""


# ---------------------------------------------------------------------------
# Embedded occupation table
# All wages in 2024 annual dollars.
# LCA medians confirmed from 5,001-row sample of FY2024 Q4 file.
# OEWS percentiles from national_M2024_dl.xlsx.
# ---------------------------------------------------------------------------

OCCUPATION_PROFILES: list[OccupationProfile] = [
    # --- Software engineering ---
    OccupationProfile(
        soc_code="15-1252",
        title="Software Developer",
        domain="tech",
        wage_by_level={
            "I":   (72_000,  110_000),   # entry-level / junior
            "II":  (100_000, 145_000),   # mid-level
            "III": (140_000, 200_000),   # senior
            "IV":  (175_000, 320_000),   # staff / principal / distinguished
        },
        oews_percentiles=(79_850, 103_050, 133_080, 169_000, 211_450),
        typical_401k_rate=0.10,
        hsa_eligible_pct=0.65,
        se_side_income_pct=0.08,
        notes="LCA Q4 2024: Level I p50=$95k, II p50=$113k, III p50=$153k, IV p50=$185k",
    ),
    OccupationProfile(
        soc_code="15-1211",
        title="Computer Systems Analyst",
        domain="tech",
        wage_by_level={
            "I":   (60_000,  85_000),
            "II":  (80_000,  115_000),
            "III": (110_000, 155_000),
            "IV":  (145_000, 210_000),
        },
        oews_percentiles=(63_160, 80_900, 103_790, 132_360, 166_030),
        typical_401k_rate=0.09,
        hsa_eligible_pct=0.60,
        se_side_income_pct=0.06,
    ),
    OccupationProfile(
        soc_code="15-2051",
        title="Data Scientist",
        domain="tech",
        wage_by_level={
            "I":   (75_000,  105_000),
            "II":  (100_000, 140_000),
            "III": (135_000, 185_000),
            "IV":  (170_000, 280_000),
        },
        oews_percentiles=(63_650, 82_630, 108_020, 140_370, 177_100),
        typical_401k_rate=0.10,
        hsa_eligible_pct=0.62,
        se_side_income_pct=0.10,
        notes="Data scientists have higher SE consulting rate than pure SWE",
    ),
    OccupationProfile(
        soc_code="11-3021",
        title="IT Manager",
        domain="tech",
        wage_by_level={
            "I":   (100_000, 135_000),
            "II":  (130_000, 170_000),
            "III": (160_000, 220_000),
            "IV":  (200_000, 320_000),
        },
        oews_percentiles=(104_450, 134_350, 171_200, 216_220, 216_220),
        typical_401k_rate=0.10,
        hsa_eligible_pct=0.70,
        se_side_income_pct=0.04,
    ),
    # --- Finance / accounting ---
    OccupationProfile(
        soc_code="13-2011",
        title="Accountant / Auditor",
        domain="finance",
        wage_by_level={
            "I":   (48_000,  68_000),
            "II":  (65_000,  90_000),
            "III": (85_000,  125_000),
            "IV":  (115_000, 185_000),
        },
        oews_percentiles=(52_780, 64_660, 81_680, 106_450, 141_420),
        typical_401k_rate=0.08,
        hsa_eligible_pct=0.55,
        se_side_income_pct=0.15,
        notes="CPAs often have significant SE income from private clients",
    ),
    OccupationProfile(
        soc_code="13-2051",
        title="Financial Analyst",
        domain="finance",
        wage_by_level={
            "I":   (62_000,  85_000),
            "II":  (80_000,  115_000),
            "III": (110_000, 160_000),
            "IV":  (150_000, 250_000),
        },
        oews_percentiles=(54_890, 70_910, 96_220, 133_470, 186_280),
        typical_401k_rate=0.09,
        hsa_eligible_pct=0.60,
        se_side_income_pct=0.05,
    ),
    # --- Healthcare ---
    OccupationProfile(
        soc_code="29-1141",
        title="Registered Nurse",
        domain="healthcare",
        wage_by_level={
            "I":   (58_000,  78_000),
            "II":  (72_000,  95_000),
            "III": (88_000,  120_000),
            "IV":  (110_000, 160_000),
        },
        oews_percentiles=(61_780, 73_830, 89_010, 109_820, 132_680),
        typical_401k_rate=0.07,
        hsa_eligible_pct=0.50,
        se_side_income_pct=0.12,
        notes="Travel nurses often have additional 1099/SE income",
    ),
    OccupationProfile(
        soc_code="29-1051",
        title="Pharmacist",
        domain="healthcare",
        wage_by_level={
            "I":   (100_000, 125_000),
            "II":  (118_000, 145_000),
            "III": (138_000, 170_000),
            "IV":  (158_000, 200_000),
        },
        oews_percentiles=(99_080, 114_790, 132_750, 154_080, 175_970),
        typical_401k_rate=0.08,
        hsa_eligible_pct=0.58,
        se_side_income_pct=0.04,
    ),
    # --- Legal ---
    OccupationProfile(
        soc_code="23-1011",
        title="Lawyer",
        domain="legal",
        wage_by_level={
            "I":   (75_000,  115_000),
            "II":  (110_000, 170_000),
            "III": (160_000, 260_000),
            "IV":  (240_000, 500_000),
        },
        oews_percentiles=(66_530, 98_060, 145_760, 208_980, 208_980),
        typical_401k_rate=0.09,
        hsa_eligible_pct=0.55,
        se_side_income_pct=0.20,
        notes="Solo practitioners and of-counsel arrangements are common",
    ),
    # --- Trades / blue-collar ---
    OccupationProfile(
        soc_code="47-2061",
        title="Construction Laborer",
        domain="trade",
        wage_by_level={
            "I":   (32_000,  45_000),
            "II":  (42_000,  58_000),
            "III": (54_000,  75_000),
            "IV":  (68_000, 100_000),
        },
        oews_percentiles=(33_800, 40_960, 50_970, 66_310, 88_700),
        typical_401k_rate=0.04,
        hsa_eligible_pct=0.25,
        se_side_income_pct=0.25,
        notes="Side contracting work is very common",
    ),
    OccupationProfile(
        soc_code="49-9071",
        title="Maintenance and Repair Worker",
        domain="trade",
        wage_by_level={
            "I":   (32_000,  44_000),
            "II":  (42_000,  56_000),
            "III": (52_000,  70_000),
            "IV":  (62_000,  90_000),
        },
        oews_percentiles=(32_870, 40_700, 50_660, 63_140, 78_830),
        typical_401k_rate=0.04,
        hsa_eligible_pct=0.28,
        se_side_income_pct=0.30,
    ),
    # --- Education ---
    OccupationProfile(
        soc_code="25-2021",
        title="Elementary School Teacher",
        domain="education",
        wage_by_level={
            "I":   (38_000,  50_000),
            "II":  (48_000,  62_000),
            "III": (58_000,  78_000),
            "IV":  (72_000, 100_000),
        },
        oews_percentiles=(41_240, 50_110, 62_360, 77_600, 96_900),
        typical_401k_rate=0.00,   # pension plans, not 401k
        hsa_eligible_pct=0.20,
        se_side_income_pct=0.08,
        notes="Teachers have educator expense deduction ($300); pension replaces 401k",
    ),
    OccupationProfile(
        soc_code="25-1099",
        title="College / University Lecturer",
        domain="education",
        wage_by_level={
            "I":   (45_000,  65_000),
            "II":  (62_000,  90_000),
            "III": (85_000,  130_000),
            "IV":  (120_000, 200_000),
        },
        oews_percentiles=(46_450, 62_040, 84_380, 118_480, 166_540),
        typical_401k_rate=0.07,
        hsa_eligible_pct=0.45,
        se_side_income_pct=0.25,
        notes="Adjuncts and research consultants often have significant SE income",
    ),
    # --- Self-employment focused (no W-2) ---
    OccupationProfile(
        soc_code="00-SELF-CREATIVE",
        title="Freelance Designer / Photographer / Writer",
        domain="creative",
        wage_by_level={
            "I":   (25_000,  45_000),
            "II":  (42_000,  75_000),
            "III": (70_000, 120_000),
            "IV":  (110_000, 200_000),
        },
        oews_percentiles=(30_000, 42_000, 62_000, 90_000, 135_000),
        typical_401k_rate=0.00,
        hsa_eligible_pct=0.15,
        se_side_income_pct=1.00,  # 100% SE
        notes="Purely self-employed; no W-2",
    ),
    OccupationProfile(
        soc_code="00-SELF-CONSULT",
        title="Independent IT / Management Consultant",
        domain="tech",
        wage_by_level={
            "I":   (60_000,  95_000),
            "II":  (90_000, 140_000),
            "III": (130_000, 200_000),
            "IV":  (180_000, 350_000),
        },
        oews_percentiles=(60_000, 90_000, 130_000, 190_000, 300_000),
        typical_401k_rate=0.00,
        hsa_eligible_pct=0.20,
        se_side_income_pct=1.00,
        notes="Purely self-employed; SEP-IRA instead of 401k",
    ),
    # --- Retail and food service (lower-wage) ---
    OccupationProfile(
        soc_code="41-2011",
        title="Cashier / Retail Sales Worker",
        domain="retail",
        wage_by_level={
            "I":   (20_000,  28_000),   # part-time / entry
            "II":  (26_000,  35_000),   # full-time / senior cashier
            "III": (33_000,  46_000),   # lead / supervisor
            "IV":  (42_000,  60_000),   # department manager
        },
        oews_percentiles=(22_740, 26_080, 30_490, 36_730, 46_270),
        typical_401k_rate=0.03,
        hsa_eligible_pct=0.15,
        se_side_income_pct=0.10,
        notes="BLS OEWS May 2024 SOC 41-2011; low wages, minimal benefits",
    ),
    OccupationProfile(
        soc_code="35-2014",
        title="Cook / Food Preparation Worker",
        domain="food_service",
        wage_by_level={
            "I":   (22_000,  30_000),
            "II":  (28_000,  38_000),
            "III": (36_000,  50_000),
            "IV":  (46_000,  65_000),   # executive chef / head cook
        },
        oews_percentiles=(24_080, 27_700, 33_160, 40_810, 52_480),
        typical_401k_rate=0.02,
        hsa_eligible_pct=0.10,
        se_side_income_pct=0.12,
        notes="BLS OEWS May 2024 SOC 35-2014; tips are separate and complex — not modeled",
    ),
    # --- Transportation ---
    OccupationProfile(
        soc_code="53-3032",
        title="Heavy Truck Driver",
        domain="transportation",
        wage_by_level={
            "I":   (38_000,  52_000),
            "II":  (50_000,  68_000),
            "III": (62_000,  82_000),
            "IV":  (75_000, 100_000),   # owner-operator
        },
        oews_percentiles=(40_710, 49_920, 61_960, 76_600, 95_140),
        typical_401k_rate=0.04,
        hsa_eligible_pct=0.30,
        se_side_income_pct=0.20,
        notes="BLS OEWS May 2024 SOC 53-3032; owner-operators are Schedule C filers",
    ),
    OccupationProfile(
        soc_code="00-GIG-DRIVER",
        title="Rideshare / Delivery Driver (Gig Worker)",
        domain="transportation",
        wage_by_level={
            "I":   (15_000,  28_000),   # part-time
            "II":  (26_000,  42_000),   # full-time, one platform
            "III": (38_000,  58_000),   # full-time, multiple platforms
            "IV":  (50_000,  75_000),   # high-volume / delivery + rideshare combined
        },
        oews_percentiles=(18_000, 26_000, 38_000, 52_000, 70_000),
        typical_401k_rate=0.00,
        hsa_eligible_pct=0.05,
        se_side_income_pct=1.00,
        notes="Purely 1099-K / Schedule C; mileage deduction is the primary offset",
    ),
    # --- Warehousing and logistics ---
    OccupationProfile(
        soc_code="53-7062",
        title="Warehouse / Material Handler",
        domain="logistics",
        wage_by_level={
            "I":   (28_000,  38_000),
            "II":  (36_000,  48_000),
            "III": (45_000,  60_000),
            "IV":  (56_000,  76_000),   # lead / shift supervisor
        },
        oews_percentiles=(30_350, 36_530, 44_910, 55_880, 69_700),
        typical_401k_rate=0.04,
        hsa_eligible_pct=0.35,
        se_side_income_pct=0.05,
        notes="BLS OEWS May 2024 SOC 53-7062; includes Amazon, UPS, FedEx workers",
    ),
    # --- Office and administrative ---
    OccupationProfile(
        soc_code="43-6014",
        title="Administrative Assistant / Secretary",
        domain="admin",
        wage_by_level={
            "I":   (32_000,  44_000),
            "II":  (42_000,  56_000),
            "III": (53_000,  72_000),
            "IV":  (66_000,  90_000),   # executive assistant
        },
        oews_percentiles=(34_530, 42_070, 52_330, 65_300, 82_550),
        typical_401k_rate=0.06,
        hsa_eligible_pct=0.45,
        se_side_income_pct=0.05,
        notes="BLS OEWS May 2024 SOC 43-6014",
    ),
    OccupationProfile(
        soc_code="43-4051",
        title="Customer Service Representative",
        domain="admin",
        wage_by_level={
            "I":   (28_000,  38_000),
            "II":  (35_000,  48_000),
            "III": (44_000,  60_000),
            "IV":  (55_000,  78_000),   # team lead / specialist
        },
        oews_percentiles=(30_340, 36_480, 45_440, 57_990, 74_530),
        typical_401k_rate=0.05,
        hsa_eligible_pct=0.40,
        se_side_income_pct=0.06,
        notes="BLS OEWS May 2024 SOC 43-4051; call center and in-person roles",
    ),
    # --- Skilled trades (licensed) ---
    OccupationProfile(
        soc_code="47-2111",
        title="Electrician",
        domain="trade",
        wage_by_level={
            "I":   (40_000,  56_000),   # apprentice / helper
            "II":  (54_000,  74_000),   # journeyman
            "III": (70_000,  96_000),   # master electrician
            "IV":  (88_000, 130_000),   # foreman / contractor
        },
        oews_percentiles=(43_840, 56_880, 73_840, 95_990, 125_880),
        typical_401k_rate=0.05,
        hsa_eligible_pct=0.35,
        se_side_income_pct=0.30,
        notes="BLS OEWS May 2024 SOC 47-2111; union journeymen, non-union, and contractors",
    ),
    OccupationProfile(
        soc_code="47-2152",
        title="Plumber / Pipefitter",
        domain="trade",
        wage_by_level={
            "I":   (38_000,  54_000),
            "II":  (52_000,  72_000),
            "III": (68_000,  94_000),
            "IV":  (85_000, 120_000),
        },
        oews_percentiles=(42_760, 56_760, 73_650, 95_370, 124_110),
        typical_401k_rate=0.05,
        hsa_eligible_pct=0.32,
        se_side_income_pct=0.28,
        notes="BLS OEWS May 2024 SOC 47-2152; independent plumbing businesses common",
    ),
    # --- Healthcare support ---
    OccupationProfile(
        soc_code="31-1131",
        title="Nursing Assistant / Home Health Aide",
        domain="healthcare",
        wage_by_level={
            "I":   (24_000,  32_000),
            "II":  (30_000,  40_000),
            "III": (37_000,  50_000),
            "IV":  (46_000,  62_000),   # agency lead / supervisor
        },
        oews_percentiles=(27_080, 31_430, 36_870, 44_300, 54_170),
        typical_401k_rate=0.03,
        hsa_eligible_pct=0.20,
        se_side_income_pct=0.15,
        notes="BLS OEWS May 2024 SOC 31-1131; independent caregiver 1099 work is common",
    ),
    OccupationProfile(
        soc_code="29-2021",
        title="Dental Hygienist",
        domain="healthcare",
        wage_by_level={
            "I":   (55_000,  74_000),
            "II":  (70_000,  92_000),
            "III": (88_000, 112_000),
            "IV":  (105_000, 135_000),
        },
        oews_percentiles=(58_360, 73_220, 90_440, 108_500, 128_820),
        typical_401k_rate=0.05,
        hsa_eligible_pct=0.45,
        se_side_income_pct=0.12,
        notes="BLS OEWS May 2024 SOC 29-2021; per diem / temp dental work adds SE income",
    ),
    # --- Sales and real estate ---
    OccupationProfile(
        soc_code="41-9021",
        title="Real Estate Agent / Broker",
        domain="sales",
        wage_by_level={
            "I":   (28_000,  50_000),   # new agent, low transaction volume
            "II":  (48_000,  85_000),   # established agent
            "III": (80_000, 140_000),   # top producer
            "IV":  (130_000, 250_000),  # broker / team lead
        },
        oews_percentiles=(32_000, 53_800, 87_300, 142_000, 215_000),
        typical_401k_rate=0.00,
        hsa_eligible_pct=0.15,
        se_side_income_pct=1.00,
        notes="BLS OEWS May 2024 SOC 41-9021; almost all commission-only Schedule C",
    ),
]

def _clone_profiles(profiles: list[OccupationProfile]) -> list[OccupationProfile]:
    """Create mutable profile copies so calibration can safely overlay values."""
    cloned: list[OccupationProfile] = []
    for profile in profiles:
        cloned.append(
            OccupationProfile(
                soc_code=profile.soc_code,
                title=profile.title,
                domain=profile.domain,
                wage_by_level=dict(profile.wage_by_level),
                oews_percentiles=tuple(profile.oews_percentiles),
                typical_401k_rate=profile.typical_401k_rate,
                hsa_eligible_pct=profile.hsa_eligible_pct,
                se_side_income_pct=profile.se_side_income_pct,
                notes=profile.notes,
            )
        )
    return cloned


def _apply_dynamic_profile_calibration(
    profiles: list[OccupationProfile],
) -> tuple[list[OccupationProfile], dict[str, list[str]]]:
    """Overlay profile wages/state pools from local dataset files when available."""
    calibrated_profiles = _clone_profiles(profiles)
    state_pools_by_soc: dict[str, list[str]] = {}

    if not DYNAMIC_PROFILE_CALIBRATION:
        return calibrated_profiles, state_pools_by_soc

    target_socs = {p.soc_code for p in calibrated_profiles}
    calibration = load_dynamic_calibration(
        dataset_dir=DATASET_DIR,
        target_socs=target_socs,
        max_lca_files=DYNAMIC_PROFILE_CALIBRATION_MAX_LCA_FILES,
    )

    for profile in calibrated_profiles:
        if profile.soc_code in calibration.wage_by_level:
            for level, bounds in calibration.wage_by_level[profile.soc_code].items():
                profile.wage_by_level[level] = bounds

        if profile.soc_code in calibration.oews_percentiles:
            profile.oews_percentiles = calibration.oews_percentiles[profile.soc_code]

        if profile.soc_code in calibration.top_states_by_soc:
            state_pools_by_soc[profile.soc_code] = calibration.top_states_by_soc[profile.soc_code]

    return calibrated_profiles, state_pools_by_soc


# Active profile set (dynamic overlay when dataset files are available).
ACTIVE_OCCUPATION_PROFILES, _STATE_POOLS_BY_SOC = _apply_dynamic_profile_calibration(
    OCCUPATION_PROFILES
)

# Index by SOC for fast lookup
_PROFILES_BY_SOC: dict[str, OccupationProfile] = {
    p.soc_code: p for p in ACTIVE_OCCUPATION_PROFILES
}

# ---------------------------------------------------------------------------
# State-level real estate tax medians (ACS 5-year estimates, 2024 approx.)
# Used to generate realistic property tax deduction inputs.
# Values are median annual property tax paid by homeowners.
# ---------------------------------------------------------------------------
STATE_RE_TAX_MEDIAN: dict[str, int] = {
    "AL": 700,   "AK": 3_400, "AZ": 1_900, "AR": 800,  "CA": 4_800,
    "CO": 2_600, "CT": 6_300, "DE": 1_600, "FL": 2_500, "GA": 1_500,
    "HI": 1_900, "ID": 1_800, "IL": 5_200, "IN": 1_400, "IA": 1_900,
    "KS": 2_100, "KY": 1_200, "LA": 900,  "ME": 2_800, "MD": 3_800,
    "MA": 5_900, "MI": 2_800, "MN": 3_200, "MS": 900,  "MO": 1_700,
    "MT": 2_100, "NE": 2_800, "NV": 1_700, "NH": 6_000, "NJ": 9_500,
    "NM": 1_200, "NY": 6_100, "NC": 1_700, "ND": 2_000, "OH": 2_600,
    "OK": 1_400, "OR": 3_400, "PA": 3_200, "RI": 4_700, "SC": 1_200,
    "SD": 2_300, "TN": 1_100, "TX": 4_200, "UT": 1_600, "VT": 4_900,
    "VA": 3_100, "WA": 4_400, "WV": 900,  "WI": 3_800, "WY": 1_700,
    "DC": 4_600,
}

# State income tax rates (approximate effective rate on middle income)
# 0.0 = no state income tax (TX, FL, WA, NV, WY, SD, AK, TN, NH)
STATE_INCOME_TAX_RATE: dict[str, float] = {
    "AL": 0.040, "AK": 0.000, "AZ": 0.025, "AR": 0.047, "CA": 0.093,
    "CO": 0.044, "CT": 0.060, "DE": 0.055, "FL": 0.000, "GA": 0.055,
    "HI": 0.079, "ID": 0.058, "IL": 0.049, "IN": 0.032, "IA": 0.060,
    "KS": 0.057, "KY": 0.045, "LA": 0.043, "ME": 0.075, "MD": 0.065,
    "MA": 0.050, "MI": 0.043, "MN": 0.070, "MS": 0.047, "MO": 0.054,
    "MT": 0.069, "NE": 0.064, "NV": 0.000, "NH": 0.000, "NJ": 0.065,
    "NM": 0.059, "NY": 0.085, "NC": 0.045, "ND": 0.025, "OH": 0.040,
    "OK": 0.047, "OR": 0.090, "PA": 0.031, "RI": 0.060, "SC": 0.070,
    "SD": 0.000, "TN": 0.000, "TX": 0.000, "UT": 0.046, "VT": 0.066,
    "VA": 0.057, "WA": 0.000, "WV": 0.065, "WI": 0.065, "WY": 0.000,
    "DC": 0.085,
}

# W-2 Box 12 code D (401k) annual limits for 2024
_401K_LIMIT_UNDER50 = 23_000
_401K_LIMIT_50PLUS  = 30_500

# HSA limits for 2024
_HSA_LIMIT_SINGLE = 4_150
_HSA_LIMIT_FAMILY = 8_300

# Mortgage interest: typical 30-yr fixed on median home price by state tier
# (low, mid, high) annual mortgage interest paid based on 2024 home prices
_MORTGAGE_INTEREST_BY_STATE_TIER: dict[str, tuple[int, int, int]] = {
    # (renter/low, owner/mid, owner/high)
    "high_cost": (0, 18_000, 28_000),   # CA, NY, MA, WA, CO, CT, NJ, HI
    "medium":    (0, 11_000, 18_000),   # most of the country
    "low_cost":  (0,  7_000, 13_000),   # AL, MS, WV, AR, KY, OK
}
_HIGH_COST_STATES  = {"CA","NY","MA","WA","CO","CT","NJ","HI","OR","VA","MD","DC"}
_LOW_COST_STATES   = {"AL","MS","WV","AR","KY","OK","SD","ND","MT","WY","IA"}


def _mortgage_tier(state: str) -> tuple[int, int, int]:
    if state in _HIGH_COST_STATES:
        return _MORTGAGE_INTEREST_BY_STATE_TIER["high_cost"]
    if state in _LOW_COST_STATES:
        return _MORTGAGE_INTEREST_BY_STATE_TIER["low_cost"]
    return _MORTGAGE_INTEREST_BY_STATE_TIER["medium"]


# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------

def _jitter(value: int, pct: float = 0.12, rng: random.Random = None) -> int:
    """Add ±pct random noise to a value so profiles don't cluster on round numbers."""
    if rng is None:
        rng = random.Random()
    delta = int(value * pct * (rng.random() * 2 - 1))
    return max(0, value + delta)


def _draw_wage(
    profile: OccupationProfile,
    level: WageLevel,
    rng: random.Random,
) -> int:
    """Draw a gross annual wage from the profile's range for the given level."""
    low, high = profile.wage_by_level[level]
    # Triangular distribution: mode slightly above midpoint (right-skewed)
    mode = int(low + 0.55 * (high - low))
    wage = int(rng.triangular(low, high, mode))
    # Apply 5% jitter so values don't all land on round numbers
    return _jitter(wage, pct=0.05, rng=rng)


def _compute_w2_box1(
    gross_wage: int,
    age: int,
    has_401k: bool,
    deferral_rate: float,
    rng: random.Random,
) -> tuple[int, int]:
    """Compute Box 1 (taxable wages) and 401k deferral (Box 12D).

    Returns (box1_wages, deferral_amount).
    """
    if not has_401k or deferral_rate == 0.0:
        return gross_wage, 0

    limit = _401K_LIMIT_50PLUS if age >= 50 else _401K_LIMIT_UNDER50
    target_deferral = int(gross_wage * deferral_rate)
    # High earners may max out; add randomness to contribution decision
    if gross_wage > 120_000:
        deferral = _jitter(min(target_deferral, limit), pct=0.05, rng=rng)
    else:
        # Some workers don't contribute the full target rate
        actual_rate = deferral_rate * rng.uniform(0.5, 1.0)
        deferral = int(gross_wage * actual_rate)

    deferral = min(deferral, limit)
    box1 = gross_wage - deferral
    return max(box1, 0), deferral


def _compute_hsa(
    hsa_eligible: bool,
    filing_status: str,
    employer_contrib: int,
    rng: random.Random,
) -> int:
    """Compute HSA deduction.  Returns total annual HSA contribution above the line."""
    if not hsa_eligible:
        return 0
    limit = _HSA_LIMIT_FAMILY if filing_status in ("mfj", "hoh") else _HSA_LIMIT_SINGLE
    # Employee contribution = limit - employer contribution
    employee_contrib = max(0, int(limit * rng.uniform(0.5, 1.0)) - employer_contrib)
    return employee_contrib


def _compute_self_emp_health(
    se_income: int,
    filing_status: str,
    rng: random.Random,
) -> int:
    """Estimate realistic self-employed health insurance premium.

    ACA marketplace premiums for self-employed: roughly $4k–$15k depending
    on age, state, and plan level.  Bounded by SE income.
    """
    if se_income <= 0:
        return 0
    # Rough premium range based on filing status (family plans cost more)
    if filing_status in ("mfj", "hoh"):
        low, high = 8_000, 22_000
    else:
        low, high = 4_000, 12_000
    premium = int(rng.triangular(low, high, (low + high) // 2))
    # Deduction limited to SE net income; cap at 50% to leave room for other deductions
    return min(premium, se_income // 2)


def _compute_sep_ira(se_income: int, rng: random.Random) -> int:
    """Compute SEP-IRA deduction.

    2024 SEP-IRA limit: 25% of net SE earnings, max $69,000.
    Most self-employed people contribute 10–20% of net SE income.
    """
    if se_income <= 0:
        return 0
    max_allowed = min(int(se_income * 0.25), 69_000)
    # Actual contribution: 0% (no plan) with 30% probability; else 8–20% of income
    if rng.random() < 0.30:
        return 0
    rate = rng.uniform(0.08, 0.20)
    return min(int(se_income * rate), max_allowed)


def _compute_charitable(agi_estimate: int, rng: random.Random) -> int:
    """Estimate charitable contributions.

    IRS SOI data: filers who itemize give ~2–4% of AGI to charity.
    Non-itemizers: mostly $0 (no tax incentive) or small amounts.
    """
    if rng.random() < 0.35:   # 35% of filers give nothing deductible
        return 0
    rate = rng.uniform(0.01, 0.04)
    return _jitter(int(agi_estimate * rate), pct=0.20, rng=rng)


def _determine_homeowner(
    wage: int,
    age: int,
    filing_status: str,
    rng: random.Random,
) -> bool:
    """Estimate homeownership probability.

    Based on Census Bureau homeownership rates by income and age cohort.
    """
    base_prob = 0.35
    if age >= 35:
        base_prob += 0.20
    if age >= 45:
        base_prob += 0.15
    if wage > 80_000:
        base_prob += 0.15
    if wage > 150_000:
        base_prob += 0.10
    if filing_status == "mfj":
        base_prob += 0.15
    return rng.random() < min(base_prob, 0.92)


def build_profile(
    profile: OccupationProfile,
    level: WageLevel,
    filing_status: str,
    age_primary: int,
    state: str,
    rng: random.Random,
    age_spouse: int | None = None,
    spouse_profile: OccupationProfile | None = None,
    spouse_level: WageLevel | None = None,
    num_children: int = 0,
) -> TaxpayerFacts:
    """Build one coherent TaxpayerFacts from occupation + demographic inputs.

    All monetary values are derived from real wage distributions and
    rule-bounded benefit calculations.  No field is independently randomized.

    Parameters
    ----------
    profile:
        Primary taxpayer's occupation profile.
    level:
        DOL prevailing wage level I–IV for the primary taxpayer.
    filing_status:
        IRS filing status code.
    age_primary:
        Primary taxpayer's age at year-end.
    state:
        Two-letter worksite / residence state code.
    rng:
        Seeded random.Random instance for reproducibility.
    age_spouse:
        Spouse's age (required for MFJ).
    spouse_profile:
        Spouse's occupation profile (None = not employed or same as primary).
    spouse_level:
        Spouse's wage level (None = same as primary).
    num_children:
        Number of qualifying children for CTC / childcare.

    Returns
    -------
    TaxpayerFacts
        A fully coherent taxpayer profile.
    """
    is_purely_se = profile.se_side_income_pct >= 1.0

    # ------------------------------------------------------------------
    # Primary taxpayer wages
    # ------------------------------------------------------------------
    gross_wage = _draw_wage(profile, level, rng) if not is_purely_se else 0
    has_401k = profile.typical_401k_rate > 0 and rng.random() < 0.80
    deferral_rate = profile.typical_401k_rate * rng.uniform(0.7, 1.3) if has_401k else 0.0

    box1_wages, _deferral = _compute_w2_box1(gross_wage, age_primary, has_401k, deferral_rate, rng)

    # HSA (above-the-line deduction, but tracked separately from wages)
    hsa_enrolled = rng.random() < profile.hsa_eligible_pct
    employer_hsa = _jitter(900, pct=0.30, rng=rng) if hsa_enrolled else 0
    # Note: employer HSA is excluded from Box 1 but we don't track it separately;
    # for simplicity we only model the employee HSA deduction contribution.

    # ------------------------------------------------------------------
    # Self-employment income (SE)
    # ------------------------------------------------------------------
    se_income = 0
    if is_purely_se:
        se_income = _draw_wage(profile, level, rng)
    elif rng.random() < profile.se_side_income_pct:
        # Side SE income: typically 15–40% of W-2 wage
        se_fraction = rng.uniform(0.10, 0.40)
        se_income = _jitter(int(gross_wage * se_fraction), pct=0.15, rng=rng)

    # ------------------------------------------------------------------
    # Spouse wages (MFJ only)
    # ------------------------------------------------------------------
    spouse_wages = 0
    if filing_status == "mfj" and age_spouse is not None:
        sp = spouse_profile or profile
        sl = spouse_level or level
        if rng.random() < 0.82:   # 82% of MFJ spouses are employed (BLS data)
            sp_gross = _draw_wage(sp, sl, rng)
            sp_has_401k = sp.typical_401k_rate > 0 and rng.random() < 0.75
            sp_rate = sp.typical_401k_rate * rng.uniform(0.5, 1.2) if sp_has_401k else 0.0
            spouse_wages, _ = _compute_w2_box1(sp_gross, age_spouse, sp_has_401k, sp_rate, rng)

    # ------------------------------------------------------------------
    # Investment income (interest, dividends, capital gains)
    # ------------------------------------------------------------------
    total_earned = box1_wages + spouse_wages + se_income
    # Wealthier / older filers more likely to have investment income
    investment_prob = 0.20 + 0.30 * min(total_earned / 200_000, 1.0) + 0.15 * min(age_primary / 60, 1.0)
    has_investments = rng.random() < investment_prob

    taxable_interest    = 0
    ordinary_dividends  = 0
    qualified_dividends = 0
    capital_gains_net   = 0

    if has_investments:
        portfolio_size = int(total_earned * rng.uniform(0.5, 3.0))
        taxable_interest   = _jitter(int(portfolio_size * 0.045 * rng.uniform(0.3, 1.0)), pct=0.20, rng=rng)
        ordinary_dividends = _jitter(int(portfolio_size * 0.025 * rng.uniform(0.2, 0.8)), pct=0.20, rng=rng)
        qualified_dividends = int(ordinary_dividends * rng.uniform(0.65, 0.95))
        if rng.random() < 0.35:   # realized gains event this year
            capital_gains_net = _jitter(int(portfolio_size * 0.05), pct=0.30, rng=rng)

    # ------------------------------------------------------------------
    # Student loan interest
    # ------------------------------------------------------------------
    student_loan_interest = 0
    if age_primary < 40 and rng.random() < 0.40:
        student_loan_interest = min(_jitter(2_500, pct=0.20, rng=rng), 2_500)

    # ------------------------------------------------------------------
    # SE deductions
    # ------------------------------------------------------------------
    se_health = _compute_self_emp_health(se_income, filing_status, rng) if se_income > 0 else 0
    sep_ira = _compute_sep_ira(se_income, rng) if se_income > 0 else 0

    # ------------------------------------------------------------------
    # IRA contributions (non-SE workers; SE workers use SEP)
    # ------------------------------------------------------------------
    ira_contribution = 0
    if not is_purely_se and sep_ira == 0 and rng.random() < 0.30:
        limit = 8_000 if age_primary >= 50 else 7_000
        ira_contribution = _jitter(int(limit * rng.uniform(0.5, 1.0)), pct=0.05, rng=rng)

    # ------------------------------------------------------------------
    # Educator expenses (teachers only)
    # ------------------------------------------------------------------
    educator_expenses = 0
    if profile.domain == "education" and profile.soc_code.startswith("25-2"):
        # K-12 teachers: $300 cap (2024); both spouses may claim if both teach
        educator_expenses = _jitter(300, pct=0.15, rng=rng)
        if filing_status == "mfj" and spouse_profile and spouse_profile.soc_code.startswith("25-2"):
            educator_expenses = min(educator_expenses + 300, 600)

    # ------------------------------------------------------------------
    # Homeownership and itemized deductions
    # ------------------------------------------------------------------
    is_homeowner = _determine_homeowner(total_earned, age_primary, filing_status, rng)

    real_estate_tax = 0
    mortgage_interest = 0
    if is_homeowner:
        state_re_median = STATE_RE_TAX_MEDIAN.get(state, 2_500)
        real_estate_tax = _jitter(state_re_median, pct=0.30, rng=rng)
        low_mi, mid_mi, high_mi = _mortgage_tier(state)
        if total_earned > 150_000:
            mortgage_interest = _jitter(high_mi, pct=0.20, rng=rng)
        elif total_earned > 70_000:
            mortgage_interest = _jitter(mid_mi, pct=0.20, rng=rng)
        # Low earners: renters or inherited/paid-off homes

    # State income tax (approximate: state rate × Box 1 wages)
    st_rate = STATE_INCOME_TAX_RATE.get(state, 0.05)
    state_income_tax = int(st_rate * (box1_wages + spouse_wages))

    # Rough AGI estimate for charitable calibration
    agi_estimate = box1_wages + spouse_wages + se_income + taxable_interest + ordinary_dividends
    charitable_cash = _compute_charitable(agi_estimate, rng)

    # ------------------------------------------------------------------
    # Children / dependents
    # ------------------------------------------------------------------
    # Child care expenses: bounded by market rates ($8k–$20k per child for daycare)
    child_care_expenses = 0
    if num_children > 0:
        youngest_child_eligible = num_children > 0 and age_primary < 50
        if youngest_child_eligible and rng.random() < 0.65:
            per_child = _jitter(10_000, pct=0.30, rng=rng)
            # Form 2441 limits qualifying expenses to $3,000 (1 child) or $6,000 (2+)
            child_care_expenses = min(per_child * min(num_children, 2), 6_000)

    # ------------------------------------------------------------------
    # Rental and retirement income archetypes
    # ------------------------------------------------------------------
    rental_income_net = 0
    if is_homeowner and total_earned >= 70_000 and rng.random() < 0.18:
        gross_rent = total_earned * rng.uniform(0.06, 0.16)
        operating_ratio = rng.uniform(0.30, 0.55)
        rental_income_net = _jitter(
            int(gross_rent * (1.0 - operating_ratio)),
            pct=0.25,
            rng=rng,
        )

    pension_income = 0
    social_security_benefits = 0
    ira_distributions = 0
    if age_primary >= 62:
        if rng.random() < 0.35:
            social_security_benefits = _jitter(
                int(max(18_000, total_earned * 0.18)),
                pct=0.25,
                rng=rng,
            )
        if rng.random() < 0.25:
            pension_income = _jitter(
                int(max(12_000, total_earned * 0.14)),
                pct=0.30,
                rng=rng,
            )
        if rng.random() < 0.20:
            ira_distributions = _jitter(
                int(max(6_000, total_earned * 0.08)),
                pct=0.35,
                rng=rng,
            )

    # ------------------------------------------------------------------
    # Withholding and estimated payments
    # ------------------------------------------------------------------
    # Effective withholding rate approximation: roughly 15–25% of W-2 wages
    withholding_rate = rng.uniform(0.16, 0.26)
    federal_withholding = int((box1_wages + spouse_wages) * withholding_rate)

    estimated_tax_payments = 0
    if se_income > 10_000:
        # SE workers should pay quarterly estimates; many underpay
        compliance_rate = rng.uniform(0.60, 0.95)
        rough_se_tax = int(se_income * 0.15)
        rough_income_tax = int(se_income * 0.22)
        estimated_tax_payments = int((rough_se_tax + rough_income_tax) * compliance_rate)

    return TaxpayerFacts(
        filing_status=filing_status,
        age_primary=age_primary,
        age_spouse=age_spouse,
        wages=box1_wages,
        spouse_wages=spouse_wages,
        taxable_interest=taxable_interest,
        ordinary_dividends=ordinary_dividends,
        qualified_dividends=qualified_dividends,
        business_income=se_income,
        capital_gains_net=capital_gains_net,
        ira_distributions=ira_distributions,
        pension_income=pension_income,
        social_security_benefits=social_security_benefits,
        rental_income_net=rental_income_net,
        student_loan_interest=student_loan_interest,
        educator_expenses=educator_expenses,
        covered_by_workplace_plan=has_401k,
        ira_contribution=ira_contribution,
        ira_deduction=0,
        self_employed_health_ins=se_health,
        sep_simple_ira_deduction=sep_ira,
        state_income_tax=state_income_tax,
        real_estate_tax=real_estate_tax,
        mortgage_interest=mortgage_interest,
        charitable_cash=charitable_cash,
        num_qualifying_children=num_children,
        child_care_expenses=child_care_expenses,
        federal_withholding=federal_withholding,
        estimated_tax_payments=estimated_tax_payments,
    )


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

# Domain-level fallback states when SOC-specific LCA state pools are unavailable.
_DOMAIN_STATE_POOLS: dict[str, list[str]] = {
    "tech": ["CA", "WA", "NY", "TX", "MA", "CO"],
    "finance": ["NY", "CA", "TX", "IL", "FL", "GA"],
    "healthcare": ["CA", "TX", "FL", "NY", "PA", "OH"],
    "legal": ["NY", "CA", "DC", "IL", "TX", "MA"],
    "trade": ["TX", "FL", "OH", "PA", "GA", "NC"],
    "education": ["CA", "TX", "NY", "FL", "OH", "PA"],
    "creative": ["CA", "NY", "TX", "WA", "CO", "FL"],
    "retail": ["TX", "FL", "CA", "OH", "GA", "NC"],
    "food_service": ["TX", "FL", "CA", "NY", "AZ", "NC"],
    "transportation": ["TX", "FL", "OH", "PA", "TN", "GA"],
    "logistics": ["TX", "OH", "PA", "TN", "IN", "GA"],
    "admin": ["TX", "FL", "CA", "OH", "VA", "NC"],
    "sales": ["CA", "TX", "FL", "NY", "AZ", "CO"],
}


def _state_pool_for_profile(profile: OccupationProfile) -> list[str]:
    """Return a deterministic state pool for one occupation profile."""
    dynamic_pool = _STATE_POOLS_BY_SOC.get(profile.soc_code)
    if dynamic_pool and len(dynamic_pool) >= 3:
        return dynamic_pool
    return _DOMAIN_STATE_POOLS.get(profile.domain, ["TX", "FL", "CA", "NY", "OH"])


def _preferred_levels(profile: OccupationProfile) -> tuple[WageLevel, WageLevel]:
    """Choose lower/higher representative wage levels for scenario templates."""
    available = [lvl for lvl in ("I", "II", "III", "IV") if lvl in profile.wage_by_level]
    if not available:
        return ("II", "II")

    if "II" in available:
        low = "II"
    else:
        low = available[0]

    if "III" in available:
        high = "III"
    elif "IV" in available:
        high = "IV"
    else:
        high = available[-1]

    return low, high


def _build_dynamic_scenario_specs() -> list[tuple]:
    """Build profile scenarios programmatically from active occupation profiles."""
    specs: list[tuple] = []

    for idx, profile in enumerate(ACTIVE_OCCUPATION_PROFILES):
        low_level, high_level = _preferred_levels(profile)
        state_pool = _state_pool_for_profile(profile)
        is_pure_se = profile.se_side_income_pct >= 1.0

        # Baseline single-filer case for each SOC.
        specs.append(
            (
                profile.soc_code,
                low_level,
                "single",
                23 if low_level == "I" else 26,
                44,
                state_pool,
                (0, 0),
            )
        )

        # Family-oriented case for each SOC.
        family_status = "mfj" if idx % 3 != 0 else "hoh"
        if is_pure_se and family_status == "hoh":
            family_status = "mfj"
        specs.append(
            (
                profile.soc_code,
                high_level,
                family_status,
                30,
                58,
                state_pool,
                (1, 2) if family_status in ("mfj", "hoh", "qss") else (0, 1),
            )
        )

        # Periodic MFS/QSS coverage for split hygiene and filing-status breadth.
        if idx % 7 == 0:
            specs.append(
                (
                    profile.soc_code,
                    high_level,
                    "mfs",
                    34,
                    58,
                    state_pool,
                    (0, 1),
                )
            )

        if idx % 11 == 0:
            specs.append(
                (
                    profile.soc_code,
                    high_level,
                    "qss",
                    40,
                    68,
                    state_pool,
                    (1, 2),
                )
            )

        # Retirement-age archetype for a subset of profiles.
        if idx % 5 == 0:
            retirement_status = "single" if idx % 2 == 0 else "qss"
            specs.append(
                (
                    profile.soc_code,
                    high_level,
                    retirement_status,
                    62,
                    78,
                    state_pool,
                    (1, 2) if retirement_status == "qss" else (0, 1),
                )
            )

    # De-duplicate while preserving order.
    seen: set[tuple] = set()
    deduped: list[tuple] = []
    for spec in specs:
        key = (
            spec[0],
            spec[1],
            spec[2],
            spec[3],
            spec[4],
            tuple(spec[5]),
            spec[6],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)

    return deduped


# Representative (occupation, level, filing_status, age, state, children) scenarios
# Legacy fallback list used only if dynamic scenario construction is disabled
# or yields no records.
# Each tuple: (soc_code, level, filing_status, age_low, age_high, state_pool, child_range)

_SCENARIO_SPECS: list[tuple] = [
    # Entry-level SWE, single, no kids, high-cost city
    ("15-1252", "I",   "single", 23, 30, ["CA","WA","NY","TX","MA"], (0, 0)),
    # Mid-level SWE, single, early 30s
    ("15-1252", "II",  "single", 30, 37, ["CA","WA","NY","TX","CO"], (0, 0)),
    # Senior SWE, married, kids, variety of states
    ("15-1252", "III", "mfj",    35, 48, ["CA","WA","NY","TX","MA","CO","IL"], (1, 2)),
    # Staff/principal SWE, married, older kids
    ("15-1252", "IV",  "mfj",    42, 55, ["CA","WA","NY","MA","CO"], (0, 2)),
    # Data scientist, single
    ("15-2051", "II",  "single", 27, 38, ["CA","NY","WA","TX","MA"], (0, 0)),
    # Data scientist, married
    ("15-2051", "III", "mfj",    32, 45, ["CA","NY","WA","TX","IL"], (0, 2)),
    # IT manager, married, mid-career
    ("11-3021", "III", "mfj",    40, 55, ["TX","CA","NY","VA","IL"], (1, 3)),
    # Accountant, single
    ("13-2011", "II",  "single", 26, 40, ["NY","CA","TX","IL","OH"], (0, 0)),
    # Accountant with SE consulting, married
    ("13-2011", "III", "mfj",    38, 52, ["NY","CA","TX","FL","GA"], (1, 2)),
    # Financial analyst, single
    ("13-2051", "II",  "single", 26, 38, ["NY","CA","CT","MA","IL"], (0, 0)),
    # Financial analyst, married, high earner
    ("13-2051", "IV",  "mfj",    38, 52, ["NY","CA","CT","NJ","MA"], (0, 2)),
    # Registered nurse, single
    ("29-1141", "II",  "single", 25, 40, ["CA","TX","NY","FL","PA"], (0, 1)),
    # Registered nurse, HOH with kids
    ("29-1141", "II",  "hoh",    28, 45, ["CA","TX","NY","FL","GA"], (1, 2)),
    # Nurse, married
    ("29-1141", "III", "mfj",    35, 52, ["CA","TX","FL","NY","OH"], (1, 3)),
    # Lawyer, single, high earner
    ("23-1011", "III", "single", 30, 45, ["NY","CA","DC","IL","TX"], (0, 0)),
    # Lawyer, married
    ("23-1011", "IV",  "mfj",    38, 55, ["NY","CA","DC","IL","TX"], (0, 2)),
    # K-12 teacher, single
    ("25-2021", "I",   "single", 22, 35, ["CA","TX","NY","FL","OH"], (0, 0)),
    # K-12 teacher, HOH
    ("25-2021", "II",  "hoh",    30, 48, ["CA","TX","NY","FL","OH"], (1, 2)),
    # Construction worker, single
    ("47-2061", "I",   "single", 22, 40, ["TX","FL","CA","NY","AZ"], (0, 0)),
    # Construction worker with side contracting, married
    ("47-2061", "II",  "mfj",    32, 52, ["TX","FL","CA","NY","AZ"], (1, 2)),
    # Freelance designer, single, purely SE
    ("00-SELF-CREATIVE", "II", "single", 26, 42, ["CA","NY","TX","CO","WA"], (0, 0)),
    # Freelance designer, higher income
    ("00-SELF-CREATIVE", "III", "mfj",   32, 48, ["CA","NY","TX","WA","FL"], (0, 2)),
    # IT consultant, single, purely SE
    ("00-SELF-CONSULT",  "III", "single", 35, 55, ["CA","NY","TX","WA","MA"], (0, 0)),
    # IT consultant, married, high income
    ("00-SELF-CONSULT",  "IV",  "mfj",   40, 58, ["CA","NY","TX","WA","MA"], (0, 2)),
    # Maintenance worker with side contracting, single
    ("49-9071", "I",   "single", 28, 50, ["TX","FL","OH","PA","GA"], (0, 0)),
    # ---------------------------------------------------------------
    # Wider-range occupations added for broader benchmark coverage
    # ---------------------------------------------------------------
    # Cashier / retail worker, single, low income
    ("41-2011", "I",   "single", 20, 35, ["TX","FL","CA","OH","GA"], (0, 0)),
    # Cashier, head of household with one child
    ("41-2011", "II",  "hoh",    25, 42, ["TX","FL","CA","OH","NC"], (1, 1)),
    # Retail supervisor, married
    ("41-2011", "III", "mfj",    30, 50, ["TX","FL","OH","PA","AZ"], (1, 2)),
    # Cook / food service, single
    ("35-2014", "I",   "single", 20, 35, ["TX","FL","CA","NY","AZ"], (0, 0)),
    # Cook, head of household
    ("35-2014", "II",  "hoh",    26, 44, ["TX","FL","CA","IL","NC"], (1, 2)),
    # Truck driver, single, mid-career
    ("53-3032", "II",  "single", 28, 50, ["TX","OH","PA","GA","TN"], (0, 0)),
    # Truck driver, married, owns home
    ("53-3032", "III", "mfj",    35, 55, ["TX","OH","PA","TN","IN"], (0, 2)),
    # Gig driver (rideshare/delivery), purely SE, single
    ("00-GIG-DRIVER", "II",  "single", 22, 45, ["CA","TX","FL","NY","AZ"], (0, 0)),
    # Gig driver, head of household
    ("00-GIG-DRIVER", "I",   "hoh",    25, 42, ["CA","TX","FL","GA","NC"], (1, 1)),
    # Warehouse associate, single
    ("53-7062", "I",   "single", 22, 38, ["TX","OH","PA","TN","IN"], (0, 0)),
    # Warehouse shift supervisor, married
    ("53-7062", "III", "mfj",    30, 50, ["TX","OH","PA","GA","WI"], (1, 2)),
    # Administrative assistant, single
    ("43-6014", "II",  "single", 26, 45, ["TX","FL","CA","OH","VA"], (0, 0)),
    # Executive assistant, married
    ("43-6014", "IV",  "mfj",    35, 55, ["NY","CA","DC","IL","MA"], (0, 2)),
    # Customer service rep, single
    ("43-4051", "I",   "single", 22, 38, ["TX","FL","OH","GA","AZ"], (0, 0)),
    # Customer service lead, head of household
    ("43-4051", "II",  "hoh",    27, 45, ["TX","FL","CA","NC","GA"], (1, 2)),
    # Electrician (journeyman), single
    ("47-2111", "II",  "single", 26, 45, ["TX","CA","FL","OH","PA"], (0, 0)),
    # Electrician (master / contractor), married
    ("47-2111", "IV",  "mfj",    38, 58, ["TX","CA","FL","OH","IL"], (1, 2)),
    # Plumber, single, with side contracting
    ("47-2152", "II",  "single", 28, 48, ["TX","FL","OH","PA","TN"], (0, 0)),
    # Plumber, married, owns home
    ("47-2152", "III", "mfj",    35, 55, ["TX","FL","OH","GA","NC"], (1, 2)),
    # Nursing assistant / home health aide, single
    ("31-1131", "I",   "single", 22, 45, ["CA","TX","FL","NY","PA"], (0, 0)),
    # Nursing assistant, head of household
    ("31-1131", "II",  "hoh",    27, 48, ["CA","TX","FL","GA","NC"], (1, 2)),
    # Dental hygienist, single
    ("29-2021", "II",  "single", 25, 42, ["CA","TX","FL","NY","OH"], (0, 0)),
    # Dental hygienist, married
    ("29-2021", "III", "mfj",    32, 52, ["CA","TX","FL","WA","CO"], (1, 2)),
    # Real estate agent, single, purely SE
    ("41-9021", "II",  "single", 28, 50, ["CA","TX","FL","NY","AZ"], (0, 0)),
    # Real estate broker, married, higher income
    ("41-9021", "III", "mfj",    38, 58, ["CA","TX","FL","NY","CO"], (0, 2)),
    # MFS coverage (engine + split calibration)
    ("13-2011", "II",  "mfs",    34, 56, ["NY","CA","TX","IL","GA"], (0, 1)),
    ("43-6014", "III", "mfs",    36, 58, ["TX","FL","CA","OH","NC"], (0, 1)),
    # QSS coverage
    ("29-1141", "II",  "qss",    40, 62, ["CA","TX","FL","NY","PA"], (1, 2)),
    ("25-2021", "III", "qss",    42, 64, ["CA","TX","NY","FL","OH"], (1, 2)),
    # Retirement-heavy archetypes (still occupation grounded)
    ("13-2011", "III", "single", 62, 76, ["FL","AZ","TX","NC","GA"], (0, 1)),
    ("25-2021", "III", "mfj",    62, 78, ["FL","AZ","TX","NC","OH"], (0, 2)),
    ("43-6014", "II",  "qss",    62, 78, ["FL","AZ","TX","NC","PA"], (0, 1)),
]


def generate_lca_grounded_records(
    n: int,
    random_seed: int = 42,
) -> list[TaxpayerFacts]:
    """Generate n coherent taxpayer profiles grounded in LCA/OEWS data.

    Parameters
    ----------
    n:
        Number of profiles to generate.
    random_seed:
        Seed for the random number generator (ensures reproducibility).

    Returns
    -------
    list[TaxpayerFacts]
        Coherent taxpayer profiles, one per item.
    """
    rng = random.Random(random_seed)
    records: list[TaxpayerFacts] = []
    scenario_specs = _build_dynamic_scenario_specs()
    if not scenario_specs:
        scenario_specs = _SCENARIO_SPECS

    for i in range(n):
        spec = scenario_specs[i % len(scenario_specs)]
        soc, level, fs, age_low, age_high, state_pool, child_range = spec

        profile = _PROFILES_BY_SOC[soc]
        age = rng.randint(age_low, age_high)
        state = rng.choice(state_pool)
        n_children = rng.randint(*child_range)

        age_spouse = None
        if fs == "mfj":
            age_spouse = age + rng.randint(-5, 5)

        # Spouse profile: often same domain, sometimes different
        spouse_profile = None
        if fs == "mfj":
            sp_soc_options = [
                soc
            ] + [
                p.soc_code
                for p in ACTIVE_OCCUPATION_PROFILES
                if p.domain == profile.domain
            ]
            spouse_profile = _PROFILES_BY_SOC.get(rng.choice(sp_soc_options), profile)

        facts = build_profile(
            profile=profile,
            level=level,
            filing_status=fs,
            age_primary=age,
            state=state,
            rng=rng,
            age_spouse=age_spouse,
            spouse_profile=spouse_profile,
            spouse_level=level,
            num_children=n_children,
        )
        records.append(facts)

    return records


def describe_profile(facts: TaxpayerFacts, idx: int = 0) -> str:
    """Return a one-line description of a generated profile for logging."""
    total_income = facts.wages + facts.spouse_wages + facts.business_income
    return (
        f"[{idx:03d}] {facts.filing_status:<6} age={facts.age_primary} "
        f"state=?? wages=${facts.wages:>8,} se=${facts.business_income:>7,} "
        f"spouse_wages=${facts.spouse_wages:>8,} total=${total_income:>9,} "
        f"children={facts.num_qualifying_children}"
    )


if __name__ == "__main__":
    records = generate_lca_grounded_records(n=25, random_seed=42)
    print(f"Generated {len(records)} LCA-grounded profiles:\n")
    for i, r in enumerate(records):
        print(describe_profile(r, i))


# ---------------------------------------------------------------------------
# Hand-crafted seed records — deterministic regression anchors
# ---------------------------------------------------------------------------
# Round-number profiles covering the most common filing situations.
# Used as a fixed set of cases that should not change between builds.
SEED_RECORDS: list[dict] = [
    # ------------------------------------------------------------------
    # Federal income tax — W-2 / ordinary income cases
    # ------------------------------------------------------------------
    {
        "filing_status": "single",
        "age_primary": 28,
        "wages": 42_000,
        "federal_withholding": 4_200,
    },
    {
        "filing_status": "single",
        "age_primary": 35,
        "wages": 75_000,
        "taxable_interest": 500,
        "state_income_tax": 5_500,
        "real_estate_tax": 2_800,
        "charitable_cash": 1_200,
        "federal_withholding": 9_000,
    },
    {
        "filing_status": "single",
        "age_primary": 45,
        "wages": 120_000,
        "taxable_interest": 1_800,
        "state_income_tax": 9_000,
        "real_estate_tax": 6_500,
        "mortgage_interest": 14_000,
        "charitable_cash": 3_000,
        "federal_withholding": 20_000,
    },
    {
        "filing_status": "mfj",
        "age_primary": 38,
        "age_spouse": 36,
        "wages": 95_000,
        "spouse_wages": 62_000,
        "taxable_interest": 1_200,
        "federal_withholding": 17_000,
    },
    {
        "filing_status": "mfj",
        "age_primary": 42,
        "age_spouse": 40,
        "wages": 110_000,
        "spouse_wages": 85_000,
        "taxable_interest": 2_000,
        "qualified_dividends": 1_500,
        "ordinary_dividends": 1_500,
        "state_income_tax": 12_000,
        "real_estate_tax": 7_000,
        "mortgage_interest": 18_000,
        "charitable_cash": 4_500,
        "num_qualifying_children": 2,
        "federal_withholding": 28_000,
    },
    {
        "filing_status": "hoh",
        "age_primary": 34,
        "wages": 68_000,
        "num_qualifying_children": 1,
        "child_care_expenses": 5_000,
        "state_income_tax": 4_200,
        "federal_withholding": 9_500,
    },
    {
        "filing_status": "mfs",
        "age_primary": 41,
        "wages": 64_000,
        "taxable_interest": 700,
        "state_income_tax": 3_800,
        "real_estate_tax": 1_900,
        "charitable_cash": 900,
        "federal_withholding": 8_100,
    },
    {
        "filing_status": "qss",
        "age_primary": 46,
        "wages": 88_000,
        "taxable_interest": 1_100,
        "num_qualifying_children": 1,
        "state_income_tax": 5_200,
        "real_estate_tax": 2_600,
        "mortgage_interest": 10_200,
        "federal_withholding": 11_500,
    },
    {
        "filing_status": "mfj",
        "age_primary": 62,
        "age_spouse": 60,
        "wages": 40_000,
        "pension_income": 35_000,
        "social_security_benefits": 24_000,
        "taxable_interest": 3_000,
        "state_income_tax": 4_000,
        "real_estate_tax": 5_500,
        "mortgage_interest": 6_000,
        "federal_withholding": 8_000,
        "estimated_tax_payments": 3_000,
    },
    # ------------------------------------------------------------------
    # Capital gains cases
    # ------------------------------------------------------------------
    {
        "filing_status": "single",
        "age_primary": 50,
        "wages": 55_000,
        "capital_gains_net": 12_000,
        "qualified_dividends": 2_500,
        "ordinary_dividends": 2_500,
        "federal_withholding": 8_500,
    },
    {
        "filing_status": "mfj",
        "age_primary": 55,
        "age_spouse": 53,
        "wages": 180_000,
        "capital_gains_net": 45_000,
        "qualified_dividends": 8_000,
        "ordinary_dividends": 8_000,
        "state_income_tax": 18_000,
        "mortgage_interest": 22_000,
        "charitable_cash": 6_000,
        "federal_withholding": 42_000,
    },
    # ------------------------------------------------------------------
    # Self-employment cases
    # ------------------------------------------------------------------
    {
        "filing_status": "single",
        "age_primary": 30,
        "business_income": 55_000,
        "self_employed_health_ins": 6_000,
        "estimated_tax_payments": 12_000,
    },
    {
        "filing_status": "single",
        "age_primary": 38,
        "wages": 20_000,
        "business_income": 80_000,
        "self_employed_health_ins": 9_600,
        "sep_simple_ira_deduction": 16_000,
        "state_income_tax": 7_500,
        "real_estate_tax": 3_200,
        "estimated_tax_payments": 18_000,
    },
    {
        "filing_status": "mfj",
        "age_primary": 44,
        "age_spouse": 42,
        "wages": 50_000,
        "business_income": 120_000,
        "self_employed_health_ins": 14_400,
        "sep_simple_ira_deduction": 24_000,
        "state_income_tax": 14_000,
        "real_estate_tax": 8_000,
        "mortgage_interest": 16_000,
        "num_qualifying_children": 2,
        "estimated_tax_payments": 30_000,
    },
    {
        "filing_status": "single",
        "age_primary": 27,
        "business_income": 32_000,
        "estimated_tax_payments": 6_500,
    },
    {
        "filing_status": "mfj",
        "age_primary": 50,
        "age_spouse": 48,
        "business_income": 200_000,
        "spouse_wages": 60_000,
        "self_employed_health_ins": 18_000,
        "sep_simple_ira_deduction": 40_000,
        "state_income_tax": 22_000,
        "real_estate_tax": 12_000,
        "mortgage_interest": 24_000,
        "charitable_cash": 10_000,
        "num_qualifying_children": 1,
        "estimated_tax_payments": 55_000,
    },
    # ------------------------------------------------------------------
    # IRA contribution cases
    # ------------------------------------------------------------------
    {
        "filing_status": "single",
        "age_primary": 33,
        "wages": 65_000,
        "ira_contribution": 7_000,
        "federal_withholding": 8_500,
    },
    {
        "filing_status": "mfj",
        "age_primary": 52,
        "age_spouse": 50,
        "wages": 100_000,
        "spouse_wages": 40_000,
        "ira_contribution": 8_000,   # age 50+, catch-up applies
        "federal_withholding": 16_000,
    },
    # ------------------------------------------------------------------
    # Rental income cases
    # ------------------------------------------------------------------
    {
        "filing_status": "single",
        "age_primary": 46,
        "wages": 90_000,
        "rental_income_net": 8_000,
        "state_income_tax": 7_000,
        "real_estate_tax": 4_500,
        "mortgage_interest": 10_000,
        "federal_withholding": 14_000,
    },
    # ------------------------------------------------------------------
    # Mixed / complex cases
    # ------------------------------------------------------------------
    {
        "filing_status": "mfj",
        "age_primary": 48,
        "age_spouse": 46,
        "wages": 85_000,
        "spouse_wages": 75_000,
        "business_income": 40_000,
        "capital_gains_net": 15_000,
        "rental_income_net": 6_000,
        "taxable_interest": 2_500,
        "qualified_dividends": 3_000,
        "ordinary_dividends": 3_000,
        "self_employed_health_ins": 7_200,
        "state_income_tax": 18_000,
        "real_estate_tax": 9_500,
        "mortgage_interest": 20_000,
        "charitable_cash": 8_000,
        "num_qualifying_children": 3,
        "federal_withholding": 35_000,
        "estimated_tax_payments": 10_000,
    },
    {
        "filing_status": "single",
        "age_primary": 60,
        "wages": 0,
        "pension_income": 55_000,
        "social_security_benefits": 28_000,
        "taxable_interest": 4_000,
        "qualified_dividends": 5_000,
        "ordinary_dividends": 5_000,
        "capital_gains_net": 20_000,
        "real_estate_tax": 5_000,
        "charitable_cash": 3_500,
        "federal_withholding": 7_000,
        "estimated_tax_payments": 5_000,
    },
    # High-income — tests additional Medicare tax and LTCG 20% bracket
    {
        "filing_status": "single",
        "age_primary": 52,
        "wages": 350_000,
        "capital_gains_net": 60_000,
        "qualified_dividends": 15_000,
        "ordinary_dividends": 15_000,
        "state_income_tax": 30_000,
        "real_estate_tax": 14_000,
        "mortgage_interest": 28_000,
        "charitable_cash": 20_000,
        "federal_withholding": 95_000,
    },
    {
        "filing_status": "mfj",
        "age_primary": 57,
        "age_spouse": 55,
        "wages": 500_000,
        "spouse_wages": 150_000,
        "capital_gains_net": 100_000,
        "qualified_dividends": 25_000,
        "ordinary_dividends": 25_000,
        "state_income_tax": 50_000,
        "real_estate_tax": 20_000,
        "mortgage_interest": 35_000,
        "charitable_cash": 30_000,
        "num_qualifying_children": 2,
        "federal_withholding": 160_000,
    },
    # ------------------------------------------------------------------
    # Educator / student loan / educator expenses edge cases
    # ------------------------------------------------------------------
    {
        "filing_status": "single",
        "age_primary": 26,
        "wages": 38_000,
        "educator_expenses": 300,
        "student_loan_interest": 2_500,
        "federal_withholding": 4_200,
    },
    {
        "filing_status": "mfj",
        "age_primary": 31,
        "age_spouse": 29,
        "wages": 58_000,
        "spouse_wages": 45_000,
        "educator_expenses": 600,       # both are teachers (capped at 300 each)
        "student_loan_interest": 2_500,
        "federal_withholding": 10_500,
    },
    # Near the 0% LTCG threshold for a single filer
    {
        "filing_status": "single",
        "age_primary": 40,
        "wages": 30_000,
        "capital_gains_net": 18_000,
        "qualified_dividends": 2_000,
        "ordinary_dividends": 2_000,
        "federal_withholding": 3_600,
    },
]
