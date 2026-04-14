"""Configuration for the synthetic tax benchmark generation pipeline.

All parameters that affect dataset shape, tax law values, or file locations
are controlled from this module. Change a value here to propagate it across
the full pipeline without touching individual modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = ROOT / "output"

# ---------------------------------------------------------------------------
# Data-driven profile calibration
# ---------------------------------------------------------------------------
# Profile generation can dynamically calibrate SOC wage ranges and state pools
# from local dataset files (LCA + OEWS). If disabled, embedded profile values
# in sources/profile_builder.py are used as deterministic defaults.
DYNAMIC_PROFILE_CALIBRATION = True
DYNAMIC_PROFILE_CALIBRATION_MAX_LCA_FILES = 2

# ---------------------------------------------------------------------------
# Tax year
# ---------------------------------------------------------------------------
TAX_YEAR = 2024

# ---------------------------------------------------------------------------
# Dataset size targets
# ---------------------------------------------------------------------------
N_TIER1_CASES = 50   # Numeric computation (deterministic engine)
N_TIER2_CASES = 40   # Statutory entailment (Yes / No)
N_TIER3_CASES = 60   # Q&A, MCQ, and scenario (IRS mined + hand-crafted)

# ---------------------------------------------------------------------------
# IRS XML sources to mine for worked examples
# Keys must match directory names in KNOWLEDGE_DIR.
# ---------------------------------------------------------------------------
IRS_EXAMPLE_SOURCES = [
    "p17",
    "p505",
    "p560",
    "i1040sc",
    "i1040sse",
    "i1040sd",
    "i8829",
    "i8995",
    "p526",
    "p503",
]

# Heading patterns (lowercase) that signal a worked example
EXAMPLE_HEADING_PATTERNS = [
    "example",
    "example.",
    "illustration",
    "illustration.",
    "sample",
]

# Minimum and maximum text length for a mined example to be usable
EXAMPLE_MIN_CHARS = 80
EXAMPLE_MAX_CHARS = 1200

# ---------------------------------------------------------------------------
# 2024 Federal income tax parameters
# All dollar thresholds are for Tax Year 2024, not adjusted.
# Sources: IRS Rev. Proc. 2023-34 and IRS Publication 17 (2024 edition).
# ---------------------------------------------------------------------------

# Standard deductions by filing status
STANDARD_DEDUCTION: dict[str, int] = {
    "single": 14_600,
    "mfj": 29_200,
    "mfs": 14_600,
    "hoh": 21_900,
    "qss": 29_200,
}

# Marginal tax brackets: list of (upper_bound_inclusive, rate) tuples.
# The last bracket uses float("inf") as the upper bound.
# IRC §1 (as adjusted by Rev. Proc. 2023-34).
TAX_BRACKETS: dict[str, list[tuple[float, float]]] = {
    "single": [
        (11_600,       0.10),
        (47_150,       0.12),
        (100_525,      0.22),
        (191_950,      0.24),
        (243_725,      0.32),
        (609_350,      0.35),
        (float("inf"), 0.37),
    ],
    "mfj": [
        (23_200,       0.10),
        (94_300,       0.12),
        (201_050,      0.22),
        (383_900,      0.24),
        (487_450,      0.32),
        (731_200,      0.35),
        (float("inf"), 0.37),
    ],
    "mfs": [
        (11_600,       0.10),
        (47_150,       0.12),
        (100_525,      0.22),
        (191_950,      0.24),
        (243_725,      0.32),
        (365_600,      0.35),
        (float("inf"), 0.37),
    ],
    "hoh": [
        (16_550,       0.10),
        (63_100,       0.12),
        (100_500,      0.22),
        (191_950,      0.24),
        (243_700,      0.32),
        (609_350,      0.35),
        (float("inf"), 0.37),
    ],
}
# Qualified surviving spouse uses MFJ brackets
TAX_BRACKETS["qss"] = TAX_BRACKETS["mfj"]

# LTCG / QD preferential rate 0% thresholds (IRC §1(h))
LTCG_0_THRESHOLD: dict[str, int] = {
    "single": 47_025,
    "mfj":    94_050,
    "mfs":    47_025,
    "hoh":    63_000,
    "qss":    94_050,
}
LTCG_15_THRESHOLD: dict[str, int] = {
    "single": 518_900,
    "mfj":    583_750,
    "mfs":    291_875,
    "hoh":    551_350,
    "qss":    583_750,
}

# SALT deduction cap (IRC §164(b)(6))
SALT_CAP = 10_000

# Child Tax Credit (IRC §24)
CTC_PER_CHILD = 2_000
CTC_PHASEOUT_THRESHOLD: dict[str, int] = {
    "single": 200_000,
    "mfj":    400_000,
    "mfs":    200_000,
    "hoh":    200_000,
    "qss":    400_000,
}

# Self-employment tax (IRC §1401)
SE_TAX_RATE_SS   = 0.124   # Social Security portion
SE_TAX_RATE_MC   = 0.029   # Medicare portion
SE_TAX_RATE      = SE_TAX_RATE_SS + SE_TAX_RATE_MC   # 15.3%
SE_WAGE_BASE     = 168_600  # SS wage base for 2024

# Net Investment Income Tax (NIIT, IRC §1411)
NIIT_RATE = 0.038

# Additional Medicare Tax on earned income (IRC §3101(b)(2) / §1401(b)(2))
ADDL_MEDICARE_RATE = 0.009
ADDL_MEDICARE_THRESHOLD: dict[str, int] = {
    "single": 200_000,
    "mfj":    250_000,
    "mfs":    125_000,
    "hoh":    200_000,
    "qss":    250_000,
}

# QBI deduction income threshold (IRC §199A)
QBI_THRESHOLD: dict[str, int] = {
    "single": 191_950,
    "mfj":    383_900,
    "mfs":    191_950,
    "hoh":    191_950,
    "qss":    383_900,
}
QBI_DEDUCTION_RATE = 0.20

# IRA contribution limit (IRC §219)
IRA_CONTRIBUTION_LIMIT = 7_000
IRA_CATCH_UP_LIMIT     = 8_000   # age 50+

# Traditional IRA deduction phaseout (MFJ, covered by workplace plan)
IRA_PHASEOUT_MFJ_LOW  = 123_000
IRA_PHASEOUT_MFJ_HIGH = 143_000
IRA_PHASEOUT_SINGLE_LOW  = 77_000
IRA_PHASEOUT_SINGLE_HIGH = 87_000
