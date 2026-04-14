"""Unified schema for all synthetic tax benchmark cases.

Every case in the benchmark, regardless of tier or style, is represented
as a BenchmarkCase. The schema is designed to support:

  Tier 1 — Numeric computation  (style: "numeric")
  Tier 2 — Statutory entailment (style: "entailment")
  Tier 3 — Q&A, MCQ, scenario   (style: "qa" | "mcq" | "scenario")

Fields that do not apply to a given style are set to None.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

Domain  = Literal["federal_income_tax", "self_employment", "both"]
Style   = Literal["numeric", "entailment", "qa", "mcq", "scenario"]
Tier    = Literal[1, 2, 3]
Difficulty = Literal["basic", "intermediate", "advanced"]
ConfidenceTier = Literal["A", "B", "C"]
VerifiedBy = Literal[
    "python_engine",   # built-in deterministic Python tax engine (this codebase)
    "policyengine",    # PolicyEngine-US microsimulation
    "tax_calculator",  # PSLmodels Tax-Calculator
    "hand_crafted",    # reviewed and authored in this codebase
    "irs_publication", # IRS published worked example
    "manual_cpa",      # Reviewed by a tax professional
]


@dataclass
class TaxpayerFacts:
    """Structured financial profile of a synthetic taxpayer.

    All monetary values are in whole dollars for Tax Year 2024 unless
    otherwise noted. Fields default to zero (not applicable) when absent.
    """

    filing_status: Literal["single", "mfj", "mfs", "hoh", "qss"] = "single"
    age_primary: int = 40
    age_spouse: int | None = None

    # Income — ordinary
    wages: int = 0
    spouse_wages: int = 0
    taxable_interest: int = 0
    ordinary_dividends: int = 0
    qualified_dividends: int = 0
    state_refund: int = 0           # taxable portion if previously deducted
    alimony_received: int = 0
    business_income: int = 0        # Schedule C net profit (before SE deduction)
    capital_gains_net: int = 0      # net long-term capital gain (Schedule D line 15)
    ira_distributions: int = 0
    pension_income: int = 0
    social_security_benefits: int = 0
    rental_income_net: int = 0      # Schedule E net (passive rules not modeled)
    farm_income: int = 0
    other_income: int = 0

    # Above-the-line deductions
    educator_expenses: int = 0
    student_loan_interest: int = 0
    alimony_paid: int = 0
    covered_by_workplace_plan: bool = False
    ira_deduction: int = 0          # computed by engine if 0 and IRA contributed
    ira_contribution: int = 0       # actual amount contributed to traditional IRA
    self_employed_health_ins: int = 0
    sep_simple_ira_deduction: int = 0

    # Below-the-line deductions (itemized)
    medical_expenses: int = 0       # before 7.5% AGI floor
    state_income_tax: int = 0
    real_estate_tax: int = 0
    mortgage_interest: int = 0
    charitable_cash: int = 0
    charitable_noncash: int = 0
    casualty_losses: int = 0        # federally declared disaster only

    # Credits
    num_qualifying_children: int = 0
    child_care_expenses: int = 0
    child_care_provider_ein: str = ""
    education_credits: int = 0

    # Withholding and payments
    federal_withholding: int = 0
    estimated_tax_payments: int = 0


@dataclass
class BenchmarkCase:
    """One benchmark evaluation case for a tax LLM.

    Fields
    ------
    id
        Unique identifier in the format ``t{tier}_{style}_{seq:04d}``.
    tier
        Ground truth confidence level (1 = deterministic, 3 = IRS-sourced / open).
    domain
        Tax area covered by this case.
    style
        Evaluation format (see module docstring for details).
    difficulty
        Subjective difficulty of the reasoning required.
    tax_year
        The applicable tax year (default 2024).
    source
        Which tool or document produced this case.
    verified_by
        How the ground truth was verified.
    profile_id
        Stable profile identifier for split hygiene. Cases generated from the
        same taxpayer profile must share this ID.
    confidence_tier
        Operational confidence label:
          - A: deterministic engine + external engine agreement
          - B: deterministic engine only
          - C: text/scenario judged
    facts
        Structured taxpayer financial profile.
    facts_narrative
        The taxpayer situation written in plain English (the LLM prompt input).
    question
        The specific question the LLM must answer.
    choices
        MCQ answer choices keyed A–D; None for non-MCQ styles.
    ground_truth
        The correct answer. A number (numeric), "Yes"/"No" (entailment),
        letter key "A"–"D" (MCQ), or a text string (qa / scenario).
    ground_truth_type
        Data type of ground_truth: "numeric_usd" | "boolean_str" | "choice_key" | "text".
    tolerance_strict_usd
        Exact-match tolerance in USD; 0 = must be exact.
    tolerance_lenient_usd
        Lenient-match tolerance in USD (e.g., ±$5 as used by TaxCalcBench).
    tolerance_pct
        Percentage tolerance (e.g., 0.10 = ±10%, as used by SARA numeric).
    reasoning_steps
        Step-by-step derivation of the ground truth. Used for chain-of-thought
        evaluation and as the reference for Tier 3 reasoning judges.
    statutory_refs
        IRC sections and IRS publications that govern the correct answer.
    explanation
        Human-readable justification of the ground truth.
    tags
        Topic-level tags for stratified analysis (e.g., "standard_deduction").
    """

    # Identity
    id: str = ""
    tier: Tier = 1
    domain: Domain = "federal_income_tax"
    style: Style = "numeric"
    difficulty: Difficulty = "basic"
    tax_year: int = 2024
    source: str = "python_engine"
    verified_by: VerifiedBy = "python_engine"
    profile_id: str = ""
    confidence_tier: ConfidenceTier = "B"
    external_engine: str = ""
    external_delta_usd: float | None = None
    external_check_passed: bool | None = None

    # Input
    facts: TaxpayerFacts | None = None
    facts_narrative: str = ""
    question: str = ""
    choices: dict[str, str] | None = None

    # Ground truth
    ground_truth: float | str | None = None
    ground_truth_type: str = "numeric_usd"
    tolerance_strict_usd: float = 0.0
    tolerance_lenient_usd: float = 5.0
    tolerance_pct: float = 0.10

    # Reasoning and references
    reasoning_steps: list[str] = field(default_factory=list)
    statutory_refs: list[str] = field(default_factory=list)
    explanation: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of this case."""
        raw = asdict(self)
        # Convert None facts to empty dict for consistent JSON shape
        if raw["facts"] is None:
            raw["facts"] = {}
        return raw

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkCase":
        """Reconstruct a BenchmarkCase from a dict (e.g., loaded from JSON)."""
        facts_data = data.pop("facts", {})
        if facts_data:
            data["facts"] = TaxpayerFacts(**facts_data)
        return cls(**data)


def save_dataset(cases: list[BenchmarkCase], path) -> None:
    """Write a list of BenchmarkCase objects to a JSON file.

    Parameters
    ----------
    cases:
        List of benchmark cases to serialize.
    path:
        Destination file path (str or Path). Parent directory must exist.
    """
    records = [c.to_dict() for c in cases]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)


def load_dataset(path) -> list[BenchmarkCase]:
    """Load a benchmark dataset from a JSON file.

    Parameters
    ----------
    path:
        Source file path (str or Path).

    Returns
    -------
    list[BenchmarkCase]
        Deserialized benchmark cases.
    """
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    return [BenchmarkCase.from_dict(r) for r in records]
