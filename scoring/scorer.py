"""Scoring functions for the synthetic tax benchmark.

Each evaluation style has its own scoring function that mirrors the
conventions used by TaxCalcBench (strict/lenient exact match) and SARA
(10% numeric tolerance, string entailment).

Scoring conventions
-------------------
Tier 1 — numeric
    strict:  |prediction - ground_truth| == 0
    lenient: |prediction - ground_truth| <= tolerance_lenient_usd (default $5)
    pct:     |prediction - ground_truth| / |ground_truth| <= tolerance_pct

Tier 2 — entailment
    exact string match after normalisation (strip, lower, first word only).
    Prediction is correct if it starts with "yes" or "no" matching the label.

Tier 3 — Q&A / scenario
    Scored by an LLM judge (not deterministic).  This module provides
    the reference-answer format and a rubric for the judge prompt.

Tier 3 — MCQ
    Exact letter match ("A"–"D") after normalisation.

All scoring functions return a ``ScoreResult`` dataclass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ScoreResult:
    """Outcome of scoring one prediction against one benchmark case.

    Attributes
    ----------
    case_id:
        The benchmark case identifier.
    style:
        Evaluation style of the case.
    tier:
        Confidence tier of the ground truth.
    correct_strict:
        True when the prediction passes the strict criterion.
    correct_lenient:
        True when the prediction passes the lenient criterion
        (same as strict for non-numeric styles).
    correct_pct:
        True when the prediction falls within ``tolerance_pct`` of the
        ground truth (numeric only; same as strict for other styles).
    prediction_raw:
        The raw prediction string or number as supplied.
    prediction_parsed:
        The cleaned/parsed prediction value used for comparison.
    ground_truth:
        The ground truth value from the case.
    delta_usd:
        Absolute difference in USD (numeric cases only; None otherwise).
    error_message:
        Non-empty if the prediction could not be parsed or compared.
    """

    case_id: str = ""
    style: str = ""
    tier: int = 1
    correct_strict: bool = False
    correct_lenient: bool = False
    correct_pct: bool = False
    prediction_raw: str = ""
    prediction_parsed: float | str | None = None
    ground_truth: float | str | None = None
    delta_usd: float | None = None
    error_message: str = ""


# ---------------------------------------------------------------------------
# Numeric scoring (Tier 1)
# ---------------------------------------------------------------------------

_DOLLAR_RE = re.compile(
    r"\$?\s*([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)


def _parse_numeric(text: str) -> float | None:
    """Extract the first numeric dollar value from a string.

    Handles formats like ``"$12,345.67"``, ``"12345.67"``, ``"12,345"``.
    Returns None if no numeric value is found.
    """
    text = text.strip()
    match = _DOLLAR_RE.search(text)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def score_numeric(
    case_id: str,
    prediction: str,
    ground_truth: float,
    tolerance_lenient_usd: float = 5.0,
    tolerance_pct: float = 0.10,
    tier: int = 1,
) -> ScoreResult:
    """Score a numeric prediction against a dollar-amount ground truth.

    Parameters
    ----------
    case_id:
        Benchmark case identifier.
    prediction:
        Raw LLM output (free text; the first dollar value is extracted).
    ground_truth:
        Correct answer in USD.
    tolerance_lenient_usd:
        Maximum allowed absolute error for the lenient score.
    tolerance_pct:
        Maximum allowed relative error for the percentage score.
    tier:
        Ground truth confidence tier.

    Returns
    -------
    ScoreResult
        Populated score result.
    """
    parsed = _parse_numeric(prediction)
    result = ScoreResult(
        case_id=case_id,
        style="numeric",
        tier=tier,
        prediction_raw=prediction,
        ground_truth=ground_truth,
    )

    if parsed is None:
        result.error_message = (
            f"Could not parse a numeric value from prediction: {prediction!r}"
        )
        return result

    result.prediction_parsed = parsed
    delta = abs(parsed - ground_truth)
    result.delta_usd = round(delta, 2)

    result.correct_strict = delta == 0.0
    result.correct_lenient = delta <= tolerance_lenient_usd
    if ground_truth == 0.0:
        result.correct_pct = parsed == 0.0
    else:
        result.correct_pct = (delta / abs(ground_truth)) <= tolerance_pct

    return result


# ---------------------------------------------------------------------------
# Entailment scoring (Tier 2)
# ---------------------------------------------------------------------------

def score_entailment(
    case_id: str,
    prediction: str,
    ground_truth: str,
    tier: int = 2,
) -> ScoreResult:
    """Score a Yes/No entailment prediction.

    The prediction is correct if its first meaningful word (after stripping
    punctuation and whitespace) matches the ground truth label ("Yes" or "No"),
    case-insensitively.

    Parameters
    ----------
    case_id:
        Benchmark case identifier.
    prediction:
        Raw LLM output.
    ground_truth:
        "Yes" or "No".
    tier:
        Ground truth confidence tier.

    Returns
    -------
    ScoreResult
    """
    result = ScoreResult(
        case_id=case_id,
        style="entailment",
        tier=tier,
        prediction_raw=prediction,
        ground_truth=ground_truth,
    )

    # Extract the first word, stripping punctuation
    first_word = re.split(r"[\s,.!?;:\-]", prediction.strip())[0].lower()
    result.prediction_parsed = first_word

    expected = ground_truth.strip().lower()
    correct = first_word == expected

    result.correct_strict = correct
    result.correct_lenient = correct
    result.correct_pct = correct

    if first_word not in ("yes", "no"):
        result.error_message = (
            f"Prediction did not begin with 'Yes' or 'No'. "
            f"First word extracted: {first_word!r}."
        )

    return result


# ---------------------------------------------------------------------------
# MCQ scoring (Tier 3)
# ---------------------------------------------------------------------------

def score_mcq(
    case_id: str,
    prediction: str,
    ground_truth: str,
    tier: int = 3,
) -> ScoreResult:
    """Score a multiple-choice prediction.

    Extracts the answer letter (A–D) from the prediction string.  Accepts:
    - A bare letter:          "B"
    - A letter with period:   "B."
    - A letter with paren:    "B)"
    - A letter at start:      "B — the correct answer is ..."
    - Answer: prefix:         "Answer: B"

    Parameters
    ----------
    case_id:
        Benchmark case identifier.
    prediction:
        Raw LLM output.
    ground_truth:
        Correct answer key, one of "A", "B", "C", "D".
    tier:
        Ground truth confidence tier.

    Returns
    -------
    ScoreResult
    """
    result = ScoreResult(
        case_id=case_id,
        style="mcq",
        tier=tier,
        prediction_raw=prediction,
        ground_truth=ground_truth,
    )

    # Try to find a standalone answer letter
    patterns = [
        r"(?i)^answer\s*[:\-]?\s*([A-D])",         # "Answer: B"
        r"(?i)\b([A-D])[.)\s]",                     # "B." "B)" "B "
        r"(?i)^([A-D])$",                           # bare letter
    ]
    parsed_letter = None
    for pat in patterns:
        match = re.search(pat, prediction.strip())
        if match:
            parsed_letter = match.group(1).upper()
            break

    result.prediction_parsed = parsed_letter

    if parsed_letter is None:
        result.error_message = (
            f"Could not extract an answer letter (A–D) from: {prediction!r}"
        )
        return result

    correct = parsed_letter == ground_truth.upper()
    result.correct_strict = correct
    result.correct_lenient = correct
    result.correct_pct = correct

    return result


# ---------------------------------------------------------------------------
# Text / scenario scoring rubric (Tier 3 Q&A and scenario)
# ---------------------------------------------------------------------------

# This rubric is designed to be passed to an LLM judge.
JUDGE_RUBRIC = """You are evaluating an LLM response to a US federal tax question.

Reference answer:
{ground_truth}

LLM response:
{prediction}

Score the LLM response on the following dimensions (0–3 each):

1. Factual accuracy  — Does the response state the correct dollar amounts,
   tax rates, thresholds, and conclusions?  Penalise any arithmetic error or
   wrong figure.
2. Statutory grounding — Does the response cite the relevant IRC section(s)
   or IRS publication?
3. Reasoning completeness — Does the response show the key computation steps
   or rule application logic?
4. Clarity — Is the answer clear enough for a non-expert taxpayer to act on?

Return a JSON object:
{{
  "factual_accuracy": <0-3>,
  "statutory_grounding": <0-3>,
  "reasoning_completeness": <0-3>,
  "clarity": <0-3>,
  "total": <0-12>,
  "pass": <true if total >= 8 else false>,
  "notes": "<brief explanation of deductions>"
}}
"""


def judge_prompt(ground_truth: str, prediction: str) -> str:
    """Format the LLM judge prompt for a Q&A or scenario case.

    Parameters
    ----------
    ground_truth:
        The reference answer from the benchmark case.
    prediction:
        The LLM response to be scored.

    Returns
    -------
    str
        A prompt string ready to send to the judge LLM.
    """
    return JUDGE_RUBRIC.format(
        ground_truth=ground_truth.strip(),
        prediction=prediction.strip(),
    )


def score_text(
    case_id: str,
    prediction: str,
    ground_truth: str,
    judge_response: dict | None = None,
    tier: int = 3,
) -> ScoreResult:
    """Score a Q&A or scenario response using a pre-computed judge result.

    Parameters
    ----------
    case_id:
        Benchmark case identifier.
    prediction:
        Raw LLM response.
    ground_truth:
        Reference answer.
    judge_response:
        Parsed JSON from the LLM judge (keys: factual_accuracy, pass, etc.).
        If None, the result will have correct_strict = False and an
        error_message indicating the judge has not been run.
    tier:
        Ground truth confidence tier.

    Returns
    -------
    ScoreResult
    """
    result = ScoreResult(
        case_id=case_id,
        style="qa_or_scenario",
        tier=tier,
        prediction_raw=prediction,
        ground_truth=ground_truth,
    )

    if judge_response is None:
        result.error_message = "Judge not run. Call judge_prompt() and submit to an LLM."
        return result

    passed = bool(judge_response.get("pass", False))
    result.correct_strict = passed
    result.correct_lenient = passed
    result.correct_pct = passed
    result.prediction_parsed = judge_response.get("total")

    return result


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------

@dataclass
class AggregateStats:
    """Summary statistics across a list of ScoreResult objects."""

    total: int = 0
    strict_correct: int = 0
    lenient_correct: int = 0
    pct_correct: int = 0

    strict_accuracy: float = 0.0
    lenient_accuracy: float = 0.0
    pct_accuracy: float = 0.0

    by_tier: dict[int, dict] = field(default_factory=dict)
    by_style: dict[str, dict] = field(default_factory=dict)
    by_difficulty: dict[str, dict] = field(default_factory=dict)
    by_tag: dict[str, dict] = field(default_factory=dict)

    errors: int = 0


def aggregate(results: list[ScoreResult]) -> AggregateStats:
    """Compute aggregate accuracy statistics from a list of score results.

    Parameters
    ----------
    results:
        One ScoreResult per evaluated case.

    Returns
    -------
    AggregateStats
        Summary broken down by tier, style, difficulty, and tag.
    """
    stats = AggregateStats(total=len(results))

    for r in results:
        if r.error_message:
            stats.errors += 1

        if r.correct_strict:
            stats.strict_correct += 1
        if r.correct_lenient:
            stats.lenient_correct += 1
        if r.correct_pct:
            stats.pct_correct += 1

    if stats.total > 0:
        stats.strict_accuracy  = round(stats.strict_correct  / stats.total, 4)
        stats.lenient_accuracy = round(stats.lenient_correct / stats.total, 4)
        stats.pct_accuracy     = round(stats.pct_correct     / stats.total, 4)

    return stats


def format_report(stats: AggregateStats) -> str:
    """Return a plain-text summary report of aggregate scoring statistics.

    Parameters
    ----------
    stats:
        Computed aggregate statistics.

    Returns
    -------
    str
        Multi-line human-readable report.
    """
    lines = [
        "=" * 60,
        "Synthetic Tax Benchmark — Scoring Report",
        "=" * 60,
        f"Total cases evaluated : {stats.total}",
        f"Parse / format errors : {stats.errors}",
        "",
        "Accuracy",
        f"  Strict  (exact)         : {stats.strict_accuracy:.1%}  "
        f"({stats.strict_correct}/{stats.total})",
        f"  Lenient (±$5 or exact)  : {stats.lenient_accuracy:.1%}  "
        f"({stats.lenient_correct}/{stats.total})",
        f"  Pct     (±10% or exact) : {stats.pct_accuracy:.1%}  "
        f"({stats.pct_correct}/{stats.total})",
        "=" * 60,
    ]
    return "\n".join(lines)
