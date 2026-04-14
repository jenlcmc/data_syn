"""Run one-command LLM evaluation on data_syn benchmark.json.

This script evaluates cases in data_syn/output/benchmark.json by:
  1. Building a style-specific prompt per case
  2. Calling an LLM for predictions
  3. Scoring predictions against case ground truth
  4. Writing per-case and aggregate metrics to a JSON report

Supported model families:
  - Claude models (model id starts with "claude")
  - Gemini models (model id starts with "gemini")

Examples
--------
    python data_syn/scripts/eval_llm.py
    python data_syn/scripts/eval_llm.py --model gemini --limit 25
    python data_syn/scripts/eval_llm.py --styles numeric entailment
    python data_syn/scripts/eval_llm.py --dry-run --limit 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from schema import BenchmarkCase, load_dataset
from scoring.scorer import (
    ScoreResult,
    aggregate,
    format_report,
    judge_prompt,
    score_entailment,
    score_mcq,
    score_numeric,
    score_text,
)
from src import config as cfg


_STYLE_CHOICES = ("numeric", "entailment", "qa", "mcq", "scenario")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM on data_syn benchmark.json and score outputs."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data_syn/output/benchmark.json"),
        help="Path to benchmark.json (default: data_syn/output/benchmark.json).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude",
        help=(
            "Model alias or full model id. Aliases: 'claude' -> "
            "src.config.CLAUDE_MODEL, 'gemini' -> src.config.GEMINI_MODEL."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "Optional model alias or full model id for Tier 3 qa/scenario judging. "
            "Defaults to --model."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_syn/output/llm_eval_results.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        choices=list(_STYLE_CHOICES),
        default=None,
        help="Optional subset of styles to evaluate.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Optional subset of tiers to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases after filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and use synthetic perfect predictions.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first model/scoring error instead of continuing.",
    )
    return parser.parse_args()


def _resolve_model_name(value: str) -> str:
    lowered = value.strip().lower()
    if lowered == "claude":
        return cfg.CLAUDE_MODEL
    if lowered == "gemini":
        return cfg.GEMINI_MODEL
    return value.strip()


def _ensure_keys(model: str, dry_run: bool) -> None:
    if dry_run:
        return

    if model.startswith("claude") and not cfg.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Set it in .env before running Claude evaluation."
        )

    if model.startswith("gemini") and not cfg.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in .env before running Gemini evaluation."
        )


def _build_case_prompt(case: BenchmarkCase) -> str:
    header = [
        "You are solving a US federal tax benchmark case for Tax Year 2024.",
        "Use only the taxpayer facts and question provided.",
    ]

    if case.style == "numeric":
        instructions = [
            "Return only one final dollar amount.",
            "Output format: $<amount>",
            "Do not include commentary.",
        ]
    elif case.style == "entailment":
        instructions = [
            "Answer with Yes or No.",
            "The first word must be exactly Yes or No.",
            "You may include one short sentence after that.",
        ]
    elif case.style == "mcq":
        instructions = [
            "Answer with one letter only: A, B, C, or D.",
            "Do not include explanation.",
        ]
    else:
        instructions = [
            "Provide a concise answer grounded in tax rules.",
            "If relevant, reference controlling IRC sections or IRS guidance.",
        ]

    sections: list[str] = ["\n".join(header), "Instructions:\n- " + "\n- ".join(instructions)]

    if case.facts_narrative.strip():
        sections.append(f"Taxpayer Facts:\n{case.facts_narrative.strip()}")

    if case.style == "mcq" and case.choices:
        ordered = [f"{key}. {value}" for key, value in sorted(case.choices.items())]
        sections.append("Choices:\n" + "\n".join(ordered))

    sections.append(f"Question:\n{case.question.strip()}")

    return "\n\n".join(sections)


def _call_model(prompt: str, model: str) -> str:
    if model.startswith("claude"):
        return _call_claude(prompt, model)
    if model.startswith("gemini"):
        return _call_gemini(prompt, model)
    raise ValueError(
        "Unsupported model id. Use a model that starts with 'claude' or 'gemini'. "
        f"Got: {model}"
    )


def _call_claude(prompt: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=model,
        temperature=0,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_gemini(prompt: str, model: str) -> str:
    try:
        from google import genai
    except ImportError:
        return _call_gemini_legacy(prompt, model)

    if not hasattr(genai, "Client"):
        return _call_gemini_legacy(prompt, model)

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


def _call_gemini_legacy(prompt: str, model: str) -> str:
    import google.generativeai as legacy_genai

    legacy_genai.configure(api_key=cfg.GEMINI_API_KEY)
    gemini_model = legacy_genai.GenerativeModel(model)
    response = gemini_model.generate_content(prompt)
    return response.text or ""


def _extract_json_object(text: str) -> dict | None:
    stripped = text.strip()

    # Direct JSON parse.
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fenced JSON block.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        block = fenced.group(1)
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # First JSON object decodable from any "{" position.
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        candidate = text[index:]
        try:
            parsed, _ = decoder.raw_decode(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def _dry_run_prediction(case: BenchmarkCase) -> str:
    if case.style == "numeric":
        return f"${case.ground_truth}"
    if case.style == "entailment":
        return str(case.ground_truth)
    if case.style == "mcq":
        return str(case.ground_truth)
    return str(case.ground_truth)


def _score_case(
    case: BenchmarkCase,
    prediction: str,
    judge_response: dict | None,
) -> ScoreResult:
    if case.style == "numeric":
        if not isinstance(case.ground_truth, (int, float)):
            raise ValueError(
                f"Case {case.id} is numeric but ground_truth is not numeric: {case.ground_truth!r}"
            )
        return score_numeric(
            case_id=case.id,
            prediction=prediction,
            ground_truth=float(case.ground_truth),
            tolerance_lenient_usd=case.tolerance_lenient_usd,
            tolerance_pct=case.tolerance_pct,
            tier=case.tier,
        )

    if case.style == "entailment":
        return score_entailment(
            case_id=case.id,
            prediction=prediction,
            ground_truth=str(case.ground_truth),
            tier=case.tier,
        )

    if case.style == "mcq":
        return score_mcq(
            case_id=case.id,
            prediction=prediction,
            ground_truth=str(case.ground_truth),
            tier=case.tier,
        )

    if case.style in ("qa", "scenario"):
        return score_text(
            case_id=case.id,
            prediction=prediction,
            ground_truth=str(case.ground_truth),
            judge_response=judge_response,
            tier=case.tier,
        )

    raise ValueError(f"Unsupported style for case {case.id}: {case.style}")


def _style_summary(score_results: list[ScoreResult]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[ScoreResult]] = {}
    for result in score_results:
        grouped.setdefault(result.style, []).append(result)

    summary: dict[str, dict[str, float | int]] = {}
    for style, values in sorted(grouped.items()):
        total = len(values)
        strict = sum(1 for value in values if value.correct_strict)
        lenient = sum(1 for value in values if value.correct_lenient)
        pct = sum(1 for value in values if value.correct_pct)
        errors = sum(1 for value in values if value.error_message)

        summary[style] = {
            "n": total,
            "strict_correct": strict,
            "lenient_correct": lenient,
            "pct_correct": pct,
            "errors": errors,
            "strict_accuracy": round(strict / total, 4) if total else 0.0,
            "lenient_accuracy": round(lenient / total, 4) if total else 0.0,
            "pct_accuracy": round(pct / total, 4) if total else 0.0,
        }

    return summary


def _empty_result_for_failure(case: BenchmarkCase, error: str) -> ScoreResult:
    return ScoreResult(
        case_id=case.id,
        style=case.style,
        tier=case.tier,
        prediction_raw="",
        ground_truth=case.ground_truth,
        error_message=error,
    )


def _filter_cases(
    cases: list[BenchmarkCase],
    styles: list[str] | None,
    tiers: list[int] | None,
    limit: int | None,
) -> list[BenchmarkCase]:
    filtered = cases

    if styles:
        style_set = set(styles)
        filtered = [case for case in filtered if case.style in style_set]

    if tiers:
        tier_set = set(tiers)
        filtered = [case for case in filtered if case.tier in tier_set]

    if limit is not None:
        filtered = filtered[:limit]

    return filtered


def main() -> int:
    args = _parse_args()

    model = _resolve_model_name(args.model)
    judge_model = _resolve_model_name(args.judge_model) if args.judge_model else model

    _ensure_keys(model, args.dry_run)
    _ensure_keys(judge_model, args.dry_run)

    print(f"Loading dataset: {args.dataset}")
    cases = load_dataset(args.dataset)
    selected_cases = _filter_cases(cases, args.styles, args.tiers, args.limit)

    if not selected_cases:
        print("No cases selected after applying filters.")
        return 1

    print(
        f"Evaluating {len(selected_cases)} case(s) with model={model} "
        f"(judge_model={judge_model}, dry_run={args.dry_run})"
    )

    per_case_payload: list[dict] = []
    score_results: list[ScoreResult] = []

    for index, case in enumerate(selected_cases, start=1):
        print(f"  [{index}/{len(selected_cases)}] {case.id} ({case.style})")

        prompt = _build_case_prompt(case)
        judge_raw = ""
        judge_response: dict | None = None

        try:
            if args.dry_run:
                prediction = _dry_run_prediction(case)
            else:
                prediction = _call_model(prompt, model)

            if case.style in ("qa", "scenario"):
                if args.dry_run:
                    judge_response = {
                        "factual_accuracy": 3,
                        "statutory_grounding": 3,
                        "reasoning_completeness": 3,
                        "clarity": 3,
                        "total": 12,
                        "pass": True,
                        "notes": "Dry run synthetic judge response.",
                    }
                    judge_raw = json.dumps(judge_response)
                else:
                    judge_input = judge_prompt(
                        ground_truth=str(case.ground_truth),
                        prediction=prediction,
                    )
                    judge_raw = _call_model(judge_input, judge_model)
                    judge_response = _extract_json_object(judge_raw)

            score = _score_case(case, prediction, judge_response)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            score = _empty_result_for_failure(case, error_msg)
            prediction = ""
            if args.fail_fast:
                raise

        score_results.append(score)

        per_case_payload.append(
            {
                "id": case.id,
                "tier": case.tier,
                "style": case.style,
                "domain": case.domain,
                "difficulty": case.difficulty,
                "question": case.question,
                "prediction": prediction,
                "score": asdict(score),
                "judge_response": judge_response,
                "judge_raw": judge_raw,
            }
        )

    stats = aggregate(score_results)
    style_breakdown = _style_summary(score_results)

    tier_counts = Counter(case.tier for case in selected_cases)
    style_counts = Counter(case.style for case in selected_cases)

    report_payload = {
        "metadata": {
            "dataset": str(args.dataset),
            "model": model,
            "judge_model": judge_model,
            "dry_run": args.dry_run,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_cases": len(selected_cases),
            "filters": {
                "styles": args.styles,
                "tiers": args.tiers,
                "limit": args.limit,
            },
        },
        "distribution": {
            "tier_counts": dict(sorted(tier_counts.items())),
            "style_counts": dict(sorted(style_counts.items())),
        },
        "aggregate": {
            "total": stats.total,
            "errors": stats.errors,
            "strict_correct": stats.strict_correct,
            "lenient_correct": stats.lenient_correct,
            "pct_correct": stats.pct_correct,
            "strict_accuracy": stats.strict_accuracy,
            "lenient_accuracy": stats.lenient_accuracy,
            "pct_accuracy": stats.pct_accuracy,
            "strict_accuracy_pct": round(100 * stats.strict_accuracy, 2),
            "lenient_accuracy_pct": round(100 * stats.lenient_accuracy, 2),
            "pct_accuracy_pct": round(100 * stats.pct_accuracy, 2),
            "style_breakdown": style_breakdown,
        },
        "cases": per_case_payload,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print("\n" + format_report(stats))
    print(f"Saved report: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
