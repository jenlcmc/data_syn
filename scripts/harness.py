"""Build and validate the locked IRS worked-example harness.

The harness is a deterministic snapshot of mined IRS examples used as a
release gate to detect miner regressions or source-format drift.

Usage
-----
    # Build (or refresh) the harness fixture
    python data_syn/scripts/harness.py build
    python data_syn/scripts/harness.py build --max-cases 40

    # Validate the harness against current knowledge/ XML
    python data_syn/scripts/harness.py run
    python data_syn/scripts/harness.py run --strict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sources.miner import IRS_EXAMPLE_SOURCES, mine_all_sources


_DEFAULT_HARNESS = Path("data_syn/fixtures/irs_locked_examples.json")


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _example_id(example: dict) -> str:
    signature = "|".join([
        str(example.get("source", "")),
        str(example.get("section_id", "")),
        _normalize(str(example.get("heading", ""))),
        _normalize(str(example.get("text", "")))[:220],
    ])
    return "irsx_" + hashlib.sha1(signature.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_harness(max_cases: int = 40) -> dict:
    """Mine current knowledge/ XML and return a locked harness dict."""
    examples = mine_all_sources(IRS_EXAMPLE_SOURCES)

    cleaned = [
        ex for ex in examples
        if ex.get("text") and ex.get("conclusion")
        and len(_normalize(ex["conclusion"])) >= 24
    ]
    cleaned.sort(key=lambda ex: (
        str(ex.get("source", "")),
        str(ex.get("section_id", "")),
        _normalize(str(ex.get("heading", ""))),
    ))

    locked = [
        {
            "example_id": _example_id(ex),
            "source": ex.get("source", ""),
            "source_label": ex.get("source_label", ""),
            "section_id": ex.get("section_id", ""),
            "heading": ex.get("heading", ""),
            "conclusion": ex.get("conclusion", ""),
            "text_prefix": str(ex.get("text", ""))[:220],
        }
        for ex in cleaned[:max_cases]
    ]

    return {
        "description": "Locked IRS worked-example harness.",
        "source_count": len(IRS_EXAMPLE_SOURCES),
        "max_cases": max_cases,
        "locked_examples": locked,
    }


# ---------------------------------------------------------------------------
# Run (validate)
# ---------------------------------------------------------------------------

def run_harness(harness_path: Path, strict_conclusion: bool = False) -> tuple[bool, dict]:
    """Validate locked harness against currently mined examples."""
    with open(harness_path, encoding="utf-8") as fh:
        harness = json.load(fh)

    locked = harness.get("locked_examples", [])
    mined = mine_all_sources(IRS_EXAMPLE_SOURCES)
    mined_by_id = {_example_id(ex): ex for ex in mined}

    failures: list[str] = []
    for entry in locked:
        ex_id = entry["example_id"]
        expected = _normalize(entry.get("conclusion", ""))

        if ex_id not in mined_by_id:
            failures.append(f"missing example: {ex_id}")
            continue

        actual = _normalize(mined_by_id[ex_id].get("conclusion", ""))
        if strict_conclusion:
            if actual != expected:
                failures.append(f"conclusion mismatch: {ex_id}")
        else:
            if not actual.startswith(expected[:32]):
                failures.append(f"conclusion drift: {ex_id}")

    report = {
        "locked_examples": len(locked),
        "mined_examples": len(mined),
        "failures": failures,
        "failure_count": len(failures),
    }
    return (len(failures) == 0), report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IRS harness build/validate.")
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build the locked harness fixture.")
    build_p.add_argument("--output", type=Path, default=_DEFAULT_HARNESS)
    build_p.add_argument("--max-cases", type=int, default=40)

    run_p = sub.add_parser("run", help="Validate harness against current XML.")
    run_p.add_argument("--harness", type=Path, default=_DEFAULT_HARNESS)
    run_p.add_argument("--strict", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = _args()

    if args.command == "build":
        harness = build_harness(max_cases=args.max_cases)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(harness, fh, indent=2, ensure_ascii=False)
        print(f"Wrote harness: {args.output}")
        print(f"  locked examples: {len(harness['locked_examples'])}")
        return 0

    ok, report = run_harness(args.harness, strict_conclusion=args.strict)
    print(f"locked: {report['locked_examples']}, mined: {report['mined_examples']}, "
          f"failures: {report['failure_count']}")
    if not ok:
        for item in report["failures"][:20]:
            print(f"  - {item}")
        return 1
    print("Harness PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
