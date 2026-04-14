"""Mine IRS XML publications for worked examples.

The IRS embeds named worked examples throughout its publications (e.g., Pub 17,
Pub 560, Schedule C instructions). Each example presents a taxpayer scenario and
states a conclusion. These examples are authoritative ground truth — they were
written and verified by IRS staff.

This module scans the IRS XML files already present in the project's
``knowledge/`` directory, identifies sections whose headings signal a worked
example (e.g., "Example.", "Example 1.", "Illustration."), and returns each
such section as a structured dict that can be turned into a Tier 3 benchmark
case by ``cases/tier3_qa.py``.

Output dict per example
-----------------------
{
    "source":      str,   # IRS source key, e.g. "p17"
    "source_label": str,  # Human-readable label, e.g. "IRS Pub. 17"
    "section_id":  str,   # Section identifier from the XML
    "heading":     str,   # Original heading text
    "text":        str,   # Body text of the example
    "dollar_amounts": list[float],  # Dollar values found in the text
    "conclusion":  str,   # Inferred conclusion sentence
    "tags":        list[str],  # Auto-detected topic tags
}
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Allow running as a standalone script from within data_syn/
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EXAMPLE_HEADING_PATTERNS,
    EXAMPLE_MAX_CHARS,
    EXAMPLE_MIN_CHARS,
    IRS_EXAMPLE_SOURCES,
    KNOWLEDGE_DIR,
)

# Source labels shared with the main project's irs_xml_parser
SOURCE_LABELS: dict[str, str] = {
    "p17":      "IRS Pub. 17",
    "p501":     "IRS Pub. 501",
    "p502":     "IRS Pub. 502",
    "p503":     "IRS Pub. 503",
    "p504":     "IRS Pub. 504",
    "p505":     "IRS Pub. 505",
    "p523":     "IRS Pub. 523",
    "p525":     "IRS Pub. 525",
    "p526":     "IRS Pub. 526",
    "p529":     "IRS Pub. 529",
    "p530":     "IRS Pub. 530",
    "p544":     "IRS Pub. 544",
    "p559":     "IRS Pub. 559",
    "p560":     "IRS Pub. 560",
    "p570":     "IRS Pub. 570",
    "p596":     "IRS Pub. 596",
    "p936":     "IRS Pub. 936",
    "p970":     "IRS Pub. 970",
    "p1099":    "IRS Pub. 1099",
    "i1040gi":  "Form 1040 Instructions",
    "i1040sc":  "Sch. C Instructions",
    "i1040sd":  "Sch. D Instructions",
    "i1040se":  "Sch. E Instructions",
    "i1040sse": "Sch. SE Instructions",
    "i8829":    "Form 8829 Instructions",
    "i8995":    "Form 8995 Instructions",
}

# Regex patterns for auto-tagging
_TAG_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("standard_deduction",   re.compile(r"standard deduction", re.I)),
    ("itemized_deductions",  re.compile(r"itemize|itemized", re.I)),
    ("self_employment",      re.compile(r"self.employ|schedule.?c|sole prop", re.I)),
    ("se_tax",               re.compile(r"self.employment tax|schedule.?se", re.I)),
    ("capital_gains",        re.compile(r"capital gain|capital loss|schedule.?d", re.I)),
    ("ira",                  re.compile(r"\bira\b|individual retirement", re.I)),
    ("home_office",          re.compile(r"home office|form 8829", re.I)),
    ("qbi",                  re.compile(r"\bqbi\b|qualified business income|section 199a", re.I)),
    ("charitable",           re.compile(r"charitable|donation|contribut", re.I)),
    ("child_tax_credit",     re.compile(r"child tax credit|ctc", re.I)),
    ("earned_income_credit", re.compile(r"earned income credit|eitc|eic", re.I)),
    ("rental",               re.compile(r"rental income|schedule.?e|passive", re.I)),
    ("depreciation",         re.compile(r"depreciat|section 179|bonus deprec", re.I)),
    ("social_security",      re.compile(r"social security|ssa|ssa-1099", re.I)),
    ("medical",              re.compile(r"medical expense|health insurance", re.I)),
    ("education",            re.compile(r"education|tuition|529|coverdell|student loan", re.I)),
    ("retirement_plan",      re.compile(r"sep|simple ira|401\(k\)|pension|retirement plan", re.I)),
    ("wages",                re.compile(r"wages|salary|w.?2", re.I)),
    ("interest_income",      re.compile(r"interest income|1099.?int", re.I)),
    ("dividend",             re.compile(r"dividend|1099.?div", re.I)),
]

_DOLLAR_RE = re.compile(r"\$[\d,]+(?:\.\d{1,2})?")
_CONCLUSION_KEYWORDS_RE = re.compile(
    r"deduct|taxable|not taxable|include|exclude|must|may|owe|credit|"
    r"allowed|disallowed|qualif|limit|rate|amount",
    re.I,
)


def _find_xml_file(source_dir: Path) -> Path | None:
    """Return the first .xml file in the given directory, or None."""
    for f in source_dir.iterdir():
        if f.suffix.lower() == ".xml":
            return f
    return None


def _get_namespace(root: ET.Element) -> str:
    """Extract the XML namespace URI from the root element tag."""
    if "}" in root.tag:
        return root.tag.split("}")[0].strip("{")
    return ""


def _is_example_heading(heading: str) -> bool:
    """Return True if the heading signals a worked example."""
    normalized = heading.strip().lower().rstrip(".")
    for pattern in EXAMPLE_HEADING_PATTERNS:
        clean_pattern = pattern.rstrip(".")
        if normalized == clean_pattern or normalized.startswith(clean_pattern + " "):
            return True
    return False


def _extract_dollar_amounts(text: str) -> list[float]:
    """Return all dollar amounts found in text, as floats."""
    amounts = []
    for match in _DOLLAR_RE.finditer(text):
        raw = match.group(0).lstrip("$").replace(",", "")
        try:
            amounts.append(float(raw))
        except ValueError:
            pass
    return amounts


def _detect_tags(text: str) -> list[str]:
    """Return a list of topic tags based on keyword patterns in the text."""
    return [tag for tag, pattern in _TAG_PATTERNS if pattern.search(text)]


def _extract_conclusion(text: str) -> str:
    """Extract a self-contained conclusion sentence from worked-example text."""
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s and s.strip()
    ]
    if not sentences:
        return ""

    for sentence in reversed(sentences):
        normalized = " ".join(sentence.lower().split())
        if normalized.startswith(("see ", "refer to ", "for more information", "go to ")):
            continue
        if len(sentence) < 24 and not _DOLLAR_RE.search(sentence):
            continue
        if _DOLLAR_RE.search(sentence) or _CONCLUSION_KEYWORDS_RE.search(sentence):
            return sentence

    return sentences[-1]


def _collect_examples(
    element: ET.Element,
    heading_tag: str,
    para_tags: set[str],
    source: str,
    results: list[dict],
) -> None:
    """Depth-first walk; collect sections whose heading signals a worked example."""
    heading_el = element.find(heading_tag)
    if heading_el is not None:
        heading = "".join(heading_el.itertext()).strip()
        if _is_example_heading(heading):
            paragraphs = [
                " ".join(child.itertext()).strip()
                for child in element
                if child.tag in para_tags
            ]
            paragraphs = [p for p in paragraphs if p]
            if paragraphs:
                text = " ".join(paragraphs)
                if EXAMPLE_MIN_CHARS <= len(text) <= EXAMPLE_MAX_CHARS:
                    conclusion = _extract_conclusion(text)
                    results.append({
                        "source":         source,
                        "source_label":   SOURCE_LABELS.get(source, source),
                        "section_id":     element.get("id", ""),
                        "heading":        heading,
                        "text":           text,
                        "dollar_amounts": _extract_dollar_amounts(text),
                        "conclusion":     conclusion,
                        "tags":           _detect_tags(text),
                    })

    for child in element:
        _collect_examples(child, heading_tag, para_tags, source, results)


def mine_source(source: str) -> list[dict]:
    """Mine one IRS XML source for worked examples.

    Parameters
    ----------
    source:
        IRS source key matching a subdirectory name in ``KNOWLEDGE_DIR``
        (e.g., ``"p17"``).

    Returns
    -------
    list[dict]
        All worked examples found in the source, each as a dict described
        in the module docstring.
    """
    source_dir = KNOWLEDGE_DIR / source
    if not source_dir.is_dir():
        return []

    xml_file = _find_xml_file(source_dir)
    if xml_file is None:
        return []

    try:
        tree = ET.parse(str(xml_file))
    except ET.ParseError:
        return []

    root = tree.getroot()
    ns = _get_namespace(root)
    heading_tag = f"{{{ns}}}hd" if ns else "hd"
    para_tags = (
        {f"{{{ns}}}{t}" for t in ("p", "iconpara")} if ns else {"p", "iconpara"}
    )

    results: list[dict] = []
    _collect_examples(root, heading_tag, para_tags, source, results)
    return results


def mine_all_sources(sources: list[str] | None = None) -> list[dict]:
    """Mine all configured IRS XML sources for worked examples.

    Parameters
    ----------
    sources:
        List of source keys to mine. Defaults to ``IRS_EXAMPLE_SOURCES``
        from config.

    Returns
    -------
    list[dict]
        All worked examples across all sources, deduplicated by
        ``(source, section_id)`` pairs.
    """
    if sources is None:
        sources = IRS_EXAMPLE_SOURCES

    seen: set[tuple[str, str]] = set()
    all_examples: list[dict] = []

    for source in sources:
        for ex in mine_source(source):
            key = (ex["source"], ex["section_id"])
            if key not in seen:
                seen.add(key)
                all_examples.append(ex)

    return all_examples


if __name__ == "__main__":
    examples = mine_all_sources()
    print(f"Mined {len(examples)} examples from {len(IRS_EXAMPLE_SOURCES)} sources.\n")
    for ex in examples[:5]:
        print(f"[{ex['source_label']}] {ex['heading']}")
        print(f"  Tags: {ex['tags']}")
        print(f"  Amounts: {ex['dollar_amounts']}")
        print(f"  Conclusion: {ex['conclusion'][:120]}")
        print()
