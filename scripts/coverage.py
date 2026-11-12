"""Measure how much of a list of real model names the token dictionary understands.

    uv run python scripts/coverage.py names.txt
    cat names.txt | uv run python scripts/coverage.py - --top 50

The input is one model name per line. Nothing is looked up: this runs the
offline name reader only, and reports the tokens it could not explain, most
common first. Those are the best candidates to add to whatmodel/tokens.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from whatmodel import parse


def read_names(source: str) -> list[str]:
    text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    names = [line.strip() for line in text.splitlines()]
    return list(dict.fromkeys(name for name in names if name and not name.startswith("#")))


def measure(names: list[str]) -> dict[str, object]:
    unknown: Counter[str] = Counter()
    examples: dict[str, str] = {}
    tokens = known = clean = ambiguous = failed = no_family = 0
    for name in names:
        try:
            parsed = parse(name)
        except ValueError:
            failed += 1
            continue
        tokens += len(parsed.tokens)
        known += sum(1 for token in parsed.tokens if token.reading is not None or token.readings)
        if not parsed.unknown:
            clean += 1
        if parsed.ambiguous:
            ambiguous += 1
        if parsed.family is None:
            no_family += 1
        for text in parsed.unknown:
            key = text.lower()
            unknown[key] += 1
            examples.setdefault(key, name)
    return {
        "names": len(names),
        "failed": failed,
        "names_fully_read": clean,
        "names_ambiguous": ambiguous,
        "names_without_family": no_family,
        "tokens": tokens,
        "tokens_known": known,
        "unknown": [
            {"token": token, "count": count, "example": examples[token]}
            for token, count in unknown.most_common()
        ],
    }


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("names", help="file with one model name per line, or - for stdin")
    parser.add_argument("--top", type=int, default=30, help="how many unknown tokens to list")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()

    result = measure(read_names(args.names))
    if args.json:
        print(json.dumps(result, indent=2))
        return
    names, tokens = int(str(result["names"])), int(str(result["tokens"]))
    print(f"names               {names}")
    for label, key in (
        ("fully read", "names_fully_read"),
        ("ambiguous", "names_ambiguous"),
        ("no family found", "names_without_family"),
        ("could not parse", "failed"),
    ):
        count = int(str(result[key]))
        print(f"  {label:<17} {count:>6}  {percent(count, names)}")
    known = int(str(result["tokens_known"]))
    print(f"tokens              {tokens}")
    print(f"  known             {known:>6}  {percent(known, tokens)}")
    unknown = result["unknown"]
    assert isinstance(unknown, list)
    print(f"\ntop unknown tokens ({len(unknown)} distinct)")
    for item in unknown[: args.top]:
        print(f"  {item['count']:>5}  {item['token']:<24} e.g. {item['example']}")


if __name__ == "__main__":
    main()
