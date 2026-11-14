"""Collect popular real model names, one per line, for scripts/coverage.py.

    uv run python scripts/fetch_names.py > names.txt

Names come from two public places: the most downloaded models on Hugging Face,
and the Ollama library with the tags of its most popular models. Ollama has no
API for its library, so that part reads the web pages and may need updating if
they change. A source that fails is skipped with a note on standard error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections.abc import Iterator

USER_AGENT = "whatmodel-coverage"
HUGGINGFACE = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=1000&{query}"
HUGGINGFACE_QUERIES = (
    "pipeline_tag=text-generation",
    "pipeline_tag=image-text-to-text",
    "filter=gguf",
)
OLLAMA = "https://ollama.com/library"
#: A name is letters, digits and a few separators. Anything else is dropped,
#: so text scraped from a page can never smuggle markup into a report.
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}")


def fetch(url: str, timeout: float = 30.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def huggingface_names() -> Iterator[str]:
    for query in HUGGINGFACE_QUERIES:
        try:
            models = json.loads(fetch(HUGGINGFACE.format(query=query)))
        except (OSError, ValueError) as exc:
            print(f"skipped Hugging Face ({query}): {exc}", file=sys.stderr)
            continue
        for model in models if isinstance(models, list) else []:
            if isinstance(model, dict) and isinstance(model.get("id"), str):
                yield model["id"]


def ollama_names(with_tags: int) -> Iterator[str]:
    try:
        page = fetch(OLLAMA)
    except OSError as exc:
        print(f"skipped the Ollama library: {exc}", file=sys.stderr)
        return
    models = list(dict.fromkeys(re.findall(r'href="/library/([A-Za-z0-9._-]+)"', page)))
    yield from models
    for model in models[:with_tags]:
        try:
            tags = fetch(f"{OLLAMA}/{model}/tags")
        except OSError as exc:
            print(f"skipped tags of {model}: {exc}", file=sys.stderr)
            continue
        pattern = r'href="/library/(' + re.escape(model) + r':[A-Za-z0-9._-]+)"'
        yield from dict.fromkeys(re.findall(pattern, tags))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--ollama-tags",
        type=int,
        default=60,
        help="read the tags of this many of the most popular Ollama models (default 60)",
    )
    args = parser.parse_args()
    names = dict.fromkeys([*ollama_names(args.ollama_tags), *huggingface_names()])
    kept = [name for name in names if SAFE_NAME.fullmatch(name)]
    if not kept:
        sys.exit("no names could be fetched")
    print("\n".join(kept))


if __name__ == "__main__":
    main()
