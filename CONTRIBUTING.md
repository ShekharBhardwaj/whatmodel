# Contributing

The most useful contribution is a new token. Every meaning whatmodel knows lives in [`whatmodel/tokens.yaml`](whatmodel/tokens.yaml), and adding one needs no Python.

## Setup

```bash
git clone https://github.com/ShekharBhardwaj/whatmodel
cd whatmodel
uv sync
uv run pytest
```

## Add a token in three steps

1. Run the name that confused you and note what is reported as `Unknown`:

   ```bash
   uv run whatmodel olmo-2:7b --offline
   ```

2. Add an entry to `whatmodel/tokens.yaml`, next to the other entries of its category:

   ```yaml
   - pattern: olmo
     category: family
     label: OLMo
     maker: Allen Institute for AI
     explanation: The open language model family from the Allen Institute for AI.
   ```

3. Run the checks and look at the result:

   ```bash
   uv run pytest
   uv run whatmodel olmo-2:7b --offline
   ```

Label the pull request `tokens` so it lands under "New tokens" in the release notes.

## Finding tokens worth adding

`scripts/coverage.py` reads a file with one model name per line, runs the offline name reader on each, and lists the tokens it could not explain, most common first:

```bash
uv run python scripts/coverage.py names.txt --top 50
```

Only add a token when you know what it means. A token left as unknown is correct; a guessed meaning is a bug.

## Entry fields

| Field | Required | Meaning |
|---|---|---|
| `pattern` | yes | A literal, or a list of literals that mean the same thing. With `regex: true`, a regular expression. |
| `category` | yes | One of the ids under `categories` at the top of the file. |
| `label` | yes | The short phrase shown in the output. Keep it under 60 characters. |
| `explanation` | yes | One plain-English sentence, ending in a full stop. |
| `regex` | no | `true` when `pattern` is a regular expression. |
| `examples` | for regex entries | Tokens the pattern must match. The tests check them, and `whatmodel tokens` prints them instead of the raw regex. |
| `value` | no | The normalized value for JSON and the library, such as `4B` for `4b`. Defaults to the literal as written, or to the matched token for a regex. |
| `after` | no | Match only directly after a token of the named category. |
| `maker` | no | Family entries only: who makes the family. Shown as `Gemma 4 (Google)`. |
| `implies` | no | A fact in another category that this token settles, such as `{architecture: moe}`. |

`label`, `explanation` and `value` may use `{text}` (the matched token, lowercased) and any named regex group. Add `!u` to uppercase one: `{text!u}`.

## How matching works

- Case is ignored. `Q4_K_M` and `q4_k_m` are the same token.
- `-`, `_` and `.` are all separators, and they are interchangeable. A dot between two digits is a decimal point, so `3.5` stays whole.
- A pattern must match a whole token. `mini` does not match inside `minicpm`.
- Multi-part patterns match across separators. Write literals the way they usually appear (`gpt-oss`). In a regex, write `_` wherever the name has a separator (`q4_k_m`).
- When a name can be split more than one way, the split with the fewest unknown parts wins, then the one with the fewest tokens. That is why `mistral-small` beats `mistral` followed by `small`.
- A known family glued to a version, as in `qwen3.5`, is split into the two. This only happens when the family is in the dictionary.

## Regex entries

Use a regex when a token carries a number:

```yaml
- pattern: '(?P<n>\d+(?:\.\d+)?)b'
  regex: true
  category: size
  value: '{n}B'
  label: '{n}B parameters'
  explanation: About {n} billion parameters (learned weights), where more usually means more capable but slower and hungrier for memory.
  examples: [4b, 12B, 1.5b]
```

Put the pattern in single quotes so YAML leaves the backslashes alone. Quote examples that look like numbers (`'2507'`).

## Rules for entries

These keep the output trustworthy. The test suite enforces most of them.

- **Say only what the token always means.** If a token means different things in different names, either leave it out or add both meanings (see below).
- **Never make whatmodel guess.** If two entries match the same token, it is reported as ambiguous with both readings. That is the right outcome for a token such as `base` (a model without chat tuning, or the standard size). It is the wrong outcome if it happens by accident, so the tests fail on any overlap that is not listed in `INTENDED_AMBIGUOUS` in `tests/test_dictionary.py`.
- **One sentence per explanation**, in plain English. Explain jargon instead of repeating it.
- **Use `after` for tokens that only make sense in context.** A bare number is a version only right after a family, so that entry has `after: family`.
- **Use `implies` when a token settles something in another category.** `A3B` implies `architecture: moe`. A hosted-only family implies `hosting: cloud`. The implied token must be an entry in that category.
- **Facts must be checkable.** For a family, name the maker. For a publisher who converts other teams' models, say so.

## Adding a category

Add it under `categories` with a `heading` and a one-sentence `description`. Categories that share a heading share an output row, and their order in the file is the display order. A category may also have a `default`, shown when no token in the name belongs to it.

## Changing code

```bash
uv run ruff check
uv run ruff format
uv run pytest
```

- Token meanings never go in Python. If a change needs a new word of explanation, it belongs in `tokens.yaml`.
- Tests must not use the network. `tests/conftest.py` blocks every socket, including localhost, so lookup tests use `httpx.MockTransport`.
- Type hints are required everywhere; ruff checks this.

Releases are described in [RELEASING.md](RELEASING.md).
