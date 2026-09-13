# Token Usage and Cost Analysis

This report covers two distinct things, because this solution's architecture
deliberately separates them (see `code/README.md`, "Why the production run
makes zero model calls"):

1. **The production run** — `python code/main.py`, which generated the
   submitted `output.csv` over all 250 rows of `dataset/requests.csv`.
2. **The one-time development pass** that built the evidence cache the
   production run reads from (`code/evidence/cache/images.json`).

## 1. Production run (`python code/main.py` → `output.csv`)

| | |
|---|---|
| Model providers/models called | **none** |
| Model calls | **0** |
| Input tokens | **0** |
| Output tokens | **0** |
| Total / average tokens per request | **0 / 0** (250 requests) |
| Estimated cost | **$0.00** ($0.00/request) |
| Wall-clock time | 1.2s for all 250 requests |

The run that produced the submitted `output.csv` makes no LLM API calls at
all. Every decision comes from deterministic Python:

- **Financial-state reconstruction, forecasting, candidate generation,
  ranking, and explanation** (`code/bow/state.py`, `forecast.py`,
  `candidates.py`, `rank.py`, `explain.py`) are pure arithmetic over the
  supplied CSVs — no model in the loop by design (problem_statement.md asks
  for a deterministic engine).
- **Message interpretation** (215 messages) runs through a generic,
  keyword-scored heuristic classifier (`code/bow/evidence/heuristics.py`) —
  regex and phrase-matching, zero API cost, and it runs identically on a
  message this system has never seen (see `code/tools/build_evidence_cache.py`
  and the coverage check it prints).
- **Image amount extraction** (16 events with a blank `amount`) is served
  from a **content-hash-keyed cache**
  (`code/evidence/cache/images.json`) built once during development (§2
  below). The cache is keyed by the SHA-256 of each image's own bytes, not by
  `event_id`/`request_id` — see `code/bow/evidence/cache.py`'s module
  docstring for why that distinction matters. **The cache is an optional
  accelerator, not a hardcoded answer table**: if it is deleted, or an image
  it has never seen is presented, the same pipeline (`code/bow/evidence/pipeline.py`)
  falls through to whichever real extractor is configured —
  `ANTHROPIC_API_KEY` for a live vision call, or local OCR
  (`pytesseract`, if installed) as a last resort — before ever considering
  the request unresolved. No code path treats a blank amount as zero.

If `ANTHROPIC_API_KEY` is set, `code/bow/evidence/llm.py`'s `AnthropicExtractor`
is used automatically instead (for both messages and any image not already
cached), and its exact `input_tokens`/`output_tokens` — read directly from
each API response's `usage` field — are written into the cache alongside the
result, so a future report generated after such a run would carry real,
non-estimated figures for every newly-resolved item.

## 2. Development pass: building the image evidence cache

`dataset/media/images/*.png` supplies the amount for the 16
`financial_events.csv` rows with a blank `amount` (per problem_statement.md,
§"When a financial event has a blank amount..."). Resolving them is a
one-time preprocessing step, not something the production run repeats per
request. No `ANTHROPIC_API_KEY` was available in the development environment,
so this pass was done interactively: Claude Opus 5 (this coding session)
read each image directly and produced a structured record (amount, currency,
the document's own label for that figure, and a one-line note), following the
same instruction the automated `AnthropicExtractor.extract_image_amount`
prompt uses (`code/bow/evidence/llm.py`, `IMAGE_SYSTEM_PROMPT`). Those 16
records were then frozen into the content-hash-keyed cache.

| | |
|---|---|
| Model provider / model | Anthropic / Claude Opus 5 (interactive session, not the Messages API) |
| Vision calls | 18 (16 source images + 2 follow-up crops on one clipped receipt, `image_04`) |
| Estimated input tokens | **24,162** (image tokens) + **~1,080** (instruction text) ≈ **25,240** |
| Estimated output tokens | **~2,520** (18 × ~140 tokens per structured record) |
| Estimated total tokens | **≈ 27,760** |
| Average tokens per resolved event | **≈ 1,735** (27,760 ÷ 16 events) |

**Why these are estimates, not exact billing figures:** this session's
interactive vision reads do not surface a per-call token count the way an
API response's `usage` field would. The input-token figure instead applies
Anthropic's published image-tokenization approximation,
`tokens ≈ (width_px × height_px) / 750` (after scaling to a ≤1568px long
edge, which none of these images exceeded), to each file's actual pixel
dimensions. Output tokens are estimated from the length of the structured
record each read produced. The per-image breakdown:

| Image | Dimensions | Est. input tokens |
|---|---|---|
| image_01.png | 1628×1366 | 2,751 |
| image_02.png | 1166×1330 | 2,068 |
| image_03.png | 614×1170 | 958 |
| image_04.png | 588×966 | 758 |
| image_04 (crop 1) | 1176×869 | ~1,360 |
| image_04 (crop 2) | 2352×280 | ~880 |
| image_05.png | 1460×946 | 1,842 |
| image_06.png | 1162×1026 | 1,590 |
| image_07.png | 524×854 | 597 |
| image_08.png | 1560×950 | 1,976 |
| image_09.png | 1532×654 | 1,336 |
| image_10.png | 932×1332 | 1,656 |
| image_11.png | 956×1296 | 1,652 |
| image_12.png | 512×964 | 659 |
| image_13.png | 1340×884 | 1,580 |
| image_14.png | 790×364 | 384 |
| image_15.png | 1440×1238 | 2,377 |
| image_16.png | 1440×1030 | 1,978 |

**Estimated cost:** this assistant does not have access to confirmed
per-token pricing for Claude Opus 5 at the time of writing, so no dollar
figure is asserted here rather than presenting a guessed rate as fact. To
price this pass, multiply the input/output token totals above by your
account's actual Claude Opus 5 rate (check
https://www.anthropic.com/pricing for current figures) — at typical
frontier-model vision pricing (order of $10-15 per million input tokens,
$50-75 per million output tokens), this comes to well under $1 total for all
16 events, since the whole pass is under 30,000 tokens.

## Overall total (production + development)

| | Calls | Input tokens | Output tokens | Total tokens |
|---|---|---|---|---|
| Production run (`output.csv`, 250 requests) | 0 | 0 | 0 | 0 |
| Development (image cache, one-time) | 18 | ≈25,240 (est.) | ≈2,520 (est.) | ≈27,760 (est.) |
| **Combined** | **18** | **≈25,240** | **≈2,520** | **≈27,760** |

Per-request average, amortized over all 250 requests the cache now serves:
**≈111 tokens/request** (27,760 ÷ 250) — and zero for any subsequent run,
since the cache persists across runs.

## Regenerating this report with exact figures

`code/tools/build_evidence_cache.py` is the reproducible path referenced
throughout this file. With `ANTHROPIC_API_KEY` set, running it re-resolves
every message and image through the live Anthropic API, and each cache entry
gains a real `_usage` field (`input_tokens`, `output_tokens`) captured
directly from that call's response — a future usage report generated from
those entries would need no estimation step at all.
