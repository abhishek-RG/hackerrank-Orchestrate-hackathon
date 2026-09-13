# Buy or Wait? — solution

A deterministic financial-decision engine with a small, generic evidence
layer for interpreting the untrusted messages and images in the dataset.

## Run it

```bash
python code/main.py
```

Reads `dataset/`, writes `output.csv` at the repository root, and exits
non-zero if any row fails structural validation. No API key, no network
access, and no extra Python packages are required for this to work exactly
as submitted — see "Why the production run makes zero model calls" below.

Optional flags:

```bash
python code/main.py --dataset path/to/other/dataset --out somewhere.csv
python code/main.py --no-cache   # force every message/image through the
                                  # live extractor path instead of the cache
```

Regression check against the 25 solved examples in `dataset/sample_requests.csv`:

```bash
python code/evaluation/main.py          # summary
python code/evaluation/main.py -v       # every mismatched field, every row
python code/evaluation/main.py -v request_06   # one request, in full
```

**Interactive dashboard** (not part of the graded pipeline — a dev tool for
browsing decisions and testing "what if" scenarios live):

```bash
python code/tools/dashboard.py --port 8765
# open http://localhost:8765/ in a browser
```

Lists all 275 requests (25 solved samples + 250 eval, filterable and
searchable), and clicking one shows its full reconstructed 90-day cash-flow
timeline, every candidate plan the engine considered (not just the winner),
a field-by-field diff against the solved truth for samples, and a **live
what-if panel** — override `requested_amount`, `desired_completion_date`,
`request_date`, `allows_partial_payment`, `current_available_balance`, or
`minimum_balance_to_keep` and hit Recompute to see the decision update
instantly, in-process, without touching `dataset/` on disk. It reuses the
exact same `bow.*` modules `main.py` does — it's a window into the real
engine, not a mock.

Rebuild the evidence cache from scratch (see below):

```bash
python code/tools/build_evidence_cache.py
ANTHROPIC_API_KEY=sk-... python code/tools/build_evidence_cache.py --force
```

Requires Python 3.9+ and only the standard library for the production run.
`Pillow` and `pytesseract` are optional (local-OCR fallback for images);
`anthropic` is optional (live LLM extraction path).

## Architecture

Ten independent stages, each in its own module, wired together only by
`bow/engine.py`:

```
dataset.py        parse every CSV into typed records; fixed-rate FX conversion
evidence/          interpret messages & images (see below)
recurrence.py     generic cadence detection (monthly / every-N-days streams)
state.py          financial-state reconstruction into a dated 90-day ledger
forecast.py       the 90-day cash-flow safety check (pure Decimal arithmetic)
candidates.py     every safe, rule-eligible payment plan for one request
rank.py           the 6-criterion ranking from problem_statement.md
explain.py        decision_explanation / spending_changes_needed text
validate.py       re-derives what a *valid* row must look like, independently
output.py         CSV formatting and writing
engine.py         orchestrates the above, per request
```

`code/main.py` loops over `dataset/requests.csv` calling `engine.Engine.decide()`
once per row and writes the result. `code/tools/evaluate.py` runs the same
engine over `dataset/sample_requests.csv` for a regression check.

### Financial-state reconstruction (`recurrence.py`, `state.py`)

For each user and request, raw events become one dated ledger of signed cash
flows over `[request_date, request_date + 90 days]`. Nothing here is a per-user
or per-request label.

- **Recurring streams** (`recurrence.py`) — settled history is grouped by
  `(category, direction)` and recognised as one of two shapes only:
  *monthly* (stable day-of-month, gaps 26–33 days, ≥2 occurrences) or
  *periodic* (a modal gap of N days explaining ≥70% of gaps, ≥4 occurrences —
  e.g. groceries every 7 or 10 days, transport every 5/14/21). Detection tries
  the **description grain** first, so independent streams inside one category
  ("Base salary" vs "Performance commission") stay separate, and falls back to
  the **category grain** when descriptions rotate inside one cadence ("Rail
  pass", "Local taxi", ...). Projected amounts use the **median** observation
  (robust to one bulk shop inside a weekly cadence).
- **Income rules** — a credit stream is projected only when one amount accounts
  for most observations (confirmed salary); variable income (gig payouts,
  commission, irregular second income) is not projected. A payroll **handover**
  (the newest record sits on the stream's cadence under a new description, e.g.
  "Previous employer payroll" → "New employer payroll") continues at the newest
  figure; a newest record marked *final* ends income. A confirmed future salary
  row with no history continues monthly from its own date.
- **Re-anchoring** — a future-dated row for the same stream (e.g. a scheduled
  "next confirmed salary") supplies the stream's date and amount, so the rest
  of the cadence follows it and nothing is counted twice.
- **Dated one-off** — pending/scheduled debits are reserved on settlement,
  pending credits are ignored until settled, cancelled/failed transactions are
  dropped (unless a message says the debit will be retried), non-cash and
  unrealized investment events never touch cash. Irregular spending with no
  detectable cadence is not forecast (no invented expenses).

Evidence from `messages.csv` is folded in on top (§ below): a payroll message
can raise, cut, delay, or end the salary stream; a lease message scales a
matching recurring cost; an unconfirmed bonus/prize/refund suppresses the
credit it refers to until it would actually settle.

### The evidence layer (`bow/evidence/`)

Two things need interpreting that plain arithmetic cannot do: 215 free-text
messages (English and Indonesian) and 16 receipts/payslips with a blank
`amount`. Interpreting them is a separate, generic layer — never a per-item
lookup:

```
evidence/schema.py      the intent taxonomy (generic financial-correspondence
                         categories: salary_change, refund_pending, ...) —
                         nothing here names a specific message or request
evidence/heuristics.py  a keyword-scored classifier over ordinary financial
                         vocabulary in both languages — a real fallback
                         mechanism, not a lookup table, and it runs on text
                         this system has never seen
evidence/llm.py         a pluggable LLM extractor (Anthropic, via
                         ANTHROPIC_API_KEY) using the same fixed, generic
                         prompt on any message or image — this is the
                         primary path when a key is configured
evidence/pipeline.py    orchestrates: cache hit -> configured LLM -> generic
                         heuristic/OCR fallback -> never fabricate
evidence/cache.py       a SHA-256 content-addressed cache — keyed by the
                         hash of the message text or image bytes, never by
                         message_id/event_id/request_id
```

**Why content-hash keys matter:** a cache keyed by `event_id` would just be a
per-row answer table wearing a JSON hat. Keyed by content hash instead, two
different messages that happen to say the same thing share one entry (a
normal cache hit), and anything the cache has never seen is a genuine miss
that must go through a real extractor — there is no way to look up "the
answer for request_87" anywhere in this file. `code/evidence/cache/*.json`
each carry this in their own `_schema` field.

**Why the cache is optional, not load-bearing:** delete it, and the pipeline
still runs correctly — messages fall through to the generic heuristic
classifier (zero cost, deterministic, works on unseen text); images fall
through to a configured LLM (`ANTHROPIC_API_KEY`) or local OCR
(`pytesseract`), never to a guess. `code/main.py --no-cache` exercises this
path directly. What's frozen in the cache today was produced by running
those same generic mechanisms once (see `usage_report.md` for exactly how,
since no API key was available during development) — regenerate it any time
with `code/tools/build_evidence_cache.py`.

**Untrusted content:** `code/bow/evidence/heuristics.py` treats a message
asking the *reader* to pay an upfront fee to "release" funds as a scam
(`upfront_fee_scam`), checked before any other intent — a message's own
embedded instructions never reach the decision engine as anything but data.

### The 90-day safety check (`forecast.py`)

Between two flow dates the balance is constant, so the minimum balance over
any date range is realized either at the range's start (the level carried in
from before it) or at a flow date inside it — an exact answer from the
sparse event ledger, no day-by-day array needed:

- `max_safe_amount` — paying `X` today shifts every future balance down by
  `X`, so the largest safe `X` is `min_balance_over(request_date, horizon) -
  minimum_balance`, a closed-form subtraction.
- `earliest_full_payment_date` — `min_balance_over(D, horizon)` is
  non-decreasing as `D` increases (the window only shrinks), so a single
  forward scan over candidate dates finds the true earliest safe date.
- `is_schedule_safe` — merges an arbitrary candidate schedule (partial,
  installment, or spending-change-adjusted) into the ledger and checks the
  whole window at once; used to validate every candidate in `candidates.py`.

### Candidates, ranking, and spending changes

`candidates.py` builds every rule-eligible plan (full payment from its
supplied option, partial payment per the fixed two-payment formula,
each accepted installment option, and a delayed full payment) and, only if
none of those is safe on its own, searches subsets of up to 3 flexible
events (respecting the user's own willingness lists and protected
categories) for the smallest change that makes one safe. `rank.py` then
applies the exact 6-criterion order from problem_statement.md — complete by
deadline (already guaranteed by construction), no spending changes, minimize
total paid, start earlier, fewer payments, lowest `payment_option_id`.

`amount_safe_to_pay` and `earliest_date_for_full_payment` are always computed
from the *un*adjusted baseline, per the spec ("before optional spending
changes") — a chosen plan that needed a stop/reduce to work can still report
a smaller baseline `amount_safe_to_pay` than what it actually pays.

### Validation (`validate.py`)

Re-derives, independently of however a row was built, what a valid row must
look like: bounds on `amount_safe_to_pay`, no event targeted by more than one
spending change, an installment schedule matching its supplied option
exactly, a partial-payment schedule with exactly two chronological payments
summing to `requested_amount`. `code/main.py` runs this over every row before
writing `output.csv` and exits non-zero on any failure — the whole 250-row
dataset currently passes with zero problems.

## Why the production run makes zero model calls

Every stage that makes the actual affordability decision — reconstruction,
forecasting, candidate generation, ranking — is plain, auditable arithmetic,
which is both what problem_statement.md asks for ("keep behavior
deterministic where possible") and the only way to get an exact, reproducible
`amount_safe_to_pay` to the cent. The only place a model has anything to
contribute is interpreting free text and pixels, and that interpretation is
resolved once — either by the generic heuristic (for text; genuinely
zero-cost and language-agnostic in its mechanism, if not its current
vocabulary) or by an LLM call whose result is then frozen — rather than
repeated live on every run. See `code/evaluation/usage_report.md` for the
full accounting of what that one-time resolution cost.

## Development tools and tests

```bash
python -m unittest discover -s code/tests -v   # 23 synthetic unit tests
python code/tools/calibrate.py                 # forecast error vs samples
python code/tools/calibrate.py --grid          # search model-structure knobs
python code/tools/audit_output.py              # schema/semantic audit of output.csv
```

`calibrate.py` inverts each uncapped published `amount_safe_to_pay` into the
reference's minimum balance (`amount_safe + minimum_balance_to_keep`) and
reports the forecast model's error against it. It measures model *structure*
(cadence detection, amount convention, same-day charges); every knob is one
global setting in `bow.state.Config`.

Sample regression (25 solved rows, `python code/evaluation/main.py`):

| field | before this revision | now |
|---|---|---|
| amount_safe_to_pay (exact to the cent) | 4 | 2 |
| affordability_status | 19 | 19 |
| recommended_payment_method | 21 | 20 |
| payment_plan | 17 | 19 |
| earliest_date_for_full_payment | 17 | 17 |
| spending_changes_needed | 22 | 21 |

The exact-cent count is not the right lens for the numeric field: forecast
mean relative error on the implied minimum balance fell from 1.97 to 0.060
(median 0.027) across the 21 uncapped samples, so amounts are now close rather
than occasionally exact and otherwise far off.

## Known limitations

- Irregular variable income (freelance, gig payouts) is not projected at all;
  the reference appears to count part of it, so these users are forecast
  conservatively (e.g. request_09, request_10 in the samples).
- Projected amounts are typically within ~3% of the reference minimum balance,
  so `amount_safe_to_pay` rarely matches to the cent.
- `heuristics.py`'s vocabulary was built from ordinary financial terms in the
  message corpus; an unseen phrasing for the same intent falls to `unknown`
  (inert) rather than misfiring — `code/tools/build_evidence_cache.py` prints
  any such misses so they are never silent.
