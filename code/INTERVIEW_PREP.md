# Interview prep — Buy or Wait?

Short, honest answers that match the code as submitted. File references are
relative to `code/`.

**1. What problem are you solving?**
For each purchase/payment request, decide how much is safe to pay today, whether
and how the full request can be completed (full payment, a two-step partial
payment, a supplied installment offer, waiting, or not proceeding), the earliest
safe date for a single full payment, and which flexible expenses must change —
all while never letting the forecast balance drop below the user's minimum over
90 days.

**2. Why isn't `balance >= price` sufficient?**
Money is already spoken for. Rent, bills and payroll land on fixed days, groceries
and commuting repeat every 5–21 days, pending debits still clear, and some
"income" (pending payouts, commissions, bonuses) must not be counted. A user can
afford a payment today and still breach their minimum the week before payday.
The check is the *lowest* projected balance over the window, not today's.

**3. Why a deterministic financial engine?**
Every graded field is arithmetic or a rule: minimum balances, dates, schedule
sums, ranking order. Those must be exact, repeatable and auditable. All money
is `Decimal`; the run is byte-identical across runs and makes zero model calls.

**4. Where does AI add value?**
At the edges where data is unstructured: interpreting free-text, multilingual
messages into a fixed set of intents, and reading amounts from receipt/payslip
images for events with a blank `amount`. `bow/evidence/llm.py` holds an optional
Anthropic extractor; without an API key a keyword heuristic classifier
(`bow/evidence/heuristics.py`) and a content-hash-keyed image cache are used.
The image cache was built once during development by reading the 16 images with
Claude (see `evaluation/usage_report.md`).

**5. How do you prevent prompt injection?**
Evidence can only produce a value from a closed set of intents
(`bow/evidence/schema.py`) plus typed slots (amount, date, percent). The state
reconstructor dispatches on those intents in fixed code; message text is never
executed, never forwarded as instructions, and cannot add a payment option,
change a rule, or skip the safety check. Scam-style messages (e.g. an upfront
fee request) have their own inert intent.

**6. Contradictory messages?**
The problem's order is applied: explicit cancellation/settlement/amendment first,
then the newer record from the same source, then settled over estimated, then
the financially safer reading. Concretely: a confirmed future-dated row re-anchors
a recurring stream and supplies its amount; a payroll handover ("Previous" →
"New employer payroll") keeps the newest record; unsettled credits are suppressed.

**7. How do you reconstruct financial state?** (`bow/state.py`)
Opening balance = `current_available_balance`. Cancelled/failed rows are dropped
(except a failed debit the bank says it will retry), non-cash and unrealized
investment values are ignored, pending credits are ignored, pending/scheduled
debits are reserved on their settlement date, foreign amounts are converted with
the dated rate, blank amounts come from the linked image (never zero). Recurring
streams are then projected, then evidence is applied.

**8. How does the 90-day simulation work?** (`bow/recurrence.py`, `bow/forecast.py`)
`recurrence.detect` groups settled history by category and direction and
recognises two shapes only: *monthly* (same day-of-month, gaps 26–33 days) and
*periodic* (a modal gap explaining ≥70% of gaps). It tries the description grain
first (so "Base salary" and "Performance commission" stay separate) and falls back
to the category grain (so a transport cadence with rotating descriptions stays one
stream). Amounts use the median (robust to one bulk shop). Variable income is not
projected. The forecast collapses flows into one net change per date; because the
balance is constant between flow dates, the exact minimum is found from those
points without a day-by-day array.

**9. `amount_safe_to_pay`?**
`clamp(min_balance(request_date..horizon) − minimum_balance_to_keep, 0, requested_amount)`.
Paying X today lowers every later balance by X, so this closed form is exact — no
search needed. A unit test checks the 1-cent boundary (`test_schedule_safety_boundary`).

**10. `earliest_date_for_full_payment`?**
The first date D (checking the request date and every flow date, which is exact
because the answer only changes there) where `min_balance(D..horizon) ≥ minimum +
requested_amount`. Computed on the baseline forecast, independent of payment
preferences and spending changes. Empty if never within the window.

**11. Selecting payment plans?** (`bow/candidates.py`, `bow/rank.py`)
Build every rule-eligible candidate — full payment, partial payment (only if the
request allows it, the user accepts it, `0 < safe < requested`, and the second
payment date ≤ deadline), each supplied installment option within
`max_installment_months` and finishing by the deadline, and wait. Each schedule is
simulated; if unsafe, the smallest subset (≤3) of permitted stop/reduce changes that
makes it safe is searched. Winners are ranked literally: no spending changes,
lower total paid, earlier start, fewer payments, lowest `payment_option_id`.
No safe candidate → `not_affordable` / `not_recommended`.

**12. Installment constraints?**
Only rows from `request_payment_options.csv` are used; the schedule is generated
from `first_payment_date`, `payment_frequency_days` and `number_of_payments`, and
`validate.py` re-checks that the output plan matches the option date-for-date and
amount-for-amount.

**13. Currency conversion?** (`bow/dataset.py: ExchangeRates`)
One class; dated rate on or before the settlement date for the given direction,
inverse if only the reverse pair exists, one hop via USD/EUR, and a `KeyError`
rather than a guess if nothing matches. Unit-tested.

**14. How do you validate output?**
`bow/validate.py` re-derives constraints independently of candidate generation:
enums, `0 ≤ safe ≤ requested`, `affordable_now ⇒ earliest == request_date`,
partial payment = exactly two payments summing to the request, installment plan
equals a supplied option, chronological plan, ≤3 spending changes on distinct
events, `not_recommended ⇒ plan none`. `main.py` exits non-zero if any row fails.
All 250 rows pass.

**15. Hidden-test-like cases?**
`tests/test_core.py` builds synthetic ledgers: one-off vs recurring, rotating
descriptions, cancelled rows, a bonus inside payroll, variable gig income, final
payroll, employer handover, FX direction and missing rates, the 1-cent safety
boundary, every ranking tie-break, and output formatting.

**16. How did you avoid overfitting?**
The 25 samples were used only to measure the *forecast model*: where the published
`amount_safe_to_pay` is below the request, it inverts to the true minimum balance
(`tools/calibrate.py`). That drove structural choices applied to every user —
detecting sub-monthly cadences, median amounts, counting same-day charges. There
are no request, user, event or message IDs in production logic.

**17. Main failure modes?**
- Irregular variable income (freelance/gig): we project none; the reference counts
  some, so we are conservative (e.g. request_09, request_10).
- Forecast amounts are typically within ~3% of the reference minimum balance
  (median relative error 0.027 over 21 samples), so `amount_safe_to_pay` rarely
  matches to the cent; categorical fields match far more often.
- Message interpretation is keyword-based without an API key, so unseen phrasings
  can fall to `unknown` (which is then ignored, not guessed).

**18. Why better than asking an LLM?**
An LLM cannot reliably sum hundreds of dated flows, hold a minimum across 90 days,
or apply a six-level tie-break identically on every row, and its errors are
unauditable. Here every number traces to a ledger line, reruns are identical, and
the model is confined to reading unstructured evidence.

**19. With more time?**
Model irregular income conservatively (e.g. a lower-quantile monthly total);
run the message classifier through the LLM with a cached, token-counted pass;
widen the synthetic suite into end-to-end requests; add property tests on the
validator.

**20. AI models/tools used?**
Claude Code (Claude Opus) as the coding assistant for audit, implementation,
calibration and tests; Claude vision for one-time extraction of the 16 image
amounts (cached). The production run that writes `output.csv` makes 0 model calls.
