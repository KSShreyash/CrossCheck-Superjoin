# Fact Knowledge Layer

Pulls facts out of PDFs, keeps every fact tied to the words it came from, and works out
where facts across documents agree, disagree, or only appear to disagree.

Built for the Superjoin engineering intern assignment.

---

## Setup and run instructions

Requires Python 3.11 or newer.

```bash
git clone <this repo>
cd factlayer
pip install -e ".[dev]"
uvicorn factlayer.api:app --reload
```

Open <http://127.0.0.1:8000>.

**You do not need an API key to see the starter results.** Every model call is cached
under a hash of its prompt, and the cache for the starter documents is committed, so a
clone replays the whole knowledge layer offline. A key is only needed to ingest a PDF
the cache has never seen.

To ingest new documents, put a key in `.env` (the file is gitignored):

```
GEMINI_API_KEY=your_key_here
```

A free key comes from <https://aistudio.google.com/apikey>.

Then either upload a PDF through the web interface, or run the whole starter set:

```bash
python scripts/ingest_starter.py ../starter-datasets/starter-datasets --dry-run
python scripts/ingest_starter.py ../starter-datasets/starter-datasets
python scripts/show_cases.py          # prints the four cases with their evidence
```

**The free tier allows 20 requests per day, per model.** Not per minute — I found that
out by hitting it mid-ingest. The whole corpus needs about 160 requests, so one model
cannot read all 511 pages in a day. Three things follow, and they shaped the design more
than anything else:

- **A budget is spent deliberately.** `--max-windows-per-doc N` reads the N densest
  windows of each document. Windows were already ordered by fact density so an
  interrupted run lost the least valuable pages; the same ordering lets a run be
  truncated on purpose. `--dry-run` prices a run before it happens.
- **Canonicalisation is one pass over the corpus, not one per document** — two requests
  instead of twelve, and a better answer besides, since the model sees every metric name
  at once rather than meeting them a document at a time.
- **One key is enough, because the quota is per model.** The client rotates through
  several models and moves on when one is spent. A quota refusal is never retried:
  every attempt is itself a counted request, so backing off spends more of exactly what
  just ran out. Set `FACTLAYER_MODELS` to change the order.

```bash
python scripts/ingest_starter.py ../starter-datasets/starter-datasets --max-windows-per-doc 2
```

Recall is therefore bounded by budget rather than by capability. Raise the cap and it
reads more; nothing in the architecture changes.

The two starter datasets are independent. All four cases below happen to come from the
Delhivery documents, but the macroeconomic set is where the period normaliser earns its
keep: the RBI writes `2025-26` for the fiscal year the IMF writes `FY2025/26`, and
without resolving both to the same dates their growth projections are never compared at
all. Ingest both.

Ingest order matters if you intend to reuse the committed cache. Canonicalisation is
incremental, so its prompt reflects what was ingested before it; the script sorts
filenames so the order is reproducible.

To see the interface with data in it before setting up a key at all:

```bash
python scripts/demo_fixture.py
FACTLAYER_DB=demo.sqlite uvicorn factlayer.api:app
```

That writes four small PDFs, runs the real pipeline against a fixed stub model, and
produces one cross-document corroboration, one genuine contradiction and several
context-reconciled pairs. It is a smoke test, not a result: the documents are synthetic
and clearly named as such.

Verify the central claim yourself — that every stored fact quotes text that is really
in its source:

```bash
python scripts/audit_grounding.py
```

It re-reads each piece of evidence and looks for it in the stored text of the page it
cites, rather than trusting the offsets recorded at extraction time.

Tests:

```bash
pytest            # full suite, no network access required
```

---

## Video demo

<!-- TODO: add the link once recorded -->

---

## The four cases

Everything below is reproduced by cloning this repository and running two commands, with
**no API key**: the model responses are committed, and the facts and relations are
recomputed from them by this code. Every quote is stored evidence, and
`python scripts/audit_grounding.py` re-checks all 490 of them against the pages they cite.

```bash
python scripts/ingest_starter.py ../starter-datasets/starter-datasets --max-windows-per-doc 25
python scripts/show_cases.py
```

**1. A fact corroborated across documents, expressed differently.**

| | annual report, p2 | earnings deck, p8 |
| --- | --- | --- |
| quote | "18,793 (1) Pin codes covered" | "Pin-code reach(1) 18,074 18,540 18,675 18,793" |
| metric | `Pin codes covered` | `Pin-code reach` |
| period | as of March 31, 2024 | Q4 FY24 |

Two documents, two names for the metric, and two ways of writing the period — one a
date, the other a quarter — that normalise to the same instant. The deck reports it as
the last point of a quarterly series; the annual report as a single figure. Settled by
rule, no model call.

**2. A genuine or likely contradiction.**

| | annual report, p4 | earnings deck, p6 |
| --- | --- | --- |
| quote | "1.6% EBITDA margin" | "₹76Cr / 0.9% Adj. EBITDA / Adj. EBITDA margin" |
| value | 1.6 per cent, FY24 | 0.9 per cent, FY24 |

Same company, same period, same unit, and nothing recorded distinguishes them, so the
rules raise it rather than explain it away. The honest reading is that one is adjusted
and the other is not — a distinction neither document attached to the number itself.
That is what makes it worth surfacing: as reported, the two disagree.

**3. An apparent contradiction explained by context.**

| | prospectus, p44 | annual report, p2 |
| --- | --- | --- |
| quote | "we provide our services in 17,488 postal index number ("PIN") codes, as of December 31, 2021" | "18,793 (1) Pin codes covered" |
| period | as of December 31, 2021 | as of March 31, 2024 |

The same metric as case 1, across a two-year gap. `different_period`, decided by rule
with no model call: a company covering more PIN codes in 2024 than in 2021 is growth,
not a contradiction. Case 1 and case 3 together are the point of the system — the same
measure corroborates when the dates agree and reconciles when they do not.

**4. An extraction or reasoning failure, and how it is handled.**

- **6 unreadable pages**, found without any model: the IMF cover page yields no text at
  all, four earnings-deck slides are images, and prospectus p63 gives 53 characters.
  Recorded as gaps with reasons rather than silently contributing nothing.
- **107 proposed facts rejected** because their quote could not be found verbatim in the
  window it came from. This is why every stored fact is grounded rather than intended to
  be, and the audit script lets you check that claim rather than take it.
- **494 pairs left undecided** as `insufficient_context`, because a fact without a
  parseable period cannot honestly be called contradictory. An earlier version treated a
  missing period as a matching one and manufactured 1,480 false contradictions — 45% of
  all pairs.

The two failures I would fix next are table column attribution, and canonicalisation
over-merging: `amount` swept together "Net Assets Amount" and "Public and Rights Issues
Amount", which is the over-merge risk named in the design, observed in practice.

### What the committed run produces

| | |
| --- | --- |
| documents / pages | 6 / 511 |
| facts stored | 490, **all 490 resolved to a page and verified against it** |
| facts carrying a period | 321 (66%) |
| relations | 629 |
| corroborates / reconciled / contradicts | 34 / 79 / 22 |
| rejected as ungrounded | 107 |
| model calls needed to reproduce | **0** — 100 are committed |

---

## Approach

### The problem is comparability, not extraction

The starter documents all have a clean text layer, so pulling numbers out is mostly
plumbing. The hard part is deciding two facts are *about the same thing* before saying
anything about whether they agree. Every interesting case turns on that:

- `₹8,142 Cr` and `81,415.38` (₹ million) are the same revenue, a hundredfold apart.
- `₹74,540.82 mn` and `₹81,415.38 mn` are both "FY24 revenue" and both correct, because
  one is standalone and the other consolidated.
- `6.4%` and `6.5%` are both "FY25 growth", one an advance estimate and one an actual.
- `6.5%` (RBI) and `6.6%` (IMF) are the same metric over the same period and genuinely
  differ.

So the design puts its weight into the fact representation and the normalisers. A fact
carries its own qualifiers — metric, entity, period, unit scale, reporting basis, data
vintage — and agreement mostly falls out of comparing them. Without that you end up
writing per-document rules, which the brief rules out and which would not survive a PDF
nobody has seen.

### The model proposes, the rules verify

The LLM is good at noticing that "revenue from services" and "Revenue from Operations"
might be the same metric, and bad at being checkable. Arithmetic is the reverse. So the
model gets wide latitude to find and propose, and every quantitative claim it makes is
re-derived independently:

- It extracts facts, but a fact is discarded unless its quote is found **verbatim** in
  the source window. Grounding is a guarantee, not an intention.
- It adjudicates pairs the rules cannot settle, but when it claims two figures reconcile
  by a scale change, the arithmetic has to agree. When it claims a difference in basis
  explains a gap, the basis has to actually differ.
- Where rule and model disagree, the pair is recorded as `needs_review` rather than
  resolved. That disagreement is a real output, not a defect to hide.

### Deciding what to decide

The rule layer classifies each candidate pair before any model call:

| Values | Periods | Qualifiers | Verdict |
| --- | --- | --- | --- |
| agree | any | none differ | `corroborates`, by rule |
| agree | any | some differ | `corroborates` with the caveat recorded |
| differ | either unknown | any | `insufficient_context` |
| differ | both known | exactly one differs | model explains it |
| differ | both known | none differ | model adjudicates a likely contradiction |
| different units | any | any | `unrelated` |

The third row matters more than it looks, and it is the row I got wrong first. An absent
period means *unknown*, not "the same period as the other fact". Treating two undated
facts as contemporaneous made every difference in their values look like a contradiction.
Measured over two starter documents, that manufactured 1,480 false contradictions — 45%
of all pairs — and left 99.5% of pairs needing a model call, which no free tier
survives. Refusing to rule on undated pairs cut adjudication sevenfold and cost nothing,
because every case worth demonstrating carries an explicit period on both sides.

`insufficient_context` is an answer, not a failure. The pair is stored and visible; the
system simply declines to claim something it cannot support.

### Engineering decisions and trade-offs

- **SQLite, not a graph database.** Relations are one table with two foreign keys and the
  queries are joins. The brief is explicit that a graph store is not itself the answer,
  and one file means a grader runs one command with nothing to provision.
- **A fixed core plus an open qualifier map.** A prospectus, an earnings deck and an IMF
  staff report do not share a fact schema. Inventing one upfront means either something
  so loose it says nothing or something that breaks on the first unfamiliar document.
  Qualifier keys accumulate as documents introduce them, and no qualifier name is
  hard-coded in the comparison logic.
- **Everything cached by content hash.** Reruns are free, tests never touch the network,
  and the committed cache is what lets this be evaluated without my account.
- **Long-context windows, not pages.** Table headers and their rows have to reach the
  model together, or every column is misattributed.
- **No frontend build step.** Jinja templates and no JavaScript. The time is better spent
  on reasoning quality, and the brief warns that visualisation alone is not the solution.
- **A pinned model, not `-latest`.** The model name is part of every cache key, so a
  floating alias would keep replaying old responses under a name that now means
  something else. `gemini-3.6-flash` was chosen after measuring: the newest flash model
  exhausted its free-tier quota within a handful of calls, and the lite variants returned
  their JSON wrapped in an array. Override with `FACTLAYER_MODEL`.
- **Extraction is defensive about its own inputs.** The model returns array-wrapped JSON,
  drops units that were declared in a table header, and transcribes `₹` as `I` often
  enough to matter. Each of those is handled at the boundary rather than assumed away,
  because the alternative is a system that works on the documents I happened to test.

### AI tools used

Claude (via Claude Code) was used throughout: to explore the starter documents, argue
through the design, write code and tests, and — most usefully — to review the plan
repeatedly before building. Six review passes found twenty-one defects, of which nine
would have shipped silently. The ones that mattered were found by *running* code rather
than reading it, and several are described in "Limitations" below because they shaped
the design. `docs/design.md` and `docs/implementation-plan.md` are the working documents
from that process and are kept in the repository.

---

## Limitations and next steps

**What does not work yet, honestly:**

- **Run-to-run variance.** The model sits in the discovery path, so the same PDF can
  yield slightly different facts across runs. Temperature is zero and the cache is
  committed, so the demo and a first clone are stable in practice, but this is a real
  property of the design rather than something solved.
- **No OCR.** Image-only pages are detected and reported as gaps rather than silently
  contributing nothing. The IMF cover page is one. Adding OCR is a contained change at
  the ingest stage.
- **Tables are the weakest link.** Multi-column financial tables flatten under text
  extraction, and mapping a row of four numbers back to standalone-vs-consolidated and
  FY24-vs-FY23 is where wrong facts are most likely to originate. Long-context windows
  mitigate this; they do not solve it.
- **Period coverage bounds everything.** A fact without a parseable period can never be
  part of a contradiction, by design. 66% of stored facts carry one, and the rest are why
  494 of 629 pairs sit in `insufficient_context`. Improving period attachment is the
  single highest-value next step, because it decides how much of the corpus the system
  can reason about at all. Coverage varies by document, not by model: the earnings deck
  labels almost everything `FY24`, the prospectus is prose.
- **Entity clustering can over-merge.** Two similarly named subsidiaries could collapse
  into one canonical entity and manufacture a false contradiction. The threshold is
  conservative and failures should surface in `needs_review`.
- **Pairing is quadratic.** 0.39s at the roughly 1,250 facts the starter set produces and
  2.2s at 3,000, so it is a non-issue at this size, but the blocking needs rewriting
  before "many documents in one layer" is real.
- **Recall is unmeasured, and budget-bound.** There is no labelled ground truth here, so
  I can say every stored fact is grounded and verify it, but not what fraction of the
  facts present were found. The committed run reads the densest windows of each document
  rather than all 158, so recall is limited by the daily quota rather than by the
  approach.
- **Cache replay depends on ingest order**, because canonicalisation is incremental.

**Next steps, in the order I would do them:**

1. Attach periods from section and table context rather than only the sentence, since
   that limit gates everything else.
2. OCR fallback for image-only pages.
3. Table-aware extraction that preserves the header-to-cell relationship explicitly
   instead of hoping long context is enough.
4. Embedding-backed blocking so pairing stops being quadratic.
5. A confidence-weighted view that ranks contradictions by how much evidence supports
   each side.

---

## Additional notes

- No credentials are in the repository. `.env` is gitignored; `.env.example` shows the
  shape.
- `docs/design.md` is the design; `docs/implementation-plan.md` is the build plan and
  carries the full record of the six review passes, including what each one got wrong.
- The commit history is the real working history rather than a squashed import.
