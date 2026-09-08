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

**The free tier allows 20 requests per day, per model.** Not per minute — I found this
out by hitting it mid-ingest. The full corpus needs about 160 requests, so reading all
511 pages on one model would take eight days. Budget flags exist because of that:

```bash
# read the 2 densest windows of each document: about 14 requests
python scripts/ingest_starter.py ../starter-datasets/starter-datasets \
    --max-windows-per-doc 2 --max-model-calls 4
```

`--dry-run` prices a run before it happens. `--max-windows-per-doc N` spends the budget
on the N densest windows of each document, `--max-model-calls` caps adjudication, and
`--limit N` ingests only the first N documents. `FACTLAYER_MODEL` selects the model,
which matters because the quota is per model.

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

Produced by `python scripts/show_cases.py` over the six starter documents. Every quote
below is stored evidence, and `python scripts/audit_grounding.py` re-checks all 701 of
them against the pages they cite.

**1. A fact corroborated across documents, expressed differently.**

| | earnings deck, p6 | annual report, p4 |
| --- | --- | --- |
| quote | "1.4 Mn Tons PTL freight tonnage in FY24" | "1,429K tonnes PTL freight delivered" |
| metric | `PTL freight tonnage` | `PTL freight delivered` |
| normalised | 1,400,000 TONNE | 1,429,000 TONNE |

Different names, different scale words, and a 2% gap that is not a disagreement: the
deck prints two significant figures where the report prints four. Comparing at the
coarser precision settles it as a corroboration, by rule, with no model call.

**2. A genuine or likely contradiction.**

| | earnings deck, p6 | annual report, p4 |
| --- | --- | --- |
| quote | "₹76Cr / 0.9% Adj. EBITDA / Adj. EBITDA margin" | "1.6% EBITDA margin" |
| value | 0.9 per cent, FY24 | 1.6 per cent, FY24 |

Same company, same period, same unit, and nothing recorded distinguishes them — so the
rules raise it rather than explain it away. The honest reading is that one figure is
adjusted and the other is not, a distinction neither document attached to the number
itself. That is what makes it worth surfacing: the disagreement is real *as reported*.

**3. An apparent contradiction explained by context.**

| | earnings deck, p6 | annual report, p2 |
| --- | --- | --- |
| quote | "1.4 Mn Tons PTL freight tonnage in FY24" | ">4.8Mn tonnes Part-truckload freight delivered since inception" |
| period | FY24 | as of March 31, 2024 (cumulative) |

One year against everything ever shipped. The period normaliser separates them and the
pair resolves as `different_period` by rule, with no model call — a difference of period
is what reporting looks like, not a conflict.

Another, across a two-year gap: `82 gateways ... as of December 31, 2021` (prospectus)
against `111 Gateways ... As of March 31, 2024` (annual report), where the model named
the reason and verification confirmed the periods genuinely differ.

**4. An extraction or reasoning failure, and how it is handled.**

- **6 unreadable pages**, detected without any model: the IMF cover page yields no text
  at all, four earnings-deck slides are images, and prospectus p63 gives 53 characters.
  Recorded as gaps with reasons rather than contributing nothing silently.
- **41 proposed facts rejected** because their quote could not be found verbatim in the
  window it came from. This is why every stored fact is grounded rather than intended
  to be.
- **710 pairs left undecided** as `insufficient_context` — facts without a parseable
  period cannot honestly be called contradictory. An earlier version treated a missing
  period as a matching one and manufactured 1,480 false contradictions, 45% of all pairs.

The failure I would fix next is table column attribution, and the second is that
`amount` over-merged "Net Assets Amount" with "Public and Rights Issues Amount" during
canonicalisation — the over-merge risk named in the design, observed in practice.

### What the corpus produced

| | |
| --- | --- |
| documents / pages | 6 / 511 |
| facts stored | 701, **all 701 resolved to a page and verified against it** |
| facts carrying a period | 471 (67%) |
| relations | 1,155 |
| corroborates / reconciled / contradicts | 218 / 182 / 45 |
| of those, across documents | 18 / 17 / 5 |
| model calls consumed | 100, all cached and replayable |

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
  part of a contradiction, by design. Improving period attachment is the single highest
  value next step, because it directly determines how much of the corpus the system can
  reason about at all.
- **Entity clustering can over-merge.** Two similarly named subsidiaries could collapse
  into one canonical entity and manufacture a false contradiction. The threshold is
  conservative and failures should surface in `needs_review`.
- **Pairing is quadratic.** 0.39s at the roughly 1,250 facts the starter set produces and
  2.2s at 3,000, so it is a non-issue at this size, but the blocking needs rewriting
  before "many documents in one layer" is real.
- **Recall is unmeasured.** There is no labelled ground truth here, so I can say every
  stored fact is grounded, but not what fraction of the facts present were found.
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
