# Fact Knowledge Layer — Design

Extract facts from PDFs, keep every fact tied to the text it came from, and work out
where facts across documents agree, disagree, or only appear to disagree.

## The actual problem

Extraction is the easy half. The starter documents all have a clean text layer, so
pulling numbers out is mostly plumbing.

The hard half is **comparability**: deciding two facts are about the same thing before
you can say anything about whether they agree. Every interesting case in the starter
data turns on this:

- `₹8,142 Cr` and `81,415.38` (₹ million) are the same revenue, 100x apart in scale.
- `₹74,540.82 mn` and `₹81,415.38 mn` are both "FY24 revenue" and both correct —
  standalone versus consolidated.
- `6.4%` and `6.5%` are both "FY25 growth" — one is an advance estimate, one an actual.
- `6.5%` (RBI) and `6.6%` (IMF) are the same metric, same period, and genuinely differ.

So the design pushes almost all of its weight into the fact representation and the
normalisers. If a fact carries its own qualifiers — metric, entity, period, unit scale,
reporting basis, data vintage — then agreement and disagreement mostly fall out of
comparing them. If it doesn't, you end up writing per-document rules, which the brief
rules out and which wouldn't survive a PDF we haven't seen.

## Shape of the system

Seven stages, one SQLite file, one FastAPI process.

```
PDF ─▶ ingest ─▶ segment ─▶ extract (LLM) ─▶ ground-check ─▶ normalise
                                                                 │
                                              relations ◀── pair ◀┘
                                                   │
                                            adjudicate (LLM)
                                                   │
                                              verify (rules)
                                                   │
                                              API + UI
```

The governing idea: **the model proposes, the rules verify.** The LLM is good at
noticing that "revenue from services" and "Revenue from Operations" might be the same
metric, and bad at being checkable. Arithmetic is the reverse. So the model gets wide
latitude to find and propose, and every quantitative claim it makes is then re-derived
independently. Where the two disagree, that disagreement is recorded rather than
resolved silently.

### 0. Ingest and segment

PyMuPDF for text blocks with page number and bounding box. Bounding boxes are worth the
dependency: they let the UI show the highlighted source span, which is what makes the
evidence trail convincing rather than merely claimed.

Per document we store a SHA-256 of the file bytes, so re-uploading the same PDF is a
no-op instead of a duplicate.

Two deterministic passes, both document-agnostic:

- **Boilerplate**: blocks whose normalised text repeats across many pages are running
  headers, footers, or disclaimers. Flagged and skipped. This is what keeps the
  safe-harbour page out of the extraction budget.
- **Extraction gaps**: a page with substantial area but almost no extractable text is
  almost certainly image-only. Flagged with a reason and surfaced in the API, not
  dropped. The IMF report's cover page is exactly this, and it is one of the four
  required cases.

Segmentation groups consecutive blocks into overlapping windows sized for long context
rather than per page. Table headers and their rows land in the same window, which is the
difference between reading a financial table correctly and misattributing every column.

### 1. Extraction

Each window goes to Gemini Flash with a strict JSON schema. A fact is:

| Field | Notes |
| --- | --- |
| `subject` | entity as written — "Delhivery Limited", "the Company", "India" |
| `metric` | what is asserted, as written |
| `value_raw`, `value_num` | as printed, and parsed |
| `unit_raw` | "₹ million", "per cent", "Mn Tons", or null |
| `period_raw` | "FY24", "year ended March 31, 2024", "2025-26", or null |
| `qualifiers` | open map — basis, vintage, segment, scope, geography, attribution |
| `claim_type` | measurement / estimate / projection / attribute / event |
| `evidence_quote` | verbatim span from the window |
| `confidence` | model's own, used for ranking not for gating |

The fixed core plus the open `qualifiers` map is deliberate. A prospectus, an earnings
deck, and an IMF staff report do not share a fact schema, and inventing one upfront means
either a schema so loose it says nothing or one that breaks on the first unfamiliar
document. Qualifier keys accumulate as documents introduce them.

**Grounding check.** `evidence_quote` must be found verbatim in the window text after
whitespace normalisation. If it isn't, the fact is rejected and written to a
`rejected_facts` table with the reason. This is a hard guarantee — every stored fact has
a real span in a real PDF — and it costs one string search. Locating the quote also
recovers exact character offsets, which map back to blocks and therefore to bounding
boxes.

Windows are processed in descending order of a cheap fact-density score, so if a rate
limit interrupts a run, the pages worth having are already done. The score orders work;
it never decides what to skip. Gating extraction on a regex-grade heuristic would lose
facts permanently, and a fact never extracted can never be reconciled.

Every call is cached under `sha256(model + prompt_version + window_text)`.

**Duplicates.** Windows overlap so a fact spanning a boundary is not lost, which puts
about a tenth of blocks — 11.6%, measured on the annual report — into two windows. Their
facts arrive twice. Left alone the twins pair with each other and register as
corroborations, so the system would report a sentence agreeing with itself. Facts are
therefore deduplicated per document on subject, metric, value, period and normalised
quote. The key is per document deliberately: the same fact in a *different* document is
the cross-document corroboration we are looking for.

**Failure handling.** Everything that goes wrong costs one window rather than the
document, because a parse error propagating out of extraction would otherwise discard a
hundred pages of work.

- A transient server error is retried with bounded backoff. It costs no allowance.
- A quota refusal is never retried; it rotates to the next model. Every attempt is
  itself a counted request, so backing off spends more of exactly what just ran out.
- Malformed JSON is not retried either — temperature is zero, so the model reproduces
  it. The window is halved and read again instead, since the densest windows are both
  the likeliest to overflow the output limit and the most worth recovering.
- A window with no cached response and no API key is skipped, not fatal. Replaying the
  committed cache with a larger budget than it was built with hits exactly this, and
  abandoning the document there would throw away the cached windows behind it. The
  document stays marked incomplete so a later run with a key picks it up.

### 2. Normalisation

Deterministic, pure, and the most heavily tested part of the codebase. Four normalisers:

- **Value and unit** → canonical magnitude in a base unit. Handles lakh (1e5), crore
  (1e7), million, billion, currency symbols and spellings, percent, tonnes, and bare
  counts. `₹81,415.38 million` and `₹8,142 Cr` both land on ~8.14e10 and match within a
  relative tolerance that absorbs the rounding in the earnings deck.

  Units arrive inconsistently and the design has to survive that. In a table the
  currency is declared once in a header — "(₹ in Million)" — and never repeated in the
  cells, so extraction frequently returns a bare `million`. Against real documents the
  earnings deck reported EBITDA as `Rs. Cr` while the annual report reported revenue as
  plain `million`, and a strict unit comparison refused to compare the two figures at
  all. That would have killed the headline corroboration.

  So a unit that reduces to nothing once scale words are stripped names a magnitude with
  no dimension, and is recorded as `UNKNOWN` rather than assumed to be a count. Unknown
  units stay comparable with anything; two *known* and different units, a rupee figure
  against a percentage, remain incomparable. This is the same rule the period logic
  follows: unknown is not the same as different. The currency is also recovered from the
  verbatim evidence quote when the unit field omits it, since the quote is real document
  text and the more reliable witness.
- **Period** → `(start, end, kind)`. Indian FY24 is 2023-04-01 to 2024-03-31. The IMF
  writes `FY2025/26` for the year Indian filings call `FY26`. Also handles quarters,
  instants ("as on March 31, 2024"), and part-years ("nine months ended").
- **Entity** → canonical id, clustered from the corpus by normalised form plus embedding
  similarity. Document-relative references like "the Company" resolve against that
  document's primary entity, which is itself an extracted fact.
- **Metric** → canonical id, same clustering. This is what lets "revenue from services"
  and "Revenue from Operations" become comparable without either being hard-coded.

Clustering is derived from the ingested corpus, so no alias list ships with the code.

Documents arrive one at a time, so clustering has to be incremental, and the naive
version of that is quietly broken. If the model is shown only the terms it has not seen
before, a metric arriving with the second document cannot join a group created by the
first: "revenue from services" becomes one canonical id, "Revenue from Operations"
becomes another, and the two never pair. That is the headline corroboration failing
silently. The prompt therefore carries the groups already assigned and asks the model to
reuse an id when one fits.

The cost is a small order dependency: because that prompt reflects what was ingested
before it, so does its cache key. The committed cache replays only if documents are
ingested in the recorded order. Extraction is unaffected, since a window's text does not
depend on what came earlier.

### 3. Pairing

All-pairs comparison is quadratic and unnecessary. Candidates come from the union of two
channels — shared canonical metric and lexical overlap, plus embedding cosine over
`subject + metric` — capped at top-k per fact. Union rather than intersection, because
each channel misses pairs the other catches.

Pairs are drawn across documents and also within a document. Intra-document conflicts are
real: the standalone and consolidated columns of the same table are the clearest
reconciliation case in the whole starter set.

### 4. Adjudication

The rule layer runs first and computes, for each pair, which qualifiers differ and
whether the canonical values agree within tolerance. That yields a provisional verdict:

| Values | Periods | Qualifiers | Verdict |
| --- | --- | --- | --- |
| agree | any | none differ | corroborates — by rule, no model call |
| agree | any | some differ | corroborates with caveat — by rule |
| differ | either unknown | any | insufficient context — recorded, no model call |
| differ | both known | only the period | reconciled by context — by rule, no model call |
| differ | both known | exactly one other | reconcilable — model explains |
| differ | both known | none differ | contradiction candidate — model adjudicates |
| different units | any | any | unrelated — no model call |
| non-numeric | any | any | model adjudicates |

Two distinctions in that table were learned by running the thing rather than reasoning
about it.

A difference that is *only* the period is settled by rule and never reaches the model.
FY23 reporting a different number from FY24 is what reporting looks like. Sending it to
the model invites a confident wrong answer: asked to compare the two, it asserted they
covered the same period and called it a contradiction.

And qualifiers are not all alike. Some describe **the measurement** — basis, vintage,
scope, segment, period — and a difference there means the two facts may not be
comparable. Others describe **provenance**: who published the claim. Those must never
block a contradiction, because two institutions publishing different numbers for the
same measure over the same period is the most interesting disagreement there is. The
RBI and the IMF differing on next year's growth is the case worth showing, and an
earlier version of this rule silently suppressed it. Provenance is still recorded and
displayed; it just does not count towards comparability.

The third row carries most of the weight, and it is the row I got wrong first time.
An absent period means *unknown*, not "the same period as the other fact". Treating
two undated facts as contemporaneous made every difference in their values look like a
contradiction: measured across two starter documents, that manufactured 1,480 false
contradictions, 45% of all pairs. Refusing to rule on them cut adjudication calls
sevenfold and cost nothing, because every case worth demonstrating carries an explicit
period on both sides.

`insufficient_context` is a real answer rather than a failure. The pair is stored and
visible, the system simply declines to claim a relationship it cannot support.

Two vocabularies exist and they are easy to confuse. The verdicts above are internal to
the rule layer. What the API filters on, the UI groups by, and this document quotes is a
single `final_verdict` per relation:

| `final_verdict` | meaning |
| --- | --- |
| `corroborates` | the two facts agree, by rule or confirmed by the model |
| `contradicts` | comparable facts that genuinely disagree |
| `reconciled_by_context` | they differ, and a named difference explains it |
| `insufficient_context` | not enough qualifiers to say anything honestly |
| `unrelated` | not about the same measurement |
| `needs_review` | rule and model disagree, or a claimed transform failed to verify |

Every relation carries one. Roughly three quarters never reach the model, so leaving
their verdict unset would hide most of the knowledge layer behind a filter matching
nothing.

Attribute facts always go to the model, because "was a director" versus "resigned with
effect from 1 July 2024" is a judgement about time and status that arithmetic cannot
make.

The model returns a relation, a reason code, an explanation, a confidence, and — when it
claims a reconciliation — the **transformation it is claiming**: a scale factor, a change
of basis, a difference in vintage.

### 5. Verification

Any transformation the model claims gets re-derived from the canonical values. If it says
`₹8,142 Cr` reconciles with `₹81,415.38 million` by a hundred-fold scale change, the
arithmetic either confirms that or it doesn't. When it doesn't, the relation is downgraded
to `needs_review` and the disagreement between rule and model is recorded on the relation
itself.

This bucket is a feature. It is where the system says "these look related and I am not
confident why", which is a more honest output than a confident wrong answer, and it is
the natural place to point during the failure-case discussion.

### 6. Storage

SQLite, one file. Tables: `documents`, `blocks`, `facts`, `evidence`, `canon_terms`,
`relations`, `llm_cache`, `rejected_facts`, `gaps`, `jobs`.

Two constraints carry weight. Documents are keyed by content hash, so re-uploading a
file is a no-op rather than a second copy. Relations are unique on their fact pair,
because relation building runs after every upload and re-pairs documents already
ingested — without that, a third document would double every relation from the first
two. Re-running refreshes a verdict instead of duplicating it.

No graph database. Relations are one table with two foreign keys, the queries are joins,
and the brief is explicit that a graph store is not itself the answer. SQLite also means
a grader runs one command with nothing to provision.

### 7. API and UI

FastAPI, with the UI served by the same process as Jinja templates. No JavaScript, no
build step, no `node_modules`, no separate dev server.

```
POST /api/documents           upload, returns job id, ingest runs in the background
GET  /api/jobs/{id}           stage, pages processed, facts found
GET  /api/documents           list with counts
GET  /api/documents/{id}/gaps pages we could not read, and why
GET  /api/facts               filter by entity, metric, period, document, free text
GET  /api/facts/{id}          fact, evidence, bbox, related facts
GET  /api/relations           filter by final_verdict (see below)
GET  /api/relations/{id}      both sides, both quotes, rule verdict, model explanation
GET  /api/stats               header counts
```

Four pages: upload and document list with live progress; a filterable fact table; a
relation detail view; and a gaps-and-rejections view.

The relation detail view is the one that matters. Two evidence quotes side by side, each
with its document and page, canonical values underneath, the rule's verdict, the model's
explanation, and whether the two agreed. That single screen is the demo.

## Testing

- Unit tests on the normalisers, which is where correctness actually lives — scale
  conversion, Indian fiscal years, the IMF's `FY2025/26`, tolerance behaviour.
- An end-to-end test over a small PDF generated in the test itself, with the LLM stubbed
  by pre-seeding the cache. Tests never touch the network.
- A grounding test asserting that a fabricated `evidence_quote` is rejected.

## The budget is the design constraint

The Gemini free tier allows **20 generate requests per day, per model, per project**.
Not per minute. That single number shaped more of this system than any other
consideration, and it was discovered the honest way — by hitting it mid-ingest.

The corpus is 511 pages and segments into 158 extraction windows. Reading all of it
would take eight days on one model. So the system is built to spend a budget rather
than to assume one does not exist:

- **Windows are ordered by fact density and the budget is spent from the top.** The
  ordering already existed so that an interrupted run would lose the least valuable
  pages; under a hard quota the same ordering lets a run be deliberately truncated.
  `--max-windows-per-doc N` reads the N densest windows of each document, taking the
  passages carrying figures and skipping narrative prose.
- **Canonicalisation is one pass over the corpus, not one per document.** It cost two
  calls per document and now costs two in total. That was forced by the budget and is
  also the better answer: the model sees every subject and metric at once instead of
  meeting them a document at a time.
- **Pairs the rules can settle never reach the model at all**, which is what makes the
  adjudication budget go far enough to matter.
- **`--dry-run` prices a run before it happens**, against the real quota.
- **Rotation across models.** The quota is counted per project *per model*, so several
  models carry several allowances. When a pairing reports a daily limit it is marked
  spent and the next is used. A cached answer under any rotated model is still a hit, so
  rotation never re-asks a question already paid for. Extra keys can be supplied the same
  way for anyone with more than one project of their own.
- **A daily limit is never retried.** Every attempt is itself a counted request, so the
  original backoff spent four more requests to be told the same thing — which is why the
  second model appeared to die after seven calls rather than twenty. Per-minute limits
  are still waited out; per-day limits move straight on.

Rotation has a cost worth naming: the model is part of every cache key, so replaying a
rotated corpus depends on the same rotation happening again. The order is therefore
fixed rather than random, and the better model goes first — extraction quality is not
uniform, and `gemini-3.6-flash` attached periods far more reliably than `3.7-flash` on
the same documents.

The cost of the compromise is honest and worth stating: with a per-document window cap,
the system reads the densest sections rather than the whole document, so recall is
bounded by budget rather than by capability. Nothing about the architecture changes when
the quota does — raise the cap and it reads more.

## Running without a key

Every model call is cached by content hash, and the cache for the starter documents is
committed. A grader clones, runs one command, and gets a fully populated knowledge layer
with no API key at all. A key is only needed to ingest a PDF the cache has never seen.

This is also the answer to the brief's requirement that a paid or rate-limited service
must not block evaluation.

## Trade-offs and known limits

- **Run-to-run variance.** The model is in the discovery path, so the same PDF can yield
  slightly different facts across runs. Temperature is zero and the cache is committed,
  so the demo and a grader's first run are stable in practice, but this is a real
  property of the design and the README should say so plainly rather than imply
  determinism.
- **No OCR.** Image-only pages are detected and reported as gaps rather than silently
  contributing nothing. Adding OCR is a contained change at the ingest stage.
- **Tables are the weakest link.** Multi-column financial tables flatten under text
  extraction, and mapping a row of four numbers back to standalone-vs-consolidated and
  FY24-vs-FY23 is where wrong facts are most likely to originate. Long-context windows
  that keep headers with rows mitigate this; they do not solve it.
- **Entity clustering can over-merge.** Two genuinely different subsidiaries with similar
  names could collapse into one canonical entity, which would manufacture a false
  contradiction. Threshold is conservative; failures should surface in `needs_review`.
- **Recall is unmeasured.** There is no labelled ground truth for these documents, so we
  can say every stored fact is grounded, but not what fraction of the facts present were
  found.
- **Period coverage limits how much can be said.** A fact without a parseable period can
  never be part of a contradiction, by design. In a dry run over two starter documents
  only 39% of facts carried one, which put most pairs in `insufficient_context`. A real
  extraction should do better than that dry run's crude stand-in, but the honest number
  belongs in the README once the real ingest has run, because it bounds how much of the
  corpus the system can reason about at all.
- **Pairing is quadratic.** 0.39s at the ~1,250 facts the starter set produces and 2.2s
  at 3,000, so it is a non-issue here, but the blocking would need rewriting before the
  "many documents in one layer" extension.
- **Cache replay depends on ingest order.** Canonicalisation is incremental, so its
  prompt — and therefore its cache key — reflects what was ingested before it. The
  committed cache replays in the recorded order; a different order re-asks those
  questions and needs a key.

## Cases to demonstrate

All four are verified present in the starter data.

1. **Corroboration, differently expressed.** Earnings deck `₹8,142 Cr FY24 revenue from
   services` against the annual report's `81,415.38` ₹ million consolidated revenue from
   operations. Same fact across a scale change and a label change. They coincide exactly
   because FY24 traded-goods revenue was nil — it was ₹16.54 mn in FY23 — so the two
   labels happen to denote the same quantity that year.
2. **Genuine contradiction.** Real GDP growth for FY2025-26: RBI projects 6.5%, the IMF
   projects 6.6%. Same metric, same period, different institutions.
3. **Apparent contradiction, explained.** FY24 revenue of ₹74,540.82 mn against
   ₹81,415.38 mn — standalone versus consolidated. Second instance if needed: FY25 growth
   at 6.4% in the Economic Survey (first advance estimate) against 6.5% reported by the
   IMF (actual), which is a difference of data vintage.
4. **Failure.** The IMF cover page yields zero characters because it is an image, and the
   annual report's financial summary table flattens into an ambiguous run of numbers.
   The first is detected and reported; the second is the honest weak point.

## Out of scope

No authentication, no multi-tenancy, no job queue beyond in-process background tasks, no
OCR, no graph store, no frontend build pipeline. Each is a deliberate omission, not an
oversight, and each is cheap to add later against these interfaces.
