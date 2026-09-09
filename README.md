# CrossCheck

A fact knowledge layer over PDFs. It extracts facts, ties each one to the text it came
from, and decides where facts across documents agree, disagree, or only appear to
disagree.

## Setup and Run Instructions

Requires Python 3.11 or newer.

```bash
git clone https://github.com/KSShreyash/CrossCheck-Superjoin.git
cd CrossCheck-Superjoin
pip install -e ".[dev]"
python scripts/serve.py
```

Open <http://127.0.0.1:8000>. It starts empty on purpose: nothing is pre-loaded, so
press **Load the sample documents** and watch the knowledge layer get built. That takes
about three seconds and needs no API key, because the six sample documents are bundled
under `starter-datasets/` and the model responses for them are committed under `cache/`.

The same thing on the command line, which also accepts a path to any other folder of
PDFs:

```bash
python scripts/ingest_starter.py
python scripts/ingest_starter.py /some/other/folder
```

Two more commands are worth running once there is data:

```bash
python scripts/show_cases.py         # the four required cases, with their evidence
python scripts/audit_grounding.py    # re-checks every stored quote against its source
```

`scripts/serve.py` is used instead of `uvicorn factlayer.api:app` because it adds `src/`
to the path itself and therefore works from a bare checkout. If `pip install -e .`
succeeded, `uvicorn factlayer.api:app` is equivalent.

To read documents the cache has not seen, put a key in `.env` (gitignored; a free one
comes from https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your_key_here
```

Then upload a PDF through the web interface. Without a key the bundled documents still
work, and an upload it cannot read reports the pages it skipped rather than failing
silently.

Run the test suite with `pytest`. It needs no network access and no key.

### Hosting it

The app is one process over a SQLite file, so any host that runs a container works.
Railway needs no configuration beyond the repository.

1. On railway.app, create a project and deploy from this GitHub repository.
2. Generate a domain under Settings, Networking.

That is all. The instance starts empty and a visitor presses **Load the sample
documents** to build the layer from the committed cache, which takes about three seconds
and makes no network calls. Set `FACTLAYER_AUTOLOAD=1` if you would rather it be
populated at boot instead.

`railway.json` supplies the start command and the health check, `requirements.txt` and
`.python-version` tell Nixpacks what to install, and the app reads `PORT` from the
environment.

The start command is `python scripts/serve.py --host 0.0.0.0` rather than
`uvicorn factlayer.api:app`, because the latter needs the package to have been installed
and Nixpacks only installs the dependencies. `serve.py` puts `src/` on the path itself,
so it works either way.

Optional variables:

| variable | effect |
| --- | --- |
| `GEMINI_API_KEY` | lets visitors upload PDFs the cache has not seen |
| `FACTLAYER_AUTOLOAD=1` | build the layer at boot instead of on a button press |
| `FACTLAYER_DB` | move the database to a mounted volume |
| `FACTLAYER_UPLOADS` | move uploaded files to a mounted volume |

Railway's disk is ephemeral, so a deploy resets the instance to its empty state and
uploaded PDFs do not survive a restart unless `FACTLAYER_UPLOADS` points at a volume.
Rebuilding costs one button press.

Render works the same way and `render.yaml` configures it: create a Blueprint from the
repository, or a Web Service with `pip install -r requirements.txt` to build and
`python scripts/serve.py --host 0.0.0.0` to start. It sets `FACTLAYER_AUTOLOAD=1`,
because a free Render instance loses its disk when it sleeps and would otherwise wake up
empty; boot then takes about five seconds instead of three.

A free instance also sleeps after fifteen minutes of no traffic and takes most of a
minute to wake. `.github/workflows/keep-alive.yml` pings `/api/stats` every ten minutes
to avoid that, once a repository variable named `DEPLOY_URL` holds the deployed address.
It is deliberately harmless without one: the job reports that nothing is configured and
passes. Two things to know if you rely on it. GitHub runs scheduled workflows on a
best-effort basis and can delay them past the sleep window, and it disables schedules
altogether after sixty days without repository activity.

Vercel is a poor fit and I did not target it: its Python functions run on a read-only
filesystem, so SQLite cannot write, and each invocation would rebuild the layer.

### A note on the free tier

The Gemini free tier allows 20 requests per day per model, not per minute. Reading all
511 pages of the starter corpus takes about 160 requests, so a single key cannot do it in
one day. Three things follow, and they shaped the design:

* `--max-windows-per-doc N` reads only the N densest windows of a document. Windows are
  ordered by how many figures they contain, so a truncated run keeps the valuable pages.
* Canonicalisation runs once over the whole corpus rather than once per document, which
  costs two requests instead of twelve.
* The client rotates through several models, since the quota is counted per model, and
  never retries a quota refusal. Every attempt is itself a counted request, so backing
  off spends more of what has just run out.

`--dry-run` prices a run before making it.

## Video Demo

https://drive.google.com/drive/folders/1Hq7L3n2OgYO-Q_YT-z-87yeXvPEElHGl?usp=sharing

The video shows a PDF being processed and the four cases below.

## Approach

### The problem is comparability, not extraction

The starter documents all have a clean text layer, so pulling numbers out of them is
mostly plumbing. The difficulty is deciding that two facts are about the same thing
before saying anything about whether they agree:

* `₹8,142 Cr` and `81,415.38` in ₹ million are the same revenue, a hundredfold apart.
* `₹74,540.82 mn` and `₹81,415.38 mn` are both FY24 revenue, and both correct, because
  one is standalone and the other consolidated.
* The RBI writes `2025-26` for the fiscal year the IMF writes `FY2025/26`.

So most of the weight sits in the fact representation and the normalisers. Every fact
carries its own qualifiers: metric, subject, period, unit and scale, reporting basis,
data vintage. Agreement then largely falls out of comparing them. Without that, the
alternative is per-document rules, which would not survive an unfamiliar PDF.

### The model proposes, the rules verify

An LLM is good at noticing that "revenue from services" and "Revenue from Operations"
may be the same metric, and poor at being checkable. Arithmetic is the reverse. So the
model is given latitude to find and propose, and every quantitative claim it makes is
re-derived independently:

1. It extracts facts, but a fact is discarded unless its quote is found verbatim in the
   source window. Grounding is enforced, not intended.
2. It adjudicates pairs the rules cannot settle. When it claims two figures reconcile by
   a change of scale, the arithmetic has to agree; when it claims a change of basis
   explains a gap, the basis has to actually differ.
3. Where rule and model disagree, the pair is recorded as `needs_review` rather than
   resolved.

### Deciding what to decide

The rule layer classifies each candidate pair before any model call.

| Values | Periods | Qualifiers | Verdict |
| --- | --- | --- | --- |
| agree | any | none differ | `corroborates`, by rule |
| agree | any | some differ | `corroborates`, caveat recorded |
| differ | either unknown | any | `insufficient_context` |
| differ | both known | only the period | `reconciled_by_context`, by rule |
| differ | both known | exactly one other | model explains it |
| differ | both known | none differ | model adjudicates |
| different units | any | any | `unrelated` |

One principle runs through the whole comparison: an unknown value is not a different
value. It applies in three places.

* A missing period means unknown, not "the same period as the other fact". An earlier
  version treated two undated facts as contemporaneous, which turned every difference in
  their values into a contradiction: 1,480 of them across two documents, 45% of all
  pairs.
* A unit that reduces to nothing once scale words are stripped, such as a bare `million`
  from a table whose header carried the currency, is an unknown dimension rather than a
  different one. Treating it as different stopped the earnings deck's `Rs. Cr` from ever
  being compared with the annual report's `million`.
* A qualifier recorded on one side and absent on the other is unknown. Counting it as a
  difference downgraded a genuine disagreement into one that looked explained.

Numbers are also compared at the precision they were printed. An earnings deck reporting
`1.4 Mn Tons` and an annual report reporting `1,429K tonnes` are the same figure at two
and four significant figures; a flat tolerance reads that 2% gap as a disagreement.

### The four cases

Reproduced by `python scripts/show_cases.py`. Every quote below is stored evidence, and
`python scripts/audit_grounding.py` re-checks all 490 of them against the pages they
cite.

**1. A fact corroborated across documents, expressed differently.**

| | annual report, p2 | earnings deck, p8 |
| --- | --- | --- |
| quote | "18,793 (1) Pin codes covered" | "Pin-code reach(1) 18,074 18,540 18,675 18,793" |
| metric | `Pin codes covered` | `Pin-code reach` |
| period | as of March 31, 2024 | Q4 FY24 |

Two documents, two names for the metric, and two ways of writing the period that
normalise to the same instant. The deck reports it as the last point of a quarterly
series. Settled by rule, with no model call.

**2. A genuine or likely contradiction.**

| | annual report, p4 | earnings deck, p6 |
| --- | --- | --- |
| quote | "1.6% EBITDA margin" | "₹76Cr / 0.9% Adj. EBITDA / Adj. EBITDA margin" |
| value | 1.6 per cent, FY24 | 0.9 per cent, FY24 |

Same company, same period, same unit, and nothing recorded distinguishes them, so the
rules raise it rather than explain it away. One figure is adjusted and the other is not,
a distinction neither document attached to the number itself. As reported, the two
disagree.

A second instance, across institutions, under the `contradicts` filter on
`/relations`: the RBI Annual Report projects real GDP growth of 6.5 per cent for
`2025-26` (p17) and the IMF Article IV projects 6.6 per cent for `FY2025/26` (p13). Both
period strings normalise to 2025-04-01. Without that step the two are never compared.

**3. An apparent contradiction explained by context.**

| | prospectus, p44 | annual report, p2 |
| --- | --- | --- |
| quote | "we provide our services in 17,488 postal index number (PIN) codes, as of December 31, 2021" | "18,793 (1) Pin codes covered" |
| period | as of December 31, 2021 | as of March 31, 2024 |

The same metric as case 1, across a two-year gap. Resolved as `different_period` by rule:
a company covering more PIN codes in 2024 than in 2021 is growth, not a contradiction.
Cases 1 and 3 together are the point of the system. The same measure corroborates when
the dates agree and reconciles when they do not.

**4. An extraction or reasoning failure, and how it is handled.**

* Six unreadable pages, found without any model. The IMF cover page yields no text at
  all, four earnings-deck slides are images, and prospectus p63 gives 53 characters.
  Each is recorded as a gap with a reason rather than silently contributing nothing.
* 29 proposed facts rejected because their quote could not be found verbatim in the
  window it came from. Reported separately from the 117 windows never read for want of a
  cached response and a key, which is unread text rather than a failed extraction.
  Combining the two would overstate the error rate fourfold.
* 506 pairs left as `insufficient_context`, because a fact without a parseable period
  cannot honestly be called contradictory.

The failures worth fixing next are table column attribution and canonicalisation
over-merging: `amount` swept together "Net Assets Amount" and "Public and Rights Issues
Amount".

### Engineering decisions and trade-offs

* **SQLite rather than a graph database.** Relations are one table with two foreign keys
  and the queries are joins. A graph store would not answer the hard question here, which
  is whether two facts are comparable at all, and one file means nothing to provision.
* **A fixed core plus an open qualifier map.** A prospectus, an earnings deck and an IMF
  staff report do not share a fact schema. Inventing one upfront gives either something
  so loose it says nothing, or something that breaks on the first unfamiliar document.
  Qualifier keys accumulate as documents introduce them, and none is named in the
  comparison logic.
* **Every model call cached by content hash.** Reruns cost nothing, tests never touch the
  network, and the committed cache is what makes the work reviewable without an account.
* **Long-context windows rather than pages.** A table header and its rows have to reach
  the model together or every column is misattributed.
* **No frontend build step.** Jinja templates and no JavaScript.
* **A pinned model rather than a floating alias.** The model name is part of every cache
  key, so an alias would replay old responses under a name that had changed meaning.

### AI tools used

An AI coding assistant was used throughout: to explore the starter documents, work
through the design, write code and tests, and review the plan before building. Several
review passes found problems that reading alone had missed, and the ones that mattered
came from running code rather than reading it. Three are described above because they
changed the design: evidence resolving to the wrong page, a missing period reading as a
matching one, and a unit comparison that refused to compare the figures the system exists
to compare.

`docs/design.md` is the design document from that process.

## Limitations and Next Steps

What does not work yet:

* **Run-to-run variance.** The model sits in the discovery path, so the same PDF can
  yield slightly different facts across runs. Temperature is zero and the cache is
  committed, so a clone is stable in practice, but this is a property of the design
  rather than something solved.
* **No OCR.** Image-only pages are detected and reported as gaps rather than silently
  contributing nothing. Adding OCR is a contained change at the ingest stage.
* **Tables are the weakest link.** Multi-column financial tables flatten under text
  extraction, and mapping a row of four numbers back to standalone against consolidated,
  and FY24 against FY23, is where wrong facts are most likely to originate.
* **Period coverage bounds everything.** A fact without a parseable period can never be
  part of a contradiction, by design. 66% of stored facts carry one, and the rest are why
  506 of 646 pairs sit in `insufficient_context`. Coverage varies by document rather than
  by model: the earnings deck labels almost everything FY24, the prospectus is prose.
* **Canonicalisation can over-merge.** Two similarly named metrics can collapse into one
  canonical id and manufacture a false comparison. `amount` is a live example.
* **Pairing is quadratic.** 0.39s at the roughly 1,250 facts the starter set produces and
  2.2s at 3,000, so it is not a problem at this size, but the blocking needs rewriting
  before many documents in one layer is realistic.
* **Recall is unmeasured and budget-bound.** There is no labelled ground truth here, so
  every stored fact can be shown to be grounded but not what fraction of the facts
  present were found. The committed run reads the densest windows of each document rather
  than all 158.

Next steps, in the order I would take them:

1. Attach periods from section and table context rather than only the sentence, since
   that limit gates everything else.
2. OCR fallback for image-only pages.
3. Table-aware extraction that preserves the header-to-cell relationship explicitly.
4. Embedding-backed blocking so pairing stops being quadratic.
5. A confidence-weighted view ranking contradictions by how much evidence supports each
   side.

## Additional Notes

**What ships, and what is computed.** The repository contains the six starter PDFs and
100 cached model responses. It contains no facts, relations or verdicts: those are
computed by the pipeline on every run, so what a reader sees is produced by the code
rather than copied from a prepared database. The cache exists so the work can be
evaluated without an API key of your own, and because the free tier cannot read this
corpus in a day.

**Nothing is hard-coded to these documents.** `tests/test_no_hardcoding.py` enforces that
rather than asserting it. It reads the source and fails if any string literal in a code
path names one of these documents, companies or metrics; docstrings explaining why a rule
exists are exempt, code is not. It also pins the only fixed vocabularies as generic roles
such as `source` and `basis`, and checks that a qualifier key the system has never seen is
compared correctly without code changes.

**Figures from the committed run:**

| | |
| --- | --- |
| documents / pages | 6 / 511 |
| facts stored | 490, all resolved to a page and verified against it |
| facts carrying a period | 321 (66%) |
| relations | 646 |
| corroborates / reconciled / contradicts | 37 / 79 / 24 |
| ungrounded facts rejected | 29 |
| model calls needed to reproduce | 0, since 100 are committed |
| tests | 140, no network access required |

No credentials are in the repository. `.env` is gitignored and `.env.example` shows the
shape.
