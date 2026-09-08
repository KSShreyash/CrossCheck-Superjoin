"""The brief forbids relying on hard-coded facts, filenames or schemas.

These tests make that checkable rather than asserted. They read the shipped
source and fail if anything specific to the starter documents has crept into
the code, so the guarantee survives future edits.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "factlayer"

# names, metrics and filenames peculiar to the documents this was built against
DOCUMENT_SPECIFIC = [
    "delhivery", "spoton", "imf", "rbi", "economic survey", "article iv",
    "prospectus", "annual report", "ebitda", "ptl freight", "pin code",
    "revenue from operations", "sortation", "gateways",
    "delhivery-", "-excerpt",
]
# ".pdf" is deliberately absent: a file extension is generic. The starter
# filenames themselves ("delhivery-", "-excerpt") are what would be damning.


def _string_literals(tree, source_lines):
    """Every string literal that is not a docstring.

    Docstrings explain why a rule exists and often name the document that
    motivated it. That is documentation, not logic, so it is exempt.
    """
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr):
                    docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                yield node.lineno, node.value


def test_no_document_specific_strings_in_code():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for lineno, value in _string_literals(tree, source.splitlines()):
            low = value.lower()
            for term in DOCUMENT_SPECIFIC:
                if term in low:
                    offenders.append(f"{path.name}:{lineno} contains {term!r}")
    assert not offenders, (
        "document-specific strings found in code paths; the system must work "
        "on PDFs it has never seen:\n  " + "\n  ".join(offenders))


def test_no_metric_or_entity_vocabulary_is_hard_coded():
    """Comparison logic must not carry a list of expected metrics.

    Qualifier keys are discovered from whatever the documents supply, so the
    only fixed vocabularies allowed are generic roles: provenance keys and the
    unit and transform kinds, none of which name a metric or a company.
    """
    from factlayer.reconcile import rules, verify
    generic = {"source", "publisher", "attribution", "reported_by", "reporter",
               "author", "institution"}
    assert rules.PROVENANCE_KEYS == generic, "provenance keys are roles, not metrics"
    assert verify._QUALIFIER_KINDS == {"basis", "vintage", "period", "scope",
                                       "segment"}, "transform kinds are generic"


def test_qualifier_comparison_accepts_keys_it_has_never_seen():
    """A new document inventing a qualifier must be handled without changes."""
    from factlayer.models import Fact
    from factlayer.reconcile.rules import qualifier_diff

    def _f(quals):
        f = Fact(1, "s", "m", "1", 1.0, None, "FY24", qualifiers=quals)
        f.canon_value, f.canon_unit = 1.0, "INR"
        f.period_start = f.period_end = "2024-03-31"
        return f

    diff = qualifier_diff(_f({"counterparty": "acme", "audited": "yes"}),
                          _f({"counterparty": "beta", "audited": "yes"}))
    assert diff == {"counterparty": ("acme", "beta")}
