"""Guards against test teardown deleting production Neo4j data.

Two incidents motivate this file:

* 2026-07-02 — a global theme deduplication run from the test suite wiped
  every production Theme node.
* 2026-08-13 — an autouse cleanup fixture in tests/test_intelligence carried
  the production prefix "malcolm-guite--" and deleted all 5 Malcolm Guite
  works (3,495 Chunk nodes) from the production graph. The same fixture's
  orphan sweep also removed 3 real Author nodes.

Neo4j Community Edition serves a single database, so TEST_NEO4J_URL usually
points at the graph holding the production corpus. Prefix scoping is the only
thing standing between teardown and real data. These tests assert that every
destructive Cypher statement in the test tree stays inside the test--
namespace, so the class of bug cannot silently return.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import PRODUCTION_AUTHOR_PREFIXES, TEST_NAMESPACE

TESTS_ROOT = Path(__file__).resolve().parent.parent

#: Statements that delete Neo4j data.
_DELETE_RE = re.compile(r"(DETACH\s+DELETE|(?<!DETACH\s)\bDELETE\s+[a-zA-Z])", re.I)

#: An orphan sweep that DELETES: matches nodes merely for being unreferenced.
#: A read-only `WHERE NOT (w)--() RETURN count(w)` assertion is fine.
_ORPHAN_SWEEP_RE = re.compile(
    r"WHERE\s+NOT\s*\(\s*\w+\s*\)\s*--\s*\(\s*\)[^\"\']*?DELETE",
    re.I | re.S,
)

#: An unscoped whole-graph wipe.
_WIPE_RE = re.compile(r"MATCH\s*\(\s*\w*\s*\)\s*DETACH\s+DELETE", re.I)

#: How far back to look for the call that executes a matched statement.
_EXEC_WINDOW = 300


def _python_files() -> list[Path]:
    return sorted(p for p in TESTS_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


#: The root conftest owns the single audited unscoped delete: the
#: reset_disposable_graph fixture, which re-proves the graph holds no
#: production Work nodes immediately before clearing it. Exempt by path so the
#: exemption cannot silently spread to other files.
_AUDITED_UNSCOPED_DELETE = {Path("conftest.py")}


def _this_file(path: Path) -> bool:
    return path.resolve() == Path(__file__).resolve()


def _is_audited(path: Path) -> bool:
    return path.relative_to(TESTS_ROOT) in _AUDITED_UNSCOPED_DELETE


def _code_only(text: str) -> str:
    """Strip ``#`` comments so prose about a bad pattern is not the bad pattern.

    The cleanup fixtures document the rules they follow ("Never use unscoped
    MATCH (n) DETACH DELETE n"), and those sentences must not read as
    violations.
    """
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if "#" in line:
            # Crude, but these files put Cypher in string literals and
            # comments at end-of-line, never a literal '#' inside Cypher.
            line = line.split("#", 1)[0]
        out.append(line)
    return "\n".join(out)


def _is_executed(text: str, start: int) -> bool:
    """True if a matched statement is actually handed to Neo4j.

    Both cleanup conftests document the dangerous pattern they replaced
    inside their module docstrings. Describing `MATCH (n) DETACH DELETE n`
    is not running it, so only flag statements near an execute call.
    """
    window = text[max(0, start - _EXEC_WINDOW) : start]
    return bool(re.search(r"execute_write|execute_read|session\.run|\.run\(", window))


def _touches_graph(text: str) -> bool:
    """True if the file could actually write to Neo4j.

    Pure-unit files (e.g. work_id slug generation) legitimately mention real
    author names as expected values without ever opening a connection.
    """
    return bool(re.search(r"neo4j|graph_repo|upsert_\w*node|execute_write", text, re.I))


def _find_executed(path: Path, pattern: re.Pattern[str]) -> bool:
    """True if `pattern` matches a statement this file actually executes."""
    text = _code_only(path.read_text(encoding="utf-8"))
    return any(_is_executed(text, m.start()) for m in pattern.finditer(text))


class TestNoDestructiveUnscopedCypher:
    def test_no_orphan_sweeps(self) -> None:
        """No test may delete nodes merely because they are unreferenced.

        An orphan sweep cannot tell a leftover test node from a production
        entity that happens to have no edges right now.
        """
        offenders = [
            str(p.relative_to(TESTS_ROOT))
            for p in _python_files()
            if not _this_file(p) and _find_executed(p, _ORPHAN_SWEEP_RE)
        ]
        assert not offenders, (
            "Orphan-sweep cleanup found in: "
            f"{offenders}. Scope entity cleanup to the {TEST_NAMESPACE!r} "
            "namespace instead (see tests/test_graph/conftest.py)."
        )

    def test_no_whole_graph_wipes(self) -> None:
        """No test may run MATCH (n) DETACH DELETE n."""
        offenders = [
            str(p.relative_to(TESTS_ROOT))
            for p in _python_files()
            if not _this_file(p) and not _is_audited(p) and _find_executed(p, _WIPE_RE)
        ]
        assert not offenders, f"Unscoped whole-graph delete found in: {offenders}"


class TestNoProductionIdsInTests:
    #: Single source of truth lives in tests/conftest.py so the runtime check
    #: in reset_disposable_graph and this static check cannot drift apart.
    PRODUCTION_PREFIXES = PRODUCTION_AUTHOR_PREFIXES

    @pytest.mark.parametrize("prefix", PRODUCTION_PREFIXES)
    def test_no_production_work_ids_created_by_tests(self, prefix: str) -> None:
        """Test fixtures must not use production author/work identifiers."""
        offenders: list[str] = []
        for path in _python_files():
            # Skip this file and the root conftest: they NAME the production
            # prefixes in order to ban them, rather than using them as data.
            if _this_file(path) or _is_audited(path):
                continue
            text = _code_only(path.read_text(encoding="utf-8"))
            if not _touches_graph(text):
                continue
            # Only flag identifier-ish usage (quoted), not prose in comments.
            if re.search(rf"""["']{re.escape(prefix)}""", text):
                offenders.append(str(path.relative_to(TESTS_ROOT)))
        assert not offenders, (
            f"Production identifier {prefix!r} used as test data in {offenders}. "
            f"Use the {TEST_NAMESPACE!r} namespace so cleanup cannot reach real data."
        )
