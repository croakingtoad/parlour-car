"""Guards against test teardown deleting production Neo4j data.

Two incidents motivate this file:

* 2026-07-02 — a global theme deduplication run from the test suite wiped
  every production Theme node.
* 2026-08-13 — an autouse cleanup fixture in tests/test_intelligence carried
  the production prefix "malcolm-guite--" and deleted all 5 Malcolm Guite
  works (3,495 Chunk nodes) from the production graph. The same fixture's
  orphan sweep also removed 3 real Author nodes.

Neo4j tests default to a separate, disposable instance. The production-graph
guard fixture refuses to run graph tests against a graph holding production
data unless the operator opts in explicitly, and prefix scoping confines
teardown to test data. These layered protections respond to the incidents
above; these tests assert that every destructive Cypher statement in the test
tree stays inside the test-- namespace, so the class of bug cannot silently
return.
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

def _python_files() -> list[Path]:
    return sorted(p for p in TESTS_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


#: The root conftest owns the single audited unscoped delete: the
#: reset_disposable_graph fixture, which re-proves every Work node is
#: disposable immediately before clearing. Exempting the whole FILE was too
#: broad — an unguarded wipe anywhere in conftest.py passed — so the exemption
#: is the fixture's source span only.
_AUDITED_FILE = Path("conftest.py")
_AUDITED_FUNCTION = "reset_disposable_graph"


def _this_file(path: Path) -> bool:
    return path.resolve() == Path(__file__).resolve()


def _is_audited(path: Path) -> bool:
    """True for the file that owns the audited exemption (name checks only)."""
    return path.relative_to(TESTS_ROOT) == _AUDITED_FILE


def _strip_docstrings(text: str) -> str:
    """Remove every docstring: module, class and function.

    These files describe the dangerous patterns they replaced ("the previous
    implementation used MATCH (n) DETACH DELETE n"), and prose about a wipe is
    not a wipe. Safe to strip wholesale because executable Cypher is always a
    plain string expression passed to execute_write, never a docstring.
    """
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                text = text.replace(doc, "", 1)
    return text


def _audited_source(path: Path, original: str) -> str:
    """Transformed source of the audited fixture, or "" if absent.

    Offsets cannot be compared after normalisation (joining literals makes the
    text unparseable), so the exemption is decided by counting matches inside
    this snippet versus the whole file.
    """
    import ast

    if path.relative_to(TESTS_ROOT) != _AUDITED_FILE:
        return ""
    try:
        tree = ast.parse(original)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == _AUDITED_FUNCTION
        ):
            return _code_only(ast.get_source_segment(original, node) or "")
    return ""


def _join_concatenated_literals(text: str) -> str:
    """Collapse adjacent string literals so split Cypher is seen as one string.

    `"MATCH (n) " "DETACH DELETE n"` and the `+`-joined form both evade a
    regex that expects contiguous text, and splitting across literals is the
    prevailing style in these very files.
    """
    return re.sub(r"""["']\s*\+?\s*["']""", "", text)


def _code_only(text: str) -> str:
    """Strip ``#`` comments so prose about a bad pattern is not the bad pattern.

    The cleanup fixtures document the rules they follow ("Never use unscoped
    MATCH (n) DETACH DELETE n"), and those sentences must not read as
    violations.
    """
    text = _strip_docstrings(text)
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
    return _join_concatenated_literals("\n".join(out))


def _executes_cypher(text: str) -> bool:
    """True if the file hands Cypher to Neo4j anywhere.

    Deliberately file-wide rather than a proximity window: assigning the
    statement to a module-level constant and executing it elsewhere defeated
    the window, so this fails closed. Prose is excluded by stripping the
    module docstring instead of by distance.
    """
    return bool(re.search(r"execute_write|execute_read|session\.run|\.run\(", text))


def _touches_graph(text: str) -> bool:
    """True if the file could actually write to Neo4j.

    Pure-unit files (e.g. work_id slug generation) legitimately mention real
    author names as expected values without ever opening a connection.
    """
    return bool(re.search(r"neo4j|graph_repo|upsert_\w*node|execute_write", text, re.I))


def _find_executed(path: Path, pattern: re.Pattern[str]) -> bool:
    """True if `pattern` matches destructive Cypher this file could execute.

    Matches inside the audited fixture span are permitted; a match anywhere
    else in the same file is not.
    """
    original = path.read_text(encoding="utf-8")
    text = _code_only(original)
    if not _executes_cypher(text):
        return False
    total = len(pattern.findall(text))
    if not total:
        return False
    audited = len(pattern.findall(_audited_source(path, original)))
    return total > audited


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
            if not _this_file(p) and _find_executed(p, _WIPE_RE)
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


class TestGuardListsMatchReality:
    """The hand-maintained lists must not silently drift from the corpus.

    A stale list is a live hazard in both directions: an author missing from
    PRODUCTION_AUTHOR_PREFIXES is one the static check will not ban from tests,
    and (before the guard was inverted) was one reset_disposable_graph would
    have treated as disposable.
    """

    async def test_production_prefixes_cover_every_author_in_pg(self) -> None:
        """Every author slug derivable from the works table must be listed."""
        import os

        import asyncpg

        from author_library.catalog.pipeline import ClassificationPipeline

        # Deliberately the PRODUCTION database, read-only: that is the corpus
        # the lists must describe. tests/conftest.py points DB_POSTGRES_URL at
        # the test DB, so build the production DSN explicitly.
        dsn = os.environ.get(
            "PARLOUR_PRODUCTION_POSTGRES_URL",
            "postgresql://author_library:author_library@localhost:5432/author_library",
        )
        try:
            conn = await asyncpg.connect(dsn)
        except Exception as exc:
            pytest.skip(f"production PostgreSQL not reachable: {exc}")

        try:
            rows = await conn.fetch("SELECT DISTINCT author FROM works WHERE author IS NOT NULL")
        finally:
            await conn.close()

        missing = set()
        for row in rows:
            slug = ClassificationPipeline._generate_work_id(row["author"], "x").split("--")[0]
            if slug and slug not in PRODUCTION_AUTHOR_PREFIXES:
                missing.add(slug)

        assert not missing, (
            f"author slug(s) present in the corpus but absent from "
            f"PRODUCTION_AUTHOR_PREFIXES: {sorted(missing)}. Add them to "
            f"tests/conftest.py — the static ban and the reset guard both key "
            f"off that list."
        )


class TestNoProductionIdsDerivedAtRuntime:
    """Catch production prefixes produced by slugifying a human-readable name.

    The static literal check is blind to this: a test writes
    author="Malcolm Guite", the classification pipeline slugifies it into
    malcolm-guite--<title>, and a production-namespace Work node lands in
    whatever graph the suite is pointed at. That is how production-named
    fixture nodes reached the real graph. Asserting on the OUTPUT of
    _generate_work_id closes the gap and needs no second list to maintain.
    """

    #: Quoted literal that could plausibly be a person's name.
    _NAME_LITERAL = re.compile(
        r"""["']([A-Z][A-Za-z.'\u2019-]*(?:[ ,]+[A-Z][A-Za-z.'\u2019-]*)+)["']"""
    )

    def test_no_author_name_slugifies_to_a_production_prefix(self) -> None:
        from author_library.catalog.pipeline import ClassificationPipeline

        offenders: dict[str, set[str]] = {}
        for path in _python_files():
            if _this_file(path) or _is_audited(path):
                continue
            text = _code_only(path.read_text(encoding="utf-8"))
            if not _touches_graph(text):
                continue
            for name in set(self._NAME_LITERAL.findall(text)):
                if len(name) > 60:
                    continue
                slug = ClassificationPipeline._generate_work_id(name, "x").split("--")[0]
                if slug in PRODUCTION_AUTHOR_PREFIXES:
                    offenders.setdefault(str(path.relative_to(TESTS_ROOT)), set()).add(
                        f"{name!r} -> {slug}--"
                    )

        assert not offenders, (
            "author name(s) in graph-touching tests slugify into the PRODUCTION "
            f"work_id namespace: { {k: sorted(v) for k, v in offenders.items()} }. "
            "Use a name whose slug lands under the test namespace (e.g. 'Test')."
        )
