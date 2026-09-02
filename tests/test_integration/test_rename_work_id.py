"""Safety integration tests for the work_id rename script.

This test intentionally targets the disposable Neo4j fixture only.  Current
production-like FK definitions are immediate, so an execution must refuse
before it attempts either a PostgreSQL or graph write.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.test_integration.conftest import SKIP_NO_DB

if TYPE_CHECKING:
    from author_library.storage.manager import StorageManager

_SCRIPT = Path(__file__).parents[2] / "scripts" / "rename_work_id.py"
_SPEC = importlib.util.spec_from_file_location("rename_work_id_integration", _SCRIPT)
assert _SPEC and _SPEC.loader
rename = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rename
_SPEC.loader.exec_module(rename)


@SKIP_NO_DB
@pytest.mark.asyncio
async def test_execution_refuses_immediate_fks_before_cross_store_writes(
    clean_storage: StorageManager,
    assert_graph_is_disposable: None,
) -> None:
    old_id = "test--rename-source"
    new_id = "test--rename-target"
    await clean_storage.pg.execute(
        """
        INSERT INTO works (
            work_id, title, author, source_class, source_class_note,
            publication_year, publisher, format_ingested, word_count,
            genre_tags, subject_headings
        ) VALUES ($1, 'Source', 'Test Author', 'secondary',
                  'safe disposable test record', 2026, 'Test', 'txt', 1,
                  ARRAY['test'], ARRAY['test'])
        """,
        old_id,
    )
    await clean_storage.neo4j.execute_write(
        "CREATE (:Work {work_id: $work_id})", {"work_id": old_id}
    )

    with pytest.raises(rename.RenameError, match="immediate foreign key"):
        await rename.run(clean_storage, old_id, new_id, execute=True)

    assert await clean_storage.pg.fetch_val(
        "SELECT count(*) FROM works WHERE work_id = $1", old_id
    ) == 1
    records = await clean_storage.neo4j.execute_read(
        "MATCH (w:Work {work_id: $work_id}) RETURN count(w) AS n",
        {"work_id": old_id},
    )
    assert records[0]["n"] == 1
