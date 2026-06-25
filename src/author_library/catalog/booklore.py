"""Booklore metadata resolver.

Queries the Booklore MariaDB for curated book metadata (title, author,
publisher, publication year, ISBN, page count) given a file path.  The
metadata is used as ingestion overrides so that the pipeline uses
human-curated data rather than whatever the PDF/EPUB metadata contains.

Gracefully degrades: if Booklore is unreachable or the file is not
found in the catalog, returns an empty dict and logs a warning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Default connection URL matching the docker-compose setup.
# Override via BOOKLORE_DB_URL env var.
_DEFAULT_URL = "mysql+aiomysql://parlour:parlour_read@172.21.0.2:3306/booklore"

_METADATA_QUERY = """
    SELECT
        bm.title,
        GROUP_CONCAT(DISTINCT a.name ORDER BY bma.sort_order SEPARATOR '; ') AS author,
        bm.publisher,
        bm.published_date,
        bm.isbn_13,
        bm.isbn_10,
        bm.page_count,
        bm.language
    FROM book b
    JOIN book_metadata bm ON b.id = bm.book_id
    JOIN book_file bf ON bf.book_id = b.id
    LEFT JOIN book_metadata_author_mapping bma ON bma.book_id = b.id
    LEFT JOIN author a ON bma.author_id = a.id
    WHERE bf.file_name = %s
    GROUP BY bm.book_id
    LIMIT 1
"""


async def resolve_metadata(
    file_path: str | Path,
    *,
    db_url: str = _DEFAULT_URL,
) -> dict[str, Any]:
    """Look up Booklore metadata for a file and return ingestion overrides.

    Args:
        file_path: Path to the book file (only the basename is matched).
        db_url: MariaDB connection URL.  Defaults to local Booklore instance.

    Returns:
        Dict of override keys (title, author, publisher, publication_year,
        isbn, language) with values from Booklore.  Empty dict if not found
        or if Booklore is unreachable.
    """
    filename = Path(file_path).name

    try:
        import aiomysql
    except ImportError:
        log.warning(
            "booklore_resolver_unavailable",
            reason="aiomysql not installed — run: uv add aiomysql",
        )
        return {}

    try:
        # Parse connection URL components
        conn_kwargs = _parse_db_url(db_url)
        conn = await aiomysql.connect(**conn_kwargs)
    except Exception as exc:
        log.warning(
            "booklore_connection_failed",
            error=str(exc),
            db_url=db_url,
        )
        return {}

    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(_METADATA_QUERY, (filename,))
            row = await cur.fetchone()

        if row is None:
            log.debug("booklore_file_not_found", filename=filename)
            return {}

        overrides: dict[str, Any] = {}

        if row.get("title"):
            overrides["title"] = row["title"]

        if row.get("author"):
            # Take first author if multiple (separated by '; ')
            authors = row["author"]
            first_author = authors.split(";")[0].strip()
            overrides["author"] = first_author

        if row.get("publisher"):
            overrides["publisher"] = row["publisher"]

        if row.get("published_date"):
            try:
                overrides["publication_year"] = row["published_date"].year
            except (AttributeError, ValueError):
                pass

        isbn = row.get("isbn_13") or row.get("isbn_10")
        if isbn:
            overrides["isbn"] = isbn

        if row.get("language"):
            overrides["language"] = row["language"]

        log.info(
            "booklore_metadata_resolved",
            filename=filename,
            title=overrides.get("title"),
            author=overrides.get("author"),
            keys=list(overrides.keys()),
        )

        return overrides

    except Exception as exc:
        log.warning(
            "booklore_query_failed",
            filename=filename,
            error=str(exc),
        )
        return {}
    finally:
        conn.close()


def _parse_db_url(url: str) -> dict[str, Any]:
    """Parse a mysql connection URL into aiomysql.connect() kwargs.

    Accepts: mysql+aiomysql://user:pass@host:port/db
             mysql://user:pass@host:port/db
             user:pass@host:port/db
    """
    import urllib.parse

    # Strip scheme prefix
    for prefix in ("mysql+aiomysql://", "mysql://", "mariadb://"):
        if url.startswith(prefix):
            url = url[len(prefix):]
            break

    # Split user:pass@host:port/db
    if "@" in url:
        userinfo, hostpart = url.rsplit("@", 1)
    else:
        userinfo, hostpart = "", url

    user, password = "", ""
    if ":" in userinfo:
        user, password = userinfo.split(":", 1)
        password = urllib.parse.unquote(password)
    else:
        user = userinfo

    if "/" in hostpart:
        hostport, db = hostpart.split("/", 1)
    else:
        hostport, db = hostpart, "booklore"

    if ":" in hostport:
        host, port_str = hostport.split(":", 1)
        port = int(port_str)
    else:
        host, port = hostport, 3306

    kwargs: dict[str, Any] = {
        "host": host or "localhost",
        "port": port,
        "user": user or "root",
        "db": db or "booklore",
    }
    if password:
        kwargs["password"] = password

    return kwargs
