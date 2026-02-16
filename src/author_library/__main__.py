"""Entry point for `python -m author_library`."""

from __future__ import annotations

import asyncio
import sys

from author_library.config import get_settings
from author_library.server import run_server


def main() -> None:
    """Run the Author Library MCP server."""
    settings = get_settings()
    try:
        asyncio.run(run_server(settings))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Fatal: {exc}", file=sys.stderr)
        sys.exit(1)


main()
