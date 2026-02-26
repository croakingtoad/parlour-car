"""MCP client test script — tests tool calls without Claude Desktop.

Usage:
    # Start the server first:
    SERVER_TRANSPORT=streamable-http SERVER_PORT=8080 uv run python -m author_library

    # Then run this:
    uv run python test_mcp_client.py
"""

import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


SERVER_URL = "http://localhost:8080/mcp"


async def test_connection():
    """Test basic MCP connection and list tools."""
    print(f"Connecting to {SERVER_URL}...")

    async with streamable_http_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected and initialized.")

            # List tools
            tools_result = await session.list_tools()
            tools = tools_result.tools
            print(f"\n{len(tools)} tools available:")
            for t in sorted(tools, key=lambda x: x.name):
                print(f"  - {t.name}")

            return tools


async def test_health_check():
    """Test health_check tool."""
    print("\n--- health_check ---")

    async with streamable_http_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool("health_check", {})
            data = json.loads(result.content[0].text)
            print(json.dumps(data, indent=2))
            return data


async def test_library_stats():
    """Test library_stats tool."""
    print("\n--- library_stats ---")

    async with streamable_http_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool("library_stats", {})
            data = json.loads(result.content[0].text)
            print(json.dumps(data, indent=2))
            return data


async def test_list_books():
    """Test list_books tool."""
    print("\n--- list_books ---")

    async with streamable_http_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool("list_books", {})
            data = json.loads(result.content[0].text)
            print(json.dumps(data, indent=2))
            return data


async def test_job_status():
    """Test job_status tool (list all jobs)."""
    print("\n--- job_status (all jobs) ---")

    async with streamable_http_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool("job_status", {})
            data = json.loads(result.content[0].text)
            print(json.dumps(data, indent=2))
            return data


async def test_ingest_book(file_path: str, author: str):
    """Test ingest_book tool."""
    print(f"\n--- ingest_book ({file_path}) ---")

    async with streamable_http_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool("ingest_book", {
                "file_path": file_path,
                "subject_author_id": author,
            })
            data = json.loads(result.content[0].text)
            print(json.dumps(data, indent=2))
            return data


async def main():
    test = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test in ("all", "connect"):
        await test_connection()

    if test in ("all", "health"):
        await test_health_check()

    if test in ("all", "stats"):
        await test_library_stats()

    if test in ("all", "books"):
        await test_list_books()

    if test in ("all", "jobs"):
        await test_job_status()

    if test == "ingest":
        file_path = sys.argv[2] if len(sys.argv) > 2 else "/home/marty/repos/parlour-car/test-corpus/fred-rogers/senate-testimony.html"
        author = sys.argv[3] if len(sys.argv) > 3 else "fred-rogers"
        await test_ingest_book(file_path, author)

    print("\n✓ All tests completed.")


if __name__ == "__main__":
    asyncio.run(main())
