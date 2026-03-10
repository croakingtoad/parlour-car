#!/usr/bin/env bash
cd "$(dirname "$0")"
set -a
source .env
set +a
exec env SERVER_TRANSPORT=streamable-http SERVER_PORT=8080 uv run python -m author_library
