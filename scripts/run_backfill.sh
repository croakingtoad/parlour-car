#!/usr/bin/env bash
cd "$(dirname "$0")/.."
set -a
source .env
set +a
exec uv run python scripts/backfill_entities.py "$@"
