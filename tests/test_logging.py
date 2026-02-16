"""Tests for logging configuration."""

from __future__ import annotations

import structlog

from author_library.logging import (
    correlation_id_var,
    get_logger,
    new_correlation_id,
    setup_logging,
)


class TestCorrelationId:
    def test_new_correlation_id_sets_contextvar(self) -> None:
        cid = new_correlation_id()
        assert len(cid) == 12
        assert correlation_id_var.get() == cid

    def test_successive_ids_differ(self) -> None:
        id1 = new_correlation_id()
        id2 = new_correlation_id()
        assert id1 != id2


class TestSetupLogging:
    def test_console_format(self) -> None:
        setup_logging(level="DEBUG", log_format="console")
        log = structlog.get_logger("test")
        assert log is not None

    def test_json_format(self) -> None:
        setup_logging(level="INFO", log_format="json")
        log = structlog.get_logger("test")
        assert log is not None


class TestGetLogger:
    def test_returns_bound_logger(self) -> None:
        setup_logging(level="DEBUG", log_format="console")
        log = get_logger("test.module")
        assert log is not None

    def test_with_initial_context(self) -> None:
        setup_logging(level="DEBUG", log_format="console")
        log = get_logger("test.module", component="parser")
        assert log is not None
