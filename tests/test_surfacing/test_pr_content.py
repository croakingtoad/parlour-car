"""Tests for N2: PR content generator."""

from __future__ import annotations

from author_library.surfacing.connection_scanner import ScanResult, StagedConnection
from author_library.surfacing.pr_content import PRContent, generate_pr_content


def _make_connection(
    *,
    confidence_level: str = "high",
    connection_type: str = "thematic_parallel",
    target_work_id: str = "target-work-1",
    explanation: str = "Shared themes.",
) -> StagedConnection:
    """Helper to create a StagedConnection."""
    labels = {
        "high": "This directly engages with",
        "medium": "This appears to connect to",
        "low": "You might find this relevant",
    }
    return StagedConnection(
        source_chunk_id="src-1",
        target_chunk_id="tgt-1",
        source_work_id="source-work",
        target_work_id=target_work_id,
        connection_type=connection_type,
        confidence_level=confidence_level,
        confidence_label=labels.get(confidence_level, "Related"),
        source_excerpt="Source passage excerpt...",
        target_excerpt="Target passage excerpt...",
        explanation=explanation,
    )


class TestPRContentGeneration:
    """Tests for generate_pr_content."""

    def test_single_connection_pr(self) -> None:
        """Single connection produces valid PR content."""
        conn = _make_connection()
        result = ScanResult(
            work_id="new-work",
            connections=[conn],
            by_confidence={"high": [conn]},
            by_target_work={"target-work-1": [conn]},
            total_found=1,
        )

        pr = generate_pr_content(result, work_title="Faith, Hope and Poetry")
        assert "connection found" in pr.title.lower()
        assert "Faith, Hope and Poetry" in pr.title
        assert pr.pr_type == "new_connection"
        assert len(pr.affected_notes) == 1

    def test_multiple_connections_pr(self) -> None:
        """Multiple connections produce correct summary."""
        conns = [
            _make_connection(confidence_level="high", target_work_id="work-a"),
            _make_connection(confidence_level="medium", target_work_id="work-b"),
            _make_connection(confidence_level="low", target_work_id="work-c"),
        ]
        result = ScanResult(
            work_id="new-work",
            connections=conns,
            by_confidence={
                "high": [conns[0]],
                "medium": [conns[1]],
                "low": [conns[2]],
            },
            by_target_work={
                "work-a": [conns[0]],
                "work-b": [conns[1]],
                "work-c": [conns[2]],
            },
            total_found=3,
        )

        pr = generate_pr_content(result)
        assert "connections found" in pr.title.lower()
        assert "Strong" in pr.body or "Strong Connections" in pr.body
        assert len(pr.affected_notes) == 3

    def test_pr_includes_author_in_title(self) -> None:
        """PR title includes author when provided."""
        conn = _make_connection()
        result = ScanResult(
            work_id="new-work",
            connections=[conn],
            by_confidence={"high": [conn]},
            by_target_work={"target-work-1": [conn]},
            total_found=1,
        )

        pr = generate_pr_content(
            result,
            work_title="Faith, Hope and Poetry",
            work_author="Malcolm Guite",
        )
        assert "Malcolm Guite" in pr.title

    def test_pr_body_has_confidence_table(self) -> None:
        """PR body includes confidence level breakdown table."""
        conns = [
            _make_connection(confidence_level="high"),
            _make_connection(confidence_level="medium"),
        ]
        result = ScanResult(
            work_id="new-work",
            connections=conns,
            by_confidence={"high": [conns[0]], "medium": [conns[1]]},
            by_target_work={"target-work-1": conns},
            total_found=2,
        )

        pr = generate_pr_content(result)
        assert "Confidence" in pr.body
        assert "Count" in pr.body

    def test_pr_labels(self) -> None:
        """PR includes appropriate labels."""
        conn = _make_connection(confidence_level="high")
        result = ScanResult(
            work_id="new-work",
            connections=[conn],
            by_confidence={"high": [conn]},
            by_target_work={"target-work-1": [conn]},
            total_found=1,
        )

        pr = generate_pr_content(result)
        assert "parlour/new-connections" in pr.labels
        assert "parlour/high-confidence" in pr.labels

    def test_empty_result_pr(self) -> None:
        """Empty result produces minimal PR content."""
        result = ScanResult(work_id="new-work", total_found=0)
        pr = generate_pr_content(result)
        assert "0 new connection" in pr.body or "**0 new" in pr.body
