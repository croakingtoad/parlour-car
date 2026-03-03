"""N3: Batch surfacing — group related connections into single PR.

Don't create one PR per connection — group related ones. E.g.,
"New connections found after ingesting [book]: 3 passages link to
[other book], 2 thematic parallels with [video]".

Coordinates with N1 (ConnectionScanner) and N2 (PR content generator)
to produce batched PRs through the vault sync workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from author_library.surfacing.connection_scanner import ConnectionScanner, ScanResult
from author_library.surfacing.pr_content import PRContent, generate_pr_content

if TYPE_CHECKING:
    from author_library.cache import CacheManager
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


@dataclass
class BatchSurfacingResult:
    """Result of a batch surfacing operation."""

    work_id: str
    scan_result: ScanResult | None = None
    pr_content: PRContent | None = None
    pr_created: bool = False
    pr_id: str | None = None
    pr_url: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        result: dict[str, Any] = {
            "work_id": self.work_id,
            "pr_created": self.pr_created,
        }
        if self.scan_result:
            result["total_connections"] = self.scan_result.total_found
            result["by_confidence"] = {
                level: len(conns)
                for level, conns in self.scan_result.by_confidence.items()
            }
            result["target_works"] = list(self.scan_result.by_target_work.keys())
        if self.pr_content:
            result["pr_title"] = self.pr_content.title
        if self.pr_id:
            result["pr_id"] = self.pr_id
        if self.pr_url:
            result["pr_url"] = self.pr_url
        if self.errors:
            result["errors"] = self.errors
        return result


class BatchSurfacer:
    """Orchestrates batch surfacing: scan → group → generate PR.

    Combines the connection scanner, PR content generator, and
    PR creation into a single cohesive pipeline triggered after
    a work is ingested.
    """

    def __init__(
        self,
        settings: Settings,
        storage: StorageManager,
        embedding_provider: EmbeddingProvider,
        cache_manager: CacheManager | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._embedding = embedding_provider
        self._cache = cache_manager
        self._scanner = ConnectionScanner(
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
            cache_manager=cache_manager,
        )

    async def surface_after_ingestion(
        self,
        work_id: str,
        *,
        work_title: str = "",
        work_author: str = "",
        confidence_threshold: float = 0.4,
        min_connections_for_pr: int = 1,
    ) -> BatchSurfacingResult:
        """Run the full batch surfacing pipeline after ingestion.

        1. Scan for new connections (N1)
        2. Generate PR content (N2)
        3. Return result (PR creation delegated to vault sync layer)

        Args:
            work_id: The newly ingested work.
            work_title: Title for PR readability.
            work_author: Author for PR readability.
            confidence_threshold: Minimum confidence to include.
            min_connections_for_pr: Skip PR if fewer connections found.

        Returns:
            BatchSurfacingResult with scan results and PR content.
        """
        result = BatchSurfacingResult(work_id=work_id)

        # Step 1: Scan for connections
        log.info("batch_surfacing_start", work_id=work_id)
        try:
            scan = await self._scanner.scan_new_connections(
                work_id,
                confidence_threshold=confidence_threshold,
            )
            result.scan_result = scan
            result.errors.extend(scan.errors)
        except Exception as exc:
            error_msg = f"Connection scan failed: {exc}"
            log.error("batch_surfacing_scan_failed", error=error_msg)
            result.errors.append(error_msg)
            return result

        # Step 2: Check if we have enough connections for a PR
        if scan.total_found < min_connections_for_pr:
            log.info(
                "batch_surfacing_below_threshold",
                work_id=work_id,
                total_found=scan.total_found,
                threshold=min_connections_for_pr,
            )
            return result

        # Step 3: Generate PR content
        try:
            pr_content = generate_pr_content(
                scan,
                work_title=work_title,
                work_author=work_author,
            )
            result.pr_content = pr_content
        except Exception as exc:
            error_msg = f"PR content generation failed: {exc}"
            log.error("batch_surfacing_pr_content_failed", error=error_msg)
            result.errors.append(error_msg)
            return result

        log.info(
            "batch_surfacing_complete",
            work_id=work_id,
            total_connections=scan.total_found,
            pr_title=pr_content.title,
            affected_notes=len(pr_content.affected_notes),
        )

        return result

    async def surface_and_create_pr(
        self,
        work_id: str,
        *,
        work_title: str = "",
        work_author: str = "",
        confidence_threshold: float = 0.4,
        min_connections_for_pr: int = 1,
        pr_manager: Any = None,
        git_sync: Any = None,
        vault_path: str = "",
    ) -> BatchSurfacingResult:
        """Run batch surfacing and create a PR if connections found.

        Extended version that integrates with the vault sync layer
        to create actual PRs. Requires pr_manager and git_sync
        instances from parlour-notes.

        Args:
            work_id: The newly ingested work.
            work_title: Title for PR readability.
            work_author: Author for PR readability.
            confidence_threshold: Minimum confidence to include.
            min_connections_for_pr: Skip PR if fewer connections found.
            pr_manager: PRManager instance for creating GitHub PRs.
            git_sync: GitSync functions for branch management.
            vault_path: Path to the vault root.

        Returns:
            BatchSurfacingResult with PR creation status.
        """
        result = await self.surface_after_ingestion(
            work_id,
            work_title=work_title,
            work_author=work_author,
            confidence_threshold=confidence_threshold,
            min_connections_for_pr=min_connections_for_pr,
        )

        if not result.pr_content or not pr_manager:
            return result

        # Create PR via parlour-notes PR manager
        try:
            from author_library.surfacing.connection_scanner import StagedConnection

            # Build a work slug for the branch name
            work_slug = work_id.replace("--", "-").replace("/", "-")
            branch_name = f"parlour/enrichment-{work_slug}"

            pr_info = await pr_manager.create_pr(
                branch_name=branch_name,
                title=result.pr_content.title,
                body=result.pr_content.body,
                affected_notes=result.pr_content.affected_notes,
                pr_type=result.pr_content.pr_type,
                labels=result.pr_content.labels,
            )

            if pr_info:
                result.pr_created = True
                result.pr_id = str(pr_info.pr_id)
                result.pr_url = pr_info.url

                log.info(
                    "batch_surfacing_pr_created",
                    work_id=work_id,
                    pr_id=result.pr_id,
                    pr_url=result.pr_url,
                )
        except Exception as exc:
            error_msg = f"PR creation failed: {exc}"
            log.error("batch_surfacing_pr_failed", error=error_msg)
            result.errors.append(error_msg)

        return result
