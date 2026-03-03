"""Surfacing — find forgotten connections across the knowledge base.

Active surfacing (N1/N2/N3): Post-ingestion scanning, PR content generation,
and batch grouping of discovered connections.

Passive surfacing (M1/M2): On-demand related content queries triggered by
user interaction.
"""

from author_library.surfacing.batch_surfacing import BatchSurfacer, BatchSurfacingResult
from author_library.surfacing.connection_scanner import ConnectionScanner, ScanResult

__all__ = [
    "BatchSurfacer",
    "BatchSurfacingResult",
    "ConnectionScanner",
    "ScanResult",
]
