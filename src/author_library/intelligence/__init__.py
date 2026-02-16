"""Author intelligence extraction for The Author Library.

Provides LLM-powered analysis of an author's corpus to extract
voice profiles, thematic indices, terminology mappings, and
cross-work thematic evolution.
"""

from __future__ import annotations

from author_library.intelligence.evolution import ThematicEvolutionAnalyzer
from author_library.intelligence.terminology import TerminologyNormalizer
from author_library.intelligence.thematic_index import ThematicIndexGenerator
from author_library.intelligence.voice_crud import VoiceProfileManager
from author_library.intelligence.voice_profile import VoiceProfileExtractor

__all__ = [
    "TerminologyNormalizer",
    "ThematicEvolutionAnalyzer",
    "ThematicIndexGenerator",
    "VoiceProfileExtractor",
    "VoiceProfileManager",
]
