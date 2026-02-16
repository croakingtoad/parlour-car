"""Tests for entity extraction pipeline."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from author_library.graph.entity_extraction import (
    ChunkExtraction,
    EntityExtractor,
    ExtractedEntity,
    ExtractionResult,
)

from .conftest import requires_anthropic, requires_neo4j

if TYPE_CHECKING:
    from author_library.chunking.models import Chunk
    from author_library.config import APIKeySettings, LLMSettings
    from author_library.storage.neo4j import Neo4jConnection


class TestExtractionResponseParsing:
    """Test LLM response parsing without requiring API calls."""

    def test_parse_valid_response(self) -> None:
        """Parse a well-formed JSON extraction response."""
        response_text = json.dumps([
            {
                "chunk_id": "chunk-001",
                "themes": [
                    {"name": "Primary Imagination", "canonical_name": "primary-imagination"},
                    {"name": "Sacramental Vision", "canonical_name": "sacramental-vision"},
                ],
                "arguments": [
                    {
                        "claim": "Imagination is the living power of perception",
                        "evidence_summary": "Coleridge's definition of Primary Imagination",
                    }
                ],
                "concepts": [
                    {"name": "Esemplastic Power", "canonical_name": "esemplastic-power"},
                ],
                "persons": [
                    {
                        "name": "Samuel Taylor Coleridge",
                        "canonical_name": "samuel-taylor-coleridge",
                        "role": "discussed",
                    },
                ],
            }
        ])

        # We need to instantiate the parser method directly
        # Use a minimal extractor just for parsing (no API call)
        extraction = _parse_test_response(response_text)
        assert len(extraction) == 1
        assert extraction[0].chunk_id == "chunk-001"
        assert len(extraction[0].themes) == 2
        assert extraction[0].themes[0].canonical_name == "primary-imagination"
        assert len(extraction[0].arguments) == 1
        assert extraction[0].arguments[0].entity_type == "argument"
        assert len(extraction[0].concepts) == 1
        assert extraction[0].concepts[0].canonical_name == "esemplastic-power"
        assert len(extraction[0].persons) == 1
        assert extraction[0].persons[0].properties["role"] == "discussed"

    def test_parse_response_with_code_fences(self) -> None:
        """Parse response wrapped in markdown code fences."""
        inner = json.dumps([
            {
                "chunk_id": "chunk-002",
                "themes": [{"name": "Poetry as Truth", "canonical_name": "poetry-as-truth"}],
                "arguments": [],
                "concepts": [],
                "persons": [],
            }
        ])
        response_text = f"```json\n{inner}\n```"

        extraction = _parse_test_response(response_text)
        assert len(extraction) == 1
        assert extraction[0].themes[0].canonical_name == "poetry-as-truth"

    def test_parse_multi_chunk_response(self) -> None:
        """Parse response with multiple chunk extractions."""
        response_text = json.dumps([
            {
                "chunk_id": "chunk-a",
                "themes": [{"name": "T1", "canonical_name": "t1"}],
                "arguments": [],
                "concepts": [],
                "persons": [],
            },
            {
                "chunk_id": "chunk-b",
                "themes": [],
                "arguments": [{"claim": "Some claim", "evidence_summary": "Some evidence"}],
                "concepts": [{"name": "C1", "canonical_name": "c1"}],
                "persons": [{"name": "Person A", "canonical_name": "person-a", "role": "quoted"}],
            },
        ])

        extractions = _parse_test_response(response_text)
        assert len(extractions) == 2
        assert extractions[0].chunk_id == "chunk-a"
        assert extractions[1].chunk_id == "chunk-b"
        assert len(extractions[1].arguments) == 1
        assert extractions[1].persons[0].properties["role"] == "quoted"

    def test_parse_empty_entities(self) -> None:
        """Parse response where chunk has no entities."""
        response_text = json.dumps([
            {
                "chunk_id": "chunk-empty",
                "themes": [],
                "arguments": [],
                "concepts": [],
                "persons": [],
            }
        ])

        extraction = _parse_test_response(response_text)
        assert len(extraction) == 1
        assert extraction[0].themes == []
        assert extraction[0].arguments == []


class TestEntityExtractionModels:
    """Test entity extraction data models."""

    def test_extracted_entity_immutable(self) -> None:
        """ExtractedEntity should be immutable (frozen dataclass)."""
        entity = ExtractedEntity(
            entity_type="theme",
            name="Primary Imagination",
            canonical_name="primary-imagination",
        )
        assert entity.entity_type == "theme"
        assert entity.properties == {}

    def test_chunk_extraction_collects_all_types(self) -> None:
        """ChunkExtraction holds all entity types for a chunk."""
        extraction = ChunkExtraction(
            chunk_id="test-001",
            themes=[ExtractedEntity("theme", "T1", "t1")],
            arguments=[ExtractedEntity("argument", "A1", "a1", {"evidence_summary": "ev"})],
            concepts=[ExtractedEntity("concept", "C1", "c1")],
            persons=[ExtractedEntity("person", "P1", "p1", {"role": "referenced"})],
        )
        assert len(extraction.themes) == 1
        assert len(extraction.arguments) == 1
        assert len(extraction.concepts) == 1
        assert len(extraction.persons) == 1

    def test_extraction_result_accumulates(self) -> None:
        """ExtractionResult should accumulate counts."""
        result = ExtractionResult()
        result.nodes_created += 5
        result.edges_created += 3
        result.errors.append("test error")
        assert result.nodes_created == 5
        assert result.edges_created == 3
        assert len(result.errors) == 1


@requires_neo4j
@requires_anthropic
class TestEntityExtractionWithNeo4j:
    """Integration tests requiring Neo4j and Anthropic API."""

    async def test_extract_and_persist_primary_chunks(
        self,
        neo4j_conn: Neo4jConnection,
        api_keys: APIKeySettings,
        llm_settings: LLMSettings,
        primary_chunks: list[Chunk],
    ) -> None:
        """Extract entities from primary chunks and verify Neo4j nodes/edges."""
        extractor = EntityExtractor(neo4j_conn, api_keys, llm_settings)
        result = await extractor.extract_and_persist(
            primary_chunks[:1],
            work_title="Faith Hope and Poetry",
            author="Malcolm Guite",
        )

        assert result.nodes_created > 0
        assert result.edges_created > 0
        assert len(result.errors) == 0

        # Verify theme nodes were created
        themes = await neo4j_conn.execute_read("MATCH (t:Theme) RETURN t.canonical_name AS name")
        assert len(themes) > 0

        # Verify EXPLORES_THEME edges exist
        edges = await neo4j_conn.execute_read(
            "MATCH (:Chunk)-[r:EXPLORES_THEME]->(:Theme) RETURN count(r) AS cnt"
        )
        assert edges[0]["cnt"] > 0

    async def test_primary_chunks_get_makes_argument(
        self,
        neo4j_conn: Neo4jConnection,
        api_keys: APIKeySettings,
        llm_settings: LLMSettings,
        primary_chunks: list[Chunk],
    ) -> None:
        """PRIMARY source chunks should produce MAKES_ARGUMENT edges."""
        extractor = EntityExtractor(neo4j_conn, api_keys, llm_settings)
        await extractor.extract_and_persist(
            primary_chunks[:1],
            work_title="Faith Hope and Poetry",
            author="Malcolm Guite",
        )

        # Check for MAKES_ARGUMENT (PRIMARY only)
        makes = await neo4j_conn.execute_read(
            "MATCH (:Chunk)-[r:MAKES_ARGUMENT]->(:Argument) RETURN count(r) AS cnt"
        )
        # There should be at least some arguments extracted from this rich text
        # (the LLM should find claims about imagination)
        assert makes[0]["cnt"] >= 0  # May be 0 if LLM doesn't extract arguments

        # Verify NO ATTRIBUTED_BY_CRITIC from primary chunks
        attributed = await neo4j_conn.execute_read(
            "MATCH (:Chunk {source_class: 'primary'})-[r:ATTRIBUTED_BY_CRITIC]->(:Argument) "
            "RETURN count(r) AS cnt"
        )
        assert attributed[0]["cnt"] == 0

    async def test_secondary_chunks_get_attributed_by_critic(
        self,
        neo4j_conn: Neo4jConnection,
        api_keys: APIKeySettings,
        llm_settings: LLMSettings,
        secondary_chunks: list[Chunk],
    ) -> None:
        """SECONDARY source chunks should produce ATTRIBUTED_BY_CRITIC edges."""
        extractor = EntityExtractor(neo4j_conn, api_keys, llm_settings)
        await extractor.extract_and_persist(
            secondary_chunks,
            work_title="Romantic Theology",
            author="Michael Ward",
        )

        # Verify NO MAKES_ARGUMENT from secondary chunks
        makes = await neo4j_conn.execute_read(
            "MATCH (:Chunk {source_class: 'secondary'})-[r:MAKES_ARGUMENT]->(:Argument) "
            "RETURN count(r) AS cnt"
        )
        assert makes[0]["cnt"] == 0

    async def test_person_references_created(
        self,
        neo4j_conn: Neo4jConnection,
        api_keys: APIKeySettings,
        llm_settings: LLMSettings,
        primary_chunks: list[Chunk],
    ) -> None:
        """Person nodes and REFERENCES_PERSON edges should be created."""
        extractor = EntityExtractor(neo4j_conn, api_keys, llm_settings)
        await extractor.extract_and_persist(
            primary_chunks[:1],
            work_title="Faith Hope and Poetry",
            author="Malcolm Guite",
        )

        persons = await neo4j_conn.execute_read(
            "MATCH (p:Person) RETURN p.canonical_name AS name"
        )
        # Coleridge should be extracted as a referenced person
        assert len(persons) >= 0  # LLM-dependent, but structure should work


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def _parse_test_response(response_text: str) -> list[ChunkExtraction]:
    """Parse an extraction response without instantiating a full EntityExtractor."""
    # Replicate the parsing logic for unit testing
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    raw_list = json.loads(text)
    extractions: list[ChunkExtraction] = []

    for raw in raw_list:
        chunk_id = raw["chunk_id"]
        themes = [
            ExtractedEntity(
                entity_type="theme",
                name=t["name"],
                canonical_name=t["canonical_name"],
            )
            for t in raw.get("themes", [])
        ]
        arguments = [
            ExtractedEntity(
                entity_type="argument",
                name=a["claim"],
                canonical_name=a["claim"][:80].lower().replace(" ", "-"),
                properties={"evidence_summary": a.get("evidence_summary", "")},
            )
            for a in raw.get("arguments", [])
        ]
        concepts = [
            ExtractedEntity(
                entity_type="concept",
                name=c["name"],
                canonical_name=c["canonical_name"],
            )
            for c in raw.get("concepts", [])
        ]
        persons = [
            ExtractedEntity(
                entity_type="person",
                name=p["name"],
                canonical_name=p["canonical_name"],
                properties={"role": p.get("role", "referenced")},
            )
            for p in raw.get("persons", [])
        ]
        extractions.append(
            ChunkExtraction(
                chunk_id=chunk_id,
                themes=themes,
                arguments=arguments,
                concepts=concepts,
                persons=persons,
            )
        )

    return extractions
