"""Chainlit chat frontend for The Author Library.

Provides a conversational RAG interface over the ingested author corpus.
Retrieval happens transparently on every message — the user just chats.
Author switching is handled via Chainlit's chat profiles dropdown.

Usage:
    chainlit run chainlit_app.py -w --port 8501
"""

from __future__ import annotations

import anthropic
import chainlit as cl
import structlog

from author_library.config import get_settings
from author_library.embeddings.registry import ProviderRegistry
from author_library.graph.queries import GraphQueryService
from author_library.intelligence.thematic_index import ThematicEntry
from author_library.intelligence.voice_crud import VoiceProfileManager
from author_library.retrieval.context_assembly import assemble_context
from author_library.retrieval.orchestrator import RetrievalOrchestrator
from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)

# Work count query: matches primary sources (work_id starts with author slug)
# and secondary sources (about_author_id in metadata).
_WORK_COUNT_SQL = (
    "SELECT COUNT(*) FROM works"
    " WHERE work_id LIKE $1 || '--%'"
    " OR source_metadata->>'about_author_id' = $1"
)


@cl.set_chat_profiles
async def chat_profiles(current_user: cl.User | None = None) -> list[cl.ChatProfile]:
    """Build chat profiles from the authors table."""
    settings = get_settings()
    storage = StorageManager(settings.database)
    await storage.connect()

    rows = await storage.pg.fetch_all(
        "SELECT id, canonical_name FROM authors ORDER BY canonical_name"
    )

    profiles = []
    for idx, row in enumerate(rows):
        author_id = row["id"]
        author_name = row["canonical_name"]
        work_count = await storage.pg.fetch_val(_WORK_COUNT_SQL, author_id)
        profiles.append(
            cl.ChatProfile(
                name=author_name,
                markdown_description=(
                    f"Chat with the corpus of **{author_name}** "
                    f"({work_count} works ingested)"
                ),
                default=idx == 0,
            )
        )

    await storage.close()
    return profiles


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize storage, embedding provider, and set author from profile."""
    settings = get_settings()
    storage = StorageManager(settings.database)
    await storage.connect()

    embedding_provider = ProviderRegistry.create(settings)

    # Store in session
    cl.user_session.set("settings", settings)
    cl.user_session.set("storage", storage)
    cl.user_session.set("embedding_provider", embedding_provider)
    cl.user_session.set("history", [])

    # Get the selected chat profile
    chat_profile = cl.user_session.get("chat_profile")

    if not chat_profile:
        # No profiles available (no authors ingested)
        await cl.Message(
            content="No authors found in the library. Ingest some works first."
        ).send()
        return

    # Look up the author_id from the profile metadata
    rows = await storage.pg.fetch_all(
        "SELECT id, canonical_name FROM authors WHERE canonical_name = $1",
        chat_profile,
    )

    if not rows:
        await cl.Message(
            content=f"Author '{chat_profile}' not found in the database."
        ).send()
        return

    author = rows[0]
    author_id = author["id"]
    author_name = author["canonical_name"]

    cl.user_session.set("author_id", author_id)
    cl.user_session.set("author_name", author_name)

    work_count = await storage.pg.fetch_val(_WORK_COUNT_SQL, author_id)

    await cl.Message(
        content=(
            f"Welcome to **The Author Library**.\n\n"
            f"You're chatting with the corpus of **{author_name}** "
            f"({work_count} works ingested).\n\n"
            f"Ask anything — I'll draw from the ingested texts and respond "
            f"in the author's voice with citations."
        )
    ).send()


@cl.step(type="tool", name="Searching corpus")
async def retrieve_passages(
    question: str,
) -> tuple:
    """Run multi-pass retrieval and assemble context."""
    settings = cl.user_session.get("settings")
    storage: StorageManager = cl.user_session.get("storage")
    embedding_provider = cl.user_session.get("embedding_provider")
    author_id = cl.user_session.get("author_id")

    orchestrator = RetrievalOrchestrator(
        settings=settings,
        embedding_provider=embedding_provider,
        embedding_repo=storage.embeddings,
        pg_pool=storage.pg,
        graph_query_service=GraphQueryService(storage.neo4j),
    )

    orchestrated = await orchestrator.retrieve(
        question,
        author_id=author_id,
    )

    voice_manager = VoiceProfileManager(settings)
    voice_profile = await voice_manager.get_current(
        author_id=author_id,
        voice_repo=storage.voice_profiles,
    )

    thematic_entries_raw = await storage.thematic.list_entries(author_id)
    thematic_entries = [
        ThematicEntry(
            theme=e.get("theme", ""),
            author_stance=e.get("author_stance", ""),
            related_themes=e.get("related_themes", []),
        )
        for e in thematic_entries_raw
    ]

    author_name = cl.user_session.get("author_name", author_id)
    context_window = assemble_context(
        orchestrated,
        voice_profile=voice_profile,
        author_name=author_name,
        thematic_entries=thematic_entries,
    )

    return context_window, orchestrated


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle user message: retrieve, assemble context, stream response."""
    author_id = cl.user_session.get("author_id")
    if not author_id:
        await cl.Message(
            content="Please select an author from the dropdown above."
        ).send()
        return

    settings = cl.user_session.get("settings")
    history: list[dict] = cl.user_session.get("history", [])

    # Step 1: Retrieve passages (shown as a tool step in the UI)
    context_window, orchestrated = await retrieve_passages(message.content)

    # Step 2: Build messages for Anthropic
    passage_texts = [p.text for p in context_window.passages]
    passages_block = "\n\n---\n\n".join(passage_texts)

    user_content = (
        f"Question: {message.content}\n\n"
        f"Retrieved passages:\n\n{passages_block}"
    )
    if context_window.thematic_summaries:
        user_content += (
            "\n\nThematic context:\n" + "\n".join(context_window.thematic_summaries)
        )

    # Include conversation history for multi-turn
    messages = list(history)
    messages.append({"role": "user", "content": user_content})

    # Step 3: Stream response from Anthropic
    api_key = settings.api_keys.anthropic_api_key.get_secret_value()
    client = anthropic.AsyncAnthropic(api_key=api_key)

    response_msg = cl.Message(content="")

    full_response = ""
    async with client.messages.stream(
        model=settings.llm.query_model,
        max_tokens=4096,
        system=context_window.system_prompt,
        messages=messages,
    ) as stream:
        async for token in stream.text_stream:
            full_response += token
            await response_msg.stream_token(token)

    # Step 4: Attach citation sources as side-panel elements
    source_elements = []
    seen_works = set()
    for passage in context_window.passages:
        # Deduplicate by work_id to avoid flooding the sidebar
        if passage.work_id in seen_works:
            continue
        seen_works.add(passage.work_id)

        # Collect all passages from this work
        work_passages = [
            p for p in context_window.passages if p.work_id == passage.work_id
        ]
        work_text = "\n\n---\n\n".join(
            f"**{p.citation_label}** (relevance: {p.relevance_score:.3f})\n\n{p.text}"
            for p in work_passages
        )
        source_elements.append(
            cl.Text(
                content=work_text,
                name=f"{passage.source_class.upper()}: {passage.citation_label}",
                display="side",
            )
        )

    response_msg.elements = source_elements
    await response_msg.send()

    # Step 5: Update conversation history (keep last 10 turns)
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": full_response})
    if len(history) > 20:  # 10 turns = 20 messages
        history = history[-20:]
    cl.user_session.set("history", history)

    log.info(
        "chat_response_sent",
        author_id=author_id,
        question_type=orchestrated.question_type.value,
        passages_used=len(context_window.passages),
        tokens_estimate=context_window.total_tokens_estimate,
    )
