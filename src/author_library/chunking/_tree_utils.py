"""Shared utilities for walking and extracting text from DocumentNode trees."""

from __future__ import annotations

from author_library.parsing.models import DocumentNode, NodeType


def collect_text(node: DocumentNode) -> str:
    """Recursively collect all text from a node and its descendants.

    Joins text fragments with newlines, preserving paragraph boundaries.
    """
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in node.children:
        child_text = collect_text(child)
        if child_text:
            parts.append(child_text)
    return "\n".join(parts)


def word_count(text: str) -> int:
    """Count whitespace-delimited words in *text*."""
    return len(text.split())


def find_nodes(node: DocumentNode, node_type: NodeType) -> list[DocumentNode]:
    """Return all descendants of *node* (inclusive) matching *node_type*."""
    result: list[DocumentNode] = []
    if node.node_type == node_type:
        result.append(node)
    for child in node.children:
        result.extend(find_nodes(child, node_type))
    return result


def find_children_of_type(
    node: DocumentNode,
    *node_types: NodeType,
) -> list[DocumentNode]:
    """Return direct children of *node* whose type is in *node_types*."""
    return [c for c in node.children if c.node_type in node_types]


def collect_paragraphs(node: DocumentNode) -> list[DocumentNode]:
    """Return all PARAGRAPH nodes under *node*, in document order."""
    return find_nodes(node, NodeType.PARAGRAPH)


def collect_footnotes(node: DocumentNode) -> dict[str, str]:
    """Return a mapping from footnote ID → footnote text for notes under *node*.

    Footnotes are identified by their ``ref`` metadata key if present,
    otherwise by their node ``id``.
    """
    footnotes: dict[str, str] = {}
    for fn_node in find_nodes(node, NodeType.FOOTNOTE):
        ref = str(fn_node.metadata.get("ref", fn_node.id))
        footnotes[ref] = collect_text(fn_node)
    for en_node in find_nodes(node, NodeType.ENDNOTE):
        ref = str(en_node.metadata.get("ref", en_node.id))
        footnotes[ref] = collect_text(en_node)
    return footnotes
