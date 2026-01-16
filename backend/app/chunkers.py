import json
import os
from enum import Enum
from typing import Any, Dict, List, Tuple
from pathlib import Path

import fitz  # PyMuPDF
import requests
from llama_parse import LlamaParse
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SemanticSplitterNodeParser,
)
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.extractors import TitleExtractor, KeywordExtractor

# Local parse fallback
from agentic_doc.parse import parse


class ChunkingError(Exception):
    """Raised when a chunking strategy fails."""


class ChunkingStrategy(str, Enum):
    SEMANTIC = "semantic"  # previously called "agentic"
    LATE = "late"
    LLM = "llm"
    HIERARCHICAL = "hierarchical"

    @classmethod
    def normalize(cls, value: str) -> "ChunkingStrategy":
        aliases = {
            "agentic": cls.SEMANTIC,
            "semantic": cls.SEMANTIC,
            "late": cls.LATE,
            "llm": cls.LLM,
            "hierarchical": cls.HIERARCHICAL,
        }
        key = value.lower()
        if key in aliases:
            return aliases[key]
        raise ChunkingError(f"Unsupported chunking strategy '{value}'.")


def _clean_metadata(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure metadata is JSON serializable."""
    cleaned: Dict[str, Any] = {}
    for key, val in raw.items():
        if isinstance(val, (str, int, float, bool)) or val is None:
            cleaned[key] = val
        else:
            try:
                cleaned[key] = json.dumps(val, default=str)
            except Exception:
                cleaned[key] = str(val)
    return cleaned


def _extract_pdf_text(pdf_path: Path) -> str:
    """Return plain text from all pages of a PDF."""
    texts: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            texts.append(page.get_text())
    return "\n".join(texts).strip()


def _nodes_to_payload(
    nodes: List[Any], strategy: str, source_filename: str
) -> Dict[str, Any]:
    """Convert llama-index nodes into API-friendly payload."""
    chunks: List[Dict[str, Any]] = []
    for idx, node in enumerate(nodes, start=1):
        metadata = _clean_metadata(getattr(node, "metadata", {}) or {})
        node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
        if node_id:
            metadata.setdefault("node_id", node_id)
        parent_id = metadata.get("parent_id") or getattr(node, "parent_id", None)
        if parent_id:
            metadata["parent_id"] = parent_id

        chunks.append(
            {
                "chunk_id": metadata.get("chunk_id") or idx,
                "chunk_type": metadata.get("chunk_type") or strategy,
                "text": getattr(node, "text", ""),
                "source_filename": metadata.get("source_filename") or source_filename,
                "grounding": [],
                "metadata": metadata,
            }
        )

    return {
        "data": [
            {
                "chunks": chunks,
                "chunking_strategy": strategy,
                "source_filename": source_filename,
            }
        ]
    }


def run_semantic(pdf_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """Use existing agentic_doc.parse pipeline (semantic chunking) with viz handled upstream."""
    parsed = parse(str(pdf_path))
    return parsed, []
# Backward-compatible alias
run_agentic = run_semantic


def run_late_chunking(pdf_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """Late chunking via Jina embeddings API + lightweight splitter."""
    jina_key = os.getenv("JINA_API_KEY")
    if not jina_key:
        raise ChunkingError("JINA_API_KEY is not configured.")

    text = _extract_pdf_text(pdf_path)
    if not text:
        raise ChunkingError("No text extracted from PDF for late chunking.")

    payload = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "late_chunking": True,
        "input": [text],
    }

    try:
        resp = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {jina_key}",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        api_result = resp.json()
    except Exception as exc:
        raise ChunkingError(f"Late chunking request failed: {exc}") from exc

    splitter = SentenceSplitter(chunk_size=800, chunk_overlap=80)
    chunk_texts = splitter.split_text(text)

    chunks: List[Dict[str, Any]] = []
    for idx, chunk_text in enumerate(chunk_texts, start=1):
        chunks.append(
            {
                "chunk_id": idx,
                "chunk_type": "late_chunk",
                "text": chunk_text,
                "source_filename": pdf_path.name,
                "grounding": [],
                "metadata": {
                    "chunking_strategy": "late_chunking",
                    "model": api_result.get("model"),
                    "usage": api_result.get("usage"),
                },
            }
        )

    return {
        "data": [
            {
                "chunks": chunks,
                "chunking_strategy": "late_chunking",
                "source_filename": pdf_path.name,
            }
        ]
    }, []


def run_llm_chunking(pdf_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """LLM-guided chunking using LlamaParse + SemanticSplitter."""
    llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not llama_key:
        raise ChunkingError("LLAMA_CLOUD_API_KEY is not configured.")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ChunkingError("OPENAI_API_KEY is required for LLM chunking.")

    parser = LlamaParse(
        api_key=llama_key,
        result_type="markdown",
        verbose=False,
        merge_tables_across_pages_in_markdown=True,
        system_prompt=(
            "You are preparing content for downstream chunking and RAG. "
            "Return clean, well-structured markdown that preserves hierarchy: "
            "use headings, ordered lists, and code fences. Preserve tables exactly, "
            "do not drop rows/columns, and include alt-text style descriptions for images or charts. "
            "Keep page order, avoid hallucinating content, and never omit section titles."
        ),
    )
    documents = parser.load_data(str(pdf_path))

    embed_model = OpenAIEmbedding(api_key=openai_key)
    transformations = [
        SentenceSplitter(chunk_size=1024, chunk_overlap=40),
        SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
        ),
        TitleExtractor(),
        KeywordExtractor(keywords=10),
        embed_model,
    ]

    pipeline = IngestionPipeline(transformations=transformations)
    nodes = pipeline.run(documents=documents)

    return _nodes_to_payload(nodes, "llm_chunking", pdf_path.name), []


def run_hierarchical_chunking(pdf_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """Hierarchical chunking using HierarchicalNodeParser."""
    llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not llama_key:
        raise ChunkingError("LLAMA_CLOUD_API_KEY is not configured.")

    parser = LlamaParse(
        api_key=llama_key,
        result_type="text",
        verbose=False,
        merge_tables_across_pages_in_markdown=True,
    )
    documents = parser.load_data(str(pdf_path))

    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 768, 256]
    )
    nodes = node_parser.get_nodes_from_documents(documents)

    return _nodes_to_payload(nodes, "hierarchical_chunking", pdf_path.name), []


def run_chunking_strategy(
    pdf_path: Path, strategy_value: str
) -> Tuple[Dict[str, Any], List[str]]:
    """Dispatch chunking based on user selection."""
    strategy = ChunkingStrategy.normalize(strategy_value)

    if strategy is ChunkingStrategy.SEMANTIC:
        return run_semantic(pdf_path)
    if strategy is ChunkingStrategy.LATE:
        return run_late_chunking(pdf_path)
    if strategy is ChunkingStrategy.LLM:
        return run_llm_chunking(pdf_path)
    if strategy is ChunkingStrategy.HIERARCHICAL:
        return run_hierarchical_chunking(pdf_path)

    raise ChunkingError(f"Unsupported chunking strategy '{strategy_value}'.")
