import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, List, Set
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Third-party parser used elsewhere in the repo
from agentic_doc.config import VisualizationConfig
from .chunkers import (
    ChunkingError,
    ChunkingStrategy,
    run_chunking_strategy,
    run_semantic,
)

# Constants
BASE_DIR = Path(__file__).parent.parent
DEFAULT_UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
VIZ_DIR = STATIC_DIR / "visualizations"
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB safeguard
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-nano"


class ParseResponse(BaseModel):
    """Response model to wrap raw parser output."""

    data: Any
    visualizations: List[str] | None = None


class ChatRequest(BaseModel):
    """Payload for chat RAG requests."""

    question: str
    parsed: Any


class ChatVisualizationRequest(BaseModel):
    """Request body for chat visualization generation."""

    chunk_ids: List[str]


def ensure_upload_dir() -> Path:
    DEFAULT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_UPLOAD_DIR


def save_temp_file(upload: UploadFile, upload_dir: Path) -> Path:
    """Persist the uploaded file to a temporary location on disk."""
    suffix = Path(upload.filename or "upload.pdf").suffix
    fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=upload_dir)
    with os.fdopen(fd, "wb") as tmp:
        total = 0
        for chunk in iter(lambda: upload.file.read(1024 * 1024), b""):
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                upload.file.close()
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Limit is {MAX_FILE_BYTES // (1024 * 1024)} MB.",
                )
            tmp.write(chunk)
    upload.file.close()
    return Path(temp_path)


def create_app() -> FastAPI:
    # Load env if present (project root and backend/app/.env), overriding blanks from shell
    load_dotenv(override=True)
    load_dotenv(BASE_DIR / ".env", override=True)
    load_dotenv(BASE_DIR / "app" / ".env", override=True)

    app = FastAPI(title="PDF Parser API", version="0.1.0")
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # CORS – allow local dev frontends; adjust origins as needed.
    env_origin = os.getenv("FRONTEND_ORIGIN")
    origins: List[str] = [
        env_origin or "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )

    def format_chunk_label(chunk_type: Any) -> str:
        """Return a human-friendly label (e.g., 'TEXT', 'TABLE', 'FIGURE')."""
        if not chunk_type:
            return "TEXT"
        label_raw = str(chunk_type).lower().strip()

        # Remove common prefixes/suffixes and separators
        for prefix in ("chunktype_", "chunktype-", "chunk_type_", "chunk_type-", "chunktype", "chunk_type"):
            if label_raw.startswith(prefix):
                label_raw = label_raw[len(prefix) :]
                break
        label_raw = label_raw.replace("_", " ").replace("-", " ").strip()

        # Collapse to alphanumerics for matching
        condensed = re.sub(r"[^a-z0-9]", "", label_raw)
        if not condensed:
            return "TEXT"

        # Map to a small, consistent vocabulary
        if "table" in condensed:
            return "TABLE"
        if "figure" in condensed or "image" in condensed or "img" in condensed:
            return "FIGURE"
        if "text" in condensed or "paragraph" in condensed or "para" in condensed:
            return "TEXT"

        # Fallback: uppercase cleaned label
        return label_raw.upper()

    def build_visualizations(pdf_path: Path, parsed_docs: Any, allowed_chunk_ids: Set[str] | None = None) -> List[str]:
        """Generate visualization images (boxed/labelled) and return URLs. Optionally filter by chunk ids."""
        if not parsed_docs or len(parsed_docs) == 0:
            return []

        parsed_doc = parsed_docs[0]
        viz_subdir = VIZ_DIR / f"{pdf_path.stem}_{uuid4().hex}"
        viz_subdir.mkdir(parents=True, exist_ok=True)

        viz_config = VisualizationConfig(
            thickness=3,
            text_bg_opacity=0.85,
            padding=4,
            font_scale=0.5,
        )
        color_map = {k: (v[2], v[1], v[0]) for k, v in viz_config.color_map.items()}  # BGR->RGB
        font_path = STATIC_DIR / "fonts" / "DejaVuSans.ttf"
        try:
            font = ImageFont.truetype(str(font_path), 15)
        except Exception:
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
            except Exception:
                font = ImageFont.load_default()

        try:
            with fitz.open(pdf_path) as doc:
                for page_idx, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=144, colorspace=fitz.csRGB)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    draw = ImageDraw.Draw(img, "RGBA")

                    page_chunks = [
                        c
                        for c in parsed_doc.chunks
                        if c.grounding and c.grounding[0].page == page_idx
                    ]
                    for c in page_chunks:
                        if allowed_chunk_ids is not None and str(c.chunk_id) not in allowed_chunk_ids:
                            continue
                        color = color_map.get(c.chunk_type, (0, 122, 255))
                        for g in c.grounding:
                            if not g.box:
                                continue
                            x0 = int(g.box.l * pix.width)
                            y0 = int(g.box.t * pix.height)
                            x1 = int(g.box.r * pix.width)
                            y1 = int(g.box.b * pix.height)

                            label = format_chunk_label(c.chunk_type)
                            text_bbox = draw.textbbox((0, 0), label, font=font)
                            text_w = text_bbox[2] - text_bbox[0]
                            text_h = text_bbox[3] - text_bbox[1]
                            label_x = x0 + 4
                            label_y = max(0, y0 - text_h - 8)

                            bg_rect = [
                                (label_x - 6, label_y - 4),
                                (label_x + text_w + 6, label_y + text_h + 4),
                            ]
                            draw.rectangle(
                                bg_rect,
                                fill=(255, 255, 255, int(viz_config.text_bg_opacity * 255)),
                                outline=(*color, 200),
                                width=1,
                            )
                            draw.text(
                                (label_x, label_y),
                                label,
                                fill=color,
                                font=font,
                            )
                            # box
                            for offset in range(viz_config.thickness):
                                draw.rectangle(
                                    [(x0 - offset, y0 - offset), (x1 + offset, y1 + offset)],
                                    outline=color,
                                    width=1,
                                )

                    out_path = viz_subdir / f"{pdf_path.stem}_viz_page_{page_idx}.png"
                    img.save(out_path)
        except Exception as exc:
            print(f"Visualization failed: {exc}")
            return []

        urls: List[str] = []
        for img_path in sorted(viz_subdir.glob("*.png")):
            relative = img_path.relative_to(STATIC_DIR)
            urls.append(f"/static/{relative.as_posix()}")
        return urls

    @app.get("/healthz")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    def require_openai_api_key() -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
        return key

    def build_documents_from_parsed(parsed: Any) -> List[Document]:
        """Convert parser output into LangChain Documents."""
        documents: List[Document] = []
        docs = parsed
        if isinstance(parsed, dict) and "data" in parsed:
            docs = parsed["data"]

        if not isinstance(docs, list):
            return documents

        for doc_idx, doc in enumerate(docs):
            chunks = doc.get("chunks") if isinstance(doc, dict) else None
            if not chunks:
                continue
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                text = chunk.get("text") or ""
                if not text:
                    continue
                grounding = chunk.get("grounding") or []
                metadata = {
                    "source_filename": chunk.get("source_filename", f"document_{doc_idx}.pdf"),
                    "chunk_type": chunk.get("chunk_type"),
                    "chunk_id": chunk.get("chunk_id"),
                    "grounding_info": json.dumps(grounding),
                    "page_number": grounding[0].get("page") if grounding else None,
                }
                documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def build_context_blob(retrieved: List[Document]) -> str:
        parts: List[str] = []
        for idx, doc in enumerate(retrieved, 1):
            parts.append(f"CHUNK {idx}:")
            parts.append(f"Text: {doc.page_content}")
            parts.append(f"Metadata: {doc.metadata}")
            parts.append("\n" + "=" * 50 + "\n")
        return "\n".join(parts)

    def extract_chunk_ids_from_llm_response(content: str) -> Set[str]:
        """Best-effort extraction of chunk_ids from LLM response content."""
        ids: Set[str] = set()
        if not content:
            return ids
        try:
            # try fenced json
            if "```json" in content:
                json_part = content.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_part)
            else:
                data = json.loads(content)
            if isinstance(data, dict):
                chunks = data.get("chunks_used") or []
                for item in chunks:
                    if isinstance(item, dict):
                        cid = item.get("chunk_id")
                        if cid:
                            ids.add(str(cid))
                    elif isinstance(item, str):
                        # look for chunk_id in string
                        import re

                        match = re.search(r"chunk_id['\"]?:\s*'([^']+)'", item)
                        if match:
                            ids.add(match.group(1))
            return ids
        except Exception:
            return ids

    def build_visualizations_for_chunk_ids(
        pdf_path: Path, parsed_docs: Any, allowed_chunk_ids: Set[str]
    ) -> List[str]:
        """Generate visualizations only for the allowed chunk ids from parsed docs."""
        def normalize_parsed_docs(val: Any) -> List[Any]:
            if isinstance(val, list):
                return val
            if isinstance(val, dict) and isinstance(val.get("data"), list):
                return val["data"]
            return []

        parsed_list = normalize_parsed_docs(parsed_docs)
        if not parsed_list or not allowed_chunk_ids:
            return []

        viz_subdir = VIZ_DIR / f"{pdf_path.stem}_{uuid4().hex}_filtered"
        viz_subdir.mkdir(parents=True, exist_ok=True)

        viz_config = VisualizationConfig(
            thickness=3,
            text_bg_opacity=0.85,
            padding=4,
            font_scale=0.5,
        )
        color_map = {k: (v[2], v[1], v[0]) for k, v in viz_config.color_map.items()}  # BGR->RGB
        font_path = STATIC_DIR / "fonts" / "DejaVuSans.ttf"
        try:
            font = ImageFont.truetype(str(font_path), 40)
        except Exception:
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
            except Exception:
                font = ImageFont.load_default()

        try:
            with fitz.open(pdf_path) as doc:
                for page_idx, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=144, colorspace=fitz.csRGB)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    draw = ImageDraw.Draw(img, "RGBA")

                    parsed_doc = parsed_list[0]
                    page_chunks = [
                        c
                        for c in parsed_doc.chunks
                        if c.grounding and c.grounding[0].page == page_idx and str(c.chunk_id) in allowed_chunk_ids
                    ]
                    for c in page_chunks:
                        color = color_map.get(c.chunk_type, (0, 122, 255))
                        for g in c.grounding:
                            if not g.box:
                                continue
                            x0 = int(g.box.l * pix.width)
                            y0 = int(g.box.t * pix.height)
                            x1 = int(g.box.r * pix.width)
                            y1 = int(g.box.b * pix.height)

                            label = format_chunk_label(c.chunk_type)
                            text_bbox = draw.textbbox((0, 0), label, font=font)
                            text_w = text_bbox[2] - text_bbox[0]
                            text_h = text_bbox[3] - text_bbox[1]
                            label_x = x0 + 4
                            label_y = max(0, y0 - text_h - 8)
                            bg_rect = [
                                (label_x - 6, label_y - 4),
                                (label_x + text_w + 6, label_y + text_h + 4),
                            ]
                            draw.rectangle(
                                bg_rect,
                                fill=(255, 255, 255, int(viz_config.text_bg_opacity * 255)),
                                outline=(*color, 200),
                                width=1,
                            )
                            draw.text(
                                (label_x, label_y),
                                label,
                                fill=color,
                                font=font,
                            )
                            for offset in range(viz_config.thickness):
                                draw.rectangle(
                                    [(x0 - offset, y0 - offset), (x1 + offset, y1 + offset)],
                                    outline=color,
                                    width=1,
                                )

                    out_path = viz_subdir / f"{pdf_path.stem}_viz_page_{page_idx}.png"
                    img.save(out_path)
        except Exception as exc:
            print(f"Filtered visualization failed: {exc}")
            return []

        urls: List[str] = []
        for img_path in sorted(viz_subdir.glob("*.png")):
            relative = img_path.relative_to(STATIC_DIR)
            urls.append(f"/static/{relative.as_posix()}")
        return urls

    def collect_markdown_strings(val: Any) -> List[str]:
        """Walk arbitrary nested structures to gather markdown fields."""
        collected: List[str] = []

        def walk(item: Any) -> None:
            if item is None:
                return
            if isinstance(item, str):
                return
            if isinstance(item, list):
                for sub in item:
                    walk(sub)
                return
            if isinstance(item, dict):
                md_val = item.get("markdown")
                if isinstance(md_val, str):
                    collected.append(md_val)
                for sub in item.values():
                    walk(sub)

        walk(val)
        return [c for c in collected if c]

    @app.post("/api/chat")
    async def chat(payload: ChatRequest) -> JSONResponse:
        api_key = require_openai_api_key()
        # Use prebuilt vector store from the last parse
        vector_store: Chroma | None = getattr(app.state, "vector_store", None)
        parsed_docs = getattr(app.state, "parsed_docs", None)
        pdf_path: Path | None = getattr(app.state, "last_pdf_path", None)
        if vector_store is None:
            raise HTTPException(status_code=400, detail="No vector store available. Parse a PDF first.")
        if parsed_docs is None or pdf_path is None:
            raise HTTPException(status_code=400, detail="Parsed document cache missing. Parse again.")

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(payload.question)

        context_blob = build_context_blob(retrieved_docs)
        prompt_text = f"""You are a helpful AI assistant that answers questions based on the provided context.
Use only the information from the context to answer questions.
If the answer cannot be found in the context, say "I cannot find that information in the provided context."
Provide detailed and accurate answers when possible.

Based on the chunks provided to you, answer the question.

Return the answer in a JSON format:

{{
\"answer\": \"[insert answer here]\",
\"chunks_used\": [
\"[insert exact json of the chunk details you used to answer the question - simply copy paste]\"
]
}}

Context:
{context_blob}

Question: {payload.question}

Answer:"""

        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, openai_api_key=api_key)
        response = llm.invoke(prompt_text)

        sources = [doc.metadata for doc in retrieved_docs]

        # Build filtered visualizations based on retrieved chunk ids
        # Derive chunk ids: prefer llm response, fallback to retrieved docs
        chunk_ids_from_llm = extract_chunk_ids_from_llm_response(response.content)
        allowed_chunk_ids: Set[str] = set(chunk_ids_from_llm)
        if not allowed_chunk_ids:
            for doc in retrieved_docs:
                cid = doc.metadata.get("chunk_id")
                if cid:
                    allowed_chunk_ids.add(str(cid))

        return JSONResponse(
            content={
                "message": response.content,
                "sources": sources,
                "chunk_ids": sorted(allowed_chunk_ids),
            }
        )

    @app.post("/api/chat/visualizations")
    async def chat_visualizations(payload: ChatVisualizationRequest) -> JSONResponse:
        parsed_docs = getattr(app.state, "parsed_docs", None)
        pdf_path: Path | None = getattr(app.state, "last_pdf_path", None)
        if parsed_docs is None or pdf_path is None:
            raise HTTPException(status_code=400, detail="Parsed document cache missing. Parse again.")

        allowed_chunk_ids: Set[str] = {str(cid) for cid in payload.chunk_ids if cid}
        if not allowed_chunk_ids:
            return JSONResponse(content={"filtered_visualizations": []})

        filtered_viz = await asyncio.to_thread(
            build_visualizations_for_chunk_ids, pdf_path, parsed_docs, allowed_chunk_ids
        )
        return JSONResponse(content={"filtered_visualizations": filtered_viz})

    @app.post("/api/parse", response_model=ParseResponse)
    async def parse_pdf(
        file: UploadFile = File(...),
        chunking_strategy: str = Form(ChunkingStrategy.SEMANTIC.value),
    ) -> JSONResponse:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        upload_dir = ensure_upload_dir()
        try:
            temp_path = save_temp_file(file, upload_dir)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

        try:
            semantic_task: asyncio.Task | None = None
            chosen_strategy = ChunkingStrategy.normalize(chunking_strategy)

            # For non-semantic strategies, kick off a background semantic pass to harvest markdown.
            if chosen_strategy is not ChunkingStrategy.SEMANTIC:
                semantic_task = asyncio.create_task(
                    asyncio.to_thread(run_semantic, temp_path)
                )

            parsed_raw, viz_urls = await asyncio.to_thread(
                run_chunking_strategy, temp_path, chunking_strategy
            )
            payload = jsonable_encoder(parsed_raw)
            if chosen_strategy is ChunkingStrategy.SEMANTIC:
                viz_urls = build_visualizations(temp_path, parsed_raw)

            # If we ran the semantic fallback, attach its markdown so the UI can render it.
            semantic_markdown = None
            if semantic_task:
                try:
                    semantic_raw, _ = await semantic_task
                    semantic_payload = jsonable_encoder(semantic_raw)
                    md_parts = collect_markdown_strings(semantic_payload)
                    if md_parts:
                        semantic_markdown = "\n\n---\n\n".join(md_parts)
                except Exception as exc:  # pragma: no cover - best-effort helper
                    print(f"Semantic markdown fallback failed: {exc}")
            if semantic_markdown:
                if isinstance(payload, dict):
                    payload = {**payload, "markdown": semantic_markdown}
                else:
                    payload = {"data": payload, "markdown": semantic_markdown}

            # Build embeddings once at parse-time for chat reuse
            documents = build_documents_from_parsed(payload)
            if documents:
                api_key = require_openai_api_key()
                embeddings = OpenAIEmbeddings(
                    model=EMBEDDING_MODEL, openai_api_key=api_key
                )
                app.state.vector_store = Chroma.from_documents(
                    documents=documents, embedding=embeddings
                )
            else:
                app.state.vector_store = None
            # cache parsed docs and pdf path for chat visualizations
            app.state.parsed_docs = parsed_raw
            app.state.last_pdf_path = temp_path
            return JSONResponse(
                content={
                    "data": payload,
                    "visualizations": viz_urls,
                    "chunking_strategy": chosen_strategy.value,
                }
            )
        except ChunkingError as exc:
            if "semantic_task" in locals() and semantic_task:
                semantic_task.cancel()
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            if "semantic_task" in locals() and semantic_task:
                semantic_task.cancel()
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {exc}")

    return app


app = create_app()

