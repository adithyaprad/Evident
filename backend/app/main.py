import json
import os
import tempfile
from pathlib import Path
from typing import Any, List
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
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
from agentic_doc.parse import parse
from agentic_doc.config import VisualizationConfig

# Constants
BASE_DIR = Path(__file__).parent.parent
DEFAULT_UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
VIZ_DIR = STATIC_DIR / "visualizations"
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB safeguard
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"


class ParseResponse(BaseModel):
    """Response model to wrap raw parser output."""

    data: Any
    visualizations: List[str] | None = None


class ChatRequest(BaseModel):
    """Payload for chat RAG requests."""

    question: str
    parsed: Any


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
    # Load env if present
    load_dotenv()

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

    def build_visualizations(pdf_path: Path, parsed_docs: Any) -> List[str]:
        """Generate visualization images (boxed/labelled) and return URLs."""
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
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        try:
            with fitz.open(pdf_path) as doc:
                for page_idx, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=144, colorspace=fitz.csRGB)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    draw = ImageDraw.Draw(img, "RGBA")

                    page_chunks = [
                        c for c in parsed_doc.chunks if c.grounding and c.grounding[0].page == page_idx
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

                            label = str(c.chunk_type).upper()
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
                                fill=(*color, int(viz_config.text_bg_opacity * 255)),
                                outline=(0, 0, 0, 180),
                                width=1,
                            )
                            draw.text(
                                (label_x, label_y),
                                label,
                                fill=(255, 255, 255),
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

    @app.post("/api/chat")
    async def chat(payload: ChatRequest) -> JSONResponse:
        api_key = require_openai_api_key()
        # Use prebuilt vector store from the last parse
        vector_store: Chroma | None = getattr(app.state, "vector_store", None)
        if vector_store is None:
            raise HTTPException(status_code=400, detail="No vector store available. Parse a PDF first.")

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
        return JSONResponse(content={"message": response.content, "sources": sources})

    @app.post("/api/parse", response_model=ParseResponse)
    async def parse_pdf(file: UploadFile = File(...)) -> JSONResponse:
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
            parsed = parse(str(temp_path))
            payload = jsonable_encoder(parsed)
            viz_urls = build_visualizations(temp_path, parsed)
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
            return JSONResponse(content={"data": payload, "visualizations": viz_urls})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {exc}")
        finally:
            # best-effort cleanup; ok to leave temp on failure
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    return app


app = create_app()

