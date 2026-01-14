import os
import tempfile
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Third-party parser used elsewhere in the repo
from agentic_doc.parse import parse

# Constants
BASE_DIR = Path(__file__).parent.parent
DEFAULT_UPLOAD_DIR = BASE_DIR / "uploads"
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB safeguard


class ParseResponse(BaseModel):
    """Response model to wrap raw parser output."""

    data: Any


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

    @app.get("/healthz")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

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
            return JSONResponse(content={"data": payload})
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

