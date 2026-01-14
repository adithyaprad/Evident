# Ground RAG Webapp

Minimal FastAPI + React/Vite UI to upload a PDF, run `agentic_doc.parse`, and view the raw JSON output (pretty-printed). Backend stores uploads in a temp folder and cleans up after parsing.

## Backend (FastAPI)
1. `cd backend`
2. (Optional) `python -m venv .venv && .venv\\Scripts\\activate`
3. `pip install -r requirements.txt`
4. Copy `env.example` to `.env` and fill in keys if needed (`FRONTEND_ORIGIN` defaults to `http://localhost:5173`).
5. Run: `uvicorn app.main:app --reload --port 8000`
   - Health check: `GET /healthz`
   - Parse: `POST /api/parse` (multipart/form-data, field `file`, PDF only, ~25MB cap)
   - Uploads land in `backend/uploads/` (temp files are deleted after parse best effort).

## Frontend (React/Vite, TS)
1. `cd frontend`
2. Copy `env.example` to `.env.local` (or `.env`) and adjust `VITE_API_BASE` (default `http://localhost:8000`).
3. `npm install`
4. `npm run dev` (defaults to `http://localhost:5173`)
5. UI: left pane for file upload/drop + parse trigger; right pane shows pretty JSON with copy-to-clipboard and loader states.

## Notes
- CORS allows `http://localhost:5173` (override via `FRONTEND_ORIGIN`).
- Only PDF files are accepted; size guarded at ~25MB.
- Parser output is returned as-is under `data` for transparency.***
