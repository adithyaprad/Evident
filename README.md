# Evident

### Case study - https://www.adithyapradeep.com/#projects/evident

Evident is a document intelligence web application designed to make retrieval-augmented generation (RAG) systems transparent, grounded, and measurable.

Instead of treating RAG as a black box, Evident exposes how documents are parsed and chunked, shows exactly which parts of a document are used to answer a question, and evaluates answer quality using explicit metrics. The goal is to make RAG systems easier to inspect, debug, and improve.

---

## What this project demonstrates

* How different chunking strategies affect retrieval and answer quality
* How grounded answers improve trust in RAG outputs
* How evaluation helps separate retrieval failures from generation failures
* How RAG systems can be designed as inspectable pipelines rather than opaque model calls

This project is intentionally scoped as a **system exploration tool**, not a production SaaS.

---

## Core features

* Upload and parse PDFs with multiple chunking strategies
* Inspect parsed output as structured JSON and annotated document overlays
* Ask questions using a lightweight RAG chatbot
* See exactly which chunks were used to answer each question
* Evaluate answers using built-in quality metrics

---

## Architecture overview

Evident consists of:

* A **FastAPI backend** responsible for parsing, chunking, retrieval, grounding, and evaluation
* A **React/Vite frontend** for document upload, inspection, chat, and visualization
* A **temporary in-memory vector store** used for retrieval during a session

All intermediate state is ephemeral by design to keep iteration fast and behavior easy to inspect.

---

## Chunking strategies

The backend supports multiple chunking approaches so their tradeoffs can be compared directly:

* **Semantic (agentic)**
  Meaning-based chunking using `agentic_doc.parse`, with visual grounding support.
* **Late chunking**
  Embedding-first approach using Jina embeddings to preserve broader context.
* **LLM-guided chunking**
  LlamaParse-based parsing with semantic splitting.
* **Hierarchical chunking**
  Multi-level chunking that preserves document structure.

---

## Evaluation metrics

Each generated answer can be scored using an LLM-based evaluation step:

* **Context relevance** – how well retrieved chunks match the question
* **Faithfulness** – whether the answer is supported by retrieved context
* **Answer relevance** – how directly the answer addresses the question
* **Correctness** – factual accuracy relative to the source document

These metrics are intended for debugging and comparison, not absolute judgment.

---

## Prerequisites

* Python 3.10+ (backend)
* Node.js 18+ (frontend)

### Required API keys

* `OPENAI_API_KEY` – embeddings, chat, evaluation
* `JINA_API_KEY` – late chunking
* `LLAMA_CLOUD_API_KEY` – LLM-based parsing
* `VISION_AGENT_API_KEY` – agentic visualizations

Optional:

* `EVAL_MODEL` – override evaluation model (default: `gpt-4o-mini`)

---

## Quick start

### Backend

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy env.example .env
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
copy env.example .env.local
npm install
npm run dev
```

Frontend runs at `http://localhost:5173` by default.

---

## API overview

* `GET /healthz` – service health check
* `POST /api/parse` – upload and parse a PDF using a selected chunking strategy
* `POST /api/chat` – ask a question against the most recent parsed document
* `POST /api/chat/visualizations` – retrieve filtered annotations for specific chunks
* `POST /api/chat/evaluate` – evaluate an answer using grounding and relevance metrics

---

## Limitations and design notes

* Vector storage is in-memory and scoped to a single backend process
* Uploaded files and visualizations are stored temporarily
* No authentication or rate limiting is enforced
* Some chunking strategies do not produce visual grounding boxes

These tradeoffs are intentional to keep the system focused on inspection and iteration.

---

## When this project is useful

Evident is useful if you want to:

* understand why a RAG system produced a specific answer
* compare chunking strategies on real documents
* debug retrieval failures visually
* explore evaluation techniques for RAG pipelines

---

## Summary

Evident treats RAG as a system that needs observability. By making chunking, grounding, and evaluation explicit, it turns RAG from a black box into something that can be inspected, reasoned about, and improved.
