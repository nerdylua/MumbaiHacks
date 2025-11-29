# MumbaiHacks · Sampada AI

End-to-end Retrieval-Augmented Generation (RAG) stack for insurance policy understanding. The frontend consumes a trio of backend services—RAG MCP server, computer-use MCP server, and the financial screener server—so keeping those daemons online is the main requirement during development demos.

## Demo of our Solution: [youtube link](https://youtu.be/2kKybRsVwoI)
---

## Repo Highlights
- **Backend (`Backend/`)** – FastMCP RAG server (`run_mcp.py`), computer-use MCP server (`run_computer_mcp.py`), unified screener API (`screener_server/`), plus the FastAPI RAG service and policy decision-tree tooling.
- **Frontend (`Frontend/`)** – Next.js 15 + React 19 app with Supabase auth, AI chat flows, marketing pages, and push notifications.
- **Scripts & Results** – Utilities for converting generated decision trees to Mermaid/PNG plus sample inference outputs under `results/`.

## Core Backend Services (MCP-first)
- **RAG MCP server (`python run_mcp.py`)** – Exposes `process_document_rag` over the Model Context Protocol so the frontend can stream doc QA via FastMCP/HTTP. Configure the port with `MCP_SERVER_PORT` (default 8000).
- **Computer MCP server (`python run_computer_mcp.py`)** – Provides computer-use tools (read/write files, drive Notepad, stream typing, PDF text extraction). Requires Windows plus `pyautogui`, `pywin32`, `PyMuPDF`, and listens on `COMPUTER_MCP_PORT` (default 8002).
- **Screener server (`Backend/screener_server`)** – FastAPI service that logs into screener.in, captures charts/financials, and answers analyst queries. Default port 8080 (configure in `run_server.py`).
- **FastAPI RAG API (`uv run main.py`)** – Traditional REST interface for `/rag/run`. Handy for automated tests and MCP tool reuse but not required for the chat UI if the MCP server is running.

## Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- Node.js 18.18+ (or latest LTS) and npm
- Supabase project (optional, only if you want persistent vector storage)
- Azure Document Intelligence credentials (optional for document parsing)

## Quick Start
| Task | Command | Notes |
| --- | --- | --- |
| Install backend deps | `cd Backend && pip install uv && uv venv --python=3.12 && uv pip install -r requirements.txt` | Installs MCP + FastAPI requirements.
| Run RAG MCP server | `python run_mcp.py` | Keeps `process_document_rag` online for the frontend (default port 8000).
| Run computer MCP server | `python run_computer_mcp.py` | Streams UI actions/Notepad typing for the chat agent (default port 8002).
| Run screener server | `cd Backend/screener_server && uvicorn server:app --host 0.0.0.0 --port 8080` | Serves financial charts, peers, and query agent.
| (Optional) Run FastAPI RAG API | `uv run main.py` | Traditional REST access at `http://localhost:8000`.
| Run backend tests | `python test_api.py` | Validates `/rag/run` behavior.
| Install frontend deps | `cd Frontend && npm install` | Installs Next.js app deps.
| Start frontend dev server | `npm run dev` | Available at `http://localhost:3000`.

> Run backend and frontend in parallel for the full experience. All commands assume execution inside their respective folders.

## Backend Setup (`Backend/`)
### 1. Environment variables
- Copy `.env.example` to `.env`.
- Fill in LLM keys (`DEFAULT_LLM_PROVIDER`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, etc.).
- Supabase for embeddings: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `SUPABASE_TABLE_NAME`, `SUPABASE_QUERY_NAME`.
- Optional integrations: `AZURE_DOCUMENT_INTELLIGENCE_*`, `LMSTUDIO_*`, `MCP_SERVER_PORT`, `COMPUTER_MCP_PORT`.

### 2. RAG MCP server (`run_mcp.py`)
```bash
cd Backend
python run_mcp.py        # defaults to streamable HTTP on port 8000
```
- Implements `process_document_rag` using `fastmcp` (see `mcp_server/server.py`).
- Frontend calls this server directly via `@modelcontextprotocol/sdk`, so keep it running whenever the chat UI is open.
- If you also run the FastAPI app on port 8000, override the MCP port via `set MCP_SERVER_PORT=8001` (PowerShell) first.

### 3. Computer MCP server (`run_computer_mcp.py`)
```bash
cd Backend
python run_computer_mcp.py   # defaults to port 8002
```
- Provides high-level tools: `read_file`, `read_pdf`, `write_file`, `stream_to_notepad`, `type_text`, directory listing, etc.
- Requires Windows plus GUI automation packages (`pyautogui`, `pywin32`, `pyperclip`, `PyMuPDF`). Install them via the main `requirements.txt` or manually (`pip install pyautogui pywin32 pyperclip pymupdf`).
- The frontend’s computer-use flows expect this endpoint URL in `actions/notifications.ts` via the MCP client.

### 4. Screener server (`Backend/screener_server`)
```bash
cd Backend/screener_server
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8080
```
- Logs into screener.in (headless browser), fetches charts/PE data, quarterly results, announcements, and exposes a `/query_agent` endpoint backed by the Google Custom Search key defined in `credentials.py`.
- The frontend fetches analyst dashboards and chart widgets from this server; adjust environment variables or credentials there when deploying.

### 5. FastAPI RAG API (`main.py`, optional)
```bash
cd Backend
pip install uv
uv venv --python=3.12
uv pip install -r requirements.txt
uv run main.py               # FastAPI on http://localhost:8000
```
- Endpoints: `GET /`, `GET /health`, `POST /rag/run`.
- Useful for CLI testing and powering additional tools (e.g., LangChain clients) outside MCP.

### 6. Testing
```bash
cd Backend
python test_api.py
```
Runs mocked calls against `/rag/run` and prints a CLI report.

### 7. Supabase (optional but recommended)
- Enable `pgvector` and create a `documents` table with a `vector(1536)` column for embeddings.
- Add an SQL RPC function (default `match_documents`) that filters by `document_id` metadata and orders by cosine distance. Adjust the vector dimension if you switch embedding models.

### 8. Decision Tree Builder
```bash
cd Backend
python build_trees.py
```
- Outputs topic-specific JSON trees under `results/decision_trees/trees/`.
- Convert to Mermaid/PNG via `scripts/decision_tree/convert_json_to_flowchart.py`.

## Frontend Setup (`Frontend/`)
1. **Environment variables** – copy `env.example` to `.env.local` and populate:
	- Supabase: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `SUPABASE_ADMIN`.
	- App config: `NEXT_PUBLIC_APP_NAME`, `NEXT_PUBLIC_APP_ICON`.
	- LLM providers + web search: `OPENAI_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`, `TAVILY_API_KEY`, etc.
	- Email: `RESEND_API_KEY`, `RESEND_DOMAIN`.

2. **Install & run**
	```bash
	cd Frontend
	npm install
	npm run dev
	```
	The app is built with Next.js 15 (App Router), Tailwind CSS, and React 19. Lint with `npm run lint` and build with `npm run build`.

3. **Optional MCP tooling**
	```bash
	npm run mcp:all   # starts Playwright MCP + computer MCP (requires uv + Python per script)
	```

## Repository Layout
```
Backend/   FastAPI RAG API, MCP servers, decision tree tools
Frontend/  Next.js application (auth, chat, marketing)
tests/     Sample evaluation prompts + answers
Policy documents/ Source PDFs used during development
results/   Saved structure-aware retrieval runs and decision trees
```

## Contributing & Next Steps
- Keep backend + frontend README files updated when adding new services.
- Prefer `.env.example` updates whenever new configuration keys are introduced.
- Run `python test_api.py` and `npm run lint` before opening a PR.
