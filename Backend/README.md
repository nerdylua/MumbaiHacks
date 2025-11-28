# RAG API Backend

A FastAPI-based RAG (Retrieval-Augmented Generation) system for document processing and Q&A.

## Installation

```bash
# Install uv if not already installed
pip install uv

# Create a virtual environment with Python 3.12
uv venv --python=3.12

# Install dependencies
uv pip install -r requirements.txt
```

## Running

```bash
# Start the FastAPI server
uv run main.py
```

The API will be available at `http://localhost:8000/rag/run`

## Testing

```bash
# Run the test suite
python test_api.py
```

This will run automated tests against the RAG API endpoints and display results in a beautiful CLI interface.

## Running the MCP server (needed for the frontend chat ui)

This project ships an MCP server for tool-style integrations. It’s separate from the FastAPI app and not started by `uv run main.py`.

```bash
# From backend/
python run_mcp.py
```

By default it listens on port 8001 (see `mcp_server/config/mcp_settings.py`).

## Decision Tree Builder

This project includes an autonomous decision tree builder agent that can analyze documents and generate decision trees for insurance policies and similar documents.

### Running the Decision Tree Builder

```bash
# Run the decision tree builder to generate trees from documents
python build_trees.py
```

This will:
- Analyze the configured insurance document
- Generate decision trees for various topics (eligibility, coverage, exclusions, etc.)
- Save JSON files to `results/decision_trees/trees/`

### Converting Decision Trees to Mermaid Diagrams

After generating decision trees, you can convert them to visual Mermaid diagrams:

```bash
# Navigate to the conversion scripts directory
cd scripts/decision_tree

# Convert all JSON decision trees to Mermaid diagrams and PNG images
python convert_json_to_flowchart.py
```

This will:
- Automatically find all decision tree folders in `results/decision_trees/trees/`
- Convert each JSON decision tree to a `.mmd` Mermaid file
- Generate PNG images from the Mermaid files

### Viewing Results

Generated decision trees are stored in:
- **JSON files**: `results/decision_trees/trees/<topic>/<topic>.json`
- **Mermaid files**: `results/decision_trees/trees/<topic>/<topic>.mmd`
- **PNG images**: `results/decision_trees/trees/<topic>/<topic>.png`

## Supabase setup (optional but recommended)

If you use Supabase as the vector store, create a table and RPC match function. Example schema/function names are controlled by `SUPABASE_TABLE_NAME` and `SUPABASE_QUERY_NAME`.

1) Enable pgvector extension and create the table (example):

```sql
-- Enable pgvector once per database
create extension if not exists vector;

-- Example table (text content + embedding + JSONB metadata)
create table if not exists public.documents (
	id uuid primary key default gen_random_uuid(),
	content text not null,
	embedding vector(1536) not null,
	metadata jsonb default '{}'::jsonb,
	created_at timestamp with time zone default now()
);

-- Helpful index for metadata filter by document_id
create index if not exists idx_documents_metadata_docid
	on public.documents ((metadata->>'document_id'));
```

2) Create a match function (example signature used by LangChain’s SupabaseVectorStore):

```sql
create or replace function public.match_documents(
	query_embedding vector(1536),
	match_count int,
	filter jsonb default '{}'::jsonb
)
returns table (
	id uuid,
	content text,
	metadata jsonb,
	similarity float
)
language sql stable as $$
	select d.id, d.content, d.metadata,
				 1 - (d.embedding <=> query_embedding) as similarity
	from public.documents d
	where (
		-- JSONB filter: expect keys like {"document_id": "..."}
		(filter ? 'document_id' and d.metadata->>'document_id' = filter->>'document_id')
		or not (filter ? 'document_id')
	)
	order by d.embedding <=> query_embedding
	limit match_count;
$$;
```

Adjust the vector dimension to your embedding model. If you use `text-embedding-3-small` it’s 1536; other models differ.

## API endpoints

- GET `/` — Basic info
- GET `/health` — Quick health with vector store type, LLM provider, and document count
- POST `/rag/run` — Main RAG endpoint (see below)

### POST /rag/run

Request body:

```json
{
	"documents": "https://.../documents/NIA_Policy.pdf",
	"questions": [
		"What is the maternity coverage?",
		"Waiting period for pre-existing?"
	],
	"k": 8,                      // optional: static k; omit for dynamic k
	"processing_mode": "traditional" // or "structure_aware"
}
```

Response (development mode):

```json
{
	"success": true,
	"answers": ["...", "..."],
	"processing_time": 2.34,
	"document_metadata": {"chunks": 123, "vector_store": "supabase", "document_id": "<hash>"},
	"raw_response": {"debug": "..."}
}
```

Behavior highlights:
- If `k` is omitted, the system uses dynamic-k selection per question.
- Retrieval is hybrid: vector candidates (scoped to `document_id`) + BM25 rerank on candidates.
- `processing_mode` can be:
	- `traditional` (default): standard chunking, retrieval, prompting
	- `structure_aware`: table-aware chunking and reranking, stitched multi-page tables, structure-aware prompt
