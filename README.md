
---

<p align="center">
  <img src="doc/logo.png" alt="logo" width="720">
</p>

<div align="center">

# ScholarRAG

**Multi-Agent RAG System for Academic Paper Q&A**

Upload academic papers, ask questions in natural language, get grounded answers with precise citations.

![Python](https://img.shields.io/badge/python-3.12-blue)
![React](https://img.shields.io/badge/react-18-61dafb)
![LangGraph](https://img.shields.io/badge/LangGraph-0.x-orange)
![Milvus](https://img.shields.io/badge/Milvus-2.x-00bfa5)
![License](https://img.shields.io/badge/license-MIT-green)

[Quick Start](#quick-start) | [Features](#features) | [Architecture](#architecture) | [API Reference](#api-reference)
</div>

> [!NOTE]
> Still undergoing continuous optimization and updates...

## What is ScholarRAG?

https://github.com/user-attachments/assets/5f9d36e9-9027-4fcd-b0f4-b0dee7d123a3

ScholarRAG is an end-to-end academic paper Q&A system. It parses PDFs with full structural awareness (sections, tables, figures), retrieves relevant passages via hybrid search, and generates cited answers through a multi-agent pipeline -- all accessible through a clean chat interface.

**Key highlights:**

- Multi-agent query decomposition with parallel retrieval and self-reflection
- Hybrid BM25 + dense retrieval with cross-encoder reranking
- Structured PDF parsing preserving section hierarchy, tables, figures, formulas, and captions
- Smart OCR fallback: fast text extraction by default, OCR only when needed
- Query classification routing: experimental/method/background queries use targeted retrieval strategies
- Multimodal figure understanding: lazy VLM invocation for visual queries and insufficient answers
- Source-level citations with paper, section, and page references
- Multi-turn conversation with memory compression


## Who is this for?

This project is **beginner-friendly** and well-suited for anyone looking to learn and practice the full Agentic RAG workflow -- from PDF ingestion, hybrid retrieval, to multi-agent orchestration with LangGraph. The codebase is modular, well-decoupled, and easy to follow, making it an ideal starting point for students and developers exploring RAG system design.

---

## Contents
- [🗞️ Features](#️-features)
- [📽️ Architecture](#️-architecture)
- [📁 Project Structure](#-license)
- [📖 Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Configuration](#configuration)
  - [Backend](#backend)
  - [Frontend](#frontend) 
  - [Use](#use)
- [⚙️ API Reference](#-api-reference)
- [📊 Evaluation](#-evaluation)
- [🗝️ Tech Stack](#️-tech-stack)
  - [LLM Orchestration Layer]()
  - [Vector Database]()
  - [PDF Parsing]()
  - [Reranking]()
  - [VLM]()
  - [Backend]()
  - [State Persistence]()
  - [Frontend]()
  - [Evaluation System]()
  - [DevOps and Deployment]()
- [⚠️ Security Notice](#️-security-notice)
- [📝 License](#-license)
- [🎉 Key Contributors](#-key-contributors)
- [🎖️ Star History](#️-star-history)

---
:gear:
## 🗞️ Features

<!-- TODO: replace with actual screenshot or GIF -->
<p align="center">
  <img src="doc/demo.gif" alt="Demo" width="800">
</p>

| Category | Details |
|---|---|
| **Retrieval** | BM25 + dense embedding fusion (RRF), cross-encoder reranking, parent-child chunk expansion |
| **PDF Parsing** | Docling-based with section hierarchy, table linearization, formula extraction, figure/caption linking |
| **Smart OCR** | Fast text extraction by default; auto-fallback to full OCR when text density is too low |
| **Figure Extraction** | bbox-based figure image cropping saved per paper (pymupdf) |
| **Query Routing** | LLM classifies queries (experimental/method/background/general) and filters retrieval accordingly |
| **VLM Integration** | Lazy figure analysis: invoked for visual queries or when text answer is insufficient; descriptions cached |
| **Agent** | LangGraph multi-agent: query classification -> decomposition -> parallel sub-agents -> synthesis |
| **Reflection** | Sub-agents self-evaluate sufficiency, retry with refined queries or trigger VLM fallback |
| **Memory** | Sliding window + LLM summary compression for multi-turn context |
| **Streaming** | SSE real-time streamed responses |
| **Citations** | Auto-generated source references (paper, section, page) |
| **Evaluation** | Built-in RAGAS metrics: Faithfulness, Relevancy, Precision, Correctness |

---

## 📽️ Architecture

<div align="center">
  <img src="doc/scholsr_rag.png" alt="Architecture Diagram" width="720">
</div>

---

## 📁 Project Structure

```
scholar-rag/
├── backend/                          # Python backend (FastAPI + LangGraph + RAG)
│   ├── app/                          # FastAPI application layer
│   │   ├── __init__.py               # Module initialization
│   │   ├── main.py                   # FastAPI entry point: register routes, CORS middleware, mount frontend static files
│   │   ├── dependencies.py           # Application lifecycle management: singleton initialization (LLM, Retriever, PDFParser, PostgreSQL checkpointer) and getter functions
│   │   ├── store.py                  # SQLite session and file metadata storage (sessions/files tables, zero-config file database)
│   │   └── routers/                  # API routing modules
│   │       ├── __init__.py
│   │       ├── chat.py               # POST /api/chat — SSE streaming conversation (build LangGraph, push answer/citations/status events token by token)
│   │       ├── sessions.py           # Session management: list, details, history messages (rebuild from PostgreSQL checkpointer), delete
│   │       ├── files.py              # PDF upload (SHA256 deduplication, Docling parsing, chunking into Milvus), file list, delete (sync cleanup vectors)
│   │       └── manage.py             # Collection management: clear Milvus collection + uploads/figures; health check (Milvus & LLM connectivity)
│   │
│   ├── agent/                        # LangGraph multi-agent layer
│   │   ├── states.py                 # State definitions: AgentState (top-level), SubAgentState (sub-agent), SubAnswer; custom merge functions
│   │   ├── graph.py                  # Graph assembly: main graph (summarize→classify→analyze→sub_agents→prepare_synthesis) + sub-graph (retrieve→generate→reflect→retry)
│   │   ├── nodes.py                  # Node implementations: query classification/decomposition, retrieval, generation, reflection (with VLM fallback), conversation summary compression, synthesis (citation remapping)
│   │   ├── prompts.py                # All prompt templates: QUERY_CLASSIFIER / ANALYZER / SYNTHESIZER / GENERATOR / REFLECTOR / SUMMARIZER
│   │   ├── tools.py                  # Tool definitions: paper_retrieval (retrieval tool with query_type routing), ContextVar context variables
│   │   └── checkpointer.py           # Checkpointer factory: create_memory_checkpointer() / create_postgres_checkpointer() (async context manager)
│   │
│   ├── rag/                          # RAG retrieval and parsing core
│   │   ├── models.py                 # Data models: PaperNode (node_id, paper_id, node_type, text, page_num, section_path, bbox, image_path, etc.)
│   │   ├── integration.py            # PDF parsing & RAG integration: TextCleaner (text cleaning), PDFParser (Docling parsing + OCR fallback + figure cropping + caption association), RAGIntegration (nodes→documents, parent/child chunking, Milvus indexing)
│   │   ├── node_generator.py         # Node content generation factory: 6 type generators (Paragraph / Table / Figure / Formula / Caption / SectionHeader), table linearization to key=value format
│   │   ├── retrieval.py              # Hybrid retriever: BM25 + Dense vector fusion (RRF), CrossEncoder reranking, parent chunk backtracking expansion, optional HyDE query expansion, Milvus filter expression building
│   │   ├── factory.py                # Singleton factory: EmbeddingService / RerankerService / MilvusStoreFactory / VisionService (VLM figure analysis); visual query judgment heuristics
│   │   ├── citation.py               # Citation extraction: CitationExtractor (extract paper/section/page citation metadata from retrieved documents and format)
│   │   ├── cache.py                  # Retrieval cache: RetrievalCache (LRU cache based on OrderedDict, MD5 key hashing)
│   │   └── incremental.py            # Incremental updates: IncrementalUpdater (delete/update Milvus parent & child collections by paper_id)
│   │
│   ├── eval/                         # Evaluation system
│   │   ├── eval_retrieval.py         # Retrieval evaluation: Recall@k, Precision@k, MRR, MAP
│   │   ├── eval_generation.py        # Generation evaluation: RAGAS metrics (Faithfulness / AnswerRelevancy / ContextPrecision / FactualCorrectness), end-to-end agent run and CSV report output
│   │   └── benchmark/                # Evaluation benchmark datasets (.gitkeep)
│   │
│   ├── test/                         # Tests
│   │   ├── test_agent.py             # End-to-end Agent test: initialize LLM + Retriever + Graph, run multi-turn conversation
│   │   ├── test_retrieval.py         # Retrieval pipeline test: parse→chunk→index→multi-mode retrieval, structured log output
│   │   ├── test_pdf_parser.py        # PDF parsing test
│   │   ├── test_vlm.py              # VLM service unit test
│   │   └── test_vlm_integration.py   # VLM integration test
│   │
│   ├── data/                         # Runtime data
│   │   └── figures/                  # Extracted figure images (stored in paper_id subdirectories, PyMuPDF 2x DPI cropping)
│   ├── uploads/                      # Uploaded PDF original files
│   ├── db/                           # SQLite database files
│   ├── log/                          # Runtime logs
│   ├── config.py                     # Environment variable configuration: Milvus / LLM / VLM / Embedding / Reranker / PostgreSQL / Upload and all parameters
│   ├── requirements.txt              # Python dependencies
│   ├── Dockerfile                    # Backend container image
│   └── .env.example                  # Environment variable template
│
├── frontend/                         # React frontend (Vite + TailwindCSS)
│   ├── src/
│   │   ├── main.jsx                  # React entry point (StrictMode mount)
│   │   ├── App.jsx                   # Main layout component: manage sessions / messages / panels state, SSE streaming reception, session switching
│   │   ├── api.js                    # API client: fetch + ReadableStream manual SSE parsing, AbortController cancellation support; encapsulates all backend interface calls
│   │   ├── index.css                 # Global styles (TailwindCSS directives)
│   │   └── components/               # UI components
│   │       ├── Sidebar.jsx           # Sidebar: session list, new conversation, delete session
│   │       ├── ChatMessages.jsx      # Message display: Markdown rendering (react-markdown), collapsible citation list
│   │       ├── ChatInput.jsx         # Input box: adaptive height textarea, Enter to send
│   │       ├── FileUpload.jsx        # File upload: drag-and-drop PDF upload, upload progress feedback
│   │       └── SettingsPanel.jsx     # Settings panel: uploaded file list, clear database
│   │
│   ├── public/                       # Static assets
│   │   └── vite.svg
│   ├── index.html                    # HTML entry point
│   ├── vite.config.js                # Vite configuration (dev server + build)
│   ├── tailwind.config.js            # TailwindCSS configuration
│   ├── postcss.config.js             # PostCSS configuration
│   ├── eslint.config.js              # ESLint 9 configuration (React / Hooks / Refresh plugins)
│   ├── nginx.conf                    # Nginx reverse proxy configuration (production deployment)
│   ├── package.json                  # Node.js dependencies and scripts
│   ├── package-lock.json
│   ├── Dockerfile                    # Frontend container image (Nginx hosts build artifacts)
│   └── README.md                     # Frontend documentation
│
├── doc/                              # Documentation resources
│   ├── logo.png                      # Project logo
│   ├── scholar_rag.png               # Architecture diagram
│   ├── architecture.png              # Architecture diagram (backup)
│   └── demo.gif                      # Demo GIF
│
├── resource/                         # Multimedia resources
│   └── ScholarRAG.mp4                # Video
│
├── docker-compose.yml                # 4-service orchestration: backend(8000) + frontend(5173) + milvus(19530) + postgres(5432), with health checks and persistent volumes
├── Makefile                          # Development shortcuts: install / dev / backend / frontend / build / docker-up / docker-down / lint / test / clean
├── LICENSE                           # MIT open source license
└── README.md                         # Project documentation
```

---

## 📖 Quick Start

### Prerequisites

- `Python 3.12+`
- `Node.js 18+`
- [Milvus 2.x](https://milvus.io/docs/install_standalone-docker.md) running on `localhost:19530`
- `PostgreSQL` running (database is created automatically on first start)
- A vLLM / Ollama / OpenAI-compatible LLM endpoint


### Configuration

All settings via `backend/.env`:

```yaml
# Milvus Configuration
MILVUS_URI=http://localhost:19530      # Milvus connection URI
COLLECTION_NAME=papers                 # Collection name prefix

# Model Paths
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 # Embedding model path
RERANKER_MODEL=BAAI/bge-reranker-v2-m3 # Reranker model path

# Retrieval Parameters
FETCH_K=20                             # Candidates before reranking
TOP_K=5                                # Retrieved documents per query
RRF_K=60

# Cache Configuration
ENABLE_CACHE=true
CACHE_MAX_SIZE=1000

# LLM Configuration
LLM_BASE_URL=http://localhost:8848/v1  # LLM endpoint (OpenAI-compatible)
LLM_MODEL=GPT-4o-mini                  # Model name
LLM_API_KEY=empty
LLM_TEMPERATURE=0.1                    # Generation temperature
LLM_MAX_TOKENS=4096
MAX_RETRIES=0                          # Reflection retry limit

# VLM Configuration
VLM_ENABLED=true                       # Enable VLM for figure analysis
VLM_BASE_URL=http://localhost:8080/v1  # VLM endpoint (OpenAI-compatible, multimodal)
VLM_MODEL=Qwen3-vl-4B                  # VLM model name
VLM_API_KEY=empty                      # VLM API key

# Postgres (checkpointer + session store)
POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/scholar_rag

# Upload
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE_MB=50

# Server
HOST=0.0.0.0
PORT=8000
```

### Option 1: Docker (recommended)

```bash
cp backend/.env.example backend/.env   # edit with your model endpoints
docker-compose up -d
```

All services start automatically. Open http://localhost:8000

### Option 2: Makefile

Requires Milvus and Postgres running locally.

```bash
cp backend/.env.example backend/.env   # edit with your model endpoints
make install                            # install all dependencies
make start                              # build frontend + start backend at http://localhost:8000
```

For development (hot reload):

```bash
make dev                                # backend :8000 + frontend dev server :5173
```

### Use

1. Open http://localhost:8000 in your browser
2. Upload PDF papers via the upload panel
3. Ask questions — get cited answers in seconds

---

## ⚙️ API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | SSE streaming chat (`{query, session_id?}`) |
| `GET` | `/api/sessions` | List sessions |
| `GET` | `/api/sessions/:id/history` | Conversation history |
| `DELETE` | `/api/sessions/:id` | Delete session |
| `POST` | `/api/files/upload` | Upload PDFs (multipart) |
| `GET` | `/api/files` | List uploaded files |
| `DELETE` | `/api/files/:id` | Delete file + vectors |
| `DELETE` | `/api/collection` | Clear vector database |
| `GET` | `/api/health` | Health check |

---

## 📊 Evaluation

```bash
cd backend

# Retrieval: Recall@k, Precision@k, MRR, MAP
python eval/eval_retrieval.py

# Generation: RAGAS (Faithfulness, Relevancy, Precision, Correctness)
python eval/eval_generation.py
```

---

## 🗝️ Tech Stack

<details>
  <summary>1. LLM Orchestration Layer (click me)</summary>

The core agent workflow of the project is built with LangGraph (`backend/agent/graph.py`), adopting a multi-agent architecture:

**Main Graph Flow:**
```
START → summarize → classify → analyze → [sub_agent × N] → prepare_synthesis → END
```

- `summarize`: Compresses conversation history exceeding the window size (6 turns) into summaries, using `RemoveMessage` to clean up old messages and prevent context overflow.
- `classify`: Classifies user queries into four types via LLM structured output (`with_structured_output`): `experimental_result`, `method`, `background`, and `general`, used for downstream retrieval routing.
- `analyze`: Decomposes complex questions into multiple sub-queries (`QueryAnalysis`), dispatching them in parallel to multiple sub-agents via the `Send` mechanism.
- `prepare_synthesis`: Aggregates all sub-agent responses, remaps citation numbers, and constructs the final synthesis prompt.

**Sub-Agent Graph Flow (each sub-query runs independently):**
```
START → retrieve → generate → reflect → [retry | done]
                                ↑                |
                          prepare_retry ←--------┘
```

- The `reflect` node uses an LLM to judge whether the answer is sufficient (`ReflectionResult`). If insufficient, it generates supplementary queries and retries, up to `MAX_RETRIES` (default 2).
- The reflection stage also includes a VLM fallback mechanism: when the text answer is insufficient and the retrieved results contain figures/charts, it automatically triggers visual model analysis.

LangChain provides the underlying abstractions: `BaseChatModel`, `Document`, `HumanMessage/AIMessage/SystemMessage`, `RecursiveCharacterTextSplitter`, etc. LLM calls are made through `langchain-openai`'s `ChatOpenAI`, compatible with any OpenAI-format API (default configured for Ollama's `qwen3:32b`).

Structured output extensively uses Pydantic models (`QueryAnalysis`, `QueryClassification`, `ReflectionResult`) to ensure the LLM returns parseable structured data.

</details>

<details>
  <summary>2. Vector Database (click me)</summary>

Milvus is deployed via Docker Compose (`milvusdb/milvus:v2.4.0` standalone mode), using embedded etcd and local storage.

**Hybrid Retrieval Architecture (`backend/rag/retrieval.py`):**

- Uses the `langchain-milvus` integration; each collection simultaneously builds dense vector indexes and BM25 sparse indexes (`BM25BuiltInFunction`).
- Retrieval fuses results from both pathways via RRF (Reciprocal Rank Fusion), with `rrf_k` defaulting to 60.
- Supports metadata filtering: by `node_type` (table/figure/caption, etc.) and `section_path`.

**Parent-Child Chunking Strategy (`backend/rag/integration.py`):**

- Documents are split into parent chunks (complete semantic units) and child chunks (500-character slices with 50-character overlap).
- Special nodes such as tables, figures, headings, and captions are not further split and are directly used as child chunks.
- During retrieval, the system first searches the child collection; upon a hit, it traces back to the parent chunk via `chunk_parent_id` to obtain more complete context.
- The two collections are named `{collection_name}_children` and `{collection_name}_parents` respectively.

**Retrieval Pipeline:**
```
Query → [Optional HyDE Expansion] → Hybrid Search (BM25+Dense) → RRF Fusion → Rerank → Parent Expansion → Deduplication → Top-K
```

It also implements retrieval caching (`RetrievalCache`) and incremental updates (`IncrementalUpdater`).

</details>

<details>
  <summary>3. PDF Parsing (click me)</summary>

**Docling (`backend/rag/integration.py`):**

- Uses `DocumentConverter` to parse PDFs, automatically identifying document structure elements: `SectionHeaderItem`, `TextItem`, `ListItem`, `TableItem`, `PictureItem`, `FormulaItem`.
- Supports OCR fallback: if the initial parse yields too little text (total characters < 1000 or < 200 characters per page), OCR is automatically enabled for re-parsing.
- Parsed elements undergo filtering (removing headers, footers, and page numbers), reading order sorting (row-column grouping based on bbox coordinates), and section hierarchy tracking.

**Node Content Generation (`backend/rag/node_generator.py`):**

Uses a factory pattern to provide specialized content generators for 6 node types:
- `ParagraphGenerator`: Appends section path context
- `TableGenerator`: Linearizes tables into `Row N: header1=val1, header2=val2` format
- `FigureGenerator`: Combines caption and surrounding descriptive text
- `FormulaGenerator`: Appends section context
- `CaptionGenerator`, `SectionHeaderGenerator`

**PyMuPDF (`fitz`):**

- Used for figure/chart image cropping: crops figure regions from PDF pages based on bbox coordinates provided by Docling.
- Coordinate system conversion: Docling uses the PDF standard coordinate system (origin at bottom-left), while PyMuPDF uses the screen coordinate system (origin at top-left), converted via `fitz_y = page_height - docling_y`.
- Renders at 2x DPI, saves as PNG, stored in the `data/figures/{paper_id}/` directory.

</details>

<details>
  <summary>4. Reranking (click me)</summary>

- Uses `sentence-transformers`' `CrossEncoder` to load the `BAAI/bge-reranker-v2-m3` model.
- During retrieval, first fetches `fetch_k × 2` candidate documents, scores them with CrossEncoder, then takes the top `fetch_k`.
- The embedding model uses `HuggingFaceEmbeddings` (`langchain-huggingface`), defaulting to `BAAI/bge-small-en-v1.5`.
- Both services adopt the singleton pattern (`EmbeddingService`, `RerankerService`) to avoid redundant loading.

</details>

<details>
  <summary>5. VLM (click me)</summary>

**VisionService (`backend/rag/factory.py`):**

- Singleton pattern; accepts any `BaseChatModel` as the backend (default `qwen-vl` via Ollama).
- Encodes figure/chart images in base64 and sends them to the VLM via OpenAI-compatible multimodal message format.
- Analysis covers: chart type, key visual elements, main findings, and visible numerical values.

**Trigger Mechanism (Dual Path):**

1. Proactive trigger: When the query contains visual keywords ("show", "chart", "figure", etc.) and retrieved results contain figures, VLM descriptions are injected during the `generate` stage.
2. Fallback trigger: When the `reflect` stage determines the answer is insufficient, it checks for unanalyzed figures, triggers VLM supplementary analysis, and regenerates (processes up to 2 images to control cost).

VLM descriptions are appended to the document context with a `[Figure Analysis]` prefix for the LLM to reference when generating the final answer.

</details>

<details>
  <summary>6. Backend (click me)</summary>

**FastAPI (`backend/app/main.py`):**

- 4 router modules: `chat` (conversation), `sessions` (session management), `files` (file upload), `manage` (collection management).
- CORS fully open (development mode).
- Supports mounting frontend static files (`frontend/dist`) for single-port deployment.

**SSE Streaming Output (`backend/app/routers/chat.py`):**

- Uses `sse-starlette`'s `EventSourceResponse` to implement Server-Sent Events.
- Streaming event types: `session_id` → `status` → `sub_queries` → `answer` (token by token) → `citations` → `done`.
- During the synthesis stage, tokens are streamed via `llm.astream()` for real-time frontend rendering.
- After the answer is complete, conversation history is persisted to the checkpointer via `graph.aupdate_state()`.

**Uvicorn:** ASGI server with hot-reload support for development mode.

</details>

<details>
  <summary>7. State Persistence (click me)</summary>

- PostgreSQL 16 (Alpine) is deployed via Docker Compose for LangGraph conversation state persistence.
- Uses `langgraph-checkpoint-postgres`'s `AsyncPostgresSaver` for async checkpoint read/write.
- Also provides an in-memory checkpointer (`MemorySaver`) as a lightweight alternative.
- The database adapter uses `psycopg` v3 (with binary and pool support).

</details>

<details>
  <summary>8. Frontend (click me)</summary>

**React 18 (`frontend/src/`):**

- Pure functional components + Hooks architecture (`useState`, `useEffect`, `useRef`, `useCallback`).
- Component structure: `App` (main layout) → `Sidebar` (session list), `ChatMessages` (message display), `ChatInput` (input box), `FileUpload` (file upload), `SettingsPanel` (settings panel).
- `react-markdown` renders Markdown content in AI responses.
- `lucide-react` provides icons (Upload, Settings, ChevronLeft, etc.).

**SSE Client (`frontend/src/api.js`):**

- Uses native `fetch` + `ReadableStream` to manually parse SSE data streams.
- Supports `AbortController` to cancel in-progress requests.
- Event-driven: updates UI state based on the `type` field (session_id/answer/citations/done/error).

**Build Toolchain:**
- Vite 5: Dev server + production builds.
- TailwindCSS 3.4 + PostCSS + Autoprefixer: Style processing.
- ESLint 9 + eslint-plugin-react/react-hooks/react-refresh: Code quality.
- Production deployment via Nginx reverse proxy (`frontend/nginx.conf` + Dockerfile).

</details>

<details>
  <summary>9. Evaluation System (click me)</summary>

**RAGAS Generation Quality Evaluation (`backend/eval/eval_generation.py`):**

- Evaluation metrics: `Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `FactualCorrectness`.
- Uses `LangchainLLMWrapper` and `LangchainEmbeddingsWrapper` to adapt evaluators.
- End-to-end evaluation: runs the complete agent graph, collects answers and context, and outputs CSV reports.

**Custom Retrieval Evaluation (`backend/eval/eval_retrieval.py`):**

- Metrics: `Recall@k`, `Precision@k`, `MRR` (Mean Reciprocal Rank), `MAP` (Mean Average Precision).
- Directly evaluates the full retrieval pipeline: hybrid search + rerank + parent expansion.

</details>

<details>
  <summary>10. DevOps and Deployment (click me)</summary>

**Docker Compose (`docker-compose.yml`):**

4-service orchestration:
- `backend`: FastAPI application, starts after Milvus and Postgres health checks pass.
- `frontend`: Nginx serving build artifacts, mapped to port 5173.
- `milvus`: v2.4.0 standalone, embedded etcd, exposes 19530 (gRPC) and 9091 (health check).
- `postgres`: 16-alpine, persistent volume `postgres_data`.

**Makefile:** Provides shortcut commands: `install`, `dev` (starts both frontend and backend), `build`, `test` (pytest), `lint`, `clean`, etc.

**Environment Configuration:** Loads `.env` files via `python-dotenv`; all configuration items can be overridden through environment variables (`backend/config.py`).

</details>

---

## ⚠️ Security Notice

ScholarRAG is designed as a **research and learning tool** and is intended to run in **trusted local or internal network environments**. It does not include production-grade security hardening out of the box. Please review the following before deployment:

### API & Authentication

- **No built-in authentication or authorization.** All API endpoints (chat, file upload, session history, collection management) are publicly accessible to anyone who can reach the server.
- **Session IDs are the only access boundary.** Anyone with a valid session ID can read its full conversation history or delete it.
- **Destructive endpoints are unprotected.** `DELETE /api/collection` will wipe the entire vector database without any confirmation or credential check.

### Credentials & Secrets

- **API keys and database credentials** (`LLM_API_KEY`, `VLM_API_KEY`, `POSTGRES_URI`) are stored in plaintext `.env` files. Never commit `.env` to version control.
- **Default credentials** in `.env.example` and `docker-compose.yml` (e.g., `postgres:postgres`) must be changed before any non-local deployment.
- There is **no secret rotation mechanism** -- rotate keys and passwords manually on a regular basis.

### Network & Transport

- **All services communicate over plain HTTP** by default (FastAPI, Milvus, PostgreSQL, LLM/VLM endpoints). Configure TLS/HTTPS via a reverse proxy (e.g., Nginx) if the system is exposed beyond localhost.
- **CORS is fully open** (`allow_origins=["*"]`). Restrict allowed origins to your frontend domain in production.
- **Milvus and PostgreSQL** are exposed without network-level access controls by default. Use firewall rules or Docker network isolation to limit access.

> [!CAUTION]
> Do not expose ScholarRAG directly to the public internet without adding authentication, TLS, and proper access controls. It is safe for local development and trusted internal networks as-is.

---

## 📝 License

This project is open source and available under the [MIT License](./LICENSE).

---

## 🎉 Key Contributors

- [PangHu1020](https://github.com/PangHu1020)
- [curme-miller](https://github.com/curme-miller)

---

## 🎖️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PangHu1020/scholar-rag&type=Date)](https://www.star-history.com/#PangHu1020/scholar-rag&Date)