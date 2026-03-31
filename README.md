# Technology Pipeline Tracker

A multi-agent intelligence system that answers the question: **"Where is this technology in its journey from academic research to mainstream product?"**

It collects live signals from 9 data sources — academic papers, GitHub repositories, startup funding news, patent filings, and public trend data — feeds them through a scoring pipeline, then uses Claude (Anthropic) to produce a structured stage assessment with cited evidence, velocity analysis, and a self-critique confidence score. Results stream to a browser dashboard in real time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser  ─── GET /  ──►  ui/index.html  (SSE streaming UI)     │
│           ◄── SSE ──────  POST /analyse/stream                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                   ┌────────────▼────────────┐
                   │  FastAPI Agent Service   │  api/agent_service.py
                   │  (A2A REST + SSE layer)  │
                   └─────────┬───────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
   ┌──────────▼──────────┐     ┌────────────▼────────────┐
   │   Researcher Agent   │     │    Analyst Agent         │
   │  agents/researcher.py│     │   agents/analyst.py      │
   │                      │     │                          │
   │  MCP tool catalogue: │     │  Call 1 — Assessment     │
   │  • run_pipeline      │     │  Claude claude-sonnet-4-6│
   │  • semantic_search   │     │  ↓                       │
   │  • graph_context     │     │  Call 2 — Self-critique   │
   │  • get_trend_history │     │  (confidence 0–100,      │
   │  • get_timeline      │     │   hallucination check)   │
   └──────────┬───────────┘     └────────────┬────────────┘
              │                              │
   ┌──────────▼──────────────────────────────▼────────────┐
   │                    Pipeline                           │
   │                                                       │
   │  fetcher.py  →  cleaner.py  →  embedder.py           │
   │  (fetch all)    (normalise)    (score + store)        │
   └──────┬────────────────────────────────────┬──────────┘
          │                                    │
          ▼                                    ▼
   ┌──────────────┐    ┌──────────┐    ┌──────────────────┐
   │  9 Data Tools │    │ MongoDB  │    │ ChromaDB (RAG)   │
   │               │    │ (raw     │    │ sentence-        │
   │  Stage 1:     │    │  docs)   │    │ transformers     │
   │  arXiv        │    └──────────┘    │ all-MiniLM-L6-v2 │
   │  Semantic     │                    └──────────────────┘
   │  Scholar      │    ┌──────────┐    ┌──────────────────┐
   │               │    │  Neo4j   │    │ SQLite           │
   │  Stage 2:     │    │ (graph   │    │ (scores,         │
   │  GitHub       │    │  RAG)    │    │  timeline,       │
   │  ProductHunt  │    └──────────┘    │  audit log)      │
   │  YC Companies │                    └──────────────────┘
   │               │
   │  Stage 3:     │
   │  NewsAPI      │    ┌──────────────────────────────────┐
   │  TechCrunch   │    │  W3C PROV Lineage                │
   │               │    │  lineage/tracker.py              │
   │  Stage 4:     │    │  Every data access logged to     │
   │  PatentsView  │    │  SQLite for hallucination audit  │
   │               │    └──────────────────────────────────┘
   │  Stage 5:     │
   │  Wikipedia    │
   │  Google Trends│
   └───────────────┘
```

### The five pipeline stages

| Stage | Label | Active when... |
|-------|-------|----------------|
| 1 | Academic | arXiv papers or Semantic Scholar citations exist |
| 2 | Developer | GitHub repos, ProductHunt launches, or YC companies found |
| 3 | Investment | NewsAPI funding articles or TechCrunch coverage found |
| 4 | Big Tech | Patent filings by large organisations detected |
| 5 | Mainstream | Wikipedia article exists with views, or Google Trends presence |

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | `python --version` to check |
| Docker Desktop | 4.x+ | Must be running before `docker compose up` |
| Anthropic API key | — | Required for the Analyst agent |
| GitHub token | — | Required for repository search |
| NewsAPI key | — | Free tier (100 req/day) is sufficient |
| Product Hunt credentials | — | OAuth client ID + secret, or developer token |
| Neo4j instance | — | Local Docker (included) or Neo4j Aura free tier |
| MongoDB instance | — | Local Docker (included) or Atlas free tier |

**Optional (higher rate limits / better results):**
- Semantic Scholar API key
- Reddit API credentials

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd DataEngineeringIndividual
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Playwright browser binaries (used by TechCrunch scraper)
playwright install chromium
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in the required values. The keys you need:

```
ANTHROPIC_KEY           — https://console.anthropic.com/ → API Keys
GITHUB_TOKEN            — https://github.com/settings/tokens (public_repo scope)
NEWS_API_KEY            — https://newsapi.org/register (free tier)
PRODUCTHUNT_CLIENT_ID   — https://www.producthunt.com/v2/oauth/applications
PRODUCTHUNT_CLIENT_SECRET
NEO4J_PASSWORD          — choose any password (used by Docker Compose Neo4j)
```

For Neo4j and MongoDB, if you are using the local Docker setup (recommended for development) the default values in `.env.example` work without changes:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
MONGO_URI=mongodb://localhost:27017
```

### 4. Start the databases

```bash
docker compose up -d mongo neo4j
```

Wait ~30 seconds for both services to pass their health checks, then verify:

```bash
docker compose ps
# Both mongo and neo4j should show "healthy"
```

### 5. Start the API server

```bash
python main.py serve
```

You should see:

```
Dashboard running at http://localhost:8000
API docs at         http://localhost:8000/docs
```

### 6. Open the dashboard

Navigate to **http://localhost:8000** in your browser.

---

## Alternative: run everything in Docker

To run the full stack (databases + API) in containers:

```bash
docker compose up --build
```

The API service waits for both databases to be healthy before starting.

---

## Example queries to try

These span all five pipeline stages and produce clear, evidence-backed assessments:

| Query | Expected stage | Why it is interesting |
|-------|---------------|-----------------------|
| `transformer neural network` | 5 — Mainstream | Foundational; huge paper volume, Wikipedia presence, Google Trends |
| `quantum error correction` | 3–4 — Investment / Big Tech | Active IBM/Google patent filings; VC funding starting |
| `diffusion models` | 4–5 — Big Tech / Mainstream | Rapid stage progression; strong velocity signal |
| `neuromorphic computing` | 2–3 — Developer / Investment | Emerging; GitHub activity growing, limited mainstream presence |
| `homomorphic encryption` | 2 — Developer | Years of academic work, growing OSS implementation, minimal VC yet |
| `large language models` | 5 — Mainstream | Saturated signal across all five stages |
| `solid-state batteries` | 3 — Investment | Heavy VC and corporate R&D; limited public product availability |

The first query for any technology takes 60–120 seconds (all 9 APIs run sequentially). Subsequent queries for the same technology within 24 hours are served from the MongoDB cache in under 2 seconds.

---

## CLI commands

```bash
# Start the web server (default port 8000)
python main.py serve

# One-shot analysis, printed to terminal
python main.py analyse "quantum computing"

# Stream Claude tokens as they arrive
python main.py analyse "quantum computing" --stream

# Run the data pipeline only (no Claude API cost)
python main.py pipeline "neuromorphic computing"

# Print the W3C PROV lineage log for a past query
python main.py lineage "diffusion models"
```

---

## Folder structure

```
DataEngineeringIndividual/
│
├── main.py                    CLI entry point (serve / analyse / pipeline / lineage)
├── requirements.txt           Python dependencies
├── Dockerfile                 Python 3.11-slim container for the API service
├── docker-compose.yml         MongoDB + Neo4j + API (three-service stack)
├── .env                       Your credentials (never commit this)
├── .env.example               Template — copy to .env and fill in keys
│
├── ui/
│   └── index.html             Single-file browser dashboard (SSE streaming UI)
│
├── api/
│   └── agent_service.py       FastAPI service — REST + SSE endpoints, result cache
│
├── agents/
│   ├── researcher.py          Researcher agent — MCP tool catalogue, data retrieval
│   └── analyst.py             Analyst agent — two-call Claude reasoning + self-critique
│
├── pipeline/
│   ├── fetcher.py             Orchestrates all 9 tool/scraper calls in sequence
│   ├── cleaner.py             Normalises and deduplicates raw documents
│   └── embedder.py            Storage distribution, scoring (0–100), velocity analysis
│
├── tools/                     One file per data source (all return clean dicts)
│   ├── arxiv_tool.py          arXiv paper search (Stage 1)
│   ├── semantic_scholar_tool.py  Citation-weighted academic search (Stage 1)
│   ├── github_tool.py         Repository search + activity metrics (Stage 2)
│   ├── producthunt_tool.py    Product Hunt launch search (Stage 2)
│   ├── news_tool.py           NewsAPI funding news (Stage 3)
│   ├── patents_tool.py        PatentsView + fallback chain (Stage 4)
│   ├── wikipedia_tool.py      Article existence + page views (Stage 5)
│   ├── trends_tool.py         Google Trends presence (Stage 5)
│   └── reddit_tool.py         Community discussion signals (supplementary)
│
├── scrapers/                  HTML scrapers for sites without public APIs
│   ├── yc_scraper.py          Y Combinator portfolio companies (Stage 2)
│   └── techcrunch_scraper.py  TechCrunch funding articles (Stage 3)
│
├── database/
│   ├── mongo_client.py        MongoDB — raw document storage (7 collections)
│   ├── sqlite_client.py       SQLite — scores, timeline, query audit log
│   ├── vector_store.py        ChromaDB — sentence-transformer embeddings for RAG
│   ├── graph_client.py        Neo4j — technology relationship graph (GraphRAG)
│   ├── chroma_db/             ChromaDB persisted data (git-ignored)
│   └── pipeline_tracker.db    SQLite database file (git-ignored)
│
├── lineage/
│   └── tracker.py             W3C PROV event log — every data access recorded
│
└── scripts/
    ├── validate_env.py        Checks all required keys before startup
    ├── ping_mongo.py          MongoDB connectivity smoke test
    └── test_tools.py          Individual tool smoke tests
```

---

## How the scoring works

Each of the 10 data sources contributes a score on a 0–100 scale based on document count and secondary signals (recency, citation count, funding relevance). The scores are grouped into five stage contributions:

```
Stage 1 (Academic)    arxiv_score + semantic_scholar_score      → 0–20 pts
Stage 2 (Developer)   github_score + producthunt_score + yc_score → 0–20 pts
Stage 3 (Investment)  news_score + techcrunch_score             → 0–20 pts
Stage 4 (Big Tech)    patents_score                             → 0–20 pts
Stage 5 (Mainstream)  wikipedia_score + trends_score            → 0–20 pts
                                                       overall  → 0–100
```

The `overall_stage` is the **highest stage** where at least one score exceeds 30 (50 for Wikipedia, 60 for Google Trends). A technology at Stage 4 has necessarily shown signal at all earlier stages.

Claude receives all scores plus the top RAG matches from ChromaDB, the Neo4j neighbourhood, and the velocity analysis as a structured evidence block. It is instructed to cite specific sources by name and never to assert anything not present in the evidence.

A second Claude call (self-critique) then evaluates the first response: it lists supporting and contradicting evidence, flags unsupported claims, and outputs a numeric confidence score (0–100) with a reliability tier (High / Medium / Low).

---

## API reference

Interactive docs are available at **http://localhost:8000/docs** while the server is running.

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Browser dashboard |
| `POST` | `/analyse/stream` | Full pipeline + Claude — SSE streaming |
| `POST` | `/analyse` | Full pipeline + Claude — blocking JSON |
| `POST` | `/research/pipeline` | Data pipeline only (no Claude) |
| `POST` | `/research/semantic_search` | RAG search over ChromaDB |
| `POST` | `/research/graph_context` | Neo4j neighbourhood |
| `GET` | `/research/trend_history` | Historical scores from SQLite |
| `GET` | `/lineage` | W3C PROV event log |
| `GET` | `/health` | Liveness check |
