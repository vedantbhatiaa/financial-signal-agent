# Financial Signal Agent — MSIN0166 Data Engineering

> An LLM-powered financial intelligence platform that synthesises stock price data, live news sentiment, and SEC regulatory filings into grounded, auditable investment signals via a natural language agent interface.

**Module**: MSIN0166 Data Engineering · UCL School of Management · 2025/26  
**Stack**: Python · LangChain · Groq (Llama 3.3) · PostgreSQL · MongoDB · ChromaDB · FastAPI · Docker

---

## Architecture

![Architecture Diagram](images/structure.jpg)

```
Data Sources              Storage Layer           Agent Layer          API + Frontend
─────────────             ─────────────           ───────────          ──────────────
Alpha Vantage API  ──►   PostgreSQL             
RSS Feeds (13)     ──►   MongoDB          ──►   LangChain Agent  ──►  FastAPI
SEC EDGAR API      ──►   MongoDB +               (3 tools,             POST /query
                         ChromaDB (RAG)           Groq LLaMA 3.3)      GET /lineage
                                                                        Frontend UI
```

---

## Project Structure

```
financial-signal-agent/
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb   # Ingest prices, news, SEC filings
│   ├── 02_pipeline.ipynb         # Transform, embed into ChromaDB, lineage
│   └── 03_agent_demo.ipynb       # Live agent queries with tool traces
│
├── agent/
│   ├── tools.py                  # 3 LangChain tool definitions
│   └── agent.py                  # FinancialAgent class + system prompt
│
├── api/
│   └── app.py                    # FastAPI REST wrapper (CORS, /query, /lineage)
│
├── frontend/
│   └── index.html                # SignalCore UI — served at localhost:8000
│
├── scripts/
│   └── run_pipeline.py           # Cross-platform pipeline automation
│
├── images/
│   └── architecture.png          # System architecture diagram
│
├── data/                         # Generated at runtime — not tracked in Git
│   ├── chroma/                   # ChromaDB vector store (persistent)
│   └── lineage/                  # lineage_log.jsonl — W3C Prov-aligned audit trail
│
├── docker-compose.yml            # PostgreSQL + MongoDB services
├── Dockerfile                    # Application container
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
└── .gitignore
```

---

## Data Sources

| Source | Format | Storage | Agent Tool |
|---|---|---|---|
| Alpha Vantage API | Structured OHLCV | PostgreSQL | `query_price_data` |
| RSS feeds — Reuters, FT, Yahoo Finance (general + 10 ticker-specific) | Unstructured text | MongoDB + ChromaDB | `search_financial_news` |
| SEC EDGAR API — 10-K and 10-Q filings | Semi-structured documents | MongoDB + ChromaDB | `search_financial_news`, `get_filing_metadata` |

**Tickers monitored**: AAPL · NVDA · MSFT · GOOGL · AMZN · TSLA · META · JPM · GS · BAC

---

## Storage Architecture

| System | Schema / Collection | Content | Scale |
|---|---|---|---|
| PostgreSQL | `prices` table — ticker, date, OHLCV, SMA-20, RSI-14, price_change_pct | ~1,000 rows · 9 tickers · 100 days | Indexes on ticker + date |
| MongoDB | `news_articles` — headline, summary, ticker, source, url_hash | ~300 documents | Unique index on url_hash |
| MongoDB | `sec_filings` — ticker, form_type, filing_date, risk_text | 20 documents (10-K + 10-Q per ticker) | Unique index on accession_no |
| ChromaDB | `news_chunks` + `filing_chunks` | ~300 vectors, 384-dim, all-MiniLM-L6-v2 | Cosine similarity |

---

## Agent Architecture

The agent is built as a custom tool-calling loop on `langchain_core` — portable across LangChain versions.

```
User question
     │
     ▼
FinancialAgent.invoke()
     │
     ├── query_price_data(sql)          → PostgreSQL → OHLCV + RSI + SMA
     ├── search_financial_news(query)   → ChromaDB RAG → news + SEC chunks
     └── get_filing_metadata(ticker)    → MongoDB → filing dates + metadata
     │
     ▼
Groq Llama 3.3 70B synthesises grounded answer with citations
     │
     ▼
FastAPI POST /query → JSON response with answer + tools_used + sources
```

**Key design decisions:**
- `parallel_tool_calls=False` — prevents malformed JSON from the LLM on multi-arg tools
- `temperature=0` — deterministic answers appropriate for financial queries
- Tool docstrings are the agent's only instructions for when/how to use each tool
- All tool outputs are JSON-serialisable for the lineage log

---

## Data Lineage

Every transformation is logged to `data/lineage/lineage_log.jsonl` in a W3C PROV-DM aligned schema:

```json
{
  "id": "uuid",
  "timestamp": "2026-03-25T18:55:17",
  "event": "transform",
  "source": "postgresql:prices (raw)",
  "destination": "postgresql:prices (enriched)",
  "rows_affected": 1000,
  "agent": "02_pipeline.ipynb",
  "notes": "Added price_change_pct, is_anomaly. 0 anomalies flagged."
}
```

Accessible via `GET http://localhost:8000/lineage`.

---

## Quickstart

### Prerequisites
- Python 3.11+
- Docker Desktop
- PowerShell (Windows) or bash (Mac/Linux)

### 1. Clone and configure

```bash
git clone https://github.com/vedantbhatiaa/financial-signal-agent.git
cd financial-signal-agent
cp .env.example .env
# Fill in GROQ_API_KEY and ALPHA_VANTAGE_KEY in .env
```

### 2. Start infrastructure

```bash
docker compose up -d postgres mongo
```

### 3. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install langchain-groq langchain-huggingface sentence-transformers
nbstripout --install
```

### 4. Run the pipeline (notebooks in order)

```
jupyter notebook
→ notebooks/01_data_ingestion.ipynb   (run all cells)
→ notebooks/02_pipeline.ipynb         (run all cells)
→ notebooks/03_agent_demo.ipynb       (run all cells)
```

### 5. Start the API and frontend

```bash
uvicorn api.app:app --reload --port 8000
```

Open `http://localhost:8000` — the SignalCore frontend loads automatically.  
API docs: `http://localhost:8000/docs`  
Lineage log: `http://localhost:8000/lineage`

---

## Example Queries

```bash
# Structured price query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the current RSI for NVDA and is it overbought?"}'

# Sentiment + price divergence (multi-tool)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Is there a divergence between NVIDIA news sentiment and its price momentum?"}'

# SEC risk analysis
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What risks did Apple disclose in its most recent 10-K filing?"}'
```

---

## Environment Variables

```bash
# Required
GROQ_API_KEY=gsk_...          # Free at console.groq.com
ALPHA_VANTAGE_KEY=...         # Free at alphavantage.co

# Infrastructure (match docker-compose.yml defaults)
POSTGRES_USER=user
POSTGRES_PASSWORD=pass
POSTGRES_DB=findb
POSTGRES_URI=postgresql://user:pass@127.0.0.1:5433/findb
MONGO_URI=mongodb://localhost:27017

# Optional
NEWS_API_KEY=...               # newsapi.org free tier
```

---

## Known Limitations

- **SEC risk text**: Filing metadata is fully ingested but the full Item 1A risk factor text extraction uses a placeholder. In production, BeautifulSoup would parse the SEC HTML filing to extract the complete risk section.
- **Price history**: Alpha Vantage free tier returns 100 trading days (compact output). A paid tier would provide full historical data.
- **News coverage**: RSS feeds capture articles available at ingestion time. A production system would run continuous scheduled ingestion to maintain balanced ticker coverage.
- **Windows PostgreSQL conflict**: If Anaconda is installed, its bundled PostgreSQL occupies port 5432. Docker is configured on port 5433 to avoid this conflict.

---

## Module Learning Outcomes Addressed

| Learning Outcome | Implementation |
|---|---|
| Evaluate data processing strategies | Three-notebook pipeline with documented design decisions |
| Understand scaling options | PostgreSQL indexes, ChromaDB vector store, stateless agent tools |
| Explain effective data pipelines | Transform → Embed → Lineage stages with W3C Prov logging |
| Evaluate platform choices | Justified choice of PostgreSQL vs MongoDB vs ChromaDB per data type |
| Evaluate technology trends | RAG, LLM agents, MCP-style tool interfaces, Docker containerisation |