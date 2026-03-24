# Financial Signal Agent — MSIN0166 Individual Assignment

An LLM-powered data engineering pipeline that enables natural language querying
over financial market data from three heterogeneous sources.

---

## Architecture

```
Data Sources          Storage              Agent Layer         API
────────────          ───────              ───────────         ───
RSS + NewsAPI  ──►  MongoDB               LangChain           FastAPI
yfinance       ──►  PostgreSQL    ──►     AgentExecutor  ──►  POST /query
SEC EDGAR      ──►  MongoDB +             (3 tools)
                    ChromaDB (RAG)
```

---

## Project Structure

```
financial-signal-agent/
├── notebooks/
│   ├── 01_data_ingestion.ipynb   # Ingest all three sources
│   ├── 02_pipeline.ipynb         # Transform, embed, lineage
│   └── 03_agent_demo.ipynb       # Live agent queries
├── agent/
│   ├── tools.py                  # LangChain tool definitions
│   └── agent.py                  # AgentExecutor + system prompt
├── api/
│   └── app.py                    # FastAPI REST wrapper
├── scripts/
│   └── run_pipeline.sh           # One-shot pipeline runner
├── data/
│   ├── raw/                      # Downloaded source data
│   ├── processed/                # Cleaned data
│   ├── chroma/                   # ChromaDB vector store
│   └── lineage/                  # lineage_log.jsonl
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Quickstart

**1. Clone and configure**
```bash
git clone <your-repo-url>
cd financial-signal-agent
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

**2. Start infrastructure**
```bash
docker-compose up -d
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
nbstripout --install   # strip notebook outputs before commits
```

**4. Run the full pipeline**
```bash
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

**5. Run the agent demo**

Open `notebooks/03_agent_demo.ipynb` in Jupyter and run all cells.

**6. Start the API**
```bash
uvicorn api.app:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

---

## Example API call

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Is there a divergence between NVIDIA news sentiment and price momentum?"}'
```

---

## Data Sources

| Source | Type | Storage | Tool |
|---|---|---|---|
| RSS + NewsAPI | Unstructured text | MongoDB | `search_financial_news` |
| yfinance (Yahoo Finance) | Structured time-series | PostgreSQL | `query_price_data` |
| SEC EDGAR (10-K, 10-Q) | Semi-structured documents | MongoDB + ChromaDB | `search_financial_news`, `get_filing_metadata` |

---

## Module Learning Outcomes

This project demonstrates:
- Multi-source data ingestion (relational, NoSQL, vector, API, web)
- Data pipeline design with transformation and lineage tracking
- LLM agent architecture with tool-based reasoning
- Containerised, reproducible environments (Docker)
- REST API exposure of an agentic data system
