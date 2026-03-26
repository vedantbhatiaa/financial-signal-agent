"""
agent/tools.py
--------------
Defines the four tools available to the LangChain agent.

Each tool is a Python function decorated with @tool. The docstring is
critical — LangChain passes it directly to the LLM so the agent knows
when and how to use each tool. Keep docstrings precise and unambiguous.

Tools:
  1. query_price_data       — SQL queries against PostgreSQL (structured)
  2. search_financial_news  — semantic RAG search over ChromaDB (unstructured)
  3. get_filing_metadata    — metadata lookup from MongoDB (semi-structured)
  4. query_knowledge_graph  — graph query from MongoDB (graph/network data)

Embeddings use HuggingFace's all-MiniLM-L6-v2 (runs locally, completely free)
instead of OpenAI embeddings. The model is ~90MB and downloads automatically
on first run via the sentence-transformers library.
"""

import os
import json
import chromadb
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from pymongo import MongoClient
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Shared connections — initialised once at module load time.
# This means the first import of this module pays the connection cost;
# subsequent tool calls reuse the same pool with no overhead.
# ---------------------------------------------------------------------------

# PostgreSQL — structured price + indicator data
_engine = create_engine(
    os.getenv("POSTGRES_URI", "postgresql://user:pass@localhost:5432/findb")
)

# MongoDB — raw news documents, SEC filing metadata, and knowledge graph
_mongo_client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017")
)
_db = _mongo_client["findb"]

# ChromaDB — persistent local vector store (no external service needed)
# Path maps to a Docker volume in production so data survives restarts
_chroma_client = chromadb.PersistentClient(path="./data/chroma")
_news_col      = _chroma_client.get_or_create_collection("news_chunks")
_filings_col   = _chroma_client.get_or_create_collection("filing_chunks")

# HuggingFace embeddings — runs entirely on your local CPU, no API key needed.
# all-MiniLM-L6-v2 produces 384-dimension vectors, fast and good quality
# for semantic similarity tasks like news retrieval.
# NOTE: must match the model used in 02_pipeline.ipynb at ingestion time —
# mixing models produces meaningless similarity scores.
_embeddings = HuggingFaceEmbeddings(
    model_name    = "all-MiniLM-L6-v2",
    model_kwargs  = {"device": "cpu"},              # change to "cuda" if you have a GPU
    encode_kwargs = {"normalize_embeddings": True}  # cosine similarity works better normalised
)


# ---------------------------------------------------------------------------
# Tool 1 — Structured price data via SQL
# ---------------------------------------------------------------------------

@tool
def query_price_data(sql: str) -> str:
    """Query PostgreSQL for historical stock price data, volume, and technical indicators.

    The database contains daily OHLCV data for AAPL, NVDA, MSFT, GOOGL, AMZN,
    TSLA, META, JPM, GS, BAC over the last 100 trading days.

    Table schema:
        prices(ticker VARCHAR, date DATE, open NUMERIC, high NUMERIC,
               low NUMERIC, close NUMERIC, volume BIGINT,
               sma_20 NUMERIC, rsi_14 NUMERIC,
               price_change_pct NUMERIC, is_anomaly BOOLEAN)

    Input must be a valid read-only SQL SELECT statement.
    Return value is a JSON string of up to 50 rows.

    Example inputs:
        SELECT ticker, close, rsi_14 FROM prices WHERE ticker='NVDA'
          ORDER BY date DESC LIMIT 5
        SELECT ticker, AVG(close) as avg_close FROM prices
          WHERE date >= NOW() - INTERVAL '30 days' GROUP BY ticker
    """
    try:
        with _engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = [dict(row._mapping) for row in result.fetchmany(50)]

        # Convert Decimal, date, and other non-JSON-serialisable types to strings
        for row in rows:
            for k, v in row.items():
                row[k] = str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v

        return json.dumps({"rows": rows, "count": len(rows)}, indent=2)

    except Exception as e:
        # Return error as a string — the agent will read it and try a corrected query
        return f"SQL error: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 2 — Unstructured news and SEC filings via RAG
# ---------------------------------------------------------------------------

@tool
def search_financial_news(query: str) -> str:
    """Semantic search over financial news articles and SEC risk factor filings.

    Use this tool when the question requires qualitative context: sentiment,
    analyst opinion, risk disclosures, market commentary, or recent events.
    Results come from two ChromaDB collections:
      - news_chunks    : news headlines and summaries (Reuters, FT, NewsAPI,
                         and ticker-specific Yahoo Finance RSS feeds)
      - filing_chunks  : SEC 10-K and 10-Q risk factor sections

    Args:
        query : natural language search query describing what you are looking for.
                Include the company name or ticker in the query text itself.

    Returns top 5 most relevant text chunks with source and date metadata.
    """
    try:
        # Embed the query using the same model used at ingestion time —
        # this is what makes the similarity search meaningful
        query_vector = _embeddings.embed_query(query)

        # No ticker filter — search across all tickers for maximum recall.
        # Ticker detection was applied at ingestion time so relevant articles
        # will surface naturally through semantic similarity.
        all_chunks = []

        for collection, source_label in [(_news_col, "news"), (_filings_col, "sec_filing")]:
            res = collection.query(
                query_embeddings = [query_vector],
                n_results        = 3,
                include          = ["documents", "metadatas", "distances"]
            )

            for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0]
            ):
                all_chunks.append({
                    "text"         : doc[:500],          # truncate for context window
                    "source"       : source_label,
                    "ticker"       : meta.get("ticker", ""),
                    "published_at" : meta.get("published_at", ""),
                    "headline"     : meta.get("headline", ""),
                    "relevance"    : round(1 - dist, 4)  # distance → similarity score
                })

        # Merge results from both collections, return top 5 by relevance
        all_chunks.sort(key=lambda x: x["relevance"], reverse=True)
        return json.dumps({"results": all_chunks[:5]}, indent=2)

    except Exception as e:
        return f"Search error: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 3 — Structured filing metadata from MongoDB
# ---------------------------------------------------------------------------

@tool
def get_filing_metadata(ticker: str) -> str:
    """Retrieve structured metadata for a company's most recent SEC 10-K filing from MongoDB.

    Use this tool when you need precise filing details: filing date, accession
    number, company name, or to confirm what period a filing covers.
    This complements search_financial_news, which retrieves the filing text.

    Args:
        ticker : stock ticker symbol (e.g. 'AAPL', 'JPM')

    Returns the most recent 10-K filing metadata as a JSON string.
    """
    try:
        doc = _db["sec_filings"].find_one(
            {"ticker": ticker.upper(), "form_type": "10-K"},
            sort       = [("filing_date", -1)],     # most recent filing first
            projection = {"_id": 0, "risk_text": 0} # exclude raw text — too large for context
        )
        if not doc:
            return f"No 10-K filing found for {ticker}"

        # default=str handles ObjectId, datetime, and Decimal serialisation
        return json.dumps(doc, default=str, indent=2)

    except Exception as e:
        return f"MongoDB error: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 4 — Company relationship knowledge graph from MongoDB
# ---------------------------------------------------------------------------

@tool
def query_knowledge_graph(question: str) -> str:
    """Query the company co-mention knowledge graph stored in MongoDB.

    The graph was built from news articles — nodes are companies (tickers),
    edges represent co-mention relationships where two companies appeared
    in the same news headline. Edge weight = number of shared articles.

    Use this tool for questions about:
    - Which companies are most frequently mentioned together
    - Relationship strength between two specific companies
    - Network-level analysis across the watchlist

    Args:
        question : natural language question about company relationships.
                   The tool returns the top co-mention pairs regardless of
                   the specific question — use the data to answer it.

    Returns graph summary with top co-mention relationships and weights.
    """
    try:
        doc = _db["knowledge_graph"].find_one(
            {"type": "company_relationship_graph"},
            projection={"_id": 0}
        )
        if not doc:
            return "Knowledge graph not found. Run 01_data_ingestion.ipynb first."

        # Sort edges by co-mention weight descending
        edges = sorted(
            doc.get("edges", []),
            key=lambda x: x.get("weight", 0),
            reverse=True
        )

        return json.dumps({
            "graph_description" : doc.get("description", ""),
            "total_nodes"       : len(doc.get("nodes", [])),
            "total_edges"       : len(edges),
            "top_relationships" : edges[:10]  # top 10 strongest co-mention pairs
        }, indent=2)

    except Exception as e:
        return f"Graph query error: {str(e)}"


# ---------------------------------------------------------------------------
# Expose all four tools as a list — imported by agent.py and mcp_server.py
# ---------------------------------------------------------------------------
TOOLS = [query_price_data, search_financial_news, get_filing_metadata, query_knowledge_graph]