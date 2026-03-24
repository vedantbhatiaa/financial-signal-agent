"""
agent/tools.py
--------------
Defines the three tools available to the LangChain agent.

Each tool is a Python function decorated with @tool. The docstring is
critical — LangChain passes it directly to the LLM so the agent knows
when and how to use each tool. Keep docstrings precise and unambiguous.

Tools:
  1. query_price_data      — SQL queries against PostgreSQL (structured)
  2. search_financial_news — semantic RAG search over ChromaDB (unstructured)
  3. get_filing_metadata   — metadata lookup from MongoDB (semi-structured)

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

# MongoDB — raw news documents and SEC filing metadata
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
    model_name      = "all-MiniLM-L6-v2",
    model_kwargs    = {"device": "cpu"},   # change to "cuda" if you have a GPU
    encode_kwargs   = {"normalize_embeddings": True}  # cosine similarity works better normalised
)


# ---------------------------------------------------------------------------
# Tool 1 — Structured price data via SQL
# ---------------------------------------------------------------------------

@tool
def query_price_data(sql: str) -> str:
    """Query PostgreSQL for historical stock price data, volume, and technical indicators.

    The database contains daily OHLCV data for AAPL, NVDA, MSFT, GOOGL, AMZN,
    TSLA, META, JPM, GS, BAC over the last 2 years.

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
def search_financial_news(query: str, ticker: str = "") -> str:
    """Semantic search over financial news articles and SEC risk factor filings.

    Use this tool when the question requires qualitative context: sentiment,
    analyst opinion, risk disclosures, market commentary, or recent events.
    Results come from two ChromaDB collections:
      - news_chunks    : news headlines and summaries (Reuters, FT, NewsAPI)
      - filing_chunks  : SEC 10-K and 10-Q risk factor sections

    Args:
        query  : natural language search query (what you are looking for)
        ticker : optional ticker symbol to narrow results (e.g. 'NVDA')

    Returns top 5 most relevant text chunks with source and date metadata.
    """
    try:
        # Embed the query using the same model used at ingestion time —
        # this is what makes the similarity search meaningful
        query_vector = _embeddings.embed_query(query)

        # Optional metadata filter — ChromaDB supports filtering by any
        # metadata field stored alongside the vectors
        where_filter = {"ticker": ticker} if ticker else None

        all_chunks = []

        for collection, source_label in [(_news_col, "news"), (_filings_col, "sec_filing")]:
            kwargs = {
                "query_embeddings" : [query_vector],
                "n_results"        : 3,
                "include"          : ["documents", "metadatas", "distances"]
            }
            if where_filter:
                kwargs["where"] = where_filter

            res = collection.query(**kwargs)

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
def get_filing_metadata(ticker: str, form_type: str = "10-K") -> str:
    """Retrieve structured metadata for a company's most recent SEC filing from MongoDB.

    Use this tool when you need precise filing details: filing date, accession
    number, company name, or to confirm what period a filing covers.
    This complements search_financial_news, which retrieves the filing text.

    Args:
        ticker    : stock ticker symbol (e.g. 'AAPL', 'JPM')
        form_type : SEC form type — '10-K' (annual) or '10-Q' (quarterly)

    Returns filing metadata as a JSON string, or an error message if not found.
    """
    try:
        doc = _db["sec_filings"].find_one(
            {"ticker": ticker.upper(), "form_type": form_type},
            sort       = [("filing_date", -1)],        # most recent filing first
            projection = {"_id": 0, "risk_text": 0}    # exclude raw text — too large for context
        )
        if not doc:
            return f"No {form_type} filing found for {ticker}"

        # default=str handles ObjectId, datetime, and Decimal serialisation
        return json.dumps(doc, default=str, indent=2)

    except Exception as e:
        return f"MongoDB error: {str(e)}"


# Expose all three tools as a list — imported by agent.py
TOOLS = [query_price_data, search_financial_news, get_filing_metadata]
