"""
agent/mcp_server.py
-------------------
Exposes the Financial Signal Agent's four data tools as a real MCP
(Model Context Protocol) server using the FastMCP library.

This allows any MCP-compatible client (Claude Desktop, other agents,
or developer tools) to discover and invoke the pipeline's capabilities
through the standard MCP protocol — demonstrating agent-to-service
and agent-to-agent (A2A) interaction patterns.

Architecture:
  MCP tools wrap the same LangChain tools used by the internal agent,
  so the data pipeline is exposed through two complementary interfaces:
    1. LangChain tools  — used by the internal FinancialAgent
    2. MCP server tools — used by external MCP-compatible clients

Usage:
    pip install mcp
    python agent/mcp_server.py

The server runs on stdio by default (standard MCP transport).
"""

from mcp.server.fastmcp import FastMCP
from agent.tools import (
    query_price_data,
    search_financial_news,
    get_filing_metadata,
    query_knowledge_graph
)

# Initialise the MCP server — name is how external clients discover it
mcp = FastMCP("financial-signal-agent")


# ---------------------------------------------------------------------------
# MCP Tool 1 — Stock price data via PostgreSQL
# ---------------------------------------------------------------------------

@mcp.tool()
def price_data(sql: str) -> str:
    """Query historical stock price data, volume, RSI, and SMA from PostgreSQL.

    Input must be a valid SQL SELECT statement against the prices table.
    Schema: prices(ticker, date, open, high, low, close, volume, sma_20,
                   rsi_14, price_change_pct, is_anomaly)
    Available tickers: AAPL, NVDA, MSFT, GOOGL, AMZN, TSLA, META, JPM, GS, BAC
    """
    return query_price_data.invoke(sql)


# ---------------------------------------------------------------------------
# MCP Tool 2 — Financial news and SEC filings via RAG
# ---------------------------------------------------------------------------

@mcp.tool()
def financial_news(query: str) -> str:
    """Semantic search over financial news articles and SEC risk factor filings.

    Searches ChromaDB using sentence embeddings — retrieval is by meaning,
    not keywords. Returns the top 5 most relevant chunks with source citations.

    Include the company name or ticker in the query for best results.
    """
    return search_financial_news.invoke(query)


# ---------------------------------------------------------------------------
# MCP Tool 3 — SEC filing metadata from MongoDB
# ---------------------------------------------------------------------------

@mcp.tool()
def filing_metadata(ticker: str) -> str:
    """Retrieve the most recent 10-K filing metadata for a company from MongoDB.

    Returns filing date, accession number, company name, and filing URL.
    Use this to confirm what period a filing covers before querying its content.

    Args:
        ticker: stock ticker symbol e.g. 'AAPL', 'JPM'
    """
    return get_filing_metadata.invoke(ticker)


# ---------------------------------------------------------------------------
# MCP Tool 4 — Company relationship knowledge graph from MongoDB
# ---------------------------------------------------------------------------

@mcp.tool()
def knowledge_graph(question: str) -> str:
    """Query the company co-mention knowledge graph built from news articles.

    Returns the strongest co-mention relationships — pairs of companies that
    frequently appear together in news headlines — with edge weights indicating
    relationship strength. Useful for network-level market analysis.
    """
    return query_knowledge_graph.invoke(question)


# ---------------------------------------------------------------------------
# Entry point — run as a standalone MCP server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Runs on stdio transport by default — standard for MCP servers
    # Connect via any MCP-compatible client pointing to this process
    mcp.run()