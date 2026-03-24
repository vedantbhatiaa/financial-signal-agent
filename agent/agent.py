"""
agent/agent.py
--------------
Builds and returns the LangChain AgentExecutor that powers the
Financial News Sentiment Trading Signal Agent.

Architecture:
  - LLM        : Groq (free tier) running Llama 3 70B
  - Framework  : LangChain tool-calling agent
  - Tools      : query_price_data, search_financial_news, get_filing_metadata
  - Memory     : ConversationBufferWindowMemory (last 6 turns)

Groq is used instead of OpenAI — it is free, requires no credit card,
and Llama 3 70B is capable enough for SQL generation and RAG reasoning.
Sign up at console.groq.com to get a free API key (starts with gsk_).
"""

import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.tools import TOOLS

load_dotenv()


# ---------------------------------------------------------------------------
# System prompt — tells the LLM what role it plays, what tools exist,
# and how it should reason before calling any tool.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial market intelligence assistant backed by
a real-time data pipeline. You have access to three tools:

1. query_price_data — for structured questions about stock prices, volume,
   moving averages, RSI, and price momentum. Always write precise SQL.

2. search_financial_news — for qualitative questions about news sentiment,
   market commentary, analyst views, and SEC disclosed risk factors.

3. get_filing_metadata — for precise details about a company's most recent
   SEC 10-K or 10-Q filing (dates, accession numbers).

Reasoning rules:
- Always call a tool before making any factual claim about prices or news.
  Do not answer from training knowledge — data may be stale.
- For divergence questions, call BOTH query_price_data AND search_financial_news
  before synthesising an answer.
- If a SQL query returns an error, analyse the schema and try a corrected query.
- Cite the source and date of evidence in your final answer.
- If the data is insufficient to answer confidently, say so clearly rather than
  speculating. Hallucination is worse than a non-answer in a trading context.

Available tickers: AAPL, NVDA, MSFT, GOOGL, AMZN, TSLA, META, JPM, GS, BAC
Price data covers the last 2 years at daily granularity.
"""


def build_agent(verbose: bool = False) -> AgentExecutor:
    """Construct and return the AgentExecutor.

    Args:
        verbose: if True, LangChain prints each tool call and observation
                 to stdout. Useful for debugging and the demo notebook.

    Returns:
        A configured AgentExecutor ready to accept
        .invoke({'input': question}) calls.
    """

    # LLM — Groq's free tier with Llama 3 70B.
    # llama3-70b-8192 means 70 billion parameters, 8192 token context window.
    # temperature=0 gives deterministic answers — important for finance queries
    # where we want consistency, not creativity.
    llm = ChatGroq(
        model       = "llama3-70b-8192",
        temperature = 0,
        groq_api_key = os.getenv("GROQ_API_KEY=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")
    )

    # Prompt template — MessagesPlaceholder slots are required by LangChain:
    # 'chat_history' holds the memory window (last 6 turns),
    # 'agent_scratchpad' holds the tool call/observation pairs mid-reasoning.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    # Memory — sliding window of last 6 turns so the agent can handle
    # follow-up questions like "now compare that to MSFT"
    memory = ConversationBufferWindowMemory(
        k               = 6,
        memory_key      = "chat_history",
        return_messages = True
    )

    # create_tool_calling_agent works with any LLM that supports tool-calling,
    # including Groq's Llama 3 — more portable than create_openai_tools_agent
    agent = create_tool_calling_agent(
        llm    = llm,
        tools  = TOOLS,
        prompt = prompt
    )

    # AgentExecutor runs the loop: call agent → call tool → feed observation
    # back → call agent again → repeat until a final answer is produced
    executor = AgentExecutor(
        agent                 = agent,
        tools                 = TOOLS,
        memory                = memory,
        verbose               = verbose,
        max_iterations        = 8,     # safety cap — prevents infinite loops
        handle_parsing_errors = True,  # recover gracefully from bad tool calls
        return_intermediate_steps = True  # exposes tool trace for auditing
    )

    return executor


# Quick CLI test — run with: python -m agent.agent
if __name__ == "__main__":
    executor = build_agent(verbose=True)
    test_q   = "What is the current RSI for AAPL and is it overbought?"
    result   = executor.invoke({"input": test_q})
    print("\nFINAL ANSWER:", result["output"])