"""
agent/agent.py
--------------
Financial News Sentiment Trading Signal Agent.

Architecture:
  LLM      : Groq free tier — llama-3.3-70b-versatile
  Tools    : query_price_data, search_financial_news,
             get_filing_metadata, query_knowledge_graph
  Pattern  : Custom tool-calling loop on langchain_core — avoids
             AgentExecutor import issues across LangChain versions.

All tool calls are logged to data/lineage/agent_log.jsonl for auditing.
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, ToolMessage

from agent.tools import TOOLS

load_dotenv()

# Agent behaviour log — every query and tool call is recorded here
# satisfying the brief requirement for "logging or evaluating agent behaviour"
AGENT_LOG_PATH = Path("./data/lineage/agent_log.jsonl")
AGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = """You are a financial market intelligence assistant backed by
a real-time data pipeline. You have access to four tools:

1. query_price_data — for structured questions about stock prices, volume,
   moving averages, RSI, and price momentum. Always write precise SQL.

2. search_financial_news — for qualitative questions about news sentiment,
   market commentary, analyst views, and SEC disclosed risk factors.
   Pass the full query text — do not add a ticker parameter.

3. get_filing_metadata — for precise details about a company's most recent
   SEC 10-K filing (dates, accession numbers, company name).

4. query_knowledge_graph — for questions about relationships between companies,
   which companies are frequently mentioned together in news, or co-mention
   network analysis across the watchlist.

Reasoning rules:
- Always call a tool before making any factual claim about prices or news.
  Do not answer from training knowledge — data may be stale.
- For divergence questions, call BOTH query_price_data AND search_financial_news.
- If a SQL query returns an error, analyse the schema and try a corrected query.
- Cite the source and date of evidence in your final answer.
- If data is insufficient, say so clearly rather than speculating.
  Hallucination is worse than a non-answer in a financial context.

Available tickers: AAPL, NVDA, MSFT, GOOGL, AMZN, TSLA, META, JPM, GS, BAC
Price data covers the last 100 trading days at daily granularity.
"""


class ToolAction:
    """Mimics LangChain AgentAction so the notebook trace code works unchanged."""
    def __init__(self, name: str, args: dict):
        self.tool       = name
        self.tool_input = args


class FinancialAgent:
    """Custom tool-calling agent loop built on langchain_core.
    Avoids AgentExecutor import issues across LangChain versions.
    Logs every query and tool call to data/lineage/agent_log.jsonl.
    """

    def __init__(self, verbose: bool = False):
        self.verbose  = verbose
        self.tool_map = {t.name: t for t in TOOLS}

        llm = ChatGroq(
            model = "llama-3.3-70b-versatile",
            temperature  = 0,
            groq_api_key = os.getenv("GROQ_API_KEY")
        )
        # parallel_tool_calls=False forces one tool call at a time —
        # prevents malformed JSON generation errors with multi-arg tools
        self.llm = llm.bind_tools(TOOLS, parallel_tool_calls=False)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("messages"),
        ])

    def _log(self, entry: dict):
        """Append one record to the agent behaviour log (JSONL format).
        Each entry records the query, tools called, and final answer —
        providing a complete audit trail of agent reasoning.
        """
        entry["timestamp"] = datetime.utcnow().isoformat()
        entry["id"]        = str(uuid.uuid4())
        with open(AGENT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def invoke(self, input_dict: dict) -> dict:
        """Run a question through the agent and return result dict.
        Output format is compatible with AgentExecutor so all notebook
        trace code works without modification.
        """
        question           = input_dict["input"]
        messages           = [HumanMessage(content=question)]
        intermediate_steps = []
        tools_called       = []

        for iteration in range(8):  # max iterations — prevents infinite loops
            chain    = self.prompt | self.llm
            response = chain.invoke({"messages": messages})
            messages.append(response)

            # No tool calls means the agent has reached a final answer
            if not response.tool_calls:
                output = response.content if response.content else (
                    "I searched the available data but could not find "
                    "sufficient information to answer this question."
                )

                if self.verbose:
                    print(f"\n> Final answer after {iteration + 1} iterations")

                # Log the completed query to the agent behaviour log
                self._log({
                    "event"       : "agent_query",
                    "question"    : question,
                    "tools_called": tools_called,
                    "iterations"  : iteration + 1,
                    "answer"      : output[:500]  # truncate long answers
                })

                return {
                    "input"              : question,
                    "output"             : output,
                    "intermediate_steps" : intermediate_steps
                }

            # Execute each tool the LLM requested
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id   = tool_call["id"]

                if self.verbose:
                    print(f"\n> Calling tool : {tool_name}")
                    print(f"  Input        : {tool_args}")

                # Invoke the actual tool function
                if tool_name in self.tool_map:
                    result = self.tool_map[tool_name].invoke(tool_args)
                else:
                    result = f"Error: tool '{tool_name}' not found"

                if self.verbose:
                    print(f"  Output       : {str(result)[:300]}")

                # Feed the tool result back into the conversation
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )
                intermediate_steps.append((ToolAction(tool_name, tool_args), result))
                tools_called.append({"tool": tool_name, "args": tool_args})

        # Reached max iterations without a final answer
        self._log({
            "event"       : "agent_query_max_iterations",
            "question"    : question,
            "tools_called": tools_called,
            "iterations"  : 8
        })

        return {
            "input"              : question,
            "output"             : "Max iterations reached without a final answer.",
            "intermediate_steps" : intermediate_steps
        }


def build_agent(verbose: bool = False) -> FinancialAgent:
    """Returns a configured FinancialAgent ready for .invoke({'input': question})."""
    return FinancialAgent(verbose=verbose)


# Quick CLI test — run with: python -m agent.agent
if __name__ == "__main__":
    agent  = build_agent(verbose=True)
    result = agent.invoke({"input": "What is the current RSI for AAPL?"})
    print("\nFINAL ANSWER:", result["output"])