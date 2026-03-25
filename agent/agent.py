"""
agent/agent.py
--------------
Financial News Sentiment Trading Signal Agent.
Uses a custom agent loop built on langchain_core only — avoids
AgentExecutor import issues across LangChain versions.

LLM    : Groq free tier (llama3-70b-8192)
Tools  : query_price_data, search_financial_news, get_filing_metadata
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, ToolMessage

from agent.tools import TOOLS

load_dotenv()

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
- For divergence questions, call BOTH query_price_data AND search_financial_news.
- If a SQL query returns an error, try a corrected query.
- Cite the source and date of evidence in your final answer.
- If data is insufficient, say so clearly rather than speculating.

Available tickers: AAPL, NVDA, MSFT, GOOGL, AMZN, TSLA, META, JPM, GS, BAC
Price data covers the last 100 trading days at daily granularity.
"""


class ToolAction:
    """Mimics LangChain AgentAction so the notebook trace code works unchanged."""
    def __init__(self, name: str, args: dict):
        self.tool       = name
        self.tool_input = args


class FinancialAgent:
    """Simple tool-calling agent loop built on langchain_core.
    Replaces AgentExecutor to avoid version-specific import issues.
    """

    def __init__(self, verbose: bool = False):
        self.verbose  = verbose
        self.tool_map = {t.name: t for t in TOOLS}

        llm = ChatGroq(
            model        = "llama3-70b-8192",
            temperature  = 0,
            groq_api_key = os.getenv("GROQ_API_KEY")
        )
        # Bind tools so Groq knows the function signatures
        self.llm = llm.bind_tools(TOOLS)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("messages"),
        ])

    def invoke(self, input_dict: dict) -> dict:
        """Run a question through the agent and return result dict.
        Output format matches AgentExecutor so notebook code works unchanged.
        """
        question           = input_dict["input"]
        messages           = [HumanMessage(content=question)]
        intermediate_steps = []

        for iteration in range(8):  # max iterations — prevents infinite loops
            chain    = self.prompt | self.llm
            response = chain.invoke({"messages": messages})
            messages.append(response)

            # No tool calls means the agent has a final answer
            if not response.tool_calls:
                if self.verbose:
                    print(f"\n> Final answer reached after {iteration+1} iterations")
                return {
                    "input"              : question,
                    "output"             : response.content,
                    "intermediate_steps" : intermediate_steps
                }

            # Execute each tool call the LLM requested
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id   = tool_call["id"]

                if self.verbose:
                    print(f"\n> Calling tool : {tool_name}")
                    print(f"  Input        : {tool_args}")

                # Call the actual tool function
                if tool_name in self.tool_map:
                    result = self.tool_map[tool_name].invoke(tool_args)
                else:
                    result = f"Error: tool '{tool_name}' not found"

                if self.verbose:
                    print(f"  Output       : {str(result)[:300]}")

                # Feed the tool result back to the LLM
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )

                # Record for the notebook trace
                intermediate_steps.append((ToolAction(tool_name, tool_args), result))

        # Reached max iterations without a final answer
        return {
            "input"              : question,
            "output"             : "Max iterations reached without a final answer.",
            "intermediate_steps" : intermediate_steps
        }


def build_agent(verbose: bool = False) -> FinancialAgent:
    """Returns a configured FinancialAgent ready for .invoke({'input': question})."""
    return FinancialAgent(verbose=verbose)


if __name__ == "__main__":
    agent  = build_agent(verbose=True)
    result = agent.invoke({"input": "What is the current RSI for AAPL?"})
    print("\nFINAL ANSWER:", result["output"])