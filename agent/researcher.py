import os
import re
import json
import requests
from dotenv import load_dotenv
from agent.tools import (
    get_stock_quote,
    get_financials,
    get_directors,
    search_news,
    calculate_ratios,
    parse_annual_report
)

load_dotenv()
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
CHUTES_URL = "https://llm.chutes.ai/v1/chat/completions"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Get live stock price and company info for a JSE-listed company",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "JSE stock ticker symbol, e.g. NCBFG"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financials",
            "description": "Get financial statements and data for a JSE company",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "JSE stock ticker symbol"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_directors",
            "description": "Get board of directors for a JSE company",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "JSE stock ticker symbol"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search for recent news articles about a company or topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. company name"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ratios",
            "description": "Calculate financial ratios from financial data",
            "parameters": {
                "type": "object",
                "properties": {
                    "financials": {"type": "object", "description": "Financial data to analyze"}
                },
                "required": ["financials"]
            }
        }
    }
]


# Map function names to actual functions
TOOL_MAP = {
    "get_stock_quote": get_stock_quote,
    "get_financials": get_financials,
    "get_directors": get_directors,
    "search_news": search_news,
    "calculate_ratios": calculate_ratios,
}

def call_chutes(messages, max_tokens=2048):
    response = requests.post(
        CHUTES_URL,
        headers={
            "Authorization": f"Bearer {CHUTES_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "zai-org/GLM-5.1-TEE",
            "messages": messages,
            "tools": TOOLS,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": False
        },
        timeout=120
    )
    data = response.json()
    if "choices" not in data:
        # Surface API errors (bad key, unknown model, rate limit) clearly
        detail = data.get("detail") or data.get("error") or data
        raise RuntimeError(f"LLM API error (HTTP {response.status_code}): {detail}")
    return data

SYSTEM_PROMPT = """You are a senior equity research analyst covering the Jamaica Stock Exchange.

WORKFLOW — be efficient, you are timed:
- In your FIRST turn, issue all the tool calls you need at once: get_stock_quote, get_financials, get_directors, and search_news. They run in parallel.
- Optionally call calculate_ratios once after you have financials.
- Call each tool AT MOST ONCE. Never repeat a tool call you have already made — the data will not change. As soon as you have the data, write the brief.

Write a tight, skimmable research brief (aim for 600-900 words) with these sections:

1. Company Overview
2. Financial Highlights
3. Key Risks
4. Board & Governance
5. Recent News
6. Valuation Summary

Be specific with real numbers from the tool data. If a tool returned no data, say so briefly rather than guessing. Use Markdown headings."""


def clean_report(text):
    """Strip any tool-call markup the model occasionally leaks into its prose
    (GLM intermittently writes calculate_ratios as text instead of a real
    tool call)."""
    if not text:
        return text
    # Remove well-formed <tool_call>...</tool_call> blocks.
    text = re.sub(r"<\|?tool_calls?[^>]*\|?>.*?</?\|?tool_calls?[^>]*\|?>", "",
                  text, flags=re.DOTALL)
    if "<tool_call" in text or "<arg_value>" in text:
        # Unclosed leak: keep the report from its first Markdown heading on,
        # then drop anything from a remaining stray marker to the end.
        heading = re.search(r"^#", text, flags=re.MULTILINE)
        if heading:
            text = text[heading.start():]
        text = re.split(r"<\|?tool_call", text)[0]
    return text.strip()


def research(symbol):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Research {symbol} on the Jamaica Stock Exchange and produce a full analyst brief."
        }
    ]

    # Cache tool results within this run so repeat calls are free and never
    # hit the external APIs twice.
    tool_cache = {}

    # Agent loop - keep going until the LLM stops calling tools.
    max_steps = 6
    for step in range(max_steps):
        result = call_chutes(messages)
        message = result["choices"][0]["message"]

        if not message.get("tool_calls"):
            # No more tool calls - the agent is done, return the final report.
            print(f"Research complete after {step + 1} steps")
            return clean_report(message["content"])

        messages.append(message)
        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            cache_key = func_name + ":" + json.dumps(args, sort_keys=True)
            if cache_key in tool_cache:
                print(f"Step {step + 1}: Cached {func_name}({args})")
                tool_result = tool_cache[cache_key]
            else:
                print(f"Step {step + 1}: Calling {func_name}({args})")
                try:
                    tool_result = TOOL_MAP[func_name](**args)
                except Exception as exc:
                    tool_result = {"error": f"{func_name} failed: {exc}"}
                tool_cache[cache_key] = tool_result

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(tool_result)
            })

    # Out of steps: ask once for a final write-up using whatever data we have.
    print("Max steps reached - requesting final brief")
    messages.append({
        "role": "user",
        "content": "Stop calling tools now and write the complete research brief using the data you already have."
    })
    final = call_chutes(messages)["choices"][0]["message"]
    return clean_report(final.get("content")) or "Agent could not complete the research brief."