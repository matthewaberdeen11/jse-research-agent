import os
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

def call_chutes(messages):
    response = requests.post(
        CHUTES_URL,
        headers={
            "Authorization": f"Bearer {CHUTES_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-ai/DeepSeek-V3-0324",
            "messages": messages,
            "tools": TOOLS,
            "max_tokens": 4096,
            "temperature": 0.3,
            "stream": False
        }
    )
    return response.json()

def research(symbol):
    messages = [
        {
            "role": "system",
            "content": """You are a senior equity research analyst covering the Jamaica Stock Exchange.
            
Given a stock ticker, use the available tools to gather comprehensive data about the company.
Always start by getting the stock quote, then get financials and directors.
Search for recent news about the company.
Once you have all the data, write a structured research brief with these sections:

1. Company Overview
2. Financial Highlights  
3. Key Risks
4. Board & Governance
5. Recent News
6. Valuation Summary

Be specific with numbers and data. Cite your sources."""
        },
        {
            "role": "user",
            "content": f"Research {symbol} on the Jamaica Stock Exchange and produce a full analyst brief."
        }
    ]

    # Agent loop - keep going until the LLM stops calling tools
    max_steps = 10
    for step in range(max_steps):
        result = call_chutes(messages)
        
        choice = result["choices"][0]
        message = choice["message"]
        
        # If the model wants to call tools
        if message.get("tool_calls"):
            messages.append(message)
            
            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                
                print(f"Step {step + 1}: Calling {func_name}({args})")
                
                # Execute the tool
                tool_result = TOOL_MAP[func_name](**args)
                
                # Send result back to the LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result)
                })
        else:
            # No more tool calls - the agent is done, return the final report
            print(f"Research complete after {step + 1} steps")
            return message["content"]
    
    return "Agent reached maximum steps without completing research."