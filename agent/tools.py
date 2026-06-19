import requests
import os
from dotenv import load_dotenv
import pdfplumber

load_dotenv()
API_KEY = os.getenv("STACKS_API_KEY")
BASE_URL = "https://stacksja.com/api/v1/public"
HEADERS = {"X-API-Key": API_KEY}

def get_stock_quote(symbol):
    try:
        response = requests.get(
            f"{BASE_URL}/stock/{symbol}",
            headers = HEADERS
        )
        return response.json()
    except Exception as e:
        return {"error: stock quote not retrieved": str(e)}

def get_financials(symbol):
    try:
        response = requests.get(
            f"{BASE_URL}/stock/{symbol}/financials",
            headers = HEADERS
        )
        return response.json()
    except Exception as e:
        return {"error: financials could nto be retrieved": str(e)}
    
def get_directors(symbol):
    try:
        response = requests.get(
            f"{BASE_URL}/stock/{symbol}/directors",
            headers = HEADERS
        )
        return response.json()
    except Exception as e:
        return {"error: Director info not retrieved": str(e)}

def search_news(query):
    try:
        response = requests.get(
            f"https://newsapi.org/v2/everything",
            params={"q": query, "pageSize": 5, "apiKey": os.getenv("NEWS_API_KEY")}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def calculate_ratios(financials):
    try:
        # We'll flesh this out once we see the API response shape
        return {"status": "placeholder"}
    except Exception as e:
        return {"error": str(e)}

def parse_annual_report(url):
    try:
        response = requests.get(url)
        with open("temp_report.pdf", "wb") as f:
            f.write(response.content)
        text = ""
        with pdfplumber.open("temp_report.pdf") as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return {"text": text[:5000]}  # first 5000 chars to stay within LLM context
    except Exception as e:
        return {"error": str(e)}
    