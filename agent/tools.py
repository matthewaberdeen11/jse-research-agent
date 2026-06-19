import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("STACKS_API_KEY")
BASE_URL = "https://stacksja.com/api/v1/public"
HEADERS = {"X-API-Key": API_KEY}

def get_stock_quote(symbol):
    