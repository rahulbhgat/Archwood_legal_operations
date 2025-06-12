import os
import time
import requests
from dotenv import load_dotenv
from requests.exceptions import HTTPError

load_dotenv()  # Load API key from .env file

def query_llama3(prompt: str, retries=3, backoff=2):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    for attempt in range(retries):
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 429:
            print(f"Rate limit hit. Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2
        else:
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    raise HTTPError("Failed after retries due to rate limits.")
