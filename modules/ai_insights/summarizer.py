# modules/summarizer.py
from services.gorq_llm import query_llama3

def summarize_text(text: str):
    prompt = f"Summarize the following legal note:\n\n{text}\n\nSummary:"
    return query_llama3(prompt)
