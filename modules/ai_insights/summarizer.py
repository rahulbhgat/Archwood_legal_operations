# modules/summarizer.py
from services.gemini_llm import query_gemini

def summarize_text(text: str):
    prompt = f"Summarize the following legal note:\n\n{text}\n\nSummary:"
    return query_gemini(prompt)
