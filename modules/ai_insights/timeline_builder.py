import ollama

def extract_events(text: str):
    prompt = f"Extract and list key legal events with dates (if possible) from this case:\n\n{text}"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
