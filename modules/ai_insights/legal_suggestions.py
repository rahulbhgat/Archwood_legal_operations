import ollama

def suggest_legal_actions(text: str):
    prompt = f"Suggest possible legal strategies or actions that can be taken in this case:\n\n{text}"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
