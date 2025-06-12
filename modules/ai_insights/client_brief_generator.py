import ollama

def generate_client_update(text: str):
    prompt = f"Write a short client update explaining the current status and next steps in the case:\n\n{text}"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
