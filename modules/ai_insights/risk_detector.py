import ollama

def detect_legal_risks(text: str):
    prompt = f"Identify any potential legal risks or weaknesses in this case:\n\n{text}"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
