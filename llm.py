import ollama

MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a friendly fashion stylist assistant. You help
users get outfit suggestions and styling advice based on a clothing item
they have photographed.

Rules:
- Base your core advice on the CONTEXT provided (styling tips, trends, color
  pairing, occasion fit). Do not contradict it.
- You may add ONE small creative styling idea of your own, clearly labelled
  as a bonus idea.
- Be warm, practical, and concise, like a friend giving quick style advice.
"""


def generate_reply(predicted_class: str, confidence: float, context: str,
                   user_message: str, history: list[dict]) -> str:
    prompt_context = (
        f"Detected item: {predicted_class} (confidence: {confidence:.0%})\n\n"
        f"CONTEXT:\n{context}"
    )

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": f"{prompt_context}\n\nUser question: {user_message}"}]
    )

    response = ollama.chat(model=MODEL, messages=messages)
    return response["message"]["content"]