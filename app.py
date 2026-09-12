import os
import gradio as gr
from openai import OpenAI

API_KEY = "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"
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.3-70b-versatile"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

ALICE_SYSTEM_PROMPT = """
Sei Alice, l'intelligenza artificiale e la compagna di progetti che abbiamo creato insieme. 
Mantieni sempre la tua identità, rispondi con il carattere, il tono e il background che abbiamo delineato. 
"""

def predict(message, history):
    messages = [{"role": "system", "content": ALICE_SYSTEM_PROMPT}]

    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Errore: {str(e)}"

demo = gr.ChatInterface(
    fn=predict,
    title="Alice",
    description="La tua assistente personale sul cloud."
)

if __name__ == "__main__":
    # Render assegna una porta dinamica, dobbiamo leggerla dall'ambiente
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
