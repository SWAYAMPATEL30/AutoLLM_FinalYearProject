def get_fastapi_server_code() -> str:
    return """import os
import sys
import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from autollm.engine.scratch_transformer import MiniGPT, CharTokenizer
from autollm.engine.inference import get_inference_engine

app = FastAPI(title="AutoLLM Generated Model Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    prompt: str
    completion: str
    full_text: str

@app.on_event("startup")
def load_model():
    global engine
    base_dir = os.path.dirname(os.path.abspath(__file__))
    engine = get_inference_engine(base_dir, domain="general")
    print("AutoLLM Model Inference Engine mounted and running on CPU.")

@app.get("/health")
def health():
    return {"status": "ready", "device": "cpu"}

@app.post("/generate", response_model=InferenceResponse)
def generate(req: InferenceRequest):
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not loaded.")

    completion = engine.generate_response(req.prompt, max_tokens=req.max_tokens, temperature=req.temperature)
    return InferenceResponse(
        prompt=req.prompt,
        completion=completion,
        full_text=completion
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
"""

def get_cli_chat_code() -> str:
    return """import os
import sys
import torch
from autollm.engine.inference import get_inference_engine

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("AutoLLM Interactive Model CLI")
    print("Type your question and press Enter. Type 'exit' to quit.")
    print("=" * 60)

    engine = get_inference_engine(base_dir, domain="general")

    while True:
        try:
            user_input = input("\\nYou > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting...")
                break

            response = engine.generate_response(user_input)
            print(f"\\nModel > {response}")
        except KeyboardInterrupt:
            print("\\nExiting...")
            break

if __name__ == "__main__":
    main()
"""

def get_interactive_ui_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoLLM - Test Your Model</title>
    <style>
        :root {
            --bg: #0a0d14;
            --surface: #121824;
            --border: #222d3d;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --accent: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 2rem 1rem; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 800px; background: var(--surface); border: 1px solid var(--border); border-radius: 1rem; overflow: hidden; box-shadow: 0 20px 40px rgba(0,0,0,0.5); }
        .header { padding: 1.5rem; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.02); }
        .badge { background: rgba(16, 185, 129, 0.15); color: var(--accent); padding: 0.35rem 0.75rem; border-radius: 2rem; font-size: 0.85rem; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.3); }
        .chat-area { height: 420px; overflow-y: auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
        .msg { max-width: 80%; padding: 0.85rem 1.15rem; border-radius: 1rem; font-size: 0.95rem; line-height: 1.5; }
        .user-msg { align-self: flex-end; background: var(--primary); color: #fff; border-bottom-right-radius: 0.2rem; }
        .bot-msg { align-self: flex-start; background: #1e293b; color: var(--text); border-bottom-left-radius: 0.2rem; border: 1px solid var(--border); }
        .footer { padding: 1.25rem; border-top: 1px solid var(--border); display: flex; gap: 0.75rem; background: rgba(0,0,0,0.2); }
        input[type="text"] { flex: 1; background: #0f172a; border: 1px solid var(--border); padding: 0.85rem 1.15rem; border-radius: 0.6rem; color: var(--text); font-size: 0.95rem; outline: none; }
        input[type="text"]:focus { border-color: var(--primary); }
        button { background: var(--primary); color: white; border: none; padding: 0.85rem 1.5rem; border-radius: 0.6rem; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        button:hover { background: var(--primary-hover); }
        .status-pill { font-size: 0.8rem; color: var(--text-muted); display: flex; align-items: center; gap: 0.4rem; }
        .dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h2>AutoLLM Model Playground</h2>
                <div class="status-pill"><span class="dot"></span> Serving on CPU (FastAPI :8001)</div>
            </div>
            <span class="badge">Model Ready</span>
        </div>
        <div class="chat-area" id="chat">
            <div class="msg bot-msg">Hello! I am your newly trained domain model. Ask me a question based on the training corpus!</div>
        </div>
        <div class="footer">
            <input type="text" id="userInput" placeholder="Ask a question..." onkeydown="if(event.key==='Enter') sendPrompt()" />
            <button onclick="sendPrompt()">Ask Model</button>
        </div>
    </div>
    <script>
        async function sendPrompt() {
            const input = document.getElementById('userInput');
            const chat = document.getElementById('chat');
            const text = input.value.trim();
            if (!text) return;

            // User bubble
            const uMsg = document.createElement('div');
            uMsg.className = 'msg user-msg';
            uMsg.innerText = text;
            chat.appendChild(uMsg);
            input.value = '';
            chat.scrollTop = chat.scrollHeight;

            // Loading bubble
            const botMsg = document.createElement('div');
            botMsg.className = 'msg bot-msg';
            botMsg.innerText = 'Thinking on CPU...';
            chat.appendChild(botMsg);
            chat.scrollTop = chat.scrollHeight;

            try {
                const res = await fetch('http://127.0.0.1:8001/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ prompt: text, max_tokens: 150, temperature: 0.7 })
                });
                const data = await res.json();
                botMsg.innerText = data.completion || data.full_text || 'No response generated.';
            } catch (err) {
                botMsg.innerText = 'Error connecting to model server at http://127.0.0.1:8001/generate. Make sure serve.py is running!';
            }
            chat.scrollTop = chat.scrollHeight;
        }
    </script>
</body>
</html>
"""
