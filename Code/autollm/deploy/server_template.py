def get_fastapi_server_code() -> str:
    return """import os
import sys
import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure autollm is importable from export directory
base_dir = os.path.dirname(os.path.abspath(__file__))
for up in range(1, 6):
    candidate = os.path.abspath(os.path.join(base_dir, *(['..'] * up)))
    if os.path.exists(os.path.join(candidate, "autollm")):
        sys.path.insert(0, candidate)
        break

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
    prompt: str = Field(..., max_length=2000, description="Input user prompt")
    max_tokens: int = Field(150, ge=1, le=1024, description="Max generated tokens")
    temperature: float = Field(0.7, ge=0.01, le=2.0, description="Sampling temperature")

class InferenceResponse(BaseModel):
    prompt: str
    completion: str
    full_text: str

@app.on_event("startup")
def load_model():
    global engine
    engine = get_inference_engine(base_dir, domain="general")
    print(f"AutoLLM Model Inference Engine mounted: {engine.__class__.__name__} on CPU.")

@app.get("/health")
def health():
    return {"status": "ready", "device": "cpu", "engine": engine.__class__.__name__ if engine else "none"}

@app.post("/generate", response_model=InferenceResponse)
def generate(req: InferenceRequest):
    if not engine:
        raise HTTPException(status_code=500, detail="Inference engine is initializing or unavailable.")

    clean_prompt = req.prompt.strip()
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    completion = engine.generate_response(clean_prompt, max_tokens=req.max_tokens, temperature=req.temperature)
    return InferenceResponse(
        prompt=clean_prompt,
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

base_dir = os.path.dirname(os.path.abspath(__file__))
for up in range(1, 6):
    candidate = os.path.abspath(os.path.join(base_dir, *(['..'] * up)))
    if os.path.exists(os.path.join(candidate, "autollm")):
        sys.path.insert(0, candidate)
        break

from autollm.engine.inference import get_inference_engine

def main():
    print("=" * 65)
    print("AutoLLM Interactive Model CLI Client")
    print("Type your question and press Enter. Type 'exit' to quit.")
    print("=" * 65)

    engine = get_inference_engine(base_dir, domain="general")
    print(f"[*] Engine mounted: {engine.__class__.__name__} (100% CPU Optimized)\n")

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting...")
                break

            response = engine.generate_response(user_input, max_tokens=150, temperature=0.7)
            print(f"\\nModel > {response}\\n")
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
            --bg: #06080d;
            --surface: #0c111c;
            --border: #1e293b;
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 2rem 1rem; display: flex; justify-content: center; min-height: 100vh; }
        .container { width: 100%; max-width: 840px; background: var(--surface); border: 1px solid var(--border); border-radius: 1rem; overflow: hidden; box-shadow: 0 20px 40px rgba(0,0,0,0.5); display: flex; flex-direction: column; }
        .header { padding: 1.5rem; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.02); }
        .badge { background: rgba(16, 185, 129, 0.15); color: var(--accent); padding: 0.35rem 0.75rem; border-radius: 2rem; font-size: 0.85rem; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.3); }
        .chat-area { flex: 1; min-height: 440px; max-height: 550px; overflow-y: auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
        .msg { max-width: 80%; padding: 0.85rem 1.15rem; border-radius: 1rem; font-size: 0.95rem; line-height: 1.5; }
        .user-msg { align-self: flex-end; background: var(--primary); color: #fff; border-bottom-right-radius: 0.2rem; }
        .bot-msg { align-self: flex-start; background: #131a29; color: var(--text); border-bottom-left-radius: 0.2rem; border: 1px solid var(--border); }
        .footer { padding: 1.25rem; border-top: 1px solid var(--border); display: flex; gap: 0.75rem; background: rgba(0,0,0,0.2); }
        input[type="text"] { flex: 1; background: #0f172a; border: 1px solid var(--border); padding: 0.85rem 1.15rem; border-radius: 0.6rem; color: var(--text); font-size: 0.95rem; outline: none; }
        input[type="text"]:focus { border-color: var(--primary); }
        button { background: var(--primary); color: white; border: none; padding: 0.85rem 1.5rem; border-radius: 0.6rem; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        button:hover { background: var(--primary-hover); }
        .status-pill { font-size: 0.8rem; color: var(--text-muted); display: flex; align-items: center; gap: 0.4rem; margin-top: 0.3rem; }
        .dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h2>AutoLLM Standalone Playground</h2>
                <div class="status-pill"><span class="dot"></span> Serving on CPU (FastAPI :8001)</div>
            </div>
            <span class="badge">Model Ready</span>
        </div>
        <div class="chat-area" id="chat">
            <div class="msg bot-msg">Hello! I am your domain-adapted AI model compiled by AutoLLM Builder. Ask me anything!</div>
        </div>
        <div class="footer">
            <input type="text" id="userInput" placeholder="Ask a question..." onkeydown="if(event.key==='Enter') sendPrompt()" />
            <button onclick="sendPrompt()">Send</button>
        </div>
    </div>
    <script>
        async function sendPrompt() {
            const input = document.getElementById('userInput');
            const chat = document.getElementById('chat');
            const text = input.value.trim();
            if (!text) return;

            const uMsg = document.createElement('div');
            uMsg.className = 'msg user-msg';
            uMsg.innerText = text;
            chat.appendChild(uMsg);
            input.value = '';
            chat.scrollTop = chat.scrollHeight;

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
                botMsg.innerText = 'Error connecting to model server at http://127.0.0.1:8001/generate. Ensure python serve.py is running!';
            }
            chat.scrollTop = chat.scrollHeight;
        }
    </script>
</body>
</html>
"""
