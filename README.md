# AutoLLM Builder 🚀
### Prompt-Driven Small Language Model Generation and Deployment Framework (CPU Optimized)

> **B.E. Final Year Capstone Project**  
> Department of Computer Engineering · SVKM's Dwarkadas J. Sanghvi College of Engineering · AY 2026–27  
> **Team Members**: Swayam Patel, Yash Manek, Veer Modi, Yug Shah

---

## 🌟 Overview

**AutoLLM Builder** is an autonomous AI agent and compiler that bridges the technical barrier of building domain-adapted AI models. A user simply provides a plain-English prompt:

> *"Create a chatbot model for medical FAQ answering under 2GB size"*

The autonomous agent executes the entire lifecycle without manual ML configuration:
1. **Understands Intent**: Parses task type, domain, language, parameters, and size limits.
2. **Analytical Feasibility Gate**: Inspects local CPU memory and cores, ensuring safe execution without throttling or crashing.
3. **Discovers & Suggests Top 3 Web Datasets**: Searches Hugging Face Hub live across community datasets, presents the top 3 best matching options with download counts, community likes, sample previews, and lets the user choose which dataset to train on.
4. **Writes Training Code**: Dynamically generates tailored PyTorch code and model architectures.
5. **Executes Sandboxed Training**: 
   - **Qwen 2.5 (0.5B-Instruct) + LoRA**: Fine-tunes attention projection adapters on CPU (~1-2 mins) for state-of-the-art natural language generation.
   - **Fast Scratch Transformer**: Fast causal autoregressive Transformer baseline (~850K parameters) training in ~10 seconds on CPU.
6. **Compiles & Packages**: Generates model weights / LoRA adapters, tokenizer, a standalone FastAPI server (`serve.py`), an interactive terminal chat (`run_cli.py`), a web playground (`index.html`), and a portable `.zip` download.

---

## ⚡ 100% CPU-Friendly Architecture

- **No GPU Required**: Designed to run smoothly and rapidly on standard student laptops and commodity CPUs.
- **Qwen 2.5 LoRA & Mini-Transformer Options**: Select between fine-tuning high-intelligence foundation models or training lightweight scratch models.
- **Zero-Crash Memory Management**: Automatically computes memory safety budgets based on system RAM.

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
git clone https://github.com/SWAYAMPATEL30/AutoLLM_FinalYearProject.git
cd AutoLLM_FinalYearProject/Code
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Studio
Start the browser dashboard:
```bash
python server.py
```
Open your browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

You can enter any prompt (or pick from built-in presets: *Medical FAQ*, *Shakespeare*, *Legal QA*, *Python Code*), watch the animated pipeline stages in real time, view streaming terminal logs, test the model in an in-browser chat room, and download the `.zip` package.

---

### 3. Run via Command Line (CLI)
You can also run the agent directly from your terminal:
```bash
python autollm/cli.py "Create a chatbot model for medical FAQ answering under 2GB size"
```

---

## 📦 Using Your Generated Model

Every run outputs a self-contained bundle in `jobs/<job_id>/export/`:

| File / Folder | Purpose |
| :--- | :--- |
| `adapter/` or `model.pt` | LoRA adapter weights (`adapter_model.safetensors`) or PyTorch weights |
| `tokenizer.json` | Fitted vocabulary and token mappings |
| `eval_metrics.json` | Training and validation metrics, parameter count, duration |
| `serve.py` | Standalone FastAPI inference server with `/generate` endpoint |
| `run_cli.py` | Interactive terminal chat script (`python run_cli.py`) |
| `index.html` | Client-side web interface to chat with the model |
| `start_server.bat` | One-click Windows batch launcher for the inference server |

### Run the standalone API:
```bash
cd jobs/<job_id>/export
python serve.py
```

### Test via curl / HTTP POST:
```bash
curl -X POST http://127.0.0.1:8001/generate \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": \"What are the common symptoms of Type 2 Diabetes?\"}"
```

---

## 📂 Project Structure

```
├── Code/
│   ├── autollm/
│   │   ├── intent/
│   │   │   ├── schemas.py          # Pydantic data models for specs, jobs, and metrics
│   │   │   └── parser.py           # Natural language prompt extractor
│   │   ├── estimator/
│   │   │   └── hardware_calc.py    # Analytical CPU RAM/core & parameter calculator
│   │   ├── datasets/
│   │   │   ├── searcher.py         # Multi-source dataset curator & domain seed packs
│   │   │   └── formatter.py        # Normalizer for instruction datasets & train/val splits
│   │   ├── engine/
│   │   │   ├── scratch_transformer.py # PyTorch Causal MiniGPT & CharTokenizer
│   │   │   ├── generator.py        # Compiles custom train_run.py scripts (LoRA & Scratch)
│   │   │   ├── runner.py           # Subprocess runner streaming logs & events
│   │   │   └── inference.py        # Qwen LoRA & grounded inference engines
│   │   ├── deploy/
│   │   │   ├── server_template.py  # Standalone FastAPI & UI code generators
│   │   │   └── package_builder.py  # Zips and bundles deployable artifacts
│   │   ├── orchestrator.py         # Master agent coordinating prompt -> model
│   │   ├── cli.py                  # Terminal command-line entrypoint
│   │   └── web/
│   │       └── index.html          # Modern dark-mode web studio
│   ├── server.py                   # Web Studio backend server
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # Code documentation
├── docs/                           # Project documentation, presentations, and reports
└── README.md                       # Main project documentation
```
