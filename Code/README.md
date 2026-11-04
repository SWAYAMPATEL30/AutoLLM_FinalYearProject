# AutoLLM Builder — Code Directory 🚀
### Core Engine, API Server & Web Studio Implementation

> **B.E. Final Year Capstone Project**  
> Department of Computer Engineering · SVKM's Dwarkadas J. Sanghvi College of Engineering · AY 2026–27  
> **Team Members**: Swayam Patel, Yash Manek, Veer Modi, Yug Shah

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-CPU%20Native-orange.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)](tests/)

---

## 🌟 Overview

The `Code/` directory houses the complete implementation of **AutoLLM Builder**:
- **`autollm/`**: Modular Python package implementing the 4-Way Decision Router, PyTorch training pipelines, Oumi-style data curation, Step 6 Dynamic INT8 quantization, and OWASP security safeguards.
- **`server.py`**: High-performance FastAPI server powering the Web Studio, streaming background job execution, and exposing interactive inference endpoints.
- **`tests/`**: Automated test suite containing 34 unit and integration tests with 100% pass rate.

---

## 🏛️ The 4-Way Decision Router

AutoLLM implements a prompt-driven decision router that analyzes the user's plain-English prompt against local CPU hardware constraints:

1. **Route 1: PEFT Qwen 2.5 (0.5B-Instruct) LoRA**:
   - Executes offline-resilient parameter-efficient fine-tuning on CPU ($540,672\text{ trainable parameters}$, $0.109\%$).
   - Adapter size: $12.98\text{ MB}$.
2. **Route 2: From-Scratch Causal MiniGPT**:
   - Trains an ~850K parameter causal decoder-only transformer using whole-word tokenization (`WordTokenizer`).
   - Applies **Step 6 Dynamic INT8 Quantization** (`torch.ao.quantization.quantize_dynamic`), reducing footprint from $4.22\text{ MB} \to 2.00\text{ MB}$ (**52.6% reduction**).
3. **Route 3: Document Vector RAG**:
   - Ingests user documents (via drag-and-drop upload or text paste), chunks content, builds normalized TF-IDF vector embeddings, and performs grounded retrieval in $1.8\text{ ms}$.
   - Synthesizes document-derived QA test sets for authentic evaluation.
4. **Route 4: Design-Only Preview**:
   - Produces analytical memory blueprints (FP32 vs INT8, AdamW optimizer footprint at $16\text{ bytes/parameter}$, and KV cache) with automated hardware feasibility gates.

---

## 🚀 Running the System

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Studio
```bash
python server.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### 3. Run Automated Tests
```bash
pytest tests/ -v
```
All 34 tests will run and pass:
```text
======================= 34 passed in 17.58s ========================
```

---

## 🛡️ OWASP & SOLID Security Safeguards

- **`autollm/security.py`**:
  - `sanitize_job_id()`: Enforces alphanumeric regex `^[a-zA-Z0-9_\-]+$`, blocking path traversal attacks.
  - `safe_join_paths()`: Verifies that resolved file paths remain within the base sandbox.
  - `sanitize_user_prompt()`: Bounds prompts to $\le 2000$ characters and strips control characters.
  - `validate_uploaded_file()`: Validates file extensions (`.txt`, `.md`, `.json`, `.csv`), enforces $5\text{ MB}$ limit, and rejects null-byte injections.
- **Polymorphic Inference (`autollm/engine/inference.py`)**:
  - Implements uniform `generate_response(user_query)` protocol across `QwenInferenceEngine`, `MiniGPTInferenceEngine`, and `RAGInferenceEngine`.

---

## 📂 Code Organization

```text
Code/
├── autollm/
│   ├── intent/
│   │   ├── schemas.py             # Pydantic v2 data models
│   │   └── parser.py              # Intent extraction & 4-way decision router
│   ├── estimator/
│   │   └── hardware_calc.py       # CPU RAM & parameter feasibility calculator
│   ├── datasets/
│   │   ├── searcher.py            # Hugging Face dataset discovery & seed packs
│   │   └── formatter.py           # Dataset split normalizer
│   ├── engine/
│   │   ├── scratch_transformer.py # MiniGPT causal transformer & WordTokenizer
│   │   ├── quantizer.py           # Step 6 Dynamic INT8 Quantizer
│   │   ├── rag_engine.py          # TF-IDF vector indexing & retrieval engine
│   │   ├── design_preview.py      # Analytical hardware budgeting & KV-cache estimator
│   │   ├── evaluator.py           # Benchmark evaluator & document QA synthesizer
│   │   ├── generator.py           # train_run.py script generator
│   │   ├── runner.py              # Subprocess execution & log streamer
│   │   ├── inference.py           # Polymorphic inference engines
│   │   └── recipe.py              # Oumi-style recipe.yaml generator
│   ├── deploy/
│   │   ├── server_template.py     # Standalone FastAPI deployment templates
│   │   └── package_builder.py     # Deployment bundler (.zip packager)
│   ├── security.py                # OWASP Top 10 sanitization & upload validator
│   ├── orchestrator.py            # Master agent coordinating prompt -> model
│   ├── cli.py                     # Terminal command-line entrypoint
│   └── web/
│       └── index.html             # Web Studio frontend (Vanilla JS / CSS)
├── tests/                         # Unit & integration test suite (34 tests)
│   ├── test_security.py
│   ├── test_scratch_engine.py
│   ├── test_inference.py
│   ├── test_evaluator.py
│   ├── test_rag.py
│   ├── test_quantizer.py
│   ├── test_design_preview.py
│   ├── test_parser.py
│   └── test_e2e_routes.py
├── server.py                      # FastAPI server & API endpoints
├── requirements.txt               # Package dependencies
└── README.md                      # This documentation
```
