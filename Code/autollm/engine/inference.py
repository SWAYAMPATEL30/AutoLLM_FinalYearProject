import os
import re
import json
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from autollm.engine.scratch_transformer import MiniGPT, load_tokenizer, WordTokenizer, CharTokenizer
from autollm.engine.rag_engine import RAGPipeline
from autollm.security import validate_inference_params

class QwenInferenceEngine:
    """
    Route 1: Qwen 2.5 (0.5B-Instruct) LoRA Neural Inference Engine on CPU.
    Executes authentic language modeling using PEFT adapter weights.
    Fully resilient to offline environments when weights are cached locally.
    """
    def __init__(self, adapter_dir: str, base_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self.adapter_dir = adapter_dir
        self.base_model_id = base_model_id
        self.device = torch.device("cpu")
        
        tok_source = adapter_dir if os.path.exists(os.path.join(adapter_dir, "tokenizer_config.json")) else base_model_id
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tok_source, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(tok_source)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
        except Exception:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                dtype=torch.float32,
                low_cpu_mem_usage=True
            )

        if os.path.exists(adapter_dir) and (
            os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")) or 
            os.path.exists(os.path.join(adapter_dir, "adapter_model.bin"))
        ):
            try:
                self.model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
            except Exception as e:
                print(f"[Inference Warning] Could not load LoRA adapter: {e}. Running base model.")
                self.model = base_model
        else:
            self.model = base_model

        self.model.eval()

    def generate_response(self, user_query: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        temp, tokens = validate_inference_params(temperature, max_tokens)
        messages = [{"role": "user", "content": user_query.strip()}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # Handle BatchEncoding dictionary or pure Tensor
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs.to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
        elif hasattr(inputs, "data") or isinstance(inputs, dict):
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids).to(self.device)
            else:
                attention_mask = attention_mask.to(self.device)
        else:
            input_ids = torch.tensor(inputs, dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=tokens,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
            
        generated_tokens = outputs[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response


class MiniGPTInferenceEngine:
    """
    Route 2: From-Scratch Causal MiniGPT Neural Inference Engine.
    Executes true autoregressive word/token generation from trained model weights.
    Supports both FP32 model.pt and dynamic INT8 model_quantized.pt.
    """
    def __init__(self, model_path: str, tokenizer_path: str):
        self.device = torch.device("cpu")
        self.tokenizer = load_tokenizer(tokenizer_path)

        embed_dim = 128
        num_layers = 4
        num_heads = 4
        max_seq_len = 256

        self.model = MiniGPT(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        )

        is_quantized = "quantized" in os.path.basename(model_path).lower()
        if is_quantized:
            try:
                self.model = torch.ao.quantization.quantize_dynamic(
                    self.model, {nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                pass

        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            cfg = state_dict.get("config", {})
            if cfg:
                embed_dim = cfg.get("embed_dim", embed_dim)
                num_layers = cfg.get("num_layers", num_layers)
                num_heads = cfg.get("num_heads", num_heads)
                max_seq_len = cfg.get("max_seq_len", max_seq_len)
                self.model = MiniGPT(
                    vocab_size=cfg.get("vocab_size", self.tokenizer.vocab_size),
                    embed_dim=embed_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    max_seq_len=max_seq_len
                )
            state_dict = state_dict["model_state_dict"]

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def generate_response(self, user_query: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        temp, tokens = validate_inference_params(temperature, max_tokens)
        prompt_text = f"Question: {user_query.strip()} Answer:"
        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        # Drop the final EOS token so causal generation continues
        if input_ids and input_ids[-1] == self.tokenizer.eos_id:
            input_ids = input_ids[:-1]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            output_tensor = self.model.generate(
                input_tensor,
                max_new_tokens=tokens,
                temperature=temp,
                top_k=25,
                top_p=0.9,
                repetition_penalty=1.2
            )

        full_text = self.tokenizer.decode(output_tensor[0].tolist())
        # Extract continuation after "Answer:"
        if "Answer:" in full_text:
            ans = full_text.split("Answer:", 1)[1].strip()
            if "Question:" in ans:
                ans = ans.split("Question:", 1)[0].strip()
            if ans:
                return ans

        return full_text.strip()


class RAGInferenceEngine:
    """
    Route 3: In-Memory Document RAG Inference Engine.
    Performs normalized vector search across indexed document passages and synthesizes grounded answers.
    """
    def __init__(self, vector_store_path: str):
        self.pipeline = RAGPipeline(index_path=vector_store_path)

    def generate_response(self, user_query: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        result = self.pipeline.generate_grounded_answer(user_query, top_k=3)
        return result["answer"]


class DomainInferenceEngine:
    """
    Intelligent Grounded Domain Fallback Engine.
    Used for conversational greetings and semantic matching when neural weights are unmounted.
    """
    def __init__(
        self,
        domain: str = "general",
        dataset_path: Optional[str] = None
    ):
        self.domain = domain.lower()
        self.knowledge_base: List[Dict[str, str]] = []
        if dataset_path and os.path.exists(dataset_path):
            self._load_knowledge_base(dataset_path)

    def _load_knowledge_base(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        inst = data.get("instruction", "").strip()
                        out = data.get("output", "").strip()
                        if inst and out:
                            if not any(k["instruction"] == inst for k in self.knowledge_base):
                                self.knowledge_base.append({"instruction": inst, "output": out})
        except Exception as e:
            print(f"[Inference Notice] Loading knowledge base: {e}")

    def generate_response(self, user_query: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        q = user_query.strip().lower()

        # Conversational greetings
        greetings = ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]
        if any(q == g or q.startswith(g + " ") for g in greetings):
            return f"Hello! I am your AI assistant specialized in {self.domain.capitalize()}. How can I assist you today?"

        # Semantic Keyword match
        best_match = None
        highest_score = 0.0
        q_words = set(re.findall(r'\b[a-z]{3,}\b', q))

        for item in self.knowledge_base:
            inst_text = item["instruction"].lower()
            inst_words = set(re.findall(r'\b[a-z]{3,}\b', inst_text))
            if not inst_words:
                continue
            common = q_words.intersection(inst_words)
            score = len(common) / (len(q_words.union(inst_words)) + 1e-5)
            if score > highest_score:
                highest_score = score
                best_match = item

        if best_match and highest_score > 0.15:
            return best_match["output"]

        return (
            f"This model was compiled for the '{self.domain}' domain. "
            f"The training corpus did not contain specific information for: '{user_query}'."
        )


def get_inference_engine(job_dir: str, domain: str = "general"):
    """
    Factory function: Inspects artifacts in job_dir and mounts the highest-capability engine:
      1. Qwen 2.5 LoRA Adapter (if adapter files exist)
      2. RAG Vector Store (if vector_store.json exists)
      3. MiniGPT Neural Engine (if model_quantized.pt or model.pt exists)
      4. Grounded Domain Fallback
    """
    candidates = [job_dir, os.path.join(job_dir, "export")]

    # 1. Check for Qwen LoRA Adapter
    for root in candidates:
        adapter_dir = os.path.join(root, "adapter")
        if os.path.exists(adapter_dir) and (
            os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")) or 
            os.path.exists(os.path.join(adapter_dir, "adapter_model.bin"))
        ):
            try:
                print(f"[Inference] Mounting Qwen 2.5 LoRA Neural Engine from: {adapter_dir}")
                return QwenInferenceEngine(adapter_dir=adapter_dir)
            except Exception as e:
                print(f"[Inference Warning] Failed to mount Qwen adapter: {e}")

    # 2. Check for RAG Vector Store
    for root in candidates:
        vec_file = os.path.join(root, "vector_store.json")
        if os.path.exists(vec_file):
            print(f"[Inference] Mounting RAG Vector Index from: {vec_file}")
            return RAGInferenceEngine(vector_store_path=vec_file)

    # 3. Check for Scratch MiniGPT Neural Weights (Quantized first, then FP32)
    for root in candidates:
        tok_file = os.path.join(root, "tokenizer.json")
        quant_model = os.path.join(root, "model_quantized.pt")
        fp32_model = os.path.join(root, "model.pt")

        target_model = quant_model if os.path.exists(quant_model) else (fp32_model if os.path.exists(fp32_model) else None)
        if target_model and os.path.exists(tok_file):
            try:
                print(f"[Inference] Mounting MiniGPT Neural Engine from: {target_model}")
                return MiniGPTInferenceEngine(model_path=target_model, tokenizer_path=tok_file)
            except Exception as e:
                print(f"[Inference Warning] Failed to mount MiniGPT: {e}")

    # 4. Fallback to Grounded Domain Knowledge
    data_dir = os.path.join(job_dir, "data")
    train_file = os.path.join(data_dir, "train.jsonl")
    knowledge_file = os.path.join(job_dir, "knowledge.jsonl")
    dataset_path = train_file if os.path.exists(train_file) else (knowledge_file if os.path.exists(knowledge_file) else None)

    print(f"[Inference] Mounting Grounded Domain Semantic Engine (domain={domain})")
    return DomainInferenceEngine(domain=domain, dataset_path=dataset_path)
