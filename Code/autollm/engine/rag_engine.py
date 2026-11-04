import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from autollm.intent.schemas import RAGSpec

class DocumentChunker:
    """
    Splits continuous text or unstructured documents into semantically coherent passages.
    """
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size_words: int = 80,
        overlap_words: int = 20
    ) -> List[Dict[str, Any]]:
        clean_text = re.sub(r'\r\n', '\n', text).strip()
        # Paragraph split
        paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip()]
        
        words = []
        for p in paragraphs:
            for w in p.split():
                words.append(w)
                
        if not words:
            return []

        chunks = []
        stride = max(10, chunk_size_words - overlap_words)
        
        for i in range(0, len(words), stride):
            chunk_words = words[i:i + chunk_size_words]
            if not chunk_words:
                continue
            chunk_content = " ".join(chunk_words)
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "text": chunk_content,
                "word_count": len(chunk_words)
            })
            if i + chunk_size_words >= len(words):
                break
                
        return chunks


class VectorStore:
    """
    Fast, lightweight, 100% offline in-memory vector index for CPU execution.
    Uses normalized TF-IDF and subword n-gram cosine similarity.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=2500,
            sublinear_tf=True
        )
        self.chunks: List[Dict[str, Any]] = []
        self.tfidf_matrix = None
        self.is_fitted = False

    def build_index(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        corpus = [c["text"] for c in chunks]
        if not corpus:
            return
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.is_fitted or not self.chunks:
            return []

        q_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(scores)[::-1][:min(top_k, len(self.chunks))]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk = self.chunks[idx].copy()
            chunk["score"] = round(score, 4)
            results.append(chunk)
            
        return results

    def save(self, filepath: str):
        data = {
            "chunks": self.chunks,
            "vocabulary": {k: int(v) for k, v in self.vectorizer.vocabulary_.items()} if self.is_fitted else {},
            "idf": self.vectorizer.idf_.tolist() if (self.is_fitted and hasattr(self.vectorizer, "idf_")) else []
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "VectorStore":
        store = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        store.chunks = data.get("chunks", [])
        vocab = data.get("vocabulary", {})
        idf = data.get("idf", [])
        
        if store.chunks and vocab:
            store.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                vocabulary=vocab,
                sublinear_tf=True
            )
            store.tfidf_matrix = store.vectorizer.fit_transform([c["text"] for c in store.chunks])
            if idf:
                store.vectorizer.idf_ = np.array(idf)
            store.is_fitted = True
        return store


class RAGPipeline:
    """
    Branch 3 in Decision Router: Zero-Training Document Grounding.
    Ingests raw text or document files, creates vector store, and provides grounded Q&A.
    """
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path
        if index_path and os.path.exists(index_path):
            self.store = VectorStore.load(index_path)
        else:
            self.store = VectorStore()

    def ingest_document(self, content: str, source_name: str, output_dir: str) -> RAGSpec:
        os.makedirs(output_dir, exist_ok=True)
        chunks = DocumentChunker.chunk_text(content)
        self.store.build_index(chunks)

        index_file = os.path.join(output_dir, "vector_store.json")
        self.store.save(index_file)
        self.index_path = index_file

        # Also save raw passages as knowledge.jsonl for uniform package compatibility
        knowledge_file = os.path.join(output_dir, "knowledge.jsonl")
        with open(knowledge_file, "w", encoding="utf-8") as f:
            for c in chunks:
                item = {
                    "instruction": f"What information is documented in section #{c['chunk_id']} of {source_name}?",
                    "output": c["text"],
                    "source": source_name,
                    "chunk_id": c["chunk_id"]
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return RAGSpec(
            document_source=source_name,
            chunk_count=len(chunks),
            embedding_dim=len(self.store.vectorizer.vocabulary_) if self.store.is_fitted else 0,
            top_k=3,
            similarity_threshold=0.1
        )

    def generate_grounded_answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        results = self.store.query(query, top_k=top_k)
        if not results or results[0]["score"] < 0.05:
            return {
                "answer": "I could not find relevant passages matching your query in the ingested document.",
                "citations": [],
                "confidence": 0.0
            }

        top_match = results[0]
        citations = [f"[Passage #{r['chunk_id']} (relevance: {int(r['score']*100)}%)]" for r in results[:2] if r["score"] > 0.08]
        
        answer = f"Based on {', '.join(citations)}:\n\n{top_match['text']}"
        return {
            "answer": answer,
            "citations": [r["text"] for r in results[:2]],
            "confidence": top_match["score"]
        }
