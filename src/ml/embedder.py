"""Hugging Face sentence-transformers wrapper."""

from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


class TextEmbedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if HAS_ST:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            return np.random.randn(
                len(texts), 384
            ).astype(np.float32)
        return self.model.encode(texts)

    def similarity(
        self, text_a: str, text_b: str
    ) -> float:
        embeddings = self.embed([text_a, text_b])
        a = embeddings[0]
        b = embeddings[1]
        cos_sim = float(
            np.dot(a, b) / (
                np.linalg.norm(a) * np.linalg.norm(b)
            )
        )
        return cos_sim

    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 3,
    ) -> List[dict]:
        all_texts = [query] + candidates
        embeddings = self.embed(all_texts)
        query_emb = embeddings[0]
        candidate_embs = embeddings[1:]

        similarities = []
        for i, emb in enumerate(candidate_embs):
            sim = float(
                np.dot(query_emb, emb) / (
                    np.linalg.norm(query_emb)
                    * np.linalg.norm(emb)
                )
            )
            similarities.append({
                "index": i,
                "text": candidates[i],
                "similarity": sim,
            })

        similarities.sort(
            key=lambda x: x["similarity"], reverse=True
        )
        return similarities[:top_k]