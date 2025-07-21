import json
import numpy as np
import faiss
from typing import List, Dict
import os


class FaissVectorEngine:
    def __init__(self, dim: int, index_path: str = "vector_db/product_index.faiss", meta_path: str = "vector_db/products.json"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatL2(dim)
        self.products: List[Dict] = []

    def index_data(self, embeddings: np.ndarray, products: List[Dict]):
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch. Expected {self.dim}, got {embeddings.shape[1]}")

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.products = products

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.products, f, ensure_ascii=False, indent=2)
        print(f"✅ FAISS index saved to {self.index_path}")
        print(f"✅ Products metadata saved to {self.meta_path}")

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("FAISS index or product metadata not found.")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.products = json.load(f)
        print("✅ FAISS index and products metadata loaded.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            print("⚠️ FAISS index is empty. Add embeddings before searching.")
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            product = self.products[idx].copy()
            product["score"] = float(dist)
            results.append(product)
        return results
