import json
import numpy as np
import faiss
from typing import List, Dict, Optional
import os


class FaissVectorEngine:
    def __init__(
        self,
        dim_combined: Optional[int] = None,
        dim_text: Optional[int] = None,
        index_path_combined: str = "vector_db/combined_index.faiss",
        index_path_text: str = "vector_db/text_only_index.faiss",
        meta_path: str = "vector_db/products.json"
    ):
        if dim_combined is None or dim_text is None:
            raise ValueError("dim_combined and dim_text must be specified.")

        self.dim_combined = dim_combined
        self.dim_text = dim_text

        self.index_path_combined = index_path_combined
        self.index_path_text = index_path_text
        self.meta_path = meta_path

        # Initialize empty indexes
        self.index_combined = faiss.IndexFlatL2(dim_combined)
        self.index_text = faiss.IndexFlatL2(dim_text)

        self.products: List[Dict] = []

    def index_data(
        self,
        embeddings_combined: Optional[np.ndarray] = None,
        embeddings_text: Optional[np.ndarray] = None,
        products: Optional[List[Dict]] = None
    ):
        if embeddings_combined is not None:
            if embeddings_combined.shape[1] != self.dim_combined:
                raise ValueError(f"Combined embeddings dim mismatch. Expected {self.dim_combined}, got {embeddings_combined.shape[1]}")
            faiss.normalize_L2(embeddings_combined)
            self.index_combined = faiss.IndexFlatL2(self.dim_combined)  # reset index before add
            self.index_combined.add(embeddings_combined)
            faiss.write_index(self.index_combined, self.index_path_combined)
            print(f"✅ Combined FAISS index saved to {self.index_path_combined}")

        if embeddings_text is not None:
            if embeddings_text.shape[1] != self.dim_text:
                raise ValueError(f"Text-only embeddings dim mismatch. Expected {self.dim_text}, got {embeddings_text.shape[1]}")
            faiss.normalize_L2(embeddings_text)
            self.index_text = faiss.IndexFlatL2(self.dim_text)  # reset index before add
            self.index_text.add(embeddings_text)
            faiss.write_index(self.index_text, self.index_path_text)
            print(f"✅ Text-only FAISS index saved to {self.index_path_text}")

        if products is not None:
            self.products = products
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.products, f, ensure_ascii=False, indent=2)
            print(f"✅ Products metadata saved to {self.meta_path}")

    def load(self):
        if os.path.exists(self.index_path_combined):
            self.index_combined = faiss.read_index(self.index_path_combined)
            print(f"✅ Loaded combined FAISS index from {self.index_path_combined}")
        else:
            print(f"⚠️ Combined index file not found at {self.index_path_combined}")

        if os.path.exists(self.index_path_text):
            self.index_text = faiss.read_index(self.index_path_text)
            print(f"✅ Loaded text-only FAISS index from {self.index_path_text}")
        else:
            print(f"⚠️ Text-only index file not found at {self.index_path_text}")

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.products = json.load(f)
            print(f"✅ Loaded products metadata from {self.meta_path}")
        else:
            print(f"⚠️ Product metadata file not found at {self.meta_path}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5, use_text_only: bool = False) -> List[Dict]:
        if use_text_only:
            index = self.index_text
            expected_dim = self.dim_text
        else:
            index = self.index_combined
            expected_dim = self.dim_combined

        if index.ntotal == 0:
            print("⚠️ FAISS index is empty. Add embeddings before searching.")
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        if query_embedding.shape[1] != expected_dim:
            raise ValueError(f"Query embedding dim mismatch. Expected {expected_dim}, got {query_embedding.shape[1]}")

        faiss.normalize_L2(query_embedding)
        distances, indices = index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            product = self.products[idx].copy()
            product["score"] = float(dist)
            results.append(product)
        return results
