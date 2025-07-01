import numpy as np
import faiss
import os
from typing import List, Dict


class FaissVectorEngine:
    def __init__(self, dim: int, index_path: str = "vector_db/product_index.faiss"):
        """
        Args:
            dim (int): Dimension of embeddings.
            index_path (str): File path to save/load FAISS index.
        """
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)
        self.products: List[Dict] = []

    def index_data(self, embeddings: np.ndarray, products: List[Dict]):
        """
        Add embeddings and associated product metadata to FAISS index.

        Args:
            embeddings (np.ndarray): shape (n_samples, dim)
            products (List[Dict]): metadata for each sample
        """
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch. Expected {self.dim}, got {embeddings.shape[1]}")

        faiss.normalize_L2(embeddings)  # Optional normalization
        self.index.add(embeddings)
        self.products = products

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        print(f"✅ FAISS index saved to {self.index_path}")

    def load(self):
        """
        Load FAISS index from disk.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("FAISS index not found.")
        self.index = faiss.read_index(self.index_path)
        print("✅ FAISS index loaded.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search top_k similar items using FAISS.

        Args:
            query_embedding (np.ndarray): shape (dim,)
            top_k (int): number of results

        Returns:
            List[Dict]: top matched products
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.products[i] for i in indices[0]]
