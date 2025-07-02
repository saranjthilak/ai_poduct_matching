# scripts/infer_text.py

from sentence_transformers import SentenceTransformer
import numpy as np
import argparse

def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text embedding")
    parser.add_argument("--text", type=str, required=True, help="Input product description/title")
    args = parser.parse_args()

    text_embedding = embed_text(args.text)
    print(f"\n✅ Text embedding (shape: {text_embedding.shape}):\n{text_embedding}")
    print("\n✅ Text embedding generated successfully! 🚀")
