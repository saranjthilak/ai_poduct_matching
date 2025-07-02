import os
import json
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from utils.embedding_utils import ClipEmbedder
from vector_db.engine import FaissVectorEngine
from mongo_store.database import MongoDB
import requests

load_dotenv()  # Load environment variables from .env file

TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "vlm_model")
TRITON_URL = f"http://localhost:8000/v2/models/{TRITON_MODEL_NAME}/infer"

def main():
    # Load product data
    products_path = os.path.join("sample_data", "products.json")
    with open(products_path) as f:
        products = json.load(f)

    # Load embeddings
    embeddings_path = os.path.join("sample_data", "embeddings.npy")
    embeddings = np.load(embeddings_path).astype("float32")

    # Initialize and index embeddings in FAISS
    engine = FaissVectorEngine(dim=embeddings.shape[1])
    engine.index_data(embeddings, products)

    # Connect to MongoDB and insert product metadata
    db = MongoDB()
    if not db.is_connected():
        print("Failed to connect to MongoDB. Exiting.")
        return

    # Clear existing products and insert fresh
    db.products.delete_many({})
    print(f"Cleared existing products.")
    db.insert_products(products)

    # Initialize embedder for query image encoding (local fallback)
    embedder = ClipEmbedder()

    # Load query image (first product image)
    query_img_path = os.path.join("sample_data", "images", products[0]["image"])
    query_img = Image.open(query_img_path).convert("RGB")

    # Call Triton to get embeddings for query image
    try:
        # Convert image to bytes or base64 as Triton expects (depends on your model input)
        # Here assuming base64-encoded image string input for simplicity:
        import base64
        with open(query_img_path, "rb") as img_file:
            img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        payload = {
            "inputs": [
                {
                    "name": "input_image",  # Change this to your actual input tensor name
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [img_b64]
                }
            ]
        }
        response = requests.post(TRITON_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        # Parse output embeddings from Triton response
        # Change 'output_embedding' to your model's actual output tensor name
        embedding_data = result["outputs"][0]["data"]
        query_embedding = np.array(embedding_data, dtype=np.float32)

    except Exception as e:
        print(f"Error during Triton inference: {e}")
        print("Falling back to local embedder...")
        query_embedding = embedder.encode_image(query_img).astype("float32")

    # Search top 5 matches
    top_matches = engine.search(query_embedding, top_k=5)

    print("Top matches:")
    for i, match in enumerate(top_matches, 1):
        print(f"{i}. {match['name']} - {match['category']} - ${match['price']}")

    print("✅ Search completed successfully! 🚀")

if __name__ == "__main__":
    main()
