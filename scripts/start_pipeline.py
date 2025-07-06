import os
import json
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from utils.embedding_utils import ClipEmbedder
from vector_db.engine import FaissVectorEngine
from mongo_store.database import MongoDB
import requests

# Load environment variables
load_dotenv()

TRITON_VISION_URL = os.getenv("TRITON_VISION_URL")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "vector_db/product_index.faiss")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "clip_vision")

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.asarray(image).astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)  # HWC → CHW
    image = np.expand_dims(image, axis=0)  # (1, 3, 224, 224)
    return image

def infer_with_triton(image_tensor: np.ndarray) -> np.ndarray:
    payload = {
        "inputs": [
            {
                "name": "input_image",
                "shape": list(image_tensor.shape),
                "datatype": "FP32",
                "data": image_tensor.flatten().tolist()
            }
        ],
        "outputs": [{"name": "image_features"}]
    }

    response = requests.post(TRITON_VISION_URL, json=payload)
    response.raise_for_status()
    result = response.json()

    embedding_data = result["outputs"][0]["data"]
    return np.array(embedding_data, dtype=np.float32).reshape((1, -1))

def main():
    # Load product metadata
    with open("sample_data/products.json") as f:
        products = json.load(f)

    # Load precomputed embeddings
    embeddings = np.load("sample_data/embeddings.npy").astype(np.float32)

    # Initialize FAISS and index the embeddings
    engine = FaissVectorEngine(dim=embeddings.shape[1])
    engine.index_data(embeddings, products)
    print(f"✅ FAISS index saved to {FAISS_INDEX_PATH}")

    # Connect to MongoDB and insert metadata
    db = MongoDB()
    if not db.is_connected():
        print("❌ MongoDB connection failed.")
        return

    db.products.delete_many({})
    print("🧹 Cleared existing products.")
    db.insert_products(products)
    print(f"📦 Inserted {len(products)} products successfully.")

    # Load query image (from first product)
    query_img_path = os.path.join("sample_data/images", products[0]["image"])
    query_img = Image.open(query_img_path).convert("RGB")

    # Try inference via Triton
    try:
        image_tensor = preprocess_image(query_img)
        query_embedding = infer_with_triton(image_tensor)
        print("✅ Inference completed via Triton server.")
    except Exception as e:
        print(f"⚠️ Triton inference failed: {e}")
        print("🔄 Falling back to local encoder.")
        embedder = ClipEmbedder()
        query_embedding = embedder.encode_image(query_img).astype("float32")

    # Perform search
    top_matches = engine.search(query_embedding, top_k=5)

    print("🔍 Top matches:")
    for i, match in enumerate(top_matches, 1):
        print(f"{i}. {match['name']} - {match['category']} - ${match['price']}")

    print("🚀 Search pipeline completed successfully!")

if __name__ == "__main__":
    main()
