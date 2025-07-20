import os
import json
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from utils.embedding_utils import ClipEmbedder
from vector_db.engine import FaissVectorEngine
from mongo_store.database import MongoDB
import requests
import datetime
import traceback

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
    print(f"[🔁] Sending image to Triton Inference Server at {TRITON_VISION_URL}...")
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

def log_event(level: str, message: str, metadata: dict = None):
    DB.db["pipeline_logs"].insert_one({
        "timestamp": datetime.datetime.utcnow(),
        "level": level,
        "message": message,
        "metadata": metadata or {}
    })

# --- Initialize resources globally ---

with open("sample_data/products.json") as f:
    PRODUCTS = json.load(f)

EMBEDDINGS = np.load("sample_data/embeddings.npy").astype(np.float32)

ENGINE = FaissVectorEngine(dim=EMBEDDINGS.shape[1])
ENGINE.index_data(EMBEDDINGS, PRODUCTS)

DB = MongoDB()
if not DB.is_connected():
    raise ConnectionError("MongoDB connection failed.")

# Ensure a clean slate for testing/demo purposes
DB.products.delete_many({})
DB.insert_products(PRODUCTS)

def run_matching_pipeline(input_image: Image.Image = None, top_k: int = 5):
    log_event("INFO", "🔄 Matching pipeline started")

    if input_image is None:
        query_img_path = os.path.join("sample_data/images", PRODUCTS[0]["image"])
        query_img = Image.open(query_img_path).convert("RGB")
    else:
        query_img = input_image.convert("RGB")

    try:
        image_tensor = preprocess_image(query_img)
        query_embedding = infer_with_triton(image_tensor)
        print("[✅] Triton inference succeeded.")
        log_event("INFO", "✅ Triton inference succeeded")
    except Exception as e:
        error_msg = str(e)
        print(f"[⚠️] Triton inference failed: {error_msg}")
        print("[🧠] Falling back to local ClipEmbedder...")
        log_event("ERROR", "❌ Triton inference failed, using fallback", {
            "error": error_msg,
            "traceback": traceback.format_exc()
        })
        embedder = ClipEmbedder()
        query_embedding = embedder.encode_image(query_img).astype(np.float32)

    top_matches = ENGINE.search(query_embedding, top_k=top_k)

    result_log = [{"name": m.get("name"), "score": m.get("score", None)} for m in top_matches]
    log_event("INFO", "✅ Matching complete", {"top_k": top_k, "results": result_log})

    sanitized = []
    for m in top_matches:
        sanitized.append({
            "name": m.get("name", ""),
            "category": m.get("category", ""),
            "price": float(m.get("price", 0)),
            "image": m.get("image", ""),
        })

    return sanitized
