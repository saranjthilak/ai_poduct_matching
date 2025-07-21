import os
import json
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from utils.embedding_utils import ClipEmbedder
from vector_db.engine import FaissVectorEngine
from mongo_store.database import MongoDB
from sentence_transformers import SentenceTransformer
import datetime
import traceback

# Triton client
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

load_dotenv()

TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL", "localhost:8000")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "clip_vision")
TRITON_INPUT_NAME = "input_image"
TRITON_OUTPUT_NAME = "image_features"

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "vector_db/product_index.faiss")

# Initialize Triton client
triton_client = httpclient.InferenceServerClient(url=TRITON_HTTP_URL)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.asarray(image).astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)  # HWC → CHW
    image = np.expand_dims(image, axis=0)  # (1, 3, 224, 224)
    return image

def infer_with_triton(image_tensor: np.ndarray) -> np.ndarray:
    inputs = [
        httpclient.InferInput(TRITON_INPUT_NAME, image_tensor.shape, np_to_triton_dtype(image_tensor.dtype))
    ]
    inputs[0].set_data_from_numpy(image_tensor)

    outputs = [httpclient.InferRequestedOutput(TRITON_OUTPUT_NAME)]

    response = triton_client.infer(model_name=TRITON_MODEL_NAME, inputs=inputs, outputs=outputs)

    result = response.as_numpy(TRITON_OUTPUT_NAME)
    return result.astype(np.float32)

def log_event(level: str, message: str, metadata: dict = None):
    DB.db["pipeline_logs"].insert_one({
        "timestamp": datetime.datetime.utcnow(),
        "level": level,
        "message": message,
        "metadata": metadata or {}
    })

# Load product metadata
with open("sample_data/products.json") as f:
    PRODUCTS = json.load(f)

# Load combined embeddings used during indexing — must be combined (image + text)
EMBEDDINGS = np.load("sample_data/combined_embeddings.npy").astype(np.float32)

# Initialize FAISS engine with combined embedding dimension (e.g., 1024)
ENGINE = FaissVectorEngine(dim=EMBEDDINGS.shape[1])
ENGINE.index_data(EMBEDDINGS, PRODUCTS)

# Initialize MongoDB connection and reindex products
DB = MongoDB()
if not DB.is_connected():
    raise ConnectionError("MongoDB connection failed.")

DB.products.delete_many({})
DB.insert_products(PRODUCTS)

# Initialize local fallback models
clip_embedder = ClipEmbedder()
text_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Get text embedding dimension for zero-padding if needed
text_dim = text_embedder.get_sentence_embedding_dimension()

def run_matching_pipeline(input_image: Image.Image = None, input_text: str = "", top_k: int = 5):
    log_event("INFO", "🔄 Matching pipeline started")

    # Use default image if none provided
    if input_image is None:
        query_img_path = os.path.join("sample_data/images", PRODUCTS[0]["image"])
        query_img = Image.open(query_img_path).convert("RGB")
    else:
        query_img = input_image.convert("RGB")

    try:
        image_tensor = preprocess_image(query_img)
        image_embedding = infer_with_triton(image_tensor)
        log_event("INFO", "✅ Triton inference succeeded")
    except Exception as e:
        error_msg = str(e)
        log_event("ERROR", "❌ Triton inference failed, using fallback", {
            "error": error_msg,
            "traceback": traceback.format_exc()
        })
        image_embedding = clip_embedder.encode_image(query_img).astype(np.float32)

    # Ensure image_embedding is 1D vector
    image_embedding = image_embedding.flatten()

    if input_text:
        text_embedding = text_embedder.encode(input_text, convert_to_numpy=True).astype(np.float32)
    else:
        text_embedding = np.zeros((0,), dtype=np.float32)  # Start empty, will pad if needed

    expected_dim = EMBEDDINGS.shape[1]

    img_emb_len = image_embedding.shape[0]
    txt_emb_len = text_embedding.shape[0]

    # Calculate needed text embedding length to match expected dimension
    needed_text_len = expected_dim - img_emb_len

    if needed_text_len < 0:
        raise ValueError(
            f"Image embedding length {img_emb_len} exceeds expected total embedding dimension {expected_dim}. "
            "Check your Triton model output size and FAISS index dimension."
        )

    # Adjust text embedding length by padding or truncating
    if txt_emb_len < needed_text_len:
        text_embedding = np.pad(text_embedding, (0, needed_text_len - txt_emb_len), mode='constant')
    else:
        text_embedding = text_embedding[:needed_text_len]

    # Final combined embedding
    combined_embedding = np.concatenate([image_embedding, text_embedding])

    # Verify final combined embedding shape matches expected FAISS dim
    if combined_embedding.shape[0] != expected_dim:
        raise ValueError(
            f"Combined embedding dimension {combined_embedding.shape[0]} does not match expected {expected_dim}"
        )

    # Search in FAISS index
    top_matches = ENGINE.search(combined_embedding.reshape(1, -1), top_k=top_k)

    result_log = [{"name": m.get("name"), "score": m.get("score", None)} for m in top_matches]
    log_event("INFO", "✅ Matching complete", {"top_k": top_k, "results": result_log})

    # Sanitize output
    sanitized = []
    for m in top_matches:
        sanitized.append({
            "name": m.get("name", ""),
            "category": m.get("category", ""),
            "price": float(m.get("price", 0)),
            "image": m.get("image", ""),
        })

    return sanitized
