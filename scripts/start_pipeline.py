import os
import json
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from utils.embedding_utils import ClipEmbedder
from vector_db.engine import FaissVectorEngine
from mongo_store.database import MongoDB
import datetime
import traceback

# Triton client
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

# Hugging Face CLIP imports
from transformers import CLIPProcessor, CLIPModel
import torch

load_dotenv()

TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL", "localhost:8000")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "clip_vision")
TRITON_INPUT_NAME = "input_image"
TRITON_OUTPUT_NAME = "image_features"

FAISS_COMBINED_INDEX_PATH = os.getenv("FAISS_COMBINED_INDEX_PATH", "vector_db/combined_index.faiss")
FAISS_TEXT_ONLY_INDEX_PATH = os.getenv("FAISS_TEXT_ONLY_INDEX_PATH", "vector_db/text_only_index.faiss")
PRODUCTS_META_PATH = os.getenv("PRODUCTS_META_PATH", "vector_db/products.json")

# Initialize Triton client
triton_client = httpclient.InferenceServerClient(url=TRITON_HTTP_URL)

# Initialize CLIP text model and processor (Hugging Face)
clip_text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_text_model.eval()

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

def get_clip_text_embedding(text: str) -> np.ndarray:
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_text_model.get_text_features(**inputs)
    emb = outputs[0].cpu().numpy()
    return emb.astype(np.float32)

def log_event(level: str, message: str, metadata: dict = None):
    DB.db["pipeline_logs"].insert_one({
        "timestamp": datetime.datetime.utcnow(),
        "level": level,
        "message": message,
        "metadata": metadata or {}
    })

# Load products metadata
with open(PRODUCTS_META_PATH, "r", encoding="utf-8") as f:
    PRODUCTS = json.load(f)

# Initialize MongoDB connection and reindex products
DB = MongoDB()
if not DB.is_connected():
    raise ConnectionError("MongoDB connection failed.")

DB.products.delete_many({})
DB.insert_products(PRODUCTS)

# Initialize FaissVectorEngine with correct paths and dims
combined_engine = FaissVectorEngine(
    dim_combined=1024,
    dim_text=512,
    index_path_combined=FAISS_COMBINED_INDEX_PATH,
    index_path_text=FAISS_TEXT_ONLY_INDEX_PATH,
    meta_path=PRODUCTS_META_PATH,
)
combined_engine.load()

# Initialize fallback model for image embeddings (ClipEmbedder)
clip_embedder = ClipEmbedder()

def run_matching_pipeline(input_image: Image.Image = None, input_text: str = "", top_k: int = 5):
    log_event("INFO", "🔄 Matching pipeline started")

    use_text_only = False
    if input_image is None and input_text:
        # Text-only search
        use_text_only = True
    elif input_image is None and not input_text:
        # No input given, use default image from first product
        input_image = Image.open(os.path.join("sample_data/images", PRODUCTS[0]["image"])).convert("RGB")

    try:
        if not use_text_only:
            # Image embedding via Triton
            image_tensor = preprocess_image(input_image)
            image_embedding = infer_with_triton(image_tensor)
            log_event("INFO", "✅ Triton inference succeeded")
            image_embedding = image_embedding.flatten()
        else:
            image_embedding = np.array([], dtype=np.float32)
    except Exception as e:
        log_event("ERROR", "❌ Triton inference failed, using fallback", {
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        if not use_text_only:
            image_embedding = clip_embedder.encode_image(input_image).astype(np.float32)
        else:
            image_embedding = np.array([], dtype=np.float32)

    # Text embedding using Hugging Face CLIP text encoder
    if input_text:
        text_embedding = get_clip_text_embedding(input_text).flatten()
    else:
        text_embedding = np.array([], dtype=np.float32)

    if use_text_only:
        # For text-only search, query is just text embedding (512 dim)
        query_vector = text_embedding
        if query_vector.shape[0] != combined_engine.dim_text:
            raise ValueError(f"Text embedding dim {query_vector.shape[0]} does not match expected {combined_engine.dim_text}")
    else:
        # For combined, concatenate image + text (with padding)
        expected_dim = combined_engine.dim_combined
        img_len = image_embedding.shape[0]
        txt_len = text_embedding.shape[0]
        needed_txt_len = expected_dim - img_len
        if needed_txt_len < 0:
            raise ValueError(f"Image embedding length {img_len} exceeds combined expected dim {expected_dim}")
        if txt_len < needed_txt_len:
            text_embedding = np.pad(text_embedding, (0, needed_txt_len - txt_len), mode='constant')
        else:
            text_embedding = text_embedding[:needed_txt_len]

        query_vector = np.concatenate([image_embedding, text_embedding])
        if query_vector.shape[0] != expected_dim:
            raise ValueError(f"Combined embedding dim {query_vector.shape[0]} does not match expected {expected_dim}")

    # Search using appropriate index
    top_matches = combined_engine.search(query_vector.reshape(1, -1), top_k=top_k, use_text_only=use_text_only)

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

if __name__ == "__main__":
    print("🔍 Running product matching pipeline...")
    # Example: run with no input to test default behavior
    results = run_matching_pipeline()
    print(results)
