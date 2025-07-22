import os
import json
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from vector_db.engine import FaissVectorEngine
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

from transformers import CLIPProcessor, CLIPModel
import torch

load_dotenv()

TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL", "localhost:8000")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "clip_vision")
TRITON_INPUT_NAME = "input_image"
TRITON_OUTPUT_NAME = "image_features"

DATA_DIR = "sample_data/images"
PRODUCTS_META_PATH = "vector_db/products.json"
FAISS_COMBINED_INDEX_PATH = "vector_db/combined_index.faiss"
FAISS_TEXT_ONLY_INDEX_PATH = "vector_db/text_only_index.faiss"

# Initialize Triton client
triton_client = httpclient.InferenceServerClient(url=TRITON_HTTP_URL)

# Initialize CLIP text model and processor from Hugging Face
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

def main():
    with open(PRODUCTS_META_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)

    combined_embeddings = []
    text_only_embeddings = []

    for prod in products:
        img_path = os.path.join(DATA_DIR, prod["image"])
        image = Image.open(img_path).convert("RGB")

        image_tensor = preprocess_image(image)
        image_embedding = infer_with_triton(image_tensor).flatten()  # expected shape: (512,)

        text_embedding = get_clip_text_embedding(prod["description"]).flatten()  # expected shape: (512,)

        combined_embeddings.append(np.concatenate([image_embedding, text_embedding]))  # (1024,)
        text_only_embeddings.append(text_embedding)

    combined_embeddings = np.array(combined_embeddings, dtype=np.float32)
    text_only_embeddings = np.array(text_only_embeddings, dtype=np.float32)

    print(f"Combined embeddings shape: {combined_embeddings.shape} (should be [num_products, 1024])")
    print(f"Text-only embeddings shape: {text_only_embeddings.shape} (should be [num_products, 512])")

    engine = FaissVectorEngine(
        dim_combined=combined_embeddings.shape[1],  # 1024
        dim_text=text_only_embeddings.shape[1],    # 512
        index_path_combined=FAISS_COMBINED_INDEX_PATH,
        index_path_text=FAISS_TEXT_ONLY_INDEX_PATH,
        meta_path=PRODUCTS_META_PATH,
    )

    engine.index_data(
        embeddings_combined=combined_embeddings,
        embeddings_text=text_only_embeddings,
        products=products,
    )

if __name__ == "__main__":
    main()
