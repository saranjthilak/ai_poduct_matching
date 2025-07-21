from PIL import Image
import numpy as np
import json
import os

from triton_clients.clip_text_embedder import TritonCLIPTextEmbedder
from triton_clients.clip_image_embedder import TritonCLIPImageEmbedder

# Initialize Triton clients with URL and model names
text_embedder = TritonCLIPTextEmbedder(url="localhost:8000", model_name="clip_text")
image_embedder = TritonCLIPImageEmbedder(url="localhost:8000", model_name="clip_vision")

image_dir = "sample_data/images"

# Load product metadata
with open("sample_data/products.json") as f:
    products = json.load(f)

embeddings = []

for product in products:
    image_path = os.path.join(image_dir, product["image"])
    image = Image.open(image_path).convert("RGB")

    # Get image embedding (should be shape (512,))
    image_embedding = image_embedder.embed_image(image)

    # Use description if available, otherwise fallback to name
    text_input = product.get("description") or product.get("name", "")
    text_embedding = text_embedder.embed_text([text_input])

    if image_embedding is None or text_embedding is None or len(text_embedding) == 0:
        print(f"⚠️ Skipping product {product.get('name')} due to empty embedding.")
        continue

    # text_embedding is a list; take first element (512-dim)
    combined = np.concatenate([image_embedding, text_embedding[0]])

    embeddings.append(combined)

embeddings = np.array(embeddings, dtype=np.float32)
print(f"Combined embeddings shape: {embeddings.shape} (should be [num_products, 1024])")

# Save combined embeddings for indexing and search
np.save("sample_data/combined_embeddings.npy", embeddings)

print(f"✅ Saved {len(embeddings)} combined embeddings to sample_data/combined_embeddings.npy")
