from utils.embedding_utils import ClipEmbedder
from PIL import Image
import json
import numpy as np
import os

embedder = ClipEmbedder()
image_dir = "sample_data/images"

with open("sample_data/products.json") as f:
    products = json.load(f)

embeddings = []

for product in products:
    image_path = os.path.join(image_dir, product["image"])
    img = Image.open(image_path).convert("RGB")
    embedding = embedder.encode_image(img)
    embeddings.append(embedding)

np.save("sample_data/embeddings.npy", np.array(embeddings))
print(f"✅ Embeddings for {len(products)} products have been saved successfully! 🚀")
