from utils.embedding_utils import ClipEmbedder
from vector_db.engine import FaissVectorEngine
from mongo_store.database import MongoDB
from PIL import Image
import numpy as np
import json
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file variables into environment

mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
print("Mongo URI is:", mongo_uri)

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

    try:
        db.insert_products(products)
    except Exception as e:
        print(f"Error inserting products into MongoDB: {e}")
        return

    # Initialize embedder for query
    embedder = ClipEmbedder()

    # Load query image and encode
    query_img_path = os.path.join("sample_data", "images", products[0]["image"])  # example using first product image
    query_img = Image.open(query_img_path).convert("RGB")
    query_embedding = embedder.encode_image(query_img).astype("float32")

    # Search top 5 matches
    top_matches = engine.search(query_embedding, top_k=5)

    print("Top matches:")
    for i, match in enumerate(top_matches, 1):
        print(f"{i}. {match['name']} - {match['category']} - ${match['price']}")

    print("✅ Search completed successfully! 🚀")

if __name__ == "__main__":
    main()
