import json
from pymongo import MongoClient

# Load product metadata from JSON file
with open("vector_db/products.json", "r", encoding="utf-8") as f:
    products = json.load(f)

# Connect to MongoDB (local Docker)
client = MongoClient("mongodb://localhost:27017/")

# Choose DB and collection
db = client["productdb"]
collection = db["products"]

# Optional: clear existing entries
collection.delete_many({})

# Insert all product documents
collection.insert_many(products)

print(f"✅ Inserted {len(products)} products into MongoDB")
