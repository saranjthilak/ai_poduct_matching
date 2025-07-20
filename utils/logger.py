from pymongo import MongoClient
from datetime import datetime
import traceback
import os

# Replace with your Docker MongoDB connection string if needed
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "ai_product_matching"
COLLECTION_NAME = "logs"

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

def log_event(event_type, message, metadata=None):
    entry = {
        "timestamp": datetime.utcnow(),
        "type": event_type,
        "message": message,
        "metadata": metadata or {},
    }
    collection.insert_one(entry)

def log_error(error, context=None):
    entry = {
        "timestamp": datetime.utcnow(),
        "type": "error",
        "message": str(error),
        "traceback": traceback.format_exc(),
        "context": context or {},
    }
    collection.insert_one(entry)
