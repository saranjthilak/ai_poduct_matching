from typing import List, Dict, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file automatically

class MongoDB:
    def __init__(self, uri: Optional[str] = None, db_name: str = None):
        """
        Initialize MongoDB connection.

        Args:
            uri (str): MongoDB URI, fallback to env var MONGO_URI if None.
            db_name (str): Database name, fallback to env var MONGO_DB_NAME or "product_db" if None.
        """
        if uri is None:
            uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        if db_name is None:
            db_name = os.getenv("MONGO_DB_NAME", "product_db")

        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.products: Collection = self.db["products"]
        self.logs: Collection = self.db["logs"]

    def is_connected(self) -> bool:
        """Check if MongoDB connection is alive."""
        try:
            self.client.admin.command('ismaster')
            return True
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            return False

    def create_indexes(self):
        """Create unique index on product 'id' field to prevent duplicates."""
        try:
            self.products.create_index("id", unique=True)
            print("Created unique index on 'id' field.")
        except Exception as e:
            print(f"Error creating indexes: {e}")

    def insert_products(self, products: List[Dict]):
        """
        Clear existing products and insert new ones.

        Args:
            products (List[Dict]): List of product dicts.
        """
        try:
            # Clear existing products first
            deleted_count = self.products.delete_many({}).deleted_count
            print(f"Cleared {deleted_count} existing products.")

            # Insert new products
            self.products.insert_many(products, ordered=False)
            print(f"Inserted {len(products)} products successfully.")
        except Exception as e:
            print(f"Error inserting products: {e}")
            self.log_event({"type": "error", "message": str(e)})

    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """
        Retrieve a product by its unique 'id'.

        Args:
            product_id (str): Product unique id.

        Returns:
            Dict or None: Product document or None if not found.
        """
        return self.products.find_one({"id": product_id})

    def get_product_by_index(self, idx: int) -> Optional[Dict]:
        """
        Retrieve a product by its index or custom field 'index'.

        Args:
            idx (int): Product index.

        Returns:
            Dict or None: Product document or None if not found.
        """
        return self.products.find_one({"index": idx})

    def find_products(self, filter_dict: Dict) -> List[Dict]:
        """
        Generic find with filter.

        Args:
            filter_dict (Dict): MongoDB filter query.

        Returns:
            List[Dict]: List of product documents.
        """
        return list(self.products.find(filter_dict))

    def find_products_by_category(self, category: str) -> List[Dict]:
        """
        Find products by category.

        Args:
            category (str): Category name.

        Returns:
            List[Dict]: List of product documents.
        """
        return list(self.products.find({"category": category}))

    def update_product(self, product_id: str, update_fields: Dict) -> bool:
        """
        Update a product's fields.

        Args:
            product_id (str): Product unique id.
            update_fields (Dict): Fields to update.

        Returns:
            bool: True if updated, False otherwise.
        """
        try:
            result = self.products.update_one({"id": product_id}, {"$set": update_fields})
            return result.modified_count > 0
        except Exception as e:
            self.log_event({"type": "error", "message": str(e)})
            return False

    def delete_product(self, product_id: str) -> bool:
        """
        Delete a product by its id.

        Args:
            product_id (str): Product unique id.

        Returns:
            bool: True if deleted, False otherwise.
        """
        try:
            result = self.products.delete_one({"id": product_id})
            return result.deleted_count > 0
        except Exception as e:
            self.log_event({"type": "error", "message": str(e)})
            return False

    def log_event(self, event: Dict):
        """
        Insert a log event.

        Args:
            event (Dict): Log data, e.g., {"type": "error", "message": "...", "timestamp": ...}
        """
        try:
            self.logs.insert_one(event)
        except Exception as e:
            print(f"Error logging event: {e}")
