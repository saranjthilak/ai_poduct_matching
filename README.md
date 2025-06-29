# 🧠 AI Infrastructure Challenge – Product Matching System

Welcome to my submission for the **AI Infrastructure Coding Challenge**, where I designed and built a full **image-to-product matching pipeline** leveraging state-of-the-art **Visual Language Models (VLMs)**, **Vector Databases**, and **MongoDB**, deployed using **NVIDIA Triton Inference Server**.

---

## 🚀 Overview

This project demonstrates an **end-to-end product matching system** that:
- Accepts an input product image
- Extracts visual + textual embeddings via a quantized VLM
- Retrieves the closest match from a **vector database**
- Fetches product metadata from **MongoDB**
- Logs operations and errors via a **MongoDB logging service**

---

## 🏗️ Architecture

```plaintext
              +-------------------------+
              |   Input Product Image   |
              +-----------+-------------+
                          |
                          v
           +------------------------------+
           | Triton Inference Server (VLM)|
           | - Vision Encoder (e.g., CLIP)|
           | - Text Encoder (e.g., BERT)  |
           +-------------+----------------+
                         |
     +-------------------+-------------------+
     |                                   |
     v                                   v
+------------+                +-------------------+
| Vector DB  |                |    MongoDB Store   |
| (e.g. FAISS)| <------------ |  Product Metadata  |
+------------+                +-------------------+
     |
     v
+---------------------+
| Nearest Neighbor     |
| Product Match Result |
+---------------------+
