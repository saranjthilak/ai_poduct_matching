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
```
## 🧩 Project Structure
```plaintext
ai-product-matching/
├── app/                     # Gradio or FastAPI demo interface
│   └── ui.py
│
├── docker/                  # Dockerfile & entrypoints
│   └── Dockerfile
│
├── mongo_store/             # MongoDB interface
│   ├── database.py          # Handles product & log storage
│   └── __init__.py
│
├── sample_data/             # Example product data
│   ├── products.json
│   ├── embeddings.npy
│   └── images/
│       └── ...
│
├── scripts/                 # Utility scripts and pipeline runners
│   ├── export_clip_onnx.py       # ✅ Export vision/text encoders to ONNX
│   ├── quantize_tensorrt.py      # ✅ Optional: Quantize via TensorRT Python API
│   └── start_pipeline.py        # Main product matching logic
│
├── triton_models/           # Model repo for NVIDIA Triton
│   ├── clip_vision/
│   │   ├── 1/
│   │   │   └── model.plan       # ✅ TensorRT engine (FP16 or INT8)
│   │   └── config.pbtxt
│   │
│   └── clip_text/           # Optional: if text encoder also quantized
│       ├── 1/
│       │   └── model.plan
│       └── config.pbtxt
│
├── vector_db/               # Vector DB logic (FAISS-based)
│   ├── faiss_engine.py
│   └── __init__.py
│
├── .env                     # ✅ Mongo URI and other secrets
├── Makefile                 # Convenience CLI
├── pyproject.toml           # Poetry dependencies
├── docker-compose.yml       # ✅ Triton + Mongo + Your App (to be added)
└── README.md

```
## ⚙️ Setup Instructions

```
1. 📦 Install dependencies
make setup
2. 🧠 Load sample data
make load-data
3. ▶️ Run pipeline
make run
4. 🎛️ Launch Gradio demo
make demo
```

---

### 🐳 Run with Docker

```markdown
🐳 Run with Docker

🛠️ Build images

make docker-build
make docker-up
```

---

### 🧪 Tech Stack

```markdown
## 🧪 Tech Stack

- **FastAPI** – API interface for embedding and matching
- **PyTorch** – VLM model execution
- **FAISS (mocked)** – Efficient similarity search
- **MongoDB** – Stores product metadata and logs
- **Triton Server** – Model serving platform
- **Gradio** – UI for uploading and testing matching
```

## 📝 License

This project is licensed under the [MIT License](LICENSE) © 2025 Saran Jaya Thilak
