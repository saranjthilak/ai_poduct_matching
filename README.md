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
├── app/                # Gradio demo interface
├── docker/             # Dockerfile
├── mongo_store/        # MongoDB interface
├── sample_data/        # Mock product metadata, embeddings, and images
├── scripts/            # Data loaders and pipeline starters
├── triton_models/      # Mock Triton model configs
├── vector_db/          # Vector DB interface
├── Makefile            # Convenient CLI commands
├── pyproject.toml      # Poetry dependencies
├── docker-compose.yml  # Multi-container orchestration
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







