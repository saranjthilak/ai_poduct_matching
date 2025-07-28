
# 🛍️ AI Product Matching System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-supported-blue.svg)](https://www.docker.com/)
[![NVIDIA Triton](https://img.shields.io/badge/NVIDIA-Triton-green.svg)](https://developer.nvidia.com/nvidia-triton-inference-server)

An end-to-end product matching pipeline that uses image and text inputs to find visually and semantically similar products from a given dataset.

Built using CLIP-based vision-language models, FAISS vector search, and served using NVIDIA Triton. This project showcases how multi-modal embeddings can power intelligent product discovery, supported by a responsive Gradio UI.

---

## 🎯 Key Highlights

- 🔍 **Multi-modal Product Matching**: Combines image and text embeddings using CLIP and BERT to enable robust similarity matching.
- ⚡ **Fast Vector Search**: Uses FAISS for efficient nearest-neighbor search on high-dimensional vectors.
- 🧠 **Inference with Triton**: Accelerated model serving with NVIDIA Triton Inference Server (ONNX + TensorRT).
- 🗃️ **Data Persistence**: MongoDB for storing and retrieving product metadata.
- 🌐 **Interactive WebUI**: Gradio frontend to test matching results quickly.
- 🐳 **Fully Containerized**: Docker support with `docker-compose` for local or cloud deployment.
- 📦 **Poetry-powered**: Clean dependency management and reproducible builds with Poetry.

---

## 📦 Sample Data

The repo comes with:

- ✅ A preloaded `products.json` for MongoDB.
- ✅ FAISS index (`product_index.faiss`) with embeddings.
- ✅ Sample product images grouped by categories (e.g., laptops, shoes, watches).
- ✅ ONNX and quantized TRT models ready to deploy via Triton.

---

## 🔗 Live Demo

Here's how the UI looks in action:
https://www.loom.com/share/1e75a628ff444e8da99317bc897fb2bc?sid=0940d9c5-d1a7-493d-871f-21429d45a47f

> To try locally, run:
```bash
poetry run python app/ui.py
```
Open: `http://localhost:7860` to interact with the interface.

---

## 🏁 Getting Started

### 📥 Clone the Repo
```bash
git clone https://github.com/saranjthilak/ai_poduct_matching.git
cd ai_poduct_matching
```

### 🐍 Setup Environment
```bash
poetry install
poetry shell
```

### 🐳 Docker Deployment
```bash
docker-compose up --build
```

This will start:
- MongoDB
- Triton Inference Server
- Gradio app

Visit:
- Gradio: [http://localhost:7860](http://localhost:7860)
- Triton: [http://localhost:8001](http://localhost:8001)

---

## 💻 How It Works

1. An image is uploaded via the Gradio UI.
2. The CLIP encoder (served by Triton) generates an image embedding.
3. FAISS retrieves the top N similar vectors from the index.
4. MongoDB provides metadata for matched products.
5. Matched results are shown with name, category, price, and similarity score.

---

## 🧪 Try It Out

```bash
# Run the Gradio app
poetry run python app/ui.py
```

Upload an image like `testimage.jpg`, and the system will show you the closest product matches from the sample database.

---

## 🧰 Scripts and Utilities

- `scripts/infer_clip.py`: Run CLIP inference locally or remotely
- `scripts/quantize_tensorrt.py`: Convert ONNX to TRT engines
- `scripts/load_embeddings.py`: Generate or reload FAISS index
- `scripts/start_pipeline.py`: One-shot embedding + indexing setup

---

## 🧠 Model Details

- Vision encoder: CLIP ViT (ONNX, FP16 quantized)
- Text encoder: BERT (ONNX)
- Quantization: TensorRT
- Serving: NVIDIA Triton Inference Server

---

## 🗂️ Folder Structure (Simplified)

```
├── app/               # Gradio UI and frontend logic
├── mongo_store/       # MongoDB handler
├── vector_db/         # FAISS index and search logic
├── scripts/           # Setup and utility scripts
├── triton_models/     # Inference model repo (Triton format)
├── sample_data/       # Product images, metadata, embeddings
├── docker/            # Dockerfile(s)
├── poetry.lock        # Poetry dependency lock
├── pyproject.toml     # Poetry project config
```

---

## 📊 Benchmarks

| Component       | Latency     |
|----------------|-------------|
| Inference       | ~45 ms      |
| Vector Search   | ~5 ms       |
| End-to-end Match| < 100 ms    |

> Tested on a local machine with NVIDIA RTX GPU.

---

## 📄 License

Licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [FAISS by Facebook](https://github.com/facebookresearch/faiss)

---

> Built with ❤️ by [Saran Jaya Thilak](https://github.com/saranjthilak)
