# AI Product Matching System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-supported-blue.svg)](https://www.docker.com/)
[![NVIDIA Triton](https://img.shields.io/badge/NVIDIA-Triton-green.svg)](https://developer.nvidia.com/nvidia-triton-inference-server)

An end-to-end AI-powered product matching system that leverages Visual Language Models (VLMs), Vector Databases, and NVIDIA Triton Inference Server to find the closest product matches from input images.

## 🚀 Features

- **Multi-modal Embedding**: Extract both visual and textual features using CLIP and BERT models
- **Efficient Search**: Lightning-fast similarity search using FAISS vector database
- **Production Ready**: Quantized models deployed with NVIDIA Triton Inference Server
- **Scalable Architecture**: MongoDB for metadata storage and comprehensive logging
- **User-friendly Interface**: Interactive Gradio web interface for testing
- **Containerized Deployment**: Full Docker support for easy deployment

## 🏗️ Architecture

```
+-------------------------+
| Input Product Image     |
+-----------+-------------+
            |
            v
+------------------------------+
| Triton Inference Server      |
| - Vision Encoder (CLIP)      |
| - Text Encoder (BERT)        |
+-------------+----------------+
            |
+-----------+------------------+
|                             |
v                             v
+------------+    +-------------------+
| Vector DB  |    | MongoDB Store     |
| (FAISS)    |<---| Product Metadata  |
+------------+    +-------------------+
            |
            v
+---------------------+
| Nearest Neighbor    |
| Product Match       |
+---------------------+
```

## 📁 Project Structure

```
ai-product-matching/
├── app/                    # Gradio application
│   ├── main.py            # Gradio interface and main app
│   ├── matching.py        # Product matching logic
│   └── utils.py           # Utility functions
├── config/                 # Configuration files
│   ├── model_config.yaml  # Model configurations
│   └── server_config.yaml # Server configurations
├── docker/                 # Docker configurations
│   ├── Dockerfile         # Main application container
│   ├── triton.Dockerfile  # Triton server container
│   └── docker-compose.yml # Multi-container setup
├── models/                 # Model management
│   ├── encoders/          # Vision and text encoders
│   ├── quantization/      # Model quantization scripts
│   └── export/            # Model export utilities
├── storage/               # Data storage components
│   ├── vector_db/         # FAISS vector database
│   ├── mongo_store/       # MongoDB interface
│   └── cache/             # Caching layer
├── triton_models/         # Triton model repository
│   ├── clip_vision/       # Vision encoder model
│   │   ├── 1/
│   │   │   └── model.plan # TensorRT engine
│   │   └── config.pbtxt   # Model configuration
│   └── clip_text/         # Text encoder model
│       ├── 1/
│       │   └── model.plan
│       └── config.pbtxt
├── sample_data/           # Sample product data
│   ├── products.json      # Product metadata
│   ├── embeddings.npy     # Pre-computed embeddings
│   └── images/            # Sample product images
├── scripts/               # Utility scripts
│   ├── setup_data.py      # Data initialization
│   ├── benchmark.py       # Performance benchmarking
│   └── validate.py        # Model validation
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── performance/       # Performance tests
├── pyproject.toml         # Poetry dependencies and config
├── poetry.lock           # Poetry lock file
├── docker-compose.yml     # Container orchestration
├── Makefile              # Build and run commands
└── README.md             # This file
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **AI/ML**: PyTorch, CLIP, BERT, TensorRT
- **Vector DB**: FAISS
- **Database**: MongoDB
- **Model Serving**: NVIDIA Triton Inference Server
- **UI**: Gradio
- **Package Management**: Poetry
- **Containerization**: Docker, Docker Compose
- **Caching**: Redis (optional)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- Docker and Docker Compose
- NVIDIA GPU (for TensorRT acceleration)
- NVIDIA Container Toolkit

### 1. Clone the Repository

```bash
git clone https://github.com/saranjthilak/ai_poduct_matching.git
cd ai_poduct_matching
```

### 2. Environment Setup

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
# Update MongoDB URI, Triton server URL, etc.
```

### 4. Quick Start with Docker

```bash
# Build and start all services
docker-compose up --build

# Access the application
# - Gradio UI: http://localhost:7860
# - API: http://localhost:8000
# - Triton Server: http://localhost:8001
```

### 5. Manual Setup (Development)

```bash
# Start MongoDB
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Start Triton Server
docker run --gpus all -d -p 8001:8001 -p 8002:8002 -p 8000:8000 \
  -v $(pwd)/triton_models:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models

# Initialize sample data
poetry run python scripts/setup_data.py

# Start the Gradio application
poetry run python app/main.py
```

## 📊 Usage

### Gradio Web Interface

1. Start the application with `poetry run python app/main.py`
2. Open your browser to `http://localhost:7860`
3. Upload a product image using the file uploader
4. Click "Find Similar Products" button
5. View the matched results with similarity scores and product details

The Gradio interface provides:
- **Image Upload**: Drag and drop or click to upload product images
- **Results Display**: Visual grid showing matched products with scores
- **Product Details**: Name, category, price, and similarity percentage
- **Interactive Interface**: Real-time processing and results



## 🔧 Model Management

### Model Quantization

```bash
# Export models to ONNX
poetry run python models/export/export_clip.py --model openai/clip-vit-base-patch32

# Quantize with TensorRT
poetry run python models/quantization/quantize_tensorrt.py \
  --model models/clip_vision.onnx \
  --precision fp16 \
  --output triton_models/clip_vision/1/model.plan
```

### Adding New Models

1. Export your model to ONNX format
2. Quantize using TensorRT
3. Add model configuration to `triton_models/`
4. Update `config/model_config.yaml`
5. Restart Triton server

## 📈 Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| Inference Latency | ~45ms |
| Search Latency | ~5ms |
| Throughput | ~100 queries/sec |
| Memory Usage | ~2GB GPU |







## 🚀 Deployment

### Production Deployment

1. **Environment Variables**:
   ```bash
   export MONGO_URI="mongodb://your-mongo-cluster"
   export TRITON_URL="http://your-triton-server:8001"
   export REDIS_URL="redis://your-redis-server:6379"
   ```

2. **Docker Compose Production**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```





## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Common Issues

### Issue: Triton Server Not Starting
```bash
# Check GPU availability
nvidia-smi

# Verify model files
ls -la triton_models/*/1/

# Check Triton logs
docker logs triton-server
```

### Issue: MongoDB Connection Error
```bash
# Verify MongoDB is running
docker ps | grep mongo

# Check connection string
echo $MONGO_URI
```

### Issue: Low Inference Performance
- Ensure TensorRT models are properly quantized
- Check GPU memory usage
- Enable batching in Triton configuration



## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for the vision-language model
- [NVIDIA Triton](https://github.com/triton-inference-server/server) for model serving
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [MongoDB](https://www.mongodb.com/) for flexible data storage



---

**Built with ❤️ by Saran Jaya Thilak**
