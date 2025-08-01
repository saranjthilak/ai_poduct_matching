# Add Poetry's install location to PATH so make finds it
export PATH := $(HOME)/.local/bin:$(PATH)

# === Paths ===
ONNX_DIR=onnx_models
ENGINE_DIR=triton_models/clip_vision/1
TRITON_REPO=triton_models
VISION_ONNX=$(ONNX_DIR)/clip_vision.onnx
TEXT_ONNX=$(ONNX_DIR)/clip_text.onnx
VISION_ENGINE=$(ENGINE_DIR)/model.plan
TEXT_ENGINE=triton_models/clip_text/1/model.plan

.PHONY: all export_onnx quantize_trt start_triton stop_triton run_pipeline clean generate_embeddings demo show_logs help

## Run full pipeline
all: run_pipeline

## Export CLIP vision & text encoders to ONNX
export_onnx:
	@echo "📦 Exporting CLIP vision and text encoders to ONNX..."
	poetry run python scripts/export_clip_onnx.py

## Quantize ONNX to TensorRT engine for both vision and text (FP16)
quantize_trt:
	@echo "⚙️ Quantizing CLIP vision encoder to TensorRT..."
	poetry run python scripts/quantize_tensorrt.py --onnx $(VISION_ONNX) --engine $(VISION_ENGINE) --model_type vision --fp16
	@echo "⚙️ Quantizing CLIP text encoder to TensorRT..."
	poetry run python scripts/quantize_tensorrt.py --onnx $(TEXT_ONNX) --engine $(TEXT_ENGINE) --model_type text --fp16

## Start Triton Inference Server and MongoDB
start_triton:
	@echo "🚀 Starting Triton Inference Server and MongoDB containers..."
	sudo docker compose up -d

## Stop all running containers
stop_triton:
	@echo "🛑 Stopping all containers..."
	sudo docker compose down

## Generate embeddings from product images
generate_embeddings:
	@echo "🧠 Generating image embeddings..."
	PYTHONPATH=$(PWD) poetry run python scripts/load_embeddings.py

## Run the product matching pipeline (depends on embeddings)
run_pipeline: generate_embeddings
	@echo "🔍 Running product matching pipeline..."
	PYTHONPATH=$(PWD) poetry run python scripts/start_pipeline.py

## Launch Gradio demo UI
demo:
	@echo "🎛️ Launching Gradio interface..."
	PYTHONPATH=$(PWD) poetry run python app/ui.py

## Show pipeline logs from MongoDB
show_logs:
	@echo "🪵 Fetching pipeline logs from MongoDB..."
	docker exec -it mongodb mongosh product_db --eval 'db.pipeline_logs.find().pretty()'

## Clean generated ONNX and engine files
clean:
	@echo "🧹 Cleaning ONNX and TensorRT engine files..."
	rm -rf $(ONNX_DIR) $(VISION_ENGINE) $(TEXT_ENGINE)

## Help: list all make targets
help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
