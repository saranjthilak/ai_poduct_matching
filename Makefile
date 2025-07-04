# === Paths ===
ONNX_DIR=onnx_models
ENGINE_DIR_VISION=triton_models/clip_vision/1
ENGINE_DIR_TEXT=triton_models/clip_text/1
TRITON_REPO=triton_models

VISION_ONNX=$(ONNX_DIR)/clip_vision.onnx
TEXT_ONNX=$(ONNX_DIR)/clip_text.onnx

VISION_ENGINE=$(ENGINE_DIR_VISION)/model.plan
TEXT_ENGINE=$(ENGINE_DIR_TEXT)/model.plan

.PHONY: all export_onnx quantize_trt start_triton stop_triton run_pipeline clean

## Export CLIP vision & text encoders to ONNX
export_onnx:
	@echo "📦 Exporting CLIP vision and text encoders to ONNX..."
	poetry run python scripts/export_clip_onnx.py

## Quantize ONNX to TensorRT engine for both vision and text (FP16)
quantize_trt:
	@echo "⚙️ Quantizing vision encoder with TensorRT (FP16) via Python script..."
	poetry run python scripts/quantize_tensorrt.py --onnx $(VISION_ONNX) --engine $(VISION_ENGINE) --model_type vision --fp16

	@echo "⚙️ Quantizing text encoder with TensorRT (FP16) via Python script..."
	poetry run python scripts/quantize_tensorrt.py --onnx $(TEXT_ONNX) --engine $(TEXT_ENGINE) --model_type text --fp16

## Start Triton Inference Server, MongoDB, and any other services from docker-compose.yml
start_triton:
	@echo "🚀 Starting Triton Inference Server and MongoDB containers..."
	docker-compose up -d

## Stop all running containers started by docker-compose
stop_triton:
	@echo "🛑 Stopping all containers..."
	docker-compose down

## Run the product matching pipeline script
run_pipeline:
	@echo "🔍 Running product matching pipeline..."
	poetry run python scripts/start_pipeline.py

## Clean generated ONNX and TensorRT engine files
clean:
	rm -rf $(ONNX_DIR) $(ENGINE_DIR_VISION)/model.plan $(ENGINE_DIR_TEXT)/model.plan
	@echo "🧹 Cleaned ONNX and TensorRT engine files."
