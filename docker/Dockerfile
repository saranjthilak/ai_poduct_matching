version: '3.9'

services:
  app:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: ai-matching-app
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    depends_on:
      - mongo

  mongo:
    image: mongo:6.0
    container_name: ai-matching-mongo
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

  triton:
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    container_name: triton-inference
    shm_size: 1g
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./triton_models:/models
    command: ["tritonserver", "--model-repository=/models"]
    ports:
      - "8001:8001" # gRPC
      - "8002:8002" # HTTP
      - "8003:8003" # Metrics

volumes:
  mongo_data:
