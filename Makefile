run:
	poetry run python app/demo_gradio.py

load-data:
	poetry run python scripts/load_embeddings.py

start-pipeline:
	poetry run python scripts/start_pipeline.py

lint:
	black . && isort .
