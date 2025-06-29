setup:
	poetry install

run:
	python scripts/start_pipeline.py

load-data:
	python scripts/load_embeddings.py

demo:
	python app/demo_gradio.py

docker-build:
	docker-compose buisetup:
	poetry install

run:
	python scripts/start_pipeline.py

load-data:
	python scripts/load_embeddings.py

demo:
	python app/demo_gradio.py

docker-build:
	docker-compose build

docker-up:
	docker-compose upld

docker-up:
	docker-compose up
