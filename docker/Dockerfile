FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

CMD ["python", "app/demo_gradio.py"]
