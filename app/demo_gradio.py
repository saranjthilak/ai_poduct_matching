import gradio as gr
from PIL import Image
from utils.embedding_utils import ClipEmbedder
from vector_db.engine import VectorEngine

embedder = ClipEmbedder()
engine = VectorEngine()
engine.load()

def match_product(image: Image.Image):
    embedding = embedder.encode_image(image)
    matches = engine.search(embedding)
    return "\n\n".join([f"{m['name']} - {m['price']}€" for m in matches])

demo = gr.Interface(fn=match_product, inputs="image", outputs="text", title="AI Product Matcher")
demo.launch()
