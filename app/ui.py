import gradio as gr
from PIL import Image
import os
from scripts.start_pipeline import run_matching_pipeline

def search_products(image):
    matches = run_matching_pipeline(image, top_k=5)
    results = []
    for match in matches:
        label = f"{match['name']} | {match['category']} | ${match['price']:.2f}"
        abs_img_path = os.path.abspath(f"sample_data/images/{match['image']}")
        if os.path.exists(abs_img_path):
            results.append((abs_img_path, label))
        else:
            print(f"⚠️ Image not found: {abs_img_path}")
    return results

with gr.Blocks() as demo:
    gr.Markdown("# 🛍️ Product Matching Demo")

    input_image = gr.Image(type="pil", label="📸 Upload Product Image")
    search_button = gr.Button("🔍 Search")
    gallery = gr.Gallery(label="Top Matches", columns=3, rows=2)

    search_button.click(fn=search_products, inputs=input_image, outputs=gallery)

demo.launch(share=True)
