import gradio as gr
from PIL import Image
from scripts.start_pipeline import run_matching_pipeline

def search_products(image):
    matches = run_matching_pipeline(image, top_k=5)
    results = []
    for match in matches:
        label = f"{match['name']} | {match['category']} | {match['price']}"
        img_path = f"sample_data/images/{match['image']}"
        results.append((label, img_path))  # Gradio expects (label, image) tuples
    return results

with gr.Blocks() as demo:
    gr.Markdown("# 🛍️ Product Matching Demo")

    input_image = gr.Image(type="pil", label="Upload Product Image")
    gallery = gr.Gallery(label="Top Matches", columns=3)
    search_button = gr.Button("Search")

    search_button.click(fn=search_products, inputs=input_image, outputs=gallery)

demo.launch(share=True)
