import gradio as gr
from PIL import Image
import os
from scripts.start_pipeline import run_matching_pipeline

def search_products(image, description):
    if image is None and not description:
        return [(None, "⚠️ Please upload an image or provide a description.")]

    matches = run_matching_pipeline(input_image=image, input_text=description, top_k=5)
    results = []
    for match in matches:
        label = f"{match['name']} | {match['category']} | ${match['price']:.2f}"
        abs_img_path = os.path.abspath(f"sample_data/images/{match['image']}")
        if os.path.exists(abs_img_path):
            results.append((abs_img_path, label))
        else:
            print(f"⚠️ Image not found: {abs_img_path}")
    return results if results else [(None, "⚠️ No matches found.")]

with gr.Blocks(title="AI Product Matching") as demo:
    gr.Markdown("# 🛍️ AI Product Matching Demo")
    gr.Markdown("Upload an image and/or enter a product description to find similar products.")

    with gr.Row():
        input_image = gr.Image(
            type="pil",
            label="📸 Upload Product Image",
            image_mode="RGB",
            height=224
        )
        input_text = gr.Textbox(
            label="📝 Product Description (Optional)",
            placeholder="e.g. white sneakers for men"
        )

    with gr.Row():
        search_button = gr.Button("🔍 Search Products")
        clear_button = gr.Button("🧹 Clear")

    gallery = gr.Gallery(label="🎯 Top Matches", columns=3, rows=2, show_label=True)

    search_button.click(
        fn=search_products,
        inputs=[input_image, input_text],
        outputs=gallery,
        show_progress=True
    )

    clear_button.click(
        fn=lambda: (None, "", []),
        inputs=[],
        outputs=[input_image, input_text, gallery]
    )

demo.launch(share=True)
