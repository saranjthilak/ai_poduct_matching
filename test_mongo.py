# test_triton.py
from PIL import Image
import numpy as np
from scripts.start_pipeline import preprocess_image, infer_with_triton

img = Image.open("sample_data/images/shoe1.jpg").convert("RGB")
img_tensor = preprocess_image(img)
embedding = infer_with_triton(img_tensor)

print("[✅] Triton returned embedding of shape:", embedding.shape)
