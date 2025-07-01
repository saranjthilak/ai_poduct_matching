# embedding_utils.py
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

class ClipEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode_image(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        return embedding.cpu().numpy().flatten()

    def encode_text(self, text: str):
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
        return embedding.cpu().numpy().flatten()
