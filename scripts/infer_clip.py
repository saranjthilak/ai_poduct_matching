# scripts/infer_clip.py

import requests
import numpy as np
from PIL import Image

TRITON_URL = "http://localhost:8000/v2/models/clip_vision/infer"
IMAGE_PATH = "testimage.jpg"  # 🔁 Replace this

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.asarray(image).astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)  # HWC → CHW
    image = np.expand_dims(image, axis=0)  # (1, 3, 224, 224)
    return image

def infer(image_tensor):
    payload = {
        "inputs": [
            {
                "name": "input_image",
                "shape": list(image_tensor.shape),
                "datatype": "FP32",
                "data": image_tensor.flatten().tolist()
            }
        ],
        "outputs": [{"name": "image_features"}]
    }

    response = requests.post(TRITON_URL, json=payload)
    response.raise_for_status()
    result = response.json()

    output = np.array(result["outputs"][0]["data"]).reshape(result["outputs"][0]["shape"])
    return output

if __name__ == "__main__":
    img = preprocess_image(IMAGE_PATH)
    features = infer(img)
    print("✅ Image embedding (shape: {}):\n{}".format(features.shape, features))
    print("✅ Image embedding generated successfully! 🚀")
