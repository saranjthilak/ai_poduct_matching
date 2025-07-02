import requests
import numpy as np
from PIL import Image

def infer_clip_vision(image_path: str, triton_url: str = "http://localhost:8000/v2/models/clip_vision/infer"):
    """
    Sends an image to Triton Inference Server for the clip_vision model and returns the output.

    Args:
        image_path (str): Path to the input image file.
        triton_url (str): URL of the Triton inference endpoint.

    Returns:
        dict: JSON response from Triton server containing inference results.
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img).astype(np.float32)

    # Optional: normalize pixels if your model expects it (uncomment if needed)
    # img_np /= 255.0

    # Convert HWC to CHW
    img_nchw = np.transpose(img_np, (2, 0, 1))

    # Add batch dimension
    input_tensor = np.expand_dims(img_nchw, axis=0)  # Shape: (1, 3, 224, 224)

    # Flatten to list
    input_data = input_tensor.flatten().tolist()

    # Prepare Triton inference request payload
    payload = {
        "inputs": [
            {
                "name": "input_image",
                "shape": list(input_tensor.shape),
                "datatype": "FP32",
                "data": input_data
            }
        ]
    }

    # Make POST request to Triton server
    response = requests.post(triton_url, json=payload)
    response.raise_for_status()  # Raise exception if request failed

    return response.json()

# Example usage
if __name__ == "__main__":
    result = infer_clip_vision("sample_data/images/example.jpg")
    print("Triton response:", result)
