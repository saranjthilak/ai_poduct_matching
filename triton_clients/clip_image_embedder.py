import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from PIL import Image
import torchvision.transforms as T

class TritonCLIPImageEmbedder:
    def __init__(self, url="localhost:8000", model_name="clip_vision", model_version="1"):
        self.client = httpclient.InferenceServerClient(url=url)
        self.model_name = model_name
        self.model_version = model_version

        self.preprocess = T.Compose([
            T.Resize(224, interpolation=Image.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]
        return tensor.numpy().astype(np.float32)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        input_tensor = self.preprocess_image(image)

        inputs = httpclient.InferInput("input_image", input_tensor.shape, "FP32")
        inputs.set_data_from_numpy(input_tensor)

        outputs = httpclient.InferRequestedOutput("image_features")

        try:
            response = self.client.infer(
                self.model_name,
                inputs=[inputs],
                outputs=[outputs],
                model_version=self.model_version
            )
            return response.as_numpy("image_features")[0]  # shape: (512,)

        except InferenceServerException as e:
            print(f"[Triton Error - image]: {e}")
            return None
