import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import clip
import torch

class TritonCLIPTextEmbedder:
    def __init__(self, url="localhost:8000", model_name="clip_text", model_version="1"):
        self.client = httpclient.InferenceServerClient(url=url)
        self.model_name = model_name
        self.model_version = model_version

    def tokenize(self, texts: list[str]) -> np.ndarray:
        # Uses OpenAI's tokenizer
        tokens = clip.tokenize(texts)  # [batch, 77], dtype=torch.int64
        return tokens.numpy().astype(np.int32)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        input_ids = self.tokenize(texts)

        inputs = httpclient.InferInput("input_text", input_ids.shape, "INT32")
        inputs.set_data_from_numpy(input_ids)

        outputs = httpclient.InferRequestedOutput("text_features")

        try:
            response = self.client.infer(
                self.model_name,
                inputs=[inputs],
                outputs=[outputs],
                model_version=self.model_version
            )
            return response.as_numpy("text_features")  # shape: [batch, 512]

        except InferenceServerException as e:
            print(f"[Triton Error - text]: {e}")
            return None
