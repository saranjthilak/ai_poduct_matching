import torch
import clip
import os

def export_vision_encoder(model, device, output_path="clip_vision.onnx"):
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model.visual,
        dummy_input,
        output_path,
        input_names=["input_image"],
        output_names=["image_features"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "image_features": {0: "batch_size"},
        },
        opset_version=14,
        verbose=False
    )
    print(f"✅ Vision encoder exported to {output_path}")

def export_text_encoder(model, device, output_path="clip_text.onnx"):
    # Create a wrapper to run full text encoder from token input to output features
    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip_model = clip_model

        def forward(self, input_ids):
            # input_ids shape: [batch, seq_len], e.g. [1,77]
            x = self.clip_model.token_embedding(input_ids).type(self.clip_model.dtype)  # embed tokens
            x = x + self.clip_model.positional_embedding  # add positional embeddings
            x = x.permute(1, 0, 2)  # NLD -> LND for transformer
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

            # pick features at EOS token
            eos_pos = input_ids.argmax(dim=-1)
            x = x[torch.arange(x.size(0)), eos_pos]  # shape [batch, dim]

            x = x @ self.clip_model.text_projection

            return x

    dummy_text = clip.tokenize(["a photo of a dog"]).to(device)  # shape [1,77]

    text_encoder = TextEncoderWrapper(model).to(device).eval()

    torch.onnx.export(
        text_encoder,
        dummy_text,
        output_path,
        input_names=["input_text"],
        output_names=["text_features"],
        dynamic_axes={
            "input_text": {0: "batch_size", 1: "seq_length"},
            "text_features": {0: "batch_size"},
        },
        opset_version=14,
        verbose=False,
    )

    print(f"✅ Text encoder exported to {output_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on device: {device}")
    model, _ = clip.load("ViT-B/32", device=device)
    model = model.float()
    model.eval()

    os.makedirs("onnx_models", exist_ok=True)

    export_vision_encoder(model, device, output_path="onnx_models/clip_vision.onnx")
    export_text_encoder(model, device, output_path="onnx_models/clip_text.onnx")
