import onnx
import os

model_dir = "onnx_models"
for file in os.listdir(model_dir):
    if file.endswith(".onnx"):
        model_path = os.path.join(model_dir, file)
        model = onnx.load(model_path)
        print(f"Outputs for {file}: {[output.name for output in model.graph.output]}")
