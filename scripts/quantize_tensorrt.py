import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path: str, engine_file_path: str, fp16_mode: bool = True):
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

    print(f"🔧 Building engine from {onnx_file_path} (FP16={fp16_mode})")

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if fp16_mode and builder.platform_has_fast_fp16:
        print("✅ FP16 supported on this platform. Enabling FP16 mode.")
        config.set_flag(trt.BuilderFlag.FP16)

    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ Failed to parse ONNX model:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()

    # Assuming your model input name is "input_image" (check with Netron or model summary)
    # You need to specify min, opt, max shapes for each dynamic input dimension
    profile.set_shape("input_image", (1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224))

    # Add the profile to config
    config.add_optimization_profile(profile)

    # Build serialized engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("❌ Engine build failed.")
        return

    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    print(f"✅ Engine saved to {engine_file_path}")

if __name__ == "__main__":
    ONNX_PATH = "onnx_models/clip_vision.onnx"
    ENGINE_PATH = "triton_models/clip_vision/1/model.plan"

    build_engine(ONNX_PATH, ENGINE_PATH, fp16_mode=True)
