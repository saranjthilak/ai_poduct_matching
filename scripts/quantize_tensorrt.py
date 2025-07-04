import tensorrt as trt
import os
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path: str, engine_file_path: str, input_name: str, input_shapes: tuple, fp16_mode: bool = True):
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

    # Unpack input_shapes: (min_shape, opt_shape, max_shape)
    min_shape, opt_shape, max_shape = input_shapes
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)

    # Add profile to config
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
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--engine", type=str, required=True, help="Output path for TensorRT engine")
    parser.add_argument("--model_type", type=str, choices=["vision", "text"], required=True, help="Model type: vision or text")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mode")

    args = parser.parse_args()

    if args.model_type == "vision":
        input_name = "input_image"
        # (min_batch, channels, height, width), etc
        input_shapes = (
            (1, 3, 224, 224),  # min
            (4, 3, 224, 224),  # opt
            (8, 3, 224, 224),  # max
        )
    else:  # text
        input_name = "input_text"
        # (min_batch, seq_len), etc
        input_shapes = (
            (1, 77),  # min
            (4, 77),  # opt
            (8, 77),  # max
        )

    build_engine(args.onnx, args.engine, input_name, input_shapes, fp16_mode=args.fp16)
