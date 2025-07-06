import tensorrt as trt
import os
import argparse
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Example calibrator skeleton - customize for your dataset
class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, batch_size, input_shape, cache_file):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cache_file = cache_file

        # Assume calibration_data is a list or numpy array of inputs
        self.data = calibration_data
        self.current_index = 0

        # Allocate device memory for a batch
        self.device_input = None
        import pycuda.driver as cuda
        import pycuda.autoinit
        self.cuda = cuda
        self.device_input = cuda.mem_alloc(np.prod(input_shape) * batch_size * np.dtype(np.float32).itemsize)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.data):
            return None

        batch = self.data[self.current_index:self.current_index + self.batch_size].astype(np.float32).ravel()
        self.cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print("Using calibration cache")
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_engine(onnx_file_path: str, engine_file_path: str, input_name: str, input_shapes: tuple,
                 fp16_mode: bool = True, int8_mode: bool = False, calibrator=None):
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

    print(f"🔧 Building engine from {onnx_file_path} (FP16={fp16_mode}, INT8={int8_mode})")

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace

    if int8_mode:
        if not builder.platform_has_fast_int8:
            print("⚠️ INT8 not supported on this platform.")
        else:
            print("✅ INT8 supported. Enabling INT8 mode.")
            config.set_flag(trt.BuilderFlag.INT8)
            if calibrator is not None:
                config.int8_calibrator = calibrator

    if fp16_mode and builder.platform_has_fast_fp16:
        print("✅ FP16 supported. Enabling FP16 mode.")
        config.set_flag(trt.BuilderFlag.FP16)

    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ Failed to parse ONNX model:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

    profile = builder.create_optimization_profile()
    min_shape, opt_shape, max_shape = input_shapes
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

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
    parser.add_argument("--model_type", type=str, choices=["vision", "text"], required=True, help="Model type")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mode")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 mode (requires calibration)")

    args = parser.parse_args()

    calibrator = None
    if args.int8:
        # TODO: load or generate calibration data (numpy arrays)
        # Example dummy calibration data for demonstration:
        if args.model_type == "vision":
            input_shape = (3, 224, 224)
            # Create dummy calibration data: 10 batches of ones
            calibration_data = np.ones((10, *input_shape), dtype=np.float32)
            batch_size = 1
            calibrator = MyCalibrator(calibration_data, batch_size, input_shape, cache_file="calibration.cache")
        else:
            input_shape = (77,)
            calibration_data = np.ones((10, *input_shape), dtype=np.float32)
            batch_size = 1
            calibrator = MyCalibrator(calibration_data, batch_size, input_shape, cache_file="calibration.cache")

    if args.model_type == "vision":
        input_name = "input_image"
        input_shapes = (
            (1, 3, 224, 224),
            (4, 3, 224, 224),
            (8, 3, 224, 224),
        )
    else:
        input_name = "input_text"
        input_shapes = (
            (1, 77),
            (4, 77),
            (8, 77),
        )

    build_engine(args.onnx, args.engine, input_name, input_shapes,
                 fp16_mode=args.fp16, int8_mode=args.int8, calibrator=calibrator)
