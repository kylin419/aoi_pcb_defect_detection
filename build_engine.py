import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

ONNX_PATH = "best.onnx"
ENGINE_PATH = "best.engine"

def build_engine():
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        print("🔧 Loading ONNX...")
        with open(ONNX_PATH, "rb") as f:
            if not parser.parse(f.read()):
                print("❌ ONNX parse failed")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return None

        print("✅ ONNX parsed")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        if builder.platform_has_fast_fp16:
            print("⚡ Using FP16")
            config.set_flag(trt.BuilderFlag.FP16)


        profile = builder.create_optimization_profile()

        input_name = network.get_input(0).name
        profile.set_shape(input_name, (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
        config.add_optimization_profile(profile)

        print("Building engine (this may take a while)...")

        engine = builder.build_engine(network, config)

        if engine is None:
            print("Build failed")
            return None

        print("Engine built successfully")

        with open(ENGINE_PATH, "wb") as f:
            f.write(engine.serialize())

        print(f"Saved to {ENGINE_PATH}")

        return engine


if __name__ == "__main__":
    build_engine()