from pathlib import Path

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    matmul_4bits_quantizer,
    quant_utils,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.preprocess import quant_pre_process

MODE = "dynamic"

_model_fp32 = "./opt1.onnx"


def _output_path(input: Path, variant: str) -> Path:
    base = input.name.removesuffix(".onnx")
    print("output_name=", base, input.parent)
    return input.parent / f"{base}.{variant}.onnx"


def preprocess(input: Path):
    output = _output_path(input, "preprocessed")
    if not output.exists():
        print(f"Preprocess {input} -> {output}")
        quant_pre_process(input, output, auto_merge=True)
    return output


def int4(input: Path) -> Path:
    quant_config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=128,  # 2's exponential and >= 16
        is_symmetric=True,  # if true, quantize to Int4. otherwsie, quantize to uint4.
        accuracy_level=4,  # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=(
            "MatMul",
            "Gather",
        ),  # specify which op types to quantize
        quant_axes=(("MatMul", 0), ("Gather", 1)),
    )

    model = quant_utils.load_model_with_shape_infer(input)
    quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(model, algo_config=quant_config)
    quant.process()

    output = _output_path(input, "int4")
    quant.model.save_model_to_file(output)
    return output


def dynamic(input: Path) -> Path:
    output = _output_path(input, "quant")
    print(f"Performing dynamic quant {input} -> {output}")
    quantize_dynamic(
        input,
        output,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=["/encoder/embeddings/patch_embeddings/projection/Conv_quant"],
        op_types_to_quantize=[
            "MatMul",
            "Attention",
            "LSTM",
            "Gather",
            "Transpose",
            "EmbedLayerNormalization",
        ],
    )
    print(output.name, ":", output.stat().st_size)
    return output


def static(input: Path, calibration: CalibrationDataReader) -> Path:
    print("Static-quantize:", input)
    # Assuming we'll run on CPU...
    q_static_opts = {"ActivationSymmetric": False, "WeightSymmetric": True}
    output = _output_path(input, "static")
    quantize_static(input, output, calibration, extra_options=q_static_opts)
    print(output.name, ":", output.stat().st_size)
    return output


if __name__ == "__main__":
    match MODE:
        case "int4":
            int4(Path(_model_fp32))

        case "dynamic":
            dynamic(Path(_model_fp32))

        case _:
            raise Exception("Invalid mode")
