from pathlib import Path

from onnxruntime.quantization import (
    QuantType,
    matmul_4bits_quantizer,
    quant_utils,
    quantize_dynamic,
)

MODE = "dynamic"

model_fp32 = "./opt1.onnx"


def output_name(variant: str) -> str:
    return f"manga-ocr.{variant}.onnx"


match MODE:
    case "int4":
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

        model = quant_utils.load_model_with_shape_infer(Path(model_fp32))
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            model, algo_config=quant_config
        )
        quant.process()
        quant.model.save_model_to_file(output_name("int4"))

    case "dynamic":
        quantize_dynamic(
            model_fp32,
            output_name("quant"),
            weight_type=QuantType.QInt8,
            nodes_to_exclude=[
                "/encoder/embeddings/patch_embeddings/projection/Conv_quant"
            ],
            op_types_to_quantize=[
                "MatMul",
                "Attention",
                "LSTM",
                "Gather",
                "Transpose",
                "EmbedLayerNormalization",
            ],
        )

    case _:
        raise Exception("Invalid mode")
