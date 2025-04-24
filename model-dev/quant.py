# import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

model_fp32 = "./opt1.onnx"
model_quant = "manga-ocr.quant.onnx"

quantize_dynamic(
    model_fp32,
    model_quant,
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
