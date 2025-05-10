from typing import cast

import ai_edge_torch
import torch
from ai_edge_torch.generative.quantize import quant_recipes
from const import OUTPUTS
from transformers.modeling_utils import PreTrainedModel
from transformers.models.vision_encoder_decoder import VisionEncoderDecoderModel


def convert_to_pt2():
    model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")
    image_tensor = torch.randn((1, 3, 224, 224), dtype=torch.float32)
    tokens_tensor = torch.tensor([[2]], dtype=torch.int)
    exported = torch.export.export(model, (image_tensor, tokens_tensor))
    torch.export.save(exported, OUTPUTS / "manga-ocr.pt2")


def _convert_encoder(model: VisionEncoderDecoderModel | PreTrainedModel):
    encoder = cast(PreTrainedModel, model.encoder).eval()
    print("encoder=", encoder.config, encoder)

    quant_config = quant_recipes.full_int8_dynamic_recipe()

    image_tensor = torch.randn((1, 3, 224, 224), dtype=torch.float32)
    edge_model = (
        ai_edge_torch.signature("encode", encoder, (image_tensor,))
        # .signature("decode", decoder, decoder.dummy_inputs.values())
        .convert(quant_config=quant_config)
    )

    output = OUTPUTS / "manga-ocr.converted.encoder.tflite"
    print(edge_model)
    edge_model.export(str(output))
    print(output, "->", output.stat().st_size)
    return output


def _convert_decoder(model: VisionEncoderDecoderModel | PreTrainedModel):
    decoder = cast(PreTrainedModel, model.decoder).eval()
    print("decoder=", decoder.config, decoder)
    print(model)

    quant_config = quant_recipes.full_int8_dynamic_recipe()

    encoder_hidden_states = torch.randn((1, 1, 768), dtype=torch.float32)
    input_token_ids = torch.tensor([[2]], dtype=torch.float32)
    edge_model = (
        ai_edge_torch.signature(
            "decode",
            decoder,
            sample_kwargs={
                "input_ids": input_token_ids,
                "encoder_hidden_states": encoder_hidden_states,
            },
        )
        # .signature("decode", decoder, decoder.dummy_inputs.values())
        .convert(quant_config=quant_config)
    )

    output = OUTPUTS / "manga-ocr.converted.decoder.tflite"
    print(edge_model)
    edge_model.export(str(output))
    print(output, "->", output.stat().st_size)
    return output


def convert():
    model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")

    # decode_input_tensor = torch.zeros((1, 768), dtype=torch.float32)

    print("Converting...")
    _convert_encoder(model)
    _convert_decoder(model)

    # # Dummy input for the model
    # dummy_image = torch.randn(1, 3, 224, 224)
    # dummy_token_ids = torch.tensor([[2]])

    # # Export the model
    # onnx_model = torch.onnx.export(
    #     model,
    #     (dummy_image, dummy_token_ids),
    #     input_names=["image", "token_ids"],
    #     output_names=["logits"],
    #     dynamic_axes={
    #         "image": {
    #             0: "batch_size",
    #         },
    #         "token_ids": {
    #             0: "batch_size",
    #             1: "sequence_length",
    #         },
    #         "logits": {
    #             0: "batch_size",
    #             1: "sequence_length",
    #         },
    #     },
    #     dynamo=True,
    # )

    # output = OUTPUTS / "manga-ocr.dynamo.onnx"
    # onnx_model.optimize()
    # onnx_model.save(output)

    # print(output, output.stat().st_size)

    # # Export the model
    # dummy_image = torch.randn(1, 3, 224, 224)
    # dummy_token_ids = torch.tensor([[2]])
    # onnx_encoder_model = torch.onnx.export(
    #     model,
    #     (dummy_image, dummy_token_ids),
    #     input_names=["image", "token_ids"],
    #     output_names=["logits"],
    #     dynamic_axes={
    #         "image": {
    #             0: "batch_size",
    #         },
    #         "token_ids": {
    #             0: "batch_size",
    #             1: "sequence_length",
    #         },
    #         "logits": {
    #             0: "batch_size",
    #             1: "sequence_length",
    #         },
    #     },
    # )

    # output = OUTPUTS / "manga-ocr.dynamo.onnx"
    # onnx_model.optimize()
    # onnx_model.save(output)

    # print(output, output.stat().st_size)
