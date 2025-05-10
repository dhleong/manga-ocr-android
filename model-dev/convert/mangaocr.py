from typing import cast

import ai_edge_torch
import torch
from ai_edge_torch.generative.quantize import quant_recipes
from const import OUTPUTS
from quant import ops
from torch.export import Dim
from transformers.modeling_utils import PreTrainedModel
from transformers.models.vision_encoder_decoder import VisionEncoderDecoderModel


def convert_to_pt2():
    model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")
    image_tensor = torch.randn((1, 3, 224, 224), dtype=torch.float32)
    tokens_tensor = torch.tensor([[2]], dtype=torch.int)
    exported = torch.export.export(model, (image_tensor, tokens_tensor))
    torch.export.save(exported, OUTPUTS / "manga-ocr.pt2")


def _convert_encoder(
    model: VisionEncoderDecoderModel | PreTrainedModel, tflite: bool = False
):
    encoder = cast(PreTrainedModel, model.encoder).eval()
    print("encoder=", encoder.config, encoder)
    image_tensor = torch.randn((1, 3, 224, 224), dtype=torch.float32)

    if tflite:
        quant_config = quant_recipes.full_int8_dynamic_recipe()

        edge_model = (
            ai_edge_torch.signature("encode", encoder, (image_tensor,))
            # .signature("decode", decoder, decoder.dummy_inputs.values())
            .convert(quant_config=quant_config)
        )

        output = OUTPUTS / "manga-ocr.converted.encoder.tflite"
        print(edge_model)
        edge_model.export(str(output))
    else:
        output = OUTPUTS / "manga-ocr.converted.encoder.onnx"
        torch.onnx.export(
            encoder,
            f=str(output),
            kwargs={
                "pixel_values": image_tensor,
            },
            input_names=["pixel_values"],
            output_names=["encoder_hidden_states"],
            dynamic_axes={
                "pixel_values": {
                    0: "batch_size",
                },
                "encoder_hidden_states": {
                    0: "batch_size",
                },
            },
        )

    print(output, "->", output.stat().st_size)
    return output


def _convert_decoder(
    model: VisionEncoderDecoderModel | PreTrainedModel, tflite: bool = False
):
    decoder = cast(PreTrainedModel, model.decoder).eval()
    print("decoder=", decoder.config, decoder)
    print(model)

    quant_config = quant_recipes.full_int8_dynamic_recipe()

    # TODO: Not sure where this 197 comes from, but something to do
    # with 16x16 kernel size * 12 layers...?
    hidden_states_dimen = 197
    encoder_hidden_states = torch.randn(
        (1, hidden_states_dimen, 768), dtype=torch.float32
    )
    input_token_ids = torch.randint(low=0, high=6144, size=(1, 300), dtype=torch.int64)
    # input_token_ids = torch.tensor([[2]], dtype=torch.int32)

    if tflite:
        edge_model = (
            ai_edge_torch.signature(
                "decode",
                decoder,
                sample_kwargs={
                    "input_ids": input_token_ids,
                    "encoder_hidden_states": encoder_hidden_states,
                },
                dynamic_shapes={
                    "input_ids": {
                        # 0: batch_size,
                        # 1: Dim.DYNAMIC,
                        1: Dim("sequence_length", min=1, max=300),
                        # 0: "batch_size",
                        # 1: "sequence_length",
                    },
                    "encoder_hidden_states": {
                        # 0: batch_size
                    },
                },
            )
            # .signature("decode", decoder, decoder.dummy_inputs.values())
            # NOTE: May need to patch ai_edge_torch/odml_torch/export.py
            # rewrite_arange to make this work:
            .convert(quant_config=quant_config)
        )
        output = OUTPUTS / "manga-ocr.converted.decoder.tflite"
        print(edge_model)
        edge_model.export(str(output))
    else:
        output = OUTPUTS / "manga-ocr.converted.decoder.onnx"
        torch.onnx.export(
            decoder,
            f=str(output),
            kwargs={
                "encoder_hidden_states": encoder_hidden_states,
                "input_ids": input_token_ids,
            },
            input_names=["input_ids", "encoder_hidden_states"],
            output_names=["logits"],
            dynamic_axes={
                "encoder_hidden_states": {
                    0: "batch_size",
                },
                "input_ids": {
                    0: "batch_size",
                    # 1: Dim.DYNAMIC,
                    1: "sequence_length",
                    # 0: "batch_size",
                    # 1: "sequence_length",
                },
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
        )

    print("Converted:", output, "->", output.stat().st_size)
    return output


def convert(tflite: bool = False):
    model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")

    print("Converting...")
    encoder_path = _convert_encoder(model, tflite=tflite)
    decoder_path = _convert_decoder(model, tflite=tflite)

    if not tflite:
        ops.dynamic(ops.preprocess(encoder_path))
        ops.dynamic(ops.preprocess(decoder_path))
