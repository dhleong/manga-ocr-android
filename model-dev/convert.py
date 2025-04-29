from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import ai_edge_torch
import numpy
import torch
from ai_edge_torch.debug import find_culprits
from ai_edge_torch.generative.quantize import quant_recipes
from torch.export import Dim
from transformers.models.vision_encoder_decoder import VisionEncoderDecoderModel


@contextmanager
def progress(label: str, can_skip: bool = False):
    start = datetime.now()
    print(label)
    try:
        yield
        print(f"... done in {datetime.now() - start}")
    except Exception as e:
        print(f"... ERROR in {datetime.now() - start}", e)
        if not can_skip:
            raise


def run(output_path: Path):
    with progress("Grabbing manga-ocr-base model"):
        # As always, with thanks to mayocream/koharu for getting started with this
        model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")

    # Dummy input for the model
    dummy_image = torch.randn(1, 3, 224, 224)
    # dummy_token_ids = torch.tensor([[2]])
    # dummy_token_ids = torch.randn(1, 512, dtype=torch.long)
    vocab_size = 6144  # TODO: Extract from model, maybe?
    dummy_token_ids = torch.randint(0, vocab_size, size=(1, 300))
    sample_inputs = (dummy_image, dummy_token_ids)

    with progress("Checking torch output"):
        model.eval()
        torch_output = model(*sample_inputs)

    with progress("Converting to ai-edge"):
        quant_config = quant_recipes.full_int8_dynamic_recipe()
        # tfl_converter_flags = {"optimizations": [tf.lite.Optimize.DEFAULT]}
        culprits = find_culprits(
            model.eval(),
            sample_inputs,
        )
        culprit = next(culprits)
        culprit.print_code()

        edge_model = ai_edge_torch.convert(
            model,
            sample_inputs,
            quant_config=quant_config,
            # _ai_edge_converter_flags=tfl_converter_flags,
            # NOTE: We probably *need* this but it complains
            # that the inferred length is 1 so something is invalid...
            dynamic_shapes={
                "pixel_values": {
                    # 0: Dim("batch_size", max=1),
                    0: Dim.STATIC,
                },
                "decoder_input_ids": {
                    # 0: Dim("batch_size", max=1),
                    0: Dim.STATIC,
                    1: Dim("seq_length", min=1, max=300),
                    # 1: Dim.DYNAMIC,
                    # 1: Dim.DYNAMIC,
                },
            },
        )

    with progress("Evaluating conversion", can_skip=True):
        edge_output = edge_model(*sample_inputs)
        if numpy.allclose(
            torch_output.detach().numpy(),
            edge_output,  # type: ignore
            atol=1e-5,
            rtol=1e-5,
        ):
            print("Inference result with Pytorch and TfLite was within tolerance")
        else:
            print("Something wrong with Pytorch --> TfLite")

    with progress("Exporting model"):
        edge_model.export(str(output_path))
