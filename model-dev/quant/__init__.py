from typing import Literal

import click
from const import OUTPUTS
from download import download_base_models
from quant import ops
from train import dataset


@click.group()
def quantize(): ...


@quantize.command()
def ogkalu():
    # import tensorflow as tf
    from transformers.models.auto.modeling_auto import AutoModel

    # Load the model
    model = AutoModel.from_pretrained("ogkalu/comic-text-and-bubble-detector")
    pretrained_path = OUTPUTS / "ogkalu.pretrained"
    model.save_pretrained(pretrained_path)

    # NOTE: THis doesn't work :(
    # tf_model = tf.saved_model.load(model, str(pretrained_path))
    # print("hi", tf_model)


@quantize.command()
def comictextdetector():
    download_base_models()

    processed = ops.preprocess(OUTPUTS / "comictextdetector.onnx")
    ops.dynamic(processed)


@quantize.command()
@click.argument("mode", type=click.Choice(["dynamic", "static"]))
def ocr(mode: Literal["dynamic", "static"]):
    download_base_models()

    processed = ops.preprocess(OUTPUTS / "manga-ocr.onnx")
    if mode == "dynamic":
        ops.dynamic(processed)

    elif mode == "static":
        manga109s_dir = dataset.download_manga109s()
        ops.static(
            processed, calibration=dataset.onnx_calibration_reader(manga109s_dir)
        )
