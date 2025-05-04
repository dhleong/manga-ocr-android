import shutil
from pathlib import Path

import click
import download
from const import OUTPUTS


@click.group()
def convert(): ...


@convert.command()
def ogkalu_yolo():
    yolov8 = download.hf("ogkalu/manga-text-detector-yolov8s", "manga-text-detector.pt")

    import ultralytics

    model = ultralytics.YOLO(str(yolov8))
    # exported = model.export(format="onnx")
    # assert exported, "Failed to convert to onnx"
    # processed = ops.preprocess(Path(exported))
    # ops.dynamic(processed)

    export_modes = [
        dict(),
        dict(half=True),
        dict(int8=True),
    ]

    outputs = []
    for mode in export_modes:
        exported_path = model.export(format="tflite", nms=True, **mode)
        output_path = OUTPUTS / Path(exported_path).name
        shutil.move(exported_path, output_path)
        outputs.append((mode, output_path))

    for mode, output_path in outputs:
        print(f"Output {mode} tflite_model to: ", output_path)
