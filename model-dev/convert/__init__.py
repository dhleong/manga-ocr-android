import shutil
from pathlib import Path

import click
import download
from const import OUTPUTS


@click.group()
def convert(): ...


@convert.command()
@click.option("--recreate", is_flag=True, default=False)
def build_yolo_dataset(recreate: bool):
    from train import dataset

    path = dataset.build_yolo_dataset(recreate=recreate)
    print(path)


@convert.command()
@click.option("--with-data", is_flag=True, default=False)
def ogkalu_yolo(with_data: bool):
    from train import dataset

    dataset = dataset.build_yolo_dataset() if with_data else None
    yolov8 = download.hf("ogkalu/manga-text-detector-yolov8s", "manga-text-detector.pt")

    import ultralytics

    model = ultralytics.YOLO(str(yolov8))
    # exported = model.export(format="onnx")
    # assert exported, "Failed to convert to onnx"
    # processed = ops.preprocess(Path(exported))
    # ops.dynamic(processed)

    export_modes = [
        # dict(),
        # dict(half=True),
        dict(int8=True),
    ]

    outputs = []
    for mode in export_modes:
        exported_path = model.export(
            format="tflite", nms=True, data=str(dataset), **mode
        )
        output_path = OUTPUTS / Path(exported_path).name
        if with_data:
            output_path = output_path.with_suffix(".with_data.tflite")

        shutil.move(exported_path, output_path)
        outputs.append((mode, output_path))

    for mode, output_path in outputs:
        print(f"Output {mode} tflite_model to: ", output_path)


@convert.command()
def manga_ocr():
    from convert.mangaocr import convert

    convert()
