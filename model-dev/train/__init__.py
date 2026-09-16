import shutil
from pathlib import Path

import click

import options
from const import OUTPUTS, YoloModelSize
from train import dataset


@click.group()
def train(): ...


@train.command()
def manga109s():
    print("Preparing to train...")

    dir_path = dataset.download_manga109s()
    dataset.build_tf_dataset(dir_path)


@train.command()
@options.yolo_model_size()
@click.option("--epochs", type=int, default=150, help="Number of training epochs")
@click.option("--imgsz", type=int, default=640, help="Image size for training")
@click.option("--batch-size", type=int, default=16, help="Batch size")
def yolo_coco(
    model_size: YoloModelSize,
    epochs: int,
    imgsz: int,
    batch_size: int,
):
    """Train YOLO model on COCO-converted data and export to TFLite."""
    from convert.coco_to_yolo import coco_to_yolo as prepare_dataset
    from train.yolo_coco import build_yolo

    print("Preparing yolo-coco dataset...")
    dataset_dir = prepare_dataset()
    assert dataset_dir, "Failed to produce training dataset"
    build_yolo(
        dataset_dir=dataset_dir,
        model_size=model_size,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
    )
