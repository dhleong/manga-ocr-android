import json
from pathlib import Path
from typing import Optional

import click
import download
import torch

DEFAULT_PATH = "https://www.21-draw.com/wp-content/uploads/2022/12/what-is-manga.jpg"


@click.group()
def check(): ...


@check.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def yolo(path: Optional[Path] = None):
    yolov8 = download.hf("ogkalu/manga-text-detector-yolov8s", "manga-text-detector.pt")
    import ultralytics

    model = ultralytics.YOLO(str(yolov8))
    results = model(path or DEFAULT_PATH)

    assert isinstance(results, list)
    result = results[0]
    # type checkers struggling...:
    if isinstance(result, torch.Tensor):
        raise ValueError("Expected Results object; got", result)
    print(result.__dict__)
    print(result.boxes)
    result.show()


@check.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
def koharu(path: Optional[Path] = None):
    import importlib.util

    import numpy as np
    from PIL import Image

    weights = download.hf(
        "mayocream/koharu-layout-rfdetr-seg-2xl-1152",
        "model.safetensors",
        outputs_path="koharu-seg",
    )
    loader_path = download.hf(
        "mayocream/koharu-layout-rfdetr-seg-2xl-1152",
        "load_model.py",
        outputs_path="koharu-seg",
    )
    spec = importlib.util.spec_from_file_location("koharu_layout_loader", loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    model = loader.load_model(weights)

    image = Image.open(path or DEFAULT_PATH).convert("RGB")
    detections = model.predict(
        image,
        threshold=0.20,
        shape=(1152, 1152),
        include_source_image=False,
    )

    class_thresholds = {0: 0.25, 1: 0.20, 2: 0.50, 3: 0.50}
    keep = np.asarray(
        [
            score >= class_thresholds[int(class_id)]
            for class_id, score in zip(detections.class_id, detections.confidence)
        ]
    )
    detections = detections[keep]
    print(detections)
