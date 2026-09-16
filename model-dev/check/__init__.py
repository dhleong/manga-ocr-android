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
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from ultralytics import YOLO

    root = Path(
        snapshot_download(
            "mayocream/koharu-yolo26s",
            allow_patterns=["*.safetensors", "*.yaml", "*.json"],
        )
    )
    config = json.loads((root / "config.json").read_text())
    model = YOLO(root / "yolo26s-seg.yaml", task="segment")
    assert model.model
    assert not isinstance(model.model, str)
    model.model.load_state_dict(load_file(root / "model.safetensors"), strict=True)
    model.model.names = {int(key): value for key, value in config["names"].items()}

    results = model.predict(path or DEFAULT_PATH, imgsz=1280, conf=0.25)
    assert isinstance(results, list)
    result = results[0]
    # type checkers struggling...:
    if isinstance(result, torch.Tensor):
        raise ValueError("Expected Results object; got", result)
    print(result.__dict__)
    print(result.boxes)
    result.show()
