from pathlib import Path
from typing import Optional

import click
import download
import torch


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
    results = model(
        path or "https://www.21-draw.com/wp-content/uploads/2022/12/what-is-manga.jpg"
    )

    assert isinstance(results, list)
    for result in results:
        # type checkers struggling...:
        if isinstance(result, torch.Tensor):
            raise ValueError("Expected Results object; got", result)
        print(result.__dict__)
        print(result.boxes)
        result.show()
        break
