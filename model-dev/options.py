import click

from const import DEFAULT_YOLO_SIZE, YOLO_MODEL_SIZES


def yolo_model_size():
    return click.option(
        "--model-size",
        type=click.Choice(list(YOLO_MODEL_SIZES)),
        default=DEFAULT_YOLO_SIZE,
        help="YOLO model size (n=nano, s=small, etc.)",
    )
