from pathlib import Path
from typing import Literal

__DEV_ROOT__ = Path(__file__).parent.absolute()

INPUTS = __DEV_ROOT__ / "inputs"
OUTPUTS = __DEV_ROOT__ / "outputs"

YoloModelSize = Literal["n", "s", "m", "l", "x"]
YOLO_MODEL_SIZES = {"n", "s", "m", "l", "x"}
DEFAULT_YOLO_SIZE = "s"
