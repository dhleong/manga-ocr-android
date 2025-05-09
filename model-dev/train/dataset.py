import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Iterable, List

import click
import download


@dataclass
class Labeled:
    page_path: Path
    box: tuple[float, float, float, float]
    """Bounding box in xyxy format (left, top, right, bottom)"""
    text: str


class Vocab:
    def __init__(self, token_to_id: Dict[str, int], id_to_token: List[str]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

    def __len__(self) -> int:
        return len(self.token_to_id)

    def tokenize(self, s: str) -> List[int]:
        token_ids = [self.token_to_id[ch] for ch in s if ch in self.token_to_id]
        return [2] + token_ids + [2]


def _generate_labeled_for_file(file: Path):
    try:
        tree = ET.parse(file).getroot()
    except ET.ParseError as e:
        raise Exception(f"Error parsing {file}", e)
    images_root = file.parent.parent / "images" / file.with_suffix("").name
    for page in tree.findall("./pages/page"):
        for frame in page.findall("./text"):
            if not frame.text:
                continue

            page_index = int(page.attrib["index"])
            page_path = images_root / f"{page_index:03d}.jpg"
            xmin = float(frame.attrib["xmin"])
            ymin = float(frame.attrib["ymin"])
            xmax = float(frame.attrib["xmax"])
            ymax = float(frame.attrib["ymax"])
            box = (xmin, ymin, xmax, ymax)
            yield Labeled(page_path=page_path, box=box, text=frame.text)


def _generate_tf_for_group(path: Path, labeleds: Iterable[Labeled], image_dimen: int):
    import tensorflow as tf

    image = tf.io.read_file(str(path))
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    for labeled in labeleds:
        cropped = tf.image.crop_and_resize(
            image,
            boxes=[labeled.box],
            box_indices=[0],
            crop_size=(image_dimen, image_dimen),
        )
        yield (cropped, labeled.text)


def generate_labeleds(manga109s_dir: Path):
    def _is_valid_path(path: Path):
        return not path.name.startswith(".")

    annotation_files = list(
        filter(_is_valid_path, manga109s_dir.glob("**/annotations/*.xml"))
    )
    with click.progressbar(annotation_files) as bar:
        for file in bar:
            yield from _generate_labeled_for_file(file)


def build_tf_dataset(manga109s_dir: Path, image_dimen=224):
    import tensorflow as tf

    def generate():
        by_path = groupby(generate_labeleds(manga109s_dir), lambda lbl: lbl.page_path)
        for path, labeleds in by_path:
            yield from _generate_tf_for_group(path, labeleds, image_dimen=image_dimen)

    return tf.data.Dataset.from_generator(
        generate,
        output_types=(tf.float32, tf.float32),
        output_shapes=([32, image_dimen, image_dimen, 3], [32, 5]),
    )


def _preprocess_onnx(labeled: Labeled):
    import numpy as np
    from PIL import Image

    image = Image.open(str(labeled.page_path))

    # Grayscale
    image = image.convert("L").convert("RGB")
    image = image.crop(labeled.box)
    image = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
    image = np.array(image, dtype=np.float32)

    # normalize into [-1, 1]
    image /= 255
    image = (image - 0.5) / 0.5

    # reshape from (224, 224, 3) to (3, 224, 224)
    image = image.transpose((2, 0, 1))

    # add batch size
    image = image[None]

    return image


def onnx_calibration_reader(manga109s_dir: Path):
    import random

    import numpy as np
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    vocab = load_vocab()
    labeleds = list(generate_labeleds(manga109s_dir))
    sampled = iter(random.sample(labeleds, 500))

    class Manga109sCalibrationReader(CalibrationDataReader):
        def get_next(self) -> Dict[str, Any]:
            labeled = next(sampled, None)
            if not labeled:
                # The types on the superclass don't indicate it, but this
                # is legit:
                return None  # type: ignore

            token_ids = np.array([vocab.tokenize(labeled.text)])
            return {"image": _preprocess_onnx(labeled), "token_ids": token_ids}

    return Manga109sCalibrationReader()


def load_vocab() -> Vocab:
    vocab_file = download.hf(
        "kha-white/manga-ocr-base",
        "vocab.txt",
    )

    with open(str(vocab_file), "r") as fp:
        id_to_token = [line.rstrip() for line in fp.readlines()]

    token_to_id = {token: i for i, token in enumerate(id_to_token)}
    return Vocab(token_to_id, id_to_token)


def download_manga109s():
    return download.hf_unzip(
        "hal-utokyo/Manga109-s",
        "Manga109s_released_2023_12_07.zip",
        repo_type="dataset",
    )
