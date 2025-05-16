import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import batched, groupby
from pathlib import Path
from typing import Any, Dict, Iterable, List

import click
import download
from const import OUTPUTS


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


def generate_grouped_labeleds(manga109s_dir: Path):
    return groupby(generate_labeleds(manga109s_dir), lambda lbl: lbl.page_path)


def build_tf_dataset(manga109s_dir: Path, image_dimen=224):
    import tensorflow as tf

    def generate():
        by_path = generate_grouped_labeleds(manga109s_dir)
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


def onnx_calibration_reader(manga109s_dir: Path, *, sample: bool = True):
    import numpy as np
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    vocab = load_vocab()
    labeleds = generate_labeleds(manga109s_dir)
    if sample:
        labeleds = list()
        sampled = iter(random.sample(labeleds, 500))
    else:
        sampled = labeleds

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


def build_yolo_dataset(*, recreate: bool = False, size: int = 8):
    from PIL import Image

    assert size % 2 == 0, "Dataset must be an even size"

    root = OUTPUTS / "yolo-dataset-root"
    yaml = root / "manga109s.yaml"
    if yaml.exists() and not recreate:
        return yaml

    manga109s_dir = download_manga109s()
    # NOTE: Something sample() does (or batched? unclear) causes an
    # iteration over the labeleds, so we eagerly materialize them into
    # a list here:
    labeleds = [
        (path, list(items)) for path, items in generate_grouped_labeleds(manga109s_dir)
    ]
    sampled = iter(random.sample(labeleds, size))
    batches = batched(sampled, 2)

    def _copy_image_to(image_path: Path, parent: Path):
        name = f"{image_path.parent.name}-{image_path.name}"
        dest = parent / name
        if not dest.exists():
            shutil.copyfile(image_path, dest)
        return name

    def _append_label_to(labeled: Labeled, dest_image: Path):
        dest = dest_image.with_suffix(".txt")

        image = Image.open(str(labeled.page_path))
        xn, yn, xm, ym = labeled.box
        xn /= image.width
        xm /= image.width
        yn /= image.height
        ym /= image.height

        line = f"0 {xn} {yn} {xm} {ym}\n"
        with open(dest, "a") as fp:
            fp.write(line)

    images_train = root / "images" / "train"
    images_val = root / "images" / "val"
    labels_train = root / "labels" / "train"
    labels_val = root / "labels" / "val"

    for path in [images_train, images_val, labels_train, labels_val]:
        if recreate:
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    with click.progressbar(batches) as bar:
        for (_, train_group), (_, val_group) in bar:
            for train in train_group:
                train_name = _copy_image_to(train.page_path, images_train)
                _append_label_to(train, labels_train / train_name)
            for val in val_group:
                val_name = _copy_image_to(val.page_path, images_val)
                _append_label_to(val, labels_val / val_name)

    yaml.write_text(f"""
path: {root}
train: images/train
val: images/val

names:
  0: text
""")

    return yaml
