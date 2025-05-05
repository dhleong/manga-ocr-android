import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Iterable, List

import click


@dataclass
class Labeled:
    page_path: Path
    box: List[int]
    """Bounding box in xyxy format (left, top, right, bottom)"""
    text: str


def _generate_labeled_for_file(file: Path):
    tree = ET.parse(file).getroot()
    images_root = file.parent.parent / "images" / file.with_suffix("").name
    for page in tree.findall("./pages/page"):
        for frame in page.findall("./text"):
            if not frame.text:
                continue

            page_path = images_root / f"{frame.attrib['page_index']:03d}.jpg"
            xmin = int(frame.attrib["xmin"])
            ymin = int(frame.attrib["ymin"])
            xmax = int(frame.attrib["xmax"])
            ymax = int(frame.attrib["ymax"])
            box = [xmin, ymin, xmax, ymax]
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
    annotation_files = list(manga109s_dir.glob("**/annotations/*.xml"))
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
