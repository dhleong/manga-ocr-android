"""
Convert COCO annotations from manga109-segmentation to YOLO format for training.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Literal, Tuple, cast

from const import OUTPUTS
from train.dataset import download_manga109s, download_segmentation_annotations

YoloSplit = Literal["train", "val"]

# We don't need these; could maybe even skip onomatopoeia (COO)
DENIED_CATEGORIES = {"panel"}


def load_coco_annotations(annotations_path: Path) -> Dict:
    """Load COCO annotations from file."""
    with open(annotations_path, "r") as f:
        return json.load(f)


def convert_bbox_to_yolo(
    bbox: List[float], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert COCO bbox [x, y, width, height] to YOLO format
    Returns (x_center, y_center, width, height) normalized to 0-1
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height


def convert_coco_to_yolo(
    *,
    manga109s_path: Path,
    coco_data: Dict,
    output_dir: Path,
    split: YoloSplit,
    min_annotations_per_image: int = 1,
) -> Tuple[int, int, Dict[int, str]]:
    """
    Convert COCO annotations to YOLO format.
    Returns (num_images, num_annotations, num_classes)
    """
    input_base = manga109s_path / "images"
    assert input_base.exists(), f"input path did not exist: {input_base}"

    output_dir = Path(output_dir)
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split

    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Build image id to info mapping
    image_info = {img["id"]: img for img in coco_data["images"]}

    # Build category id to index mapping
    # COCO categories: 1=text, 2=onomatopoeia, 3=bubble, 4=panel
    categories = coco_data["categories"]
    cat_id_to_idx = {}
    for idx, cat in enumerate(categories):
        if cat["name"] not in DENIED_CATEGORIES:
            cat_id_to_idx[cat["id"]] = idx

    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    num_images_processed = 0
    num_annotations_processed = 0

    # Process each image
    for img_id, img_info in image_info.items():
        file_name = cast(str, img_info["file_name"])
        source_path = input_base / file_name

        # Check if image exists in Manga109s
        if not source_path.exists():
            continue

        # Copy image to output
        dest_image_path = images_dir / source_path.name
        if not dest_image_path.exists():
            shutil.copy2(source_path, dest_image_path)

        # Get annotations for this image
        annotations = annotations_by_image.get(img_id, [])

        # Filter out images with too few annotations
        if len(annotations) < min_annotations_per_image:
            # Clean up copied image
            if dest_image_path.exists():
                dest_image_path.unlink()
            continue

        # Convert annotations to YOLO format
        yolo_lines = []
        img_width = img_info["width"]
        img_height = img_info["height"]

        for ann in annotations:
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_idx:
                continue

            class_idx = cat_id_to_idx[cat_id]
            bbox = ann["bbox"]

            x_center, y_center, width, height = convert_bbox_to_yolo(
                bbox, img_width, img_height
            )

            # Validate normalized values
            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                continue

            yolo_lines.append(
                f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        # Write label file
        if yolo_lines:
            label_path = labels_dir / f"{dest_image_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

            num_images_processed += 1
            num_annotations_processed += len(yolo_lines)

    # Create dataset YAML
    class_names = {
        cat_id_to_idx[cat["id"]]: cat["name"]
        for cat in categories
        if cat["name"] not in DENIED_CATEGORIES
    }
    return num_images_processed, num_annotations_processed, class_names


def coco_to_yolo(recreate: bool = False):
    """Convert COCO annotations to YOLO format."""

    output_dir = OUTPUTS / "yolo-coco-dataset"
    yaml_path = output_dir / "dataset.yaml"
    if yaml_path.exists() and not recreate:
        return
    elif yaml_path.exists():
        print("Recreating...")

    print("Fetching images")
    # FIXME: probably, move this dir entry into the download_
    manga109s_dir = download_manga109s() / "Manga109s_released_2026_05_21"

    print("Fetching segmentation annotations...")
    seg_dir = download_segmentation_annotations()

    splits: List[Tuple[YoloSplit, str]] = [("train", "train"), ("val", "validation")]
    class_names: Dict[int, str] = {}
    for split, filename in splits:
        annotations_path = seg_dir / f"{filename}.coco.json"
        if not annotations_path.exists():
            print(f"Error: Annotations not found at {annotations_path}")
            return

        print(f"Loading COCO annotations from {annotations_path}")
        coco_data = load_coco_annotations(annotations_path)

        print("Converting COCO to YOLO format...")
        num_images, num_annotations, class_names = convert_coco_to_yolo(
            manga109s_path=manga109s_dir,
            coco_data=coco_data,
            output_dir=output_dir,
            split=split,
        )

        print(f"Images ({split}) processed: {num_images}")
        print(f"Annotations ({split}) processed: {num_annotations}")
        print(f"Output directory: {output_dir}")

    if not class_names:
        raise ValueError("Failed to load class names")

    yaml_content = f"""
path: {output_dir}
train: train/images
val: val/images

names:
""".lstrip()
    for idx, name in class_names.items():
        yaml_content += f"  {idx}: {name}\n"

    yaml_path.write_text(yaml_content)
