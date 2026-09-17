import shutil
from pathlib import Path

from const import OUTPUTS, YoloModelSize

PROJECT_DIR = OUTPUTS / "yolo-coco-training"

# YOLO_VERSION = "v8"
YOLO_VERSION = "26"


def get_model_path(model_size: YoloModelSize):
    model_dir = PROJECT_DIR / f"manga109-coco-{model_size}"
    return model_dir / "weights" / "best.pt"


def train_yolo(
    *,
    dataset_dir: Path,
    model_size: YoloModelSize,
    epochs: int,
    imgsz: int,
    batch_size: int,
    retrain: bool = False,
):
    """Train YOLO model and export to TFLite."""

    from ultralytics import YOLO

    best_model_path = get_model_path(model_size)
    if not best_model_path.exists() or retrain:
        if not retrain:
            print(f"Model-{model_size} not found @ {best_model_path}")

        print(f"Training YOLO{model_size} on {dataset_dir}")

        # Check dataset exists
        yaml_path = dataset_dir / "dataset.yaml"
        if not yaml_path.exists():
            print(f"Error: Dataset YAML not found at {yaml_path}")
            return

        # Load a pretrained model
        model_name = f"yolo{YOLO_VERSION}{model_size}"
        model_filename = f"{model_name}.pt"
        print(f"Loading model {model_name}...")
        model_path = OUTPUTS / model_filename
        if not model_path.exists():
            # This constructor downloads a pretrained model to the current dir
            print(f"Downloading pretrained {model_name}")
            model = YOLO(model_filename)
            Path(model_filename).rename(model_path)
        model = YOLO(model_path)

        # Train
        print(f"Training for {epochs} epochs...")
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device="mps",  # enables apple silicon
            project=str(PROJECT_DIR),
            name=f"manga109-coco-{model_size}",
            exist_ok=True,
            verbose=True,
        )

        # Get best model path
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        if not best_model_path.exists():
            best_model_path = Path(results.save_dir) / "weights" / "last.pt"

        print(f"Best model saved to {best_model_path}")

    print("Training complete!")
    return best_model_path


def export_to_tflite(
    *,
    model_size: str,
    imgsz: int,
    model_path: Path,
    retrain: bool,
):
    output_path = OUTPUTS / f"manga109-yolo{model_size}.tflite"
    if output_path.exists() and not retrain:
        print(f"Found tflite model @{output_path}")
        return output_path

    from ultralytics import YOLO

    # Export to TFLite
    print("Exporting to TFLite...")

    # Create model for export
    export_model = YOLO(str(model_path))

    # Export with int8 quantization
    # Note: YOLO export to TFLite with quantization requires representative data
    # For now, export without quantization and let user quantize
    export_path = export_model.export(
        format="tflite",
        imgsz=imgsz,
        dynamic=False,
        nms=True,
        # int8=True,  # Requires calibration data
    )

    export_path = Path(export_path)

    shutil.move(str(export_path), str(output_path))

    print(f"TFLite model exported to {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print(f"Model: {output_path}")
    return output_path


def build_yolo(
    *,
    dataset_dir: Path,
    model_size: YoloModelSize,
    epochs: int,
    imgsz: int,
    batch_size: int,
    retrain: bool = False,
):
    model_pt_path = train_yolo(
        dataset_dir=dataset_dir,
        model_size=model_size,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        retrain=retrain,
    )
    assert model_pt_path, "No model output"

    export_to_tflite(
        model_size=model_size,
        imgsz=imgsz,
        model_path=model_pt_path,
        retrain=retrain,
    )
