import shutil
from pathlib import Path

from const import OUTPUTS


def train_yolo(
    *,
    dataset_dir: Path,
    model_size: str,
    epochs: int,
    imgsz: int,
    batch_size: int,
    retrain: bool = False,
):
    """Train YOLO model and export to TFLite."""

    from ultralytics import YOLO

    project_dir = OUTPUTS / "yolo-coco-training"
    model_dir = project_dir / f"manga109-coco-{model_size}"
    best_model_path = model_dir / "weights" / "best.pt"

    if not best_model_path.exists() or retrain:
        print(f"Training YOLO{model_size} on {dataset_dir}")

        # Check dataset exists
        yaml_path = dataset_dir / "dataset.yaml"
        if not yaml_path.exists():
            print(f"Error: Dataset YAML not found at {yaml_path}")
            return

        # Load YOLO model
        model_name = f"yolov8{model_size}"
        print(f"Loading model {model_name}...")
        model = YOLO(model_name + ".pt")

        # Train
        print(f"Training for {epochs} epochs...")
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=str(project_dir),
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
):
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
    output_path = OUTPUTS / f"manga109-yolo{model_size}.tflite"

    shutil.move(str(export_path), str(output_path))

    print(f"TFLite model exported to {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print(f"Model: {output_path}")
    return output_path


def build_yolo(
    *,
    dataset_dir: Path,
    model_size: str,
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
    )
