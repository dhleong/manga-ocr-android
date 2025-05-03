import click
import download


@click.group()
def check(): ...


@check.command()
def yolo():
    yolov8 = download.hf("ogkalu/manga-text-detector-yolov8s", "manga-text-detector.pt")
    import ultralytics

    model = ultralytics.YOLO(str(yolov8))
    results = model(
        "https://www.21-draw.com/wp-content/uploads/2022/12/what-is-manga.jpg"
    )
    results[0].show()
