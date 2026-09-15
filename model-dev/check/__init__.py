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
    assert isinstance(results, list)
    for result in results:
        # type chekcers struggling...:
        if not hasattr(result, "boxes") or not hasattr(result, "show"):
            raise ValueError("Expected Results object; got", result)
        print(result.__dict__)
        print(result.boxes)
        result.show()
        break
