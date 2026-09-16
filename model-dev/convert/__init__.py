import click


@click.group()
def convert(): ...


@convert.command()
@click.option("--recreate", is_flag=True, default=False)
def coco_to_yolo(recreate: bool):
    from convert.coco_to_yolo import coco_to_yolo as convert

    path = convert(recreate=recreate)
    print(path)


@convert.command()
@click.option("--recreate", is_flag=True, default=False)
def build_yolo_dataset(recreate: bool):
    from train import dataset

    path = dataset.build_yolo_dataset(recreate=recreate)
    print(path)


@convert.command()
def manga_ocr():
    from convert.mangaocr import convert

    convert()
