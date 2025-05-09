import click


@click.group()
def reauthor(): ...


@reauthor.command()
def manga_ocr():
    from reauthor.mangaocr import reauthor

    reauthor()
