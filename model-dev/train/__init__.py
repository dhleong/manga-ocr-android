import click
from train import dataset


@click.command()
def train():
    print("Preparing to train...")

    dir_path = dataset.download_manga109s()
    dataset.build_tf_dataset(dir_path)
