import click
import download


def _fetch():
    dir_path = download.hf_unzip(
        "hal-utokyo/Manga109-s",
        "Manga109s_released_2023_12_07.zip",
        repo_type="dataset",
    )
    return dir_path


@click.command()
def train():
    print("Preparing to train...")
    from train.dataset import build_tf_dataset

    dir_path = _fetch()
    build_tf_dataset(dir_path)
