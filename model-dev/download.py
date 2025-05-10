import zipfile
from typing import Literal, Optional

import huggingface_hub as hub
from const import OUTPUTS

KOHARU = "mayocream/koharu"


def hf(repo: str, file: str, repo_type: Optional[Literal["dataset", "model"]] = None):
    for i in range(0, 1):
        try:
            print(f"Downloading {repo}/{file}...")
            OUTPUTS.mkdir(parents=True, exist_ok=True)
            hub.hf_hub_download(repo, file, local_dir=OUTPUTS, repo_type=repo_type)
            return (OUTPUTS / file).absolute()
        except (hub.errors.RepositoryNotFoundError, hub.errors.GatedRepoError):
            if i > 0:
                print("No auth after login; giving up :(")
                raise

            print(f"Unable to access {file} in {repo}; you may need to auth!")
            print("whoami:", hub.whoami())
            print("You can get a token from: https://huggingface.co/settings/tokens")
            hub.login(new_session=False)
            print("Logged in; trying again...")


def hf_unzip(
    repo: str,
    file: str,
    repo_type: Optional[Literal["dataset", "model"]] = None,
    delete_zip: bool = True,
):
    """
    Download the `file` (see `hf`) and unzip it, deleting the zip file afterward.
    """
    dir_path = (OUTPUTS / file).with_suffix("")
    if not dir_path.exists():
        zip_path = hf(repo, file, repo_type)
        assert zip_path

        print(f"Unzipping {zip_path}")
        with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
            zip_ref.extractall(str(dir_path))

        if delete_zip:
            zip_path.unlink()

    return dir_path


def download_base_models():
    hf(KOHARU, "manga-ocr.onnx")
    hf(KOHARU, "comictextdetector.onnx")
