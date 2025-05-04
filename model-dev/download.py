from const import OUTPUTS
from huggingface_hub import hf_hub_download

KOHARU = "mayocream/koharu"


def hf(repo: str, file: str):
    print(f"Downloading {repo}/{file}...")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo, file, local_dir=OUTPUTS)
    return (OUTPUTS / file).absolute()


def download_base_models():
    hf(KOHARU, "manga-ocr.onnx")
    hf(KOHARU, "comictextdetector.onnx")
