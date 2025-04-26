from pathlib import Path

import quant
from huggingface_hub import hf_hub_download

KOHARU = "mayocream/koharu"


def _download(repo: str, file: str):
    print(f"Downloading {repo}/{file}...")
    hf_hub_download(repo, file, local_dir="./model-dev")


def download_base_models():
    _download(KOHARU, "manga-ocr.onnx")
    _download(KOHARU, "comictextdetector.onnx")


if __name__ == "__main__":
    download_base_models()

    processed = quant.preprocess(Path("model-dev/comictextdetector.onnx"))
    quant.dynamic(processed)
