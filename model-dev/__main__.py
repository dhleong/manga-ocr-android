from pathlib import Path

from huggingface_hub import hf_hub_download

KOHARU = "mayocream/koharu"

TFLITE_PATH = Path("model-dev/mangaocr.tflite").absolute()


def _download(repo: str, file: str):
    print(f"Downloading {repo}/{file}...")
    hf_hub_download(repo, file, local_dir="./model-dev")


def download_base_models():
    _download(KOHARU, "manga-ocr.onnx")
    _download(KOHARU, "comictextdetector.onnx")


def quant_cmd(_):
    print("Preparing to quantize from koharu")
    import quant

    download_base_models()

    processed = quant.preprocess(Path("model-dev/comictextdetector.onnx"))
    quant.dynamic(processed)


def convert_cmd(_):
    print("Preparing to convert manga-ocr-base")
    import convert

    convert.run(output_path=TFLITE_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model dev")
    subparsers = parser.add_subparsers(required=True)

    quant_args = subparsers.add_parser("quant", help="Perform model quantization")
    quant_args.set_defaults(func=quant_cmd)

    convert_args = subparsers.add_parser("convert", help="Perform model conversion")
    convert_args.set_defaults(func=convert_cmd)

    args = parser.parse_args()
    args.func(args)
