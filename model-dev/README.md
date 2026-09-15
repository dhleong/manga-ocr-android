# Model Dev

## Getting started

[Install uv][1]:

```
brew install uv
```

Run the entrypoint command:

```
uv run model-dev
```

The primary models in use can be generated with:

```
uv run model-dev convert ogkalu-yolo
uv run model-dev convert manga-ocr
```

The first builds a TFLite-compatible model based on [YOLO][yolo] by converting an [existing model][ogkalu] I found on [huggingface][hf].

The second splits the Encoder-Decoder model from [manga-ocr][manga-ocr] and quantizes them into [separate models][split-pr] for maximum speed.

## Activating the venv

Helpful for getting editors to find dependencies:

```
uv venv  # You only need to do this once, to create the venv
source .venv/bin/activate
```

[install-uv]: https://docs.astral.sh/uv/getting-started/installation
[hf]: https://huggingface.co
[manga-ocr]: https://github.com/kha-white/manga-ocr
[ogkalu]: https://huggingface.co/ogkalu/manga-text-detector-yolov8s
[split-pr]: https://github.com/dhleong/manga-ocr-android/pull/2
[yolo]: https://docs.ultralytics.com/models/yolov8/
