# Model Dev

## Getting started

Install poetry and setup the venv

```
gradle bootstrapModel
```

Run the entrypoint command:

```
poetry run python model-dev
```

The primary models in use can be generated with:

```
poetry run python model-dev convert ogkalu-yolo
poetry run python model-dev convert manga-ocr
```

The first builds a TFLite-compatible model based on [YOLO][yolo] by converting an [existing model][ogkalu] I found on [huggingface][hf].

The second splits the Encoder-Decoder model from [manga-ocr][manga-ocr] and quantizes them into [separate models][split-pr] for maximum speed.

## Activating the venv

Helpful for getting editors to find dependencies:

```
eval $(poetry env activate)
```

[hf]: https://huggingface.co
[manga-ocr]: https://github.com/kha-white/manga-ocr
[ogkalu]: https://huggingface.co/ogkalu/manga-text-detector-yolov8s
[split-pr]: https://github.com/dhleong/manga-ocr-android/pull/2
[yolo]: https://docs.ultralytics.com/models/yolov8/
